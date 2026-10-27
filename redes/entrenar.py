"""Entrena redes con neuroevolución (Deep GA): población de redes, se quedan las mejores y se mutan sus pesos.

    uv run python -m redes.entrenar --nombre red01 --workers 10
    uv run python -m redes.entrenar --nombre red01 --continuar [--generaciones 2000]

Cada generación todas las redes juegan los mismos `escenarios` (el comienzo de la sala y otros con una espera al
azar, nuevos cada generación) y su fitness es el promedio. La élite pasa sin cambios pero se vuelve a evaluar:
así una red que solo memorizó un escenario no se queda arriba. Opciones = campos de `redes.config.Config`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import fields

import numpy as np

from iwbtgb_rl import IWannaBeTheGBEnv

from . import almacen
from .config import Config
from .episodio import Escenario, Evaluacion, evaluar_en_worker, iniciar_worker, red_para
from .modelo import Cache

# No se pueden cambiar al continuar: definen la forma de la red o el juego que ve
FIJOS = ("start_room", "sala_objetivo", "difficulty", "seed_juego", "cuadros", "filtros1", "filtros2", "oculta",
         "entorno", "desde", "desde_gen")
# Lo que se copia de la corrida `desde` (la red tiene que ser igual para seguir su linaje)
FORMA_RED = ("cuadros", "filtros1", "filtros2", "oculta", "frames_por_decision", "vista")


def preparar_corrida(cfg: Config, carpeta) -> dict:
    """Ids de las rooms y mapa de distancias a la salida (el mismo cálculo que el GA, `ga/distancia.py`)."""
    from ga import distancia
    ruta = carpeta / "distancia.npz"
    if cfg.room_entrenamiento >= 0 and ruta.exists():
        return distancia.cargar(ruta)
    mapa = resolver_sala(cfg)
    distancia.guardar(ruta, mapa)
    return mapa


def resolver_sala(cfg: Config) -> dict:
    """Completa en `cfg` los ids y el tamaño de `start_room` y `sala_objetivo`, y arma su mapa de distancias."""
    from ga import distancia
    env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                           seed=cfg.seed_juego)
    try:
        info = env.emu.info()
        cfg.room_entrenamiento, cfg.room_ancho, cfg.room_alto = info.id, info.width, info.height
        cfg.room_objetivo = int(env.emu.exec(f"return {cfg.sala_objetivo}"))
        jugador = env.emu.query("player")[0]
        return distancia.construir(env.emu, cfg.room_objetivo, info.width, info.height, (jugador.x, jugador.y))
    finally:
        env.close()


def anotar(carpeta, texto: str) -> None:
    """A la terminal (o a la app) y a `log.txt` de la corrida."""
    print(texto, flush=True)
    with open(carpeta / "log.txt", "a") as f:
        f.write(texto + "\n")


def acciones_grabadas(evals: list[Evaluacion]) -> np.ndarray:
    """Acciones del primer escenario de cada red, rellenadas con 255 (fin)."""
    largo = max(len(e.partidas[0].acciones) for e in evals)
    m = np.full((len(evals), max(largo, 1)), 255, np.uint8)
    for i, e in enumerate(evals):
        a = e.partidas[0].acciones
        m[i, :len(a)] = a
    return m


def elegir_padres(cfg: Config, orden: list[int], evals: list[Evaluacion]) -> list[int]:
    """Los `padres` mejores, primero el mejor de cada nicho (dónde terminó el primer escenario)."""
    vistos, primero, resto = set(), [], []
    for i in orden:
        x, y = evals[i].partidas[0].final
        n = (evals[i].llego, int(x // cfg.nicho_px), int(y // cfg.nicho_px))
        (resto if n in vistos else primero).append(i)
        vistos.add(n)
    return (primero + resto)[:cfg.padres]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--continuar", action="store_true", help="seguir desde la última generación guardada")
    p.add_argument("--sobrescribir", action="store_true", help="borrar la corrida anterior con el mismo nombre")
    for f in fields(Config):
        if f.name in ("vista", "room_entrenamiento", "room_objetivo", "room_ancho", "room_alto", "genoma_inicial"):
            continue
        if f.name == "nombre":
            p.add_argument("--nombre", default=f.default)
            continue
        tipo = {"int": int, "float": float, "str": str}.get(str(f.type).split(" ")[0], str)
        p.add_argument(f"--{f.name.replace('_', '-')}", type=tipo, default=None)
    args = p.parse_args()

    carpeta = almacen.dir_corrida(args.nombre)
    hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
    if hechas and not args.continuar:
        if not args.sobrescribir:
            sys.exit(f"{carpeta} ya tiene generaciones: usa --continuar o --sobrescribir")
        shutil.rmtree(carpeta)
        hechas = []
    cfg = almacen.cargar_config(carpeta) if hechas else Config(nombre=args.nombre)
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "nombre":
            if hechas and f.name in FIJOS and v != getattr(cfg, f.name):
                sys.exit(f"al continuar no se puede cambiar {f.name} (forma de la red o juego): crea otra corrida")
            setattr(cfg, f.name, v)
    if cfg.desde and not hechas:
        partir_de(cfg)
    carpeta.mkdir(parents=True, exist_ok=True)
    mapa = preparar_corrida(cfg, carpeta)
    almacen.guardar_config(carpeta, cfg)
    red = red_para(cfg)
    cache = Cache(16)

    rng = np.random.default_rng(cfg.seed + 1000 * (hechas[-1] + 1 if hechas else 0))
    if hechas:
        gen0 = hechas[-1] + 1
        anteriores = almacen.genomas(carpeta, hechas[-1])
        previa = almacen.cargar_generacion(carpeta, hechas[-1])
        genomas, linaje = siguiente(cfg, rng, anteriores, list(previa["fitness"]), previa)
    elif cfg.genoma_inicial:  # parte de la red de otra corrida: ella sin cambios + mutaciones suyas
        gen0 = 0
        base = almacen.genoma_de(cfg.genoma_inicial)
        semillas = rng.integers(2**62, size=cfg.poblacion)
        sigmas = np.full(cfg.poblacion, cfg.sigma, np.float32)
        sigmas[0] = 0.0
        genomas = [base if k == 0 else base + ((int(s), cfg.sigma),) for k, s in enumerate(semillas)]
        linaje = (np.full(cfg.poblacion, -1, np.int32), semillas.astype(np.int64), sigmas)
    else:
        gen0 = 0
        semillas = rng.integers(2**31, size=cfg.poblacion)
        genomas = [(int(s),) for s in semillas]
        linaje = (np.full(cfg.poblacion, -1, np.int32), semillas.astype(np.int64), np.zeros(cfg.poblacion, np.float32))

    gimnasio = cfg.entorno == "gimnasio"
    donde = (f"gimnasio: {cfg.escenarios} salas nuevas por generación, dificultad {cfg.gim_dificultad_min:g}–"
             f"{cfg.gim_dificultad_max:g}, hasta {cfg.max_segundos:.0f} s" if gimnasio else
             f"{cfg.escenarios} escenarios de hasta {cfg.max_segundos:.0f} s  room {cfg.room_entrenamiento} -> "
             f"{cfg.sala_objetivo} ({cfg.room_objetivo})")
    anotar(carpeta, f"corrida {carpeta}  población {cfg.poblacion}  padres {cfg.padres}  élite {cfg.elite}  "
                    f"sigma {cfg.sigma}  workers {cfg.workers}  red de {red.n_pesos} pesos  {donde}"
                    + (f"  partiendo de {cfg.desde}" if cfg.desde else ""))
    inicio_total = time.time()
    # Un hilo de cálculo por worker: si no, numpy (OpenBLAS) abre uno por núcleo en cada worker, la máquina queda
    # sobrecargada y algunos workers se trababan al cerrar. Los workers heredan estas variables al crearse.
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    ctx = mp.get_context("spawn")
    with ctx.Pool(cfg.workers, initializer=iniciar_worker, initargs=(cfg, mapa)) as pool:
        for gen in range(gen0, cfg.generaciones):
            t0 = time.time()
            escenarios = [Escenario(0)] + [Escenario(int(rng.integers(0, cfg.espera_max + 1)))
                                           for _ in range(cfg.escenarios - 1)]
            if gimnasio:  # salas nuevas cada generación (iguales para todas las redes): no puede memorizarlas
                escenarios = [Escenario(0, int(rng.integers(2**31)),
                                        float(rng.uniform(cfg.gim_dificultad_min, cfg.gim_dificultad_max)))
                              for _ in range(cfg.escenarios)]
            elif cfg.reintentos and cfg.espera_reintento:  # esperas al reaparecer, también nuevas cada generación
                for e in escenarios:
                    e.semilla = int(rng.integers(2**31))
            evals: list[Evaluacion | None] = [None] * len(genomas)
            # Hermanos juntos: el worker reconstruye los pesos del padre una vez y los demás salen de su caché
            tareas = sorted(range(len(genomas)), key=lambda i: int(linaje[0][i]))
            for i, ev in pool.imap_unordered(evaluar_en_worker, [(i, genomas[i], escenarios) for i in tareas],
                                             chunksize=max(1, min(8, len(genomas) // (4 * cfg.workers)))):
                evals[i] = ev
            fitness = np.array([e.fitness for e in evals], np.float32)
            orden = [int(i) for i in np.argsort(-fitness, kind="stable")]

            datos = {
                "padre": linaje[0], "semilla": linaje[1], "sigma": linaje[2], "fitness": fitness,
                "avance": np.array([e.avance for e in evals], np.float32),
                "avances": np.array([[q.avance for q in e.partidas] for e in evals], np.float32),
                "fines": np.array([[q.fin for q in e.partidas] for e in evals]),
                "final": np.array([e.partidas[0].final for e in evals], np.float32),
                "esperas": np.array([e.espera for e in escenarios], np.int32),
                "semillas": np.array([e.semilla for e in escenarios], np.int64),
                "muertes": np.array([[q.muertes for q in e.partidas] for e in evals], np.int16),
                "saves": np.array([[q.saves for q in e.partidas] for e in evals], np.int8),
                "quieto": np.array([[q.quieto for q in e.partidas] for e in evals], np.float32),
                "celdas": np.array([[q.celdas for q in e.partidas] for e in evals], np.int16),
                "acciones": acciones_grabadas(evals),
                # traza por frame del primer escenario de cada red, una tras otra (para verlas todas juntas)
                "traza_largo": np.array([len(e.partidas[0].traza) for e in evals], np.int64),
                "trazas": np.concatenate([e.partidas[0].traza for e in evals]),
            }
            if gimnasio:
                datos["gimnasio"] = np.array(1)
                datos["dificultades"] = np.array([e.dificultad for e in escenarios], np.float32)
                # las trazas de todas las salas: red por red, sala por sala (largos en `traza_largo`, (redes, salas))
                datos["traza_largo"] = np.array([[len(q.traza) for q in e.partidas] for e in evals], np.int64)
                datos["trazas"] = np.concatenate([q.traza for e in evals for q in e.partidas])
            almacen.guardar_generacion(carpeta, gen, datos)
            mejor = almacen.indice_mejor(datos)  # la que llega más lejos (no la de más fitness)
            ultima = gen == cfg.generaciones - 1
            info = {"gen": gen, "agente": mejor, "fitness": float(fitness[mejor]), "genoma": genomas[mejor]}
            pesos = red.pesos(genomas[mejor], cache)
            almacen.guardar_red(carpeta, "mejor", pesos, info)
            if gen % cfg.guardar_red_cada == 0 or ultima:
                almacen.guardar_red(carpeta, f"gen_{gen:04d}", pesos, info)

            fines = Counter(q.fin for e in evals for q in e.partidas)
            total = sum(fines.values())
            nichos = len({(e.llego, int(e.partidas[0].final[0] // cfg.nicho_px),
                           int(e.partidas[0].final[1] // cfg.nicho_px)) for e in evals})
            fila = {
                "gen": gen, "fitness_max": round(float(fitness.max()), 3), "fitness_media": round(float(fitness.mean()), 3),
                "avance_max": round(100 * float(datos["avances"].max()), 2),
                "avance_media": round(100 * float(datos["avance"].mean()), 2),
                "avance_mejor": round(100 * evals[mejor].avance, 2),
                "llegaron": sum(e.llego for e in evals), "nichos": nichos,
                "murio_pct": round(100 * fines["murió"] / total, 1),
                "estancado_pct": round(100 * fines["estancado"] / total, 1),
                "tiempo_pct": round(100 * fines["tiempo"] / total, 1),
                "salio_pct": round(100 * fines["salió"] / total, 1),
                "frames_medio": round(float(np.mean([q.frames for e in evals for q in e.partidas])), 1),
                "muertes_media": round(float(np.mean([q.muertes for e in evals for q in e.partidas])), 2),
                "saves_max": max(q.saves for e in evals for q in e.partidas),
                "saves_media": round(float(np.mean([q.saves for e in evals for q in e.partidas])), 2),
                "quieto_pct": round(100 * float(np.mean([50 * q.quieto / max(q.frames, 1)
                                                         for e in evals for q in e.partidas])), 1),
                "celdas_media": round(float(np.mean([q.celdas for e in evals for q in e.partidas])), 1),
                "celdas_max": max(q.celdas for e in evals for q in e.partidas),
                "salas_pct": round(100 * float(np.mean([q.fin == "llegó" for e in evals for q in e.partidas])), 1),
                "salas_mejor": sum(q.fin == "llegó" for q in evals[mejor].partidas),
                "segundos_reloj": round(time.time() - t0, 1),
            }
            almacen.agregar_metricas(carpeta, fila)
            fines_txt = (f"muertes {fila['muertes_media']:4.1f} quieto {fila['quieto_pct']:3.0f} % saves "
                         f"{fila['saves_media']:4.2f} (máx {fila['saves_max']})" if cfg.reintentos else
                         f"murió {fila['murio_pct']:4.0f} % estancado {fila['estancado_pct']:4.0f} %")
            if gimnasio:
                fines_txt = (f"salas superadas {fila['salas_pct']:4.1f} % (la mejor {fila['salas_mejor']}/"
                             f"{cfg.escenarios}) murió {fila['murio_pct']:3.0f} % estancado "
                             f"{fila['estancado_pct']:3.0f} %")
            elif cfg.r_explorar:
                fines_txt += f" celdas {fila['celdas_media']:5.1f} (máx {fila['celdas_max']})"
            anotar(carpeta, f"gen {gen:4d} | fitness máx {fila['fitness_max']:7.2f} media {fila['fitness_media']:7.2f} | "
                            f"avance del mejor {fila['avance_mejor']:5.1f} % (máx {fila['avance_max']:5.1f} %) | "
                            f"nichos {nichos:3d} | {fines_txt} | {fila['segundos_reloj']:6.1f} s")

            if not ultima:
                genomas, linaje = siguiente(cfg, rng, genomas, list(fitness), datos, evals, orden)
        # Fin normal: los workers salen solos y cierran su emulador (sin el SIGTERM de `terminate`)
        pool.close()
        pool.join()
    anotar(carpeta, f"listo en {(time.time() - inicio_total) / 60:.1f} min")


def partir_de(cfg: Config) -> None:
    """Corrida nueva que parte de la mejor red de `cfg.desde`: copia la forma de la red y su genoma."""
    from .visualizar import red_de
    origen = almacen.dir_corrida(cfg.desde)
    if not almacen.generaciones_guardadas(origen):
        sys.exit(f"--desde {cfg.desde}: esa corrida no tiene generaciones")
    fuente = almacen.cargar_config(origen)
    for k in FORMA_RED:
        setattr(cfg, k, getattr(fuente, k))
    gen, genoma, _ = red_de(origen, cfg.desde_gen)
    cfg.desde_gen = gen
    cfg.genoma_inicial = [genoma[0]] + [list(m) for m in genoma[1:]]


def siguiente(cfg: Config, rng, genomas, fitness, datos, evals=None, orden=None):
    """Población siguiente: élite sin cambios + hijos mutados de los padres. Devuelve (genomas, linaje)."""
    if orden is None:
        orden = [int(i) for i in np.argsort(-np.asarray(fitness), kind="stable")]
    if evals is not None:
        padres = elegir_padres(cfg, orden, evals)
    else:  # --continuar: se reconstruye con lo guardado (el nicho sale de `final`)
        vistos, primero, resto = set(), [], []
        for i in orden:
            x, y = datos["final"][i]
            n = (int(x // cfg.nicho_px), int(y // cfg.nicho_px))
            (resto if n in vistos else primero).append(i)
            vistos.add(n)
        padres = (primero + resto)[:cfg.padres]
    elite = list(orden[:cfg.elite])
    if cfg.elite_mas_lejos:  # la que llega más lejos pasa siempre (además de la élite por fitness) y tiene hijos
        lejos = almacen.indice_mejor(datos)
        if lejos not in elite:
            elite.insert(0, lejos)
        if lejos not in padres:
            padres = [lejos] + list(padres)
    nuevos, padre, semilla, sigma = [], [], [], []
    for i in elite:
        nuevos.append(genomas[i])
        padre.append(i), semilla.append(-1), sigma.append(0.0)
    while len(nuevos) < cfg.poblacion:
        i = int(padres[rng.integers(len(padres))])
        s = int(rng.integers(2**62))
        nuevos.append(genomas[i] + ((s, cfg.sigma),))
        padre.append(i), semilla.append(s), sigma.append(cfg.sigma)
    return nuevos, (np.array(padre, np.int32), np.array(semilla, np.int64), np.array(sigma, np.float32))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C o "Detener": las generaciones completas ya están guardadas
        print("detenido; para seguir: --continuar", flush=True)
        sys.exit(130)
