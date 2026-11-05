"""Entrena redes con neuroevolución (Deep GA): población de redes, se quedan las mejores y se mutan sus pesos.

    uv run python -m iwbtgb.rayos entrenar --nombre rayos01 --entorno gimnasio --workers 12
    uv run python -m iwbtgb.rayos entrenar --nombre rayos01 --continuar [--generaciones 2000]
    uv run python -m iwbtgb.rayos entrenar --nombre rayos01_s01 --entorno nivel --desde rayos01
    uv run python -m iwbtgb.redes entrenar --nombre red08 --entorno nivel        (igual con la grilla)

Cada generación todas las redes juegan los mismos escenarios (en el gimnasio, salas nuevas; en el nivel, el comienzo
y otros con una espera al azar) y su fitness es el promedio. La élite pasa sin cambios pero se vuelve a evaluar: así
una red que solo memorizó un escenario no se queda arriba. Con `gim_curriculo`, cuando la población supera
`curriculo_umbral` % de las salas durante `curriculo_gens` generaciones, la dificultad máxima sube `curriculo_paso`
(y desde `curriculo_trampas` aparecen las trampas). Opciones = campos de la config de la línea.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from collections import Counter

import numpy as np

from .. import opciones
from ..workers import un_hilo_por_worker
from . import almacen
from .episodio import Escenario, Evaluacion
from .genoma import Cache, genoma_de
from .salas import preparar_corrida
from .trabajador import evaluar_en_worker, iniciar_worker

SIN_OPCION = ("nombre", "vista", "room_entrenamiento", "room_objetivo", "room_ancho", "room_alto", "genoma_inicial")


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


def elegir_padres(cfg, orden: list[int], finales, llegaron) -> list[int]:
    """Los `padres` mejores, primero el mejor de cada nicho (dónde terminó el primer escenario y si llegó)."""
    vistos, primero, resto = set(), [], []
    for i in orden:
        x, y = finales[i]
        n = (bool(llegaron[i]), int(x // cfg.nicho_px), int(y // cfg.nicho_px))
        (resto if n in vistos else primero).append(i)
        vistos.add(n)
    return (primero + resto)[:cfg.padres]


def siguiente(cfg, rng, genomas, datos, finales=None):
    """Población siguiente: élite sin cambios + hijos mutados de los padres. Devuelve (genomas, linaje). `finales`:
    dónde terminó el primer escenario de cada red (al continuar, el guardado en `datos`)."""
    orden = [int(i) for i in np.argsort(-np.asarray(datos["fitness"]), kind="stable")]
    llegaron = (np.asarray(datos["fines"]) == "llegó").any(axis=1)
    padres = elegir_padres(cfg, orden, datos["final"] if finales is None else finales, llegaron)
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


def metricas(cfg, gen: int, evals, datos, fitness, mejor: int, segundos: float) -> dict:
    partidas = [q for e in evals for q in e.partidas]
    fines = Counter(q.fin for q in partidas)
    total = sum(fines.values())
    nichos = len({(e.llego, int(e.partidas[0].final[0] // cfg.nicho_px), int(e.partidas[0].final[1] // cfg.nicho_px))
                  for e in evals})
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
        "frames_medio": round(float(np.mean([q.frames for q in partidas])), 1),
        "muertes_media": round(float(np.mean([q.muertes for q in partidas])), 2),
        "saves_max": max(q.saves for q in partidas),
        "saves_media": round(float(np.mean([q.saves for q in partidas])), 2),
        "quieto_pct": round(100 * float(np.mean([50 * q.quieto / max(q.frames, 1) for q in partidas])), 1),
        "celdas_media": round(float(np.mean([q.celdas for q in partidas])), 1),
        "celdas_max": max(q.celdas for q in partidas),
        "salas_pct": round(100 * float(np.mean([q.fin == "llegó" for q in partidas])), 1),
        "salas_mejor": sum(q.fin == "llegó" for q in evals[mejor].partidas),
        "segundos_reloj": round(segundos, 1),
    }
    if cfg.entorno == "gimnasio":
        fila.update(dificultad_min=cfg.gim_dificultad_min, dificultad_max=round(cfg.gim_dificultad_max, 3),
                    trampas=int(cfg.gim_trampas))
    return fila


def linea_log(cfg, fila: dict) -> str:
    if cfg.entorno == "gimnasio":
        fines = (f"salas superadas {fila['salas_pct']:4.1f} % (la mejor {fila['salas_mejor']}/{cfg.escenarios}) "
                 f"murió {fila['murio_pct']:3.0f} % estancado {fila['estancado_pct']:3.0f} % | dificultad "
                 f"{cfg.gim_dificultad_min:g}–{cfg.gim_dificultad_max:g}{' +trampas' if cfg.gim_trampas else ''}")
    elif cfg.reintentos:
        fines = (f"muertes {fila['muertes_media']:4.1f} quieto {fila['quieto_pct']:3.0f} % saves "
                 f"{fila['saves_media']:4.2f} (máx {fila['saves_max']})")
        if cfg.r_explorar:
            fines += f" celdas {fila['celdas_media']:5.1f} (máx {fila['celdas_max']})"
    else:
        fines = f"murió {fila['murio_pct']:4.0f} % estancado {fila['estancado_pct']:4.0f} %"
    return (f"gen {fila['gen']:4d} | fitness máx {fila['fitness_max']:7.2f} media {fila['fitness_media']:7.2f} | "
            f"avance del mejor {fila['avance_mejor']:5.1f} % (máx {fila['avance_max']:5.1f} %) | "
            f"nichos {fila['nichos']:3d} | {fines} | {fila['segundos_reloj']:6.1f} s")


def curriculo(cfg, historial: list[tuple[float, float]]) -> str | None:
    """Sube la dificultad máxima del gimnasio si en las últimas `curriculo_gens` generaciones (todas con la
    dificultad actual) la población superó en promedio `curriculo_umbral` % de las salas. Cambia `cfg`; devuelve
    qué cambió (o None)."""
    if not cfg.gim_curriculo or cfg.gim_dificultad_max >= 1.0 or len(historial) < cfg.curriculo_gens:
        return None
    ultimas = historial[-cfg.curriculo_gens:]
    if any(abs(d - cfg.gim_dificultad_max) > 1e-6 for d, _ in ultimas):
        return None
    media = float(np.mean([s for _, s in ultimas]))
    if media < cfg.curriculo_umbral:
        return None
    antes = cfg.gim_dificultad_max
    cfg.gim_dificultad_max = round(min(1.0, antes + cfg.curriculo_paso), 3)
    texto = f"{media:.0f} % de salas en {cfg.curriculo_gens} gen. → dificultad máxima {antes:g} → {cfg.gim_dificultad_max:g}"
    if not cfg.gim_trampas and cfg.gim_dificultad_max >= cfg.curriculo_trampas - 1e-6:
        cfg.gim_trampas = 1
        texto += ", con trampas"
    return texto


def partir_de(linea, cfg) -> None:
    """Corrida nueva que parte de la mejor red de `cfg.desde` (de la misma línea): copia lo que ve, la forma de la
    red y su genoma."""
    origen = linea.dir_corrida(cfg.desde)
    if not almacen.generaciones_guardadas(origen):
        sys.exit(f"--desde {cfg.desde}: esa corrida no tiene generaciones")
    fuente = linea.cargar_config(origen)
    for k in linea.forma_red:
        setattr(cfg, k, getattr(fuente, k))
    gen, genoma, _ = almacen.red_de(origen, cfg.desde_gen)
    cfg.desde_gen = gen
    cfg.genoma_inicial = [genoma[0]] + [list(m) for m in genoma[1:]]


def main(linea, argv=None) -> None:
    p = argparse.ArgumentParser(prog=f"python -m {linea.paquete} entrenar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", default=linea.Config.nombre)
    p.add_argument("--continuar", action="store_true", help="seguir desde la última generación guardada")
    p.add_argument("--sobrescribir", action="store_true", help="borrar la corrida anterior con el mismo nombre")
    opciones.agregar(p, linea.Config, omitir=SIN_OPCION)
    args = p.parse_args(argv)

    carpeta = linea.dir_corrida(args.nombre)
    hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
    if hechas and not args.continuar:
        if not args.sobrescribir:
            sys.exit(f"{carpeta} ya tiene generaciones: usa --continuar o --sobrescribir")
        shutil.rmtree(carpeta)
        hechas = []
    cfg = linea.cargar_config(carpeta) if hechas else linea.Config(nombre=args.nombre)
    for k, v in opciones.dadas(args, linea.Config, omitir=SIN_OPCION).items():
        if hechas and k in linea.fijos and v != getattr(cfg, k):
            sys.exit(f"al continuar no se puede cambiar {k} (lo que ve, la red o el juego): crea otra corrida")
        setattr(cfg, k, v)
    if (problema := cfg.problema()) is not None:
        sys.exit(problema)
    if cfg.desde and not hechas:
        partir_de(linea, cfg)
    carpeta.mkdir(parents=True, exist_ok=True)
    mapa = preparar_corrida(cfg, carpeta)
    almacen.guardar_config(carpeta, cfg)
    red = linea.red_para(cfg)
    cache = Cache(16)

    rng = np.random.default_rng(cfg.seed + 1000 * (hechas[-1] + 1 if hechas else 0))
    if hechas:
        gen0 = hechas[-1] + 1
        anteriores = almacen.genomas(carpeta, hechas[-1])
        genomas, linaje = siguiente(cfg, rng, anteriores, almacen.cargar_generacion(carpeta, hechas[-1]))
    elif cfg.genoma_inicial:  # parte de la red de otra corrida: ella sin cambios + mutaciones suyas
        gen0 = 0
        base = genoma_de(cfg.genoma_inicial)
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
             f"{cfg.gim_dificultad_max:g}{' con trampas' if cfg.gim_trampas else ''}"
             f"{f' (al menos {cfg.gim_trampas_min} por sala)' if cfg.gim_trampas and cfg.gim_trampas_min else ''}"
             f", hasta {cfg.max_segundos:.0f} s"
             + (f", currículo (sube {cfg.curriculo_paso:g} con {cfg.curriculo_umbral:g} % de salas en "
                f"{cfg.curriculo_gens} gen.)" if cfg.gim_curriculo else "") if gimnasio else
             f"{cfg.escenarios} escenarios de hasta {cfg.max_segundos:.0f} s  room {cfg.room_entrenamiento} -> "
             f"{cfg.sala_objetivo} ({cfg.room_objetivo})")
    anotar(carpeta, f"corrida {carpeta}  población {cfg.poblacion}  padres {cfg.padres}  élite {cfg.elite}  "
                    f"sigma {cfg.sigma}  workers {cfg.workers}  red de {red.n_pesos} pesos  {donde}"
                    + (f"  partiendo de {cfg.desde}" if cfg.desde else ""))
    historial = [(m.get("dificultad_max", -1.0), m.get("salas_pct", 0.0)) for m in almacen.leer_metricas(carpeta)]
    inicio_total = time.time()
    un_hilo_por_worker()
    with mp.get_context("spawn").Pool(cfg.workers, initializer=iniciar_worker,
                                      initargs=(linea.paquete, cfg, mapa)) as pool:
        for gen in range(gen0, cfg.generaciones):
            t0 = time.time()
            if gimnasio:  # salas nuevas cada generación (iguales para todas las redes): no puede memorizarlas
                escenarios = [Escenario(0, int(rng.integers(2**31)),
                                        float(rng.uniform(cfg.gim_dificultad_min, cfg.gim_dificultad_max)))
                              for _ in range(cfg.escenarios)]
            else:
                escenarios = [Escenario(0)] + [Escenario(int(rng.integers(0, cfg.espera_max + 1)))
                                               for _ in range(cfg.escenarios - 1)]
                if cfg.reintentos and cfg.espera_reintento:  # esperas al reaparecer, también nuevas cada generación
                    for e in escenarios:
                        e.semilla = int(rng.integers(2**31))
            evals: list[Evaluacion | None] = [None] * len(genomas)
            # Hermanos juntos: el worker reconstruye los pesos del padre una vez y los demás salen de su caché. La
            # config va con cada tarea (pesa poco): el currículo la cambia entre generaciones
            tareas = sorted(range(len(genomas)), key=lambda i: int(linaje[0][i]))
            for i, ev in pool.imap_unordered(evaluar_en_worker, [(i, genomas[i], escenarios, cfg) for i in tareas],
                                             chunksize=max(1, min(8, len(genomas) // (4 * cfg.workers)))):
                evals[i] = ev
            fitness = np.array([e.fitness for e in evals], np.float32)

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
                datos["dificultades"] = np.array([e.dificultad for e in escenarios], np.float64)
                datos["salida_borde"] = np.array(int(cfg.gim_salida_borde))   # cómo se generaron sus salas
                datos["trampas"] = np.array(int(cfg.gim_trampas))
                datos["trampas_min"] = np.array(int(cfg.gim_trampas_min))
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

            fila = metricas(cfg, gen, evals, datos, fitness, mejor, time.time() - t0)
            almacen.agregar_metricas(carpeta, fila)
            anotar(carpeta, linea_log(cfg, fila))
            historial.append((cfg.gim_dificultad_max, fila["salas_pct"]))
            if gimnasio and (cambio := curriculo(cfg, historial)):
                almacen.guardar_config(carpeta, cfg)
                anotar(carpeta, f"currículo: {cambio}")

            if not ultima:
                genomas, linaje = siguiente(cfg, rng, genomas, datos, [e.partidas[0].final for e in evals])
        # Fin normal: los workers salen solos y cierran su emulador (sin el SIGTERM de `terminate`)
        pool.close()
        pool.join()
    anotar(carpeta, f"listo en {(time.time() - inicio_total) / 60:.1f} min")
