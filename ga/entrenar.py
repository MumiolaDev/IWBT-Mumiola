"""Entrena una población de agentes con un algoritmo genético.

    uv run python -m ga.entrenar --nombre salida01                 # experimento por defecto (rStage01, 50 agentes)
    uv run python -m ga.entrenar --nombre prueba --poblacion 20 --generaciones 5 --max-segundos 30
    uv run python -m ga.entrenar --nombre salida01 --continuar --generaciones 500
    uv run python -m ga.entrenar --nombre s03 --start-room rStage03 --sala-objetivo rStage04

Cada generación queda en corridas/<nombre>/gen_NNNN.npz (genomas, fitness y trazas de todos los agentes) y
una fila en metricas.csv. Los videos se generan después con `python -m ga.visualizar`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import fields

import numpy as np

from . import almacen
from .nucleo import (Config, Resultado, cruzar, evaluar_en_worker, genes_al_azar, genes_usados, iniciar_worker, mutar,
                     nicho, resolver_rooms, resolver_zonas, torneo)


def resumen(gen: int, pob: list[Resultado], evaluados: int, reloj: float, cfg: Config) -> dict:
    fit = np.array([r.fitness for r in pob])
    mejor = max(pob, key=Resultado.clave)
    return {
        "progreso_max": round(100 * max(r.progreso_max for r in pob), 2),
        "progreso_final_max": round(100 * max(r.progreso for r in pob), 2),
        "progreso_media": round(100 * float(np.mean([r.progreso for r in pob])), 2),
        "nichos": len({nicho(r, cfg) for r in pob}),
        "gen": gen,
        "fitness_max": round(float(fit.max()), 2),
        "fitness_media": round(float(fit.mean()), 2),
        "fitness_mediana": round(float(np.median(fit)), 2),
        "fitness_min": round(float(fit.min()), 2),
        "pantallas_max": max(r.pantallas for r in pob),
        "pantallas_media": round(float(np.mean([r.pantallas for r in pob])), 3),
        "llegaron": sum(r.llego for r in pob),
        "celdas_max": max(r.celdas for r in pob),
        "celdas_media": round(float(np.mean([r.celdas for r in pob])), 2),
        "muertes_pct": round(100.0 * np.mean([r.murio for r in pob]), 1),
        "segundos_medio": round(float(np.mean([r.frames for r in pob])) / 50, 2),
        "segundos_mejor": round(mejor.frames / 50, 2),
        "evaluados": evaluados,
        "segundos_reloj": round(reloj, 1),
    }


def elegir_padres(pob: list[Resultado], cfg: Config) -> list[int]:
    """Los `cfg.padres` mejores, pero sin repetir nicho (lugar donde terminaron): así sobreviven caminos
    distintos aunque no sean el mejor. Si no hay suficientes nichos, se completa con los mejores restantes."""
    orden = sorted(range(len(pob)), key=lambda i: pob[i].clave(), reverse=True)
    elegidos, vistos = [], set()
    for i in orden:
        n = nicho(pob[i], cfg)
        if n not in vistos:
            vistos.add(n)
            elegidos.append(i)
        if len(elegidos) == cfg.padres:
            return elegidos
    resto = [i for i in orden if i not in elegidos]
    return elegidos + resto[: cfg.padres - len(elegidos)]


def huella(g, frames: int) -> bytes:
    """Identifica la parte del genoma que alcanza a jugarse (dos huellas iguales = la misma partida)."""
    n = genes_usados(g, frames)
    return g.acciones[:n].tobytes() + g.frames[:n].tobytes()


def siguiente_generacion(rng: np.random.Generator, pob: list[Resultado], cfg: Config):
    """Devuelve (élite que pasa sin reevaluar, tareas de evaluación para el resto)."""
    if cfg.seleccion == "nichos":
        return siguiente_generacion_nichos(rng, pob, cfg)
    orden = sorted(range(len(pob)), key=lambda i: pob[i].clave(), reverse=True)
    elite = []
    for i in orden[: cfg.elite]:
        r = pob[i]
        elite.append(Resultado(r.genoma, r.fitness, r.murio, r.frames, r.llego, r.pantallas, r.celdas,
                               r.frame_progreso, r.traza, (i,), "elite"))
    tareas = []
    while len(elite) + len(tareas) < cfg.poblacion:
        ia = torneo(rng, pob, cfg.torneo)
        a = pob[ia]
        if rng.random() < cfg.prob_cruce:
            ib = torneo(rng, pob, cfg.torneo)
            b = pob[ib]
            hijo = cruzar(rng, a, b)
            base = Resultado(hijo, a.fitness, a.murio, a.frames, a.llego, a.pantallas, a.celdas,
                             a.frame_progreso, a.traza)
            genoma = mutar(rng, base, cfg)
            padres, origen = (ia, ib), "cruce"
        else:
            genoma = mutar(rng, a, cfg)
            padres, origen = (ia,), "mutacion"
        tareas.append((genoma, int(rng.integers(2**31)), padres, origen))
    return elite, tareas


def siguiente_generacion_nichos(rng: np.random.Generator, pob: list[Resultado], cfg: Config):
    """Los padres (mejores de nichos distintos) pasan como élite y se reparten los hijos por igual.
    Un hijo que repetiría exactamente una partida ya vista se vuelve a mutar."""
    padres = elegir_padres(pob, cfg)
    elite = []
    for i in padres[: cfg.elite]:
        r = pob[i]
        elite.append(Resultado(r.genoma, r.fitness, r.murio, r.frames, r.llego, r.pantallas, r.celdas,
                               r.frame_progreso, r.traza, (i,), "elite", r.progreso, r.progreso_max))
    vistas = {huella(pob[i].genoma, pob[i].frames) for i in range(len(pob))}
    tareas = []
    turno = list(padres)
    rng.shuffle(turno)
    k = 0
    while len(elite) + len(tareas) < cfg.poblacion:
        ia = turno[k % len(turno)]
        k += 1
        a = pob[ia]
        for _ in range(10):
            if len(padres) > 1 and rng.random() < cfg.prob_cruce:
                ib = int(rng.choice([j for j in padres if j != ia]))
                b = pob[ib]
                hijo = cruzar(rng, a, b)
                base = Resultado(hijo, a.fitness, a.murio, a.frames, a.llego, a.pantallas, a.celdas,
                                 a.frame_progreso, a.traza)
                genoma = mutar(rng, base, cfg)
                padres_hijo, origen = (ia, ib), "cruce"
            else:
                genoma = mutar(rng, a, cfg)
                padres_hijo, origen = (ia,), "mutacion"
            h = huella(genoma, cfg.max_frames)
            if h not in vistas:
                break
        vistas.add(h)
        tareas.append((genoma, int(rng.integers(2**31)), padres_hijo, origen))
    return elite, tareas


HIJOS_POR_TAREA = 4


def evaluar_todas(pool, tareas, pob: list[Resultado] | None) -> list[Resultado]:
    """Agrupa los hijos por padre principal (comparten el comienzo de la partida) y devuelve los resultados
    en el orden de `tareas`. El resultado no depende de cómo se repartan entre workers."""
    familias: dict[int, list[int]] = {}
    for i, t in enumerate(tareas):
        familias.setdefault(t[2][0] if pob is not None and t[2] else -1 - i, []).append(i)
    trabajos = []
    for padre, idx in familias.items():
        genoma_padre = pob[padre].genoma if padre >= 0 else None
        for c in range(0, len(idx), HIJOS_POR_TAREA):
            parte = idx[c:c + HIJOS_POR_TAREA]
            trabajos.append((parte, (genoma_padre, [tareas[i] for i in parte])))
    # Las familias más grandes primero, para repartir mejor la carga
    trabajos.sort(key=lambda t: -len(t[0]))
    resultados: list[Resultado | None] = [None] * len(tareas)
    for parte, res in zip((t[0] for t in trabajos), pool.imap(evaluar_en_worker, [t[1] for t in trabajos])):
        for i, r in zip(parte, res):
            resultados[i] = r
    return resultados


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--continuar", action="store_true", help="seguir desde la última generación guardada")
    p.add_argument("--sobrescribir", action="store_true", help="borrar la corrida anterior con el mismo nombre")
    for f in fields(Config):
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

    if hechas:
        cfg = almacen.cargar_config(carpeta)
    else:
        cfg = Config(nombre=args.nombre)
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "nombre":
            setattr(cfg, f.name, v)
    if cfg.room_entrenamiento < 0:
        resolver_rooms(cfg)
    resolver_zonas(cfg)
    carpeta.mkdir(parents=True, exist_ok=True)
    almacen.guardar_config(carpeta, cfg)

    rng = np.random.default_rng(cfg.seed + 1000 * (hechas[-1] + 1 if hechas else 0))
    print(f"corrida {carpeta}  población {cfg.poblacion}  workers {cfg.workers}  "
          f"tope {cfg.max_segundos:.0f} s de juego  room {cfg.room_entrenamiento} -> {cfg.sala_objetivo} "
          f"({cfg.room_objetivo})  zonas prohibidas: {cfg.zonas_prohibidas or 'ninguna'}", flush=True)

    inicio_total = time.time()
    with mp.get_context("spawn").Pool(cfg.workers, initializer=iniciar_worker, initargs=(cfg,)) as pool:
        if hechas:
            gen = hechas[-1]
            pob = almacen.cargar_generacion(carpeta, gen)
            gen += 1
        else:
            gen = 0
            t = time.time()
            tareas = [(genes_al_azar(rng, cfg.genes_iniciales, cfg), int(rng.integers(2**31)), (), "azar")
                      for _ in range(cfg.poblacion)]
            pob = evaluar_todas(pool, tareas, None)
            almacen.guardar_generacion(carpeta, 0, pob)
            fila = resumen(0, pob, len(tareas), time.time() - t, cfg)
            almacen.agregar_metricas(carpeta, fila)
            imprimir(fila)
            gen = 1

        while gen < cfg.generaciones:
            t = time.time()
            elite, tareas = siguiente_generacion(rng, pob, cfg)
            pob = elite + evaluar_todas(pool, tareas, pob)
            almacen.guardar_generacion(carpeta, gen, pob)
            fila = resumen(gen, pob, len(tareas), time.time() - t, cfg)
            almacen.agregar_metricas(carpeta, fila)
            imprimir(fila)
            gen += 1
    print(f"listo en {(time.time() - inicio_total) / 60:.1f} min", flush=True)


def imprimir(f: dict) -> None:
    print(f"gen {f['gen']:4d} | fitness máx {f['fitness_max']:8.1f} media {f['fitness_media']:7.1f} | "
          f"camino máx {f['progreso_max']:5.1f} % | nichos {f['nichos']:3d} | "
          f"pantallas máx {f['pantallas_max']} llegaron {f['llegaron']} | "
          f"muertes {f['muertes_pct']:5.1f} % | vida media {f['segundos_medio']:6.1f} s | "
          f"{f['segundos_reloj']:6.1f} s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C o "Detener" en la app: las generaciones completas ya están guardadas
        print("detenido; para seguir: --continuar", flush=True)
        sys.exit(130)
