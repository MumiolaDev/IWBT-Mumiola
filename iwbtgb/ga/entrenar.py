"""Entrena una población de agentes con un algoritmo genético.

    uv run python -m iwbtgb.ga entrenar --nombre salida01              # rStage01 → rStage02, 50 agentes
    uv run python -m iwbtgb.ga entrenar --nombre prueba --poblacion 20 --generaciones 5 --max-segundos 30
    uv run python -m iwbtgb.ga entrenar --nombre salida01 --continuar --generaciones 500
    uv run python -m iwbtgb.ga entrenar --nombre s03 --start-room rStage03 --sala-objetivo rStage04

Cada generación queda en corridas/<nombre>/gen_NNNN.npz (genomas, fitness y trazas de todos los agentes) y
una fila en metricas.csv. Los videos se generan después con `python -m iwbtgb.ga visualizar`. Opciones = campos de
`Config`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import replace

import numpy as np

from .. import opciones
from ..medios import FPS
from . import almacen
from .nucleo import (Config, Inicio, Resultado, cruzar, evaluar_en_worker, genes_al_azar, genes_usados, heredar,
                     iniciar_worker, mutar, mutar_desde_save, nicho, resolver_rooms, resolver_saves, resolver_zonas)


def resumen(gen: int, pob: list[Resultado], evaluados: int, reloj: float, cfg: Config) -> dict:
    fit = np.array([r.fitness for r in pob])
    mejor = max(pob, key=Resultado.clave)
    return {
        "progreso_max": round(100 * max(r.progreso_max for r in pob), 2),
        "progreso_final_max": round(100 * max(r.progreso for r in pob), 2),
        "progreso_media": round(100 * float(np.mean([r.progreso for r in pob])), 2),
        "nichos": len({nicho(r, cfg) for r in pob}),
        "saves_max": max(r.saves for r in pob),
        "saves_media": round(float(np.mean([r.saves for r in pob])), 2),
        "reintentos_media": round(float(np.mean([r.muertes for r in pob])), 2),
        "desde_save_pct": round(100.0 * np.mean([r.inicio is not None for r in pob]), 1),
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
        "segundos_medio": round(float(np.mean([r.frames for r in pob])) / FPS, 2),
        "segundos_mejor": round(mejor.frames / FPS, 2),
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


def huella(g, frames: int, inicio: Inicio | None = None) -> bytes:
    """Identifica la parte del genoma que alcanza a jugarse y desde dónde parte (dos huellas iguales = la misma
    partida)."""
    n = genes_usados(g, frames)
    lugar = repr((inicio.gen, inicio.agente, inicio.save)).encode() if inicio else b""
    return lugar + g.acciones[:n].tobytes() + g.frames[:n].tobytes()


class Engendrar:
    """Crea un hijo de `pob[ia]`: desde su save más avanzado (si activó uno nuevo, con prob. `prob_desde_save`),
    por cruce con otro padre que parta del mismo lugar (`pareja()`), o por mutación. `gen` = generación de `pob`."""

    def __init__(self, pob: list[Resultado], cfg: Config, gen: int):
        self.pob, self.cfg, self.gen = pob, cfg, gen
        self.herencias: dict[int, tuple | None] = {}

    def __call__(self, rng: np.random.Generator, ia: int, pareja) -> tuple:
        pob, cfg, a = self.pob, self.cfg, self.pob[ia]
        if cfg.prob_desde_save > 0:
            if ia not in self.herencias:
                self.herencias[ia] = heredar(a, cfg, self.gen, ia)
            her = self.herencias[ia]
            if her and rng.random() < cfg.prob_desde_save:
                inicio, g = her
                return mutar_desde_save(rng, a, g, cfg), (ia,), "save", inicio
        if rng.random() < cfg.prob_cruce:
            ib = pareja()
            if ib is not None and ib != ia and pob[ib].inicio == a.inicio:
                b = pob[ib]
                hijo = cruzar(rng, a, b)
                base = Resultado(hijo, a.fitness, a.murio, a.frames, a.llego, a.pantallas, a.celdas,
                                 a.frame_progreso, a.traza)
                return mutar(rng, base, cfg), (ia, ib), "cruce", a.inicio
        return mutar(rng, a, cfg), (ia,), "mutacion", a.inicio


def siguiente_generacion(rng: np.random.Generator, pob: list[Resultado], cfg: Config, gen: int):
    """(élite que pasa sin reevaluar, tareas de evaluación para el resto); `gen` = generación de `pob`. Los padres
    (mejores de nichos distintos) pasan como élite y se reparten los hijos por igual. Un hijo que repetiría
    exactamente una partida ya vista se vuelve a mutar."""
    padres = elegir_padres(pob, cfg)
    elite = []
    for i in padres[: cfg.elite]:
        r = pob[i]
        elite.append(replace(r, padres=(i,), origen="elite"))
    vistas = {huella(r.genoma, r.frames, r.inicio) for r in pob}
    engendrar = Engendrar(pob, cfg, gen)
    tareas = []
    turno = list(padres)
    rng.shuffle(turno)
    k = 0
    while len(elite) + len(tareas) < cfg.poblacion:
        ia = turno[k % len(turno)]
        k += 1
        pareja = [j for j in padres if j != ia and pob[j].inicio == pob[ia].inicio]
        for _ in range(10):
            genoma, padres_hijo, origen, inicio = engendrar(
                rng, ia, lambda: int(rng.choice(pareja)) if pareja else None)
            h = huella(genoma, cfg.max_frames, inicio)
            if h not in vistas:
                break
        vistas.add(h)
        tareas.append((genoma, int(rng.integers(2**31)), padres_hijo, origen, inicio))
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
        genoma_padre, inicio_padre = (pob[padre].genoma, pob[padre].inicio) if padre >= 0 else (None, None)
        for c in range(0, len(idx), HIJOS_POR_TAREA):
            parte = idx[c:c + HIJOS_POR_TAREA]
            trabajos.append((parte, (genoma_padre, inicio_padre, [tareas[i] for i in parte])))
    # Las familias más grandes primero, para repartir mejor la carga
    trabajos.sort(key=lambda t: -len(t[0]))
    resultados: list[Resultado | None] = [None] * len(tareas)
    for parte, res in zip((t[0] for t in trabajos), pool.imap(evaluar_en_worker, [t[1] for t in trabajos])):
        for i, r in zip(parte, res):
            resultados[i] = r
    return resultados


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m iwbtgb.ga entrenar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", default=Config.nombre)
    p.add_argument("--continuar", action="store_true", help="seguir desde la última generación guardada")
    p.add_argument("--sobrescribir", action="store_true", help="borrar la corrida anterior con el mismo nombre")
    opciones.agregar(p, Config, omitir=("nombre",))
    args = p.parse_args(argv)

    carpeta = almacen.dir_corrida(args.nombre)
    if (carpeta / "explorar.json").exists():
        sys.exit(f"{carpeta} es una corrida de Go-Explore: usa python -m iwbtgb.ga explorar --nombre {args.nombre} "
                 "--continuar")
    hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
    if hechas and not args.continuar:
        if not args.sobrescribir:
            sys.exit(f"{carpeta} ya tiene generaciones: usa --continuar o --sobrescribir")
        shutil.rmtree(carpeta)
        hechas = []

    cfg = almacen.cargar_config(carpeta) if hechas else Config(nombre=args.nombre)
    for k, v in opciones.dadas(args, Config, omitir=("nombre",)).items():
        setattr(cfg, k, v)
    if cfg.room_entrenamiento < 0:
        resolver_rooms(cfg)
    resolver_zonas(cfg)
    resolver_saves(cfg)
    carpeta.mkdir(parents=True, exist_ok=True)
    almacen.guardar_config(carpeta, cfg)

    rng = np.random.default_rng(cfg.seed + 1000 * (hechas[-1] + 1 if hechas else 0))
    print(f"corrida {carpeta}  población {cfg.poblacion}  workers {cfg.workers}  "
          f"tope {cfg.max_segundos:.0f} s de juego  room {cfg.room_entrenamiento} -> {cfg.sala_objetivo} "
          f"({cfg.room_objetivo})  zonas prohibidas: {cfg.zonas_prohibidas or 'ninguna'}  reintentos: {cfg.resets}  "
          f"save points: {len(cfg.save_points.split(';')) if cfg.save_points else 0}  "
          f"parten del save heredado: {100 * cfg.prob_desde_save:.0f} %", flush=True)

    inicio_total = time.time()
    with mp.get_context("spawn").Pool(cfg.workers, initializer=iniciar_worker, initargs=(cfg,)) as pool:
        if hechas:
            gen = hechas[-1]
            pob = almacen.cargar_generacion(carpeta, gen)
            # Agentes de antes del save heredado (o élite que viene de entonces): no se sabe qué saves activaron.
            # Se vuelven a jugar, con las reglas actuales, así su fitness también queda al día
            viejos = [i for i, r in enumerate(pob)
                      if r.saves > len(r.nuevos) + (len(r.inicio.saves) if r.inicio else 0)]
            if cfg.prob_desde_save > 0 and viejos:
                print(f"re-evaluando {len(viejos)} agentes de la generación {gen} para saber qué saves activaron",
                      flush=True)
                tareas = [(pob[i].genoma, i, pob[i].padres, pob[i].origen, pob[i].inicio) for i in viejos]
                for i, r in zip(viejos, evaluar_todas(pool, tareas, None)):
                    pob[i] = r
                # Los hijos que partan de sus saves apuntan a esta generación (video del recorrido completo)
                almacen.guardar_generacion(carpeta, gen, pob)
            gen += 1
        else:
            gen = 0
            t = time.time()
            tareas = [(genes_al_azar(rng, cfg.genes_iniciales, cfg), int(rng.integers(2**31)), (), "azar", None)
                      for _ in range(cfg.poblacion)]
            pob = evaluar_todas(pool, tareas, None)
            almacen.guardar_generacion(carpeta, 0, pob)
            fila = resumen(0, pob, len(tareas), time.time() - t, cfg)
            almacen.agregar_metricas(carpeta, fila)
            imprimir(fila)
            gen = 1

        while gen < cfg.generaciones:
            t = time.time()
            elite, tareas = siguiente_generacion(rng, pob, cfg, gen - 1)
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
          f"saves máx {f['saves_max']} media {f['saves_media']:4.2f} | reintentos {f['reintentos_media']:4.2f} | "
          f"desde save {f['desde_save_pct']:5.1f} % | "
          f"pantallas máx {f['pantallas_max']} llegaron {f['llegaron']} | "
          f"muertes {f['muertes_pct']:5.1f} % | vida media {f['segundos_medio']:6.1f} s | "
          f"{f['segundos_reloj']:6.1f} s", flush=True)



