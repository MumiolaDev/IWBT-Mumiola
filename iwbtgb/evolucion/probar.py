"""Probar una red ya entrenada, sin entrenarla más: en salas del gimnasio que nunca vio, o en los niveles del juego.

    probar --corrida rayos01 --gimnasio [--dificultad 0.5] [--escenarios 40] [--verificar]   # examen
    probar --corrida rayos01 --sala rStage01 rStage02 [--max-segundos 60]
    probar --corrida red01 --sala todas [--gen 300] [--video]
    probar --corrida red01                                                                    # su sala, 10 esperas

`--gimnasio`: `--escenarios` salas generadas desde `--semilla` (no son las de su entrenamiento: con la misma semilla
es un examen fijo para comparar generaciones), con la duración y la paciencia de su corrida; se mide cuántas supera
(con `--verificar`, las salas sin solución encontrada no cuentan). En un nivel: `--escenarios` partidas (la primera
sin espera, las demás con esperas repartidas hasta `espera_max`) y su avance hacia la salida; probar un nivel antes de
entrenar en él es su «primer intento». Resultados en <corridas>/<nombre>/pruebas/ (la app los muestra en una tabla).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from ..emulador import SALAS
from . import almacen
from .episodio import Escenario
from .gimnasio import generar, resolver
from .salas import config_en_sala, esperas
from .visualizar import reglas_de_salas


def guardar(linea, corrida: str, nombre: str, r: dict) -> None:
    destino = linea.dir_corrida(corrida) / "pruebas"
    destino.mkdir(exist_ok=True)
    (destino / f"{nombre}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))


def resumen(corrida: str, gen: int, fitness: float, sala: str, objetivo: str, entrenada: bool, max_segundos: float,
            partidas: list[dict]) -> dict:
    avances = [p["avance"] for p in partidas] or [0.0]
    return {
        "corrida": corrida, "gen": gen, "fitness": fitness, "sala": sala, "objetivo": objetivo,
        "entrenada": entrenada, "max_segundos": max_segundos, "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "avance_medio": round(float(np.mean(avances)), 4), "avance_max": round(float(np.max(avances)), 4),
        "llegaron": sum(p["fin"] == "llegó" for p in partidas), "fines": dict(Counter(p["fin"] for p in partidas)),
        "partidas": partidas,
    }


def probar_sala(linea, corrida: str, gen: int, genoma, fitness: float, sala: str | None, objetivo: str | None,
                n: int, max_segundos: float | None) -> dict:
    carpeta = linea.dir_corrida(corrida)
    cfg, mapa, entrenada = config_en_sala(linea, carpeta, linea.cargar_config(carpeta), sala, objetivo)
    if max_segundos:
        cfg.max_segundos = max_segundos
    jugador = linea.jugador_nivel(cfg, mapa)
    partidas = []
    try:
        for e in esperas(n, cfg.espera_max):
            p = jugador.jugar(genoma, Escenario(e))
            partidas.append({"espera": e, "avance": round(p.avance, 4), "fin": p.fin, "frames": p.frames,
                             "muertes": p.muertes})
            print(f"  {cfg.start_room} espera {e:3d}: avance {100 * p.avance:5.1f} %  {p.fin:9s} "
                  f"{p.frames / 50:5.1f} s", flush=True)
    finally:
        jugador.close()
    r = resumen(corrida, gen, fitness, cfg.start_room, cfg.sala_objetivo, entrenada, cfg.max_segundos, partidas)
    guardar(linea, corrida, f"{cfg.start_room}_gen{gen:04d}", r)
    return r


def probar_gimnasio(linea, corrida: str, gen: int, fitness: float, n: int, dificultad: float, semilla: int,
                    verificar: bool) -> dict:
    cfg, gim, politica = linea.preparar_gimnasio(corrida, gen)
    borde, trampas, minimo = reglas_de_salas(cfg)
    rng = np.random.default_rng(semilla)
    partidas, s = [], semilla
    try:
        while len(partidas) < n:
            sala = generar(s, float(rng.uniform(0, 1)) if dificultad < 0 else dificultad, borde, trampas, minimo)
            s += 1
            gim.construir(sala)
            if verificar and resolver(gim, 600, sala.semilla) is None:
                print(f"  {sala.nombre}: sin solución encontrada, no cuenta", flush=True)
                continue
            it = gim.jugar(politica, cfg.max_frames, paciencia=int(cfg.paciencia * 50))
            partidas.append({"semilla": sala.semilla, "dificultad": round(sala.dificultad, 3),
                             "obstaculos": sala.obstaculos, "avance": round(it.avance, 4), "fin": it.fin,
                             "frames": it.frames})
            print(f"  {sala.nombre}: avance {100 * it.avance:5.1f} %  {it.fin:9s} {it.frames / 50:5.1f} s", flush=True)
    finally:
        gim.close()
    dif = "azar" if dificultad < 0 else f"{dificultad:.2f}"
    r = resumen(corrida, politica.gen, fitness, f"gimnasio (dificultad {dif})", "", cfg.entorno == "gimnasio",
                cfg.max_segundos, partidas)
    guardar(linea, corrida, f"gimnasio_dif{dif}_gen{politica.gen:04d}", r)
    return r


def main(linea, argv=None) -> None:
    p = argparse.ArgumentParser(prog=f"python -m {linea.paquete} probar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", required=True)
    p.add_argument("--gen", type=int, default=-1, help="generación de la red (-1 = la última)")
    p.add_argument("--sala", nargs="*", default=[], help="salas (rStage02 …) o «todas»; sin nada, la suya")
    p.add_argument("--objetivo", default=None, help="sala de llegada (por defecto, la siguiente)")
    p.add_argument("--escenarios", type=int, default=10)
    p.add_argument("--max-segundos", type=float, default=None, help="tope por partida (por defecto, el de la corrida)")
    p.add_argument("--video", action="store_true", help="además, video de la partida sin espera en cada sala")
    p.add_argument("--gimnasio", action="store_true", help="probar en salas generadas (--escenarios = cuántas)")
    p.add_argument("--dificultad", type=float, default=0.5, help="gimnasio: 0 a 1, o -1 = al azar en cada sala")
    p.add_argument("--semilla", type=int, default=1_000_000, help="gimnasio: semilla de la primera sala")
    p.add_argument("--verificar", action="store_true", help="gimnasio: no contar salas sin solución encontrada")
    a = p.parse_args(argv)

    from .visualizar import video_mejor, video_prueba
    gen, genoma, fitness = almacen.red_de(linea.dir_corrida(a.corrida), a.gen)
    if a.gimnasio:
        print(f"probando la mejor red de la gen {gen} de {a.corrida} en {a.escenarios} salas del gimnasio", flush=True)
        r = probar_gimnasio(linea, a.corrida, gen, fitness, a.escenarios, a.dificultad, a.semilla, a.verificar)
        print(f"{r['sala']}: superó {r['llegaron']} de {len(r['partidas'])} salas · avance medio "
              f"{100 * r['avance_medio']:.1f} %", flush=True)
        if a.video:
            video_prueba(linea, a.corrida, a.dificultad, a.semilla, min(a.escenarios, 10), gen, a.verificar)
        return
    salas = SALAS if a.sala == ["todas"] else (a.sala or [None])
    print(f"probando la mejor red de la gen {gen} de {a.corrida} en {len(salas)} sala(s), {a.escenarios} "
          f"escenarios cada una", flush=True)
    hechas = []
    for sala in salas:
        try:
            r = probar_sala(linea, a.corrida, gen, genoma, fitness, sala, a.objetivo, a.escenarios, a.max_segundos)
        except Exception as e:  # una sala rara (sin salida reconocible, etc.) no detiene las demás
            print(f"{sala}: no se pudo probar ({e})", flush=True)
            continue
        hechas.append(r)
        print(f"{r['sala']}{' (entrenada)' if r['entrenada'] else ''}: avance medio {100 * r['avance_medio']:.1f} % "
              f"(máx {100 * r['avance_max']:.1f} %), llegó {r['llegaron']} de {len(r['partidas'])}", flush=True)
        if a.video:
            video_mejor(linea, a.corrida, gen, sala=r["sala"], objetivo=r["objetivo"], max_segundos=a.max_segundos)
    if len(hechas) > 1:
        print("resumen: " + " · ".join(f"{r['sala']} {100 * r['avance_medio']:.0f} %" for r in hechas), flush=True)
