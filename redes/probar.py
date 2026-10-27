"""Probar una red ya entrenada: en su sala con muchas esperas distintas, o en salas que nunca vio.

    uv run python -m redes.probar --corrida red01                          # su sala, 10 esperas
    uv run python -m redes.probar --corrida red01 --sala rStage02 rStage03 --max-segundos 60
    uv run python -m redes.probar --corrida red01 --sala todas [--gen 300] [--video]
    uv run python -m redes.probar --corrida gym01 --gimnasio [--dificultad 0.5] [--escenarios 20] [--verificar]

En cada sala la red juega `--escenarios` partidas (la primera sin espera, las demás con esperas repartidas entre 0
y `espera_max` frames) y se mide su avance hacia la salida, igual que al entrenar. Los resultados quedan en
corridas_redes/<nombre>/pruebas/<sala>_gen<N>.json (la app los muestra en una tabla) y, con `--video`, el video
de la partida sin espera en corridas_redes/<nombre>/videos/.

`--gimnasio`: la red juega `--escenarios` salas generadas (semillas desde `--semilla`, que no son las de su
entrenamiento; dificultad fija o -1 = al azar), con la duración y la paciencia de su corrida; se mide cuántas
supera. Con `--verificar` las salas sin solución encontrada no cuentan. Queda en pruebas/gimnasio_dif<d>_gen<N>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from . import almacen
from .episodio import Escenario, Jugador
from .salas import SALAS, config_en_sala, esperas


def probar_sala(corrida: str, gen: int, genoma, fitness: float, sala: str | None, objetivo: str | None,
                n: int, max_segundos: float | None) -> dict:
    carpeta = almacen.dir_corrida(corrida)
    cfg, mapa, entrenada = config_en_sala(carpeta, almacen.cargar_config(carpeta), sala, objetivo)
    if max_segundos:
        cfg.max_segundos = max_segundos
    jugador = Jugador(cfg, mapa)
    partidas = []
    try:
        for e in esperas(n, cfg.espera_max):
            p = jugador.jugar(genoma, Escenario(e))
            partidas.append({"espera": e, "avance": round(p.avance, 4), "fin": p.fin, "frames": p.frames})
            print(f"  {cfg.start_room} espera {e:3d}: avance {100 * p.avance:5.1f} %  {p.fin:9s} "
                  f"{p.frames / 50:5.1f} s", flush=True)
    finally:
        jugador.close()
    avances = [p["avance"] for p in partidas]
    r = {
        "corrida": corrida, "gen": gen, "fitness": fitness, "sala": cfg.start_room, "objetivo": cfg.sala_objetivo,
        "entrenada": entrenada, "max_segundos": cfg.max_segundos, "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "avance_medio": round(float(np.mean(avances)), 4), "avance_max": round(float(np.max(avances)), 4),
        "llegaron": sum(p["fin"] == "llegó" for p in partidas), "fines": dict(Counter(p["fin"] for p in partidas)),
        "partidas": partidas,
    }
    destino = carpeta / "pruebas"
    destino.mkdir(exist_ok=True)
    (destino / f"{cfg.start_room}_gen{gen:04d}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    return r


def probar_gimnasio(corrida: str, gen: int, fitness: float, n: int, dificultad: float, semilla: int,
                    verificar: bool) -> dict:
    from .gimnasio import Gimnasio, PoliticaRed, generar, resolver
    politica = PoliticaRed(corrida, gen)
    cfg = politica.cfg
    gim = Gimnasio(politica.config_vista)
    politica.vista = gim.vista
    rng = np.random.default_rng(semilla)
    partidas, s = [], semilla
    try:
        while len(partidas) < n:
            gim_cfg = cfg.entorno == "gimnasio"   # salas como las de su entrenamiento
            borde = bool(cfg.gim_salida_borde) if gim_cfg else True
            trampas = bool(cfg.gim_trampas) if gim_cfg else True
            sala = generar(s, float(rng.uniform(0, 1)) if dificultad < 0 else dificultad, borde, trampas)
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
    avances = [p["avance"] for p in partidas]
    dif = "azar" if dificultad < 0 else f"{dificultad:.2f}"
    r = {
        "corrida": corrida, "gen": politica.gen, "fitness": fitness, "sala": f"gimnasio (dificultad {dif})",
        "objetivo": "", "entrenada": cfg.entorno == "gimnasio", "max_segundos": cfg.max_segundos,
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "avance_medio": round(float(np.mean(avances)), 4),
        "avance_max": round(float(np.max(avances)), 4), "llegaron": sum(p["fin"] == "llegó" for p in partidas),
        "fines": dict(Counter(p["fin"] for p in partidas)), "partidas": partidas,
    }
    destino = almacen.dir_corrida(corrida) / "pruebas"
    destino.mkdir(exist_ok=True)
    (destino / f"gimnasio_dif{dif}_gen{politica.gen:04d}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    a = p.parse_args()

    from .visualizar import red_de, video_mejor
    carpeta = almacen.dir_corrida(a.corrida)
    gen, genoma, fitness = red_de(carpeta, a.gen)
    if a.gimnasio:
        print(f"probando la mejor red de la gen {gen} de {a.corrida} en {a.escenarios} salas del gimnasio", flush=True)
        r = probar_gimnasio(a.corrida, gen, fitness, a.escenarios, a.dificultad, a.semilla, a.verificar)
        print(f"{r['sala']}: superó {r['llegaron']} de {len(r['partidas'])} salas · avance medio "
              f"{100 * r['avance_medio']:.1f} %", flush=True)
        if a.video:
            from .gimnasio_ver import video as video_gim
            video_gim(a.corrida, a.dificultad, a.semilla, min(a.escenarios, 10), a.verificar, gen)
        return
    salas = SALAS if a.sala == ["todas"] else (a.sala or [None])
    print(f"probando la mejor red de la gen {gen} de {a.corrida} en {len(salas)} sala(s), {a.escenarios} "
          f"escenarios cada una", flush=True)
    resumen = []
    for sala in salas:
        try:
            r = probar_sala(a.corrida, gen, genoma, fitness, sala, a.objetivo, a.escenarios, a.max_segundos)
        except Exception as e:  # una sala rara (sin salida reconocible, etc.) no detiene las demás
            print(f"{sala}: no se pudo probar ({e})", flush=True)
            continue
        resumen.append(r)
        print(f"{r['sala']}{' (entrenada)' if r['entrenada'] else ''}: avance medio {100 * r['avance_medio']:.1f} % "
              f"(máx {100 * r['avance_max']:.1f} %), llegó {r['llegaron']} de {len(r['partidas'])}", flush=True)
        if a.video:
            video_mejor(a.corrida, gen, 0, sala=r["sala"], objetivo=r["objetivo"], max_segundos=a.max_segundos)
    if len(resumen) > 1:
        print("resumen: " + " · ".join(f"{r['sala']} {100 * r['avance_medio']:.0f} %" for r in resumen), flush=True)


if __name__ == "__main__":
    main()
