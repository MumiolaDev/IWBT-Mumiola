"""Niveles donde juega una red: el de su corrida u otros que nunca vio (para medir si generaliza).

Cada nivel necesita sus ids de room y su mapa de distancias a la salida (`iwbtgb.distancia`). El de la sala de la
corrida está en su carpeta; los demás se arman la primera vez (unos segundos) y quedan en
<corridas de la línea>/_salas/<dificultad>_<sala>_<objetivo>/, compartidos por las corridas de la línea.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .. import distancia
from ..emulador import arrancar


def siguiente_sala(sala: str) -> str:
    """rStage03 → rStage04 (la salida normal de cada stage)."""
    m = re.fullmatch(r"rStage(\d+)", sala)
    if not m:
        raise ValueError(f"no sé cuál es la sala siguiente a {sala}: indica --objetivo")
    return f"rStage{int(m[1]) + 1:02d}"


def esperas(n: int, espera_max: int) -> list[int]:
    """`n` esperas repartidas entre 0 y `espera_max` (la primera, 0): las mismas en cada prueba, para comparar."""
    return [int(round(e)) for e in np.linspace(0, espera_max, n)] if n > 1 else [0]


def resolver_sala(cfg) -> dict:
    """Completa en `cfg` los ids y el tamaño de `start_room` y `sala_objetivo`, y arma su mapa de distancias."""
    emu = arrancar(cfg.difficulty, cfg.start_room, cfg.seed_juego)
    try:
        info = emu.info()
        cfg.room_entrenamiento, cfg.room_ancho, cfg.room_alto = info.id, info.width, info.height
        cfg.room_objetivo = int(emu.exec(f"return {cfg.sala_objetivo}"))
        jugador = emu.query("player")[0]
        return distancia.construir(emu, cfg.room_objetivo, info.width, info.height, (jugador.x, jugador.y))
    finally:
        emu.close()


def preparar_corrida(cfg, carpeta: Path) -> dict:
    """Ids de las rooms y mapa de distancias de la sala de la corrida (también en las del gimnasio: con él se
    juega después en el nivel)."""
    ruta = carpeta / "distancia.npz"
    if cfg.room_entrenamiento >= 0 and ruta.exists():
        return distancia.cargar(ruta)
    mapa = resolver_sala(cfg)
    distancia.guardar(ruta, mapa)
    return mapa


def config_en_sala(linea, carpeta: Path, cfg, sala: str | None = None,
                   objetivo: str | None = None) -> tuple[object, dict, bool]:
    """(config para jugar en `sala`, mapa de distancias, si es el nivel donde entrenó)."""
    sala = sala or cfg.start_room
    objetivo = objetivo or (cfg.sala_objetivo if sala == cfg.start_room else siguiente_sala(sala))
    nueva = linea.Config.from_dict(cfg.to_dict())
    nueva.entorno = "nivel"
    if sala == cfg.start_room and objetivo == cfg.sala_objetivo and (carpeta / "distancia.npz").exists():
        return nueva, distancia.cargar(carpeta / "distancia.npz"), cfg.entorno == "nivel"
    nueva.start_room, nueva.sala_objetivo, nueva.zonas_prohibidas = sala, objetivo, "auto"
    destino = linea.corridas / "_salas" / f"{cfg.difficulty}_{sala}_{objetivo}"
    if (destino / "sala.json").exists() and (destino / "distancia.npz").exists():
        for k, v in json.loads((destino / "sala.json").read_text()).items():
            setattr(nueva, k, v)
        return nueva, distancia.cargar(destino / "distancia.npz"), False
    print(f"preparando {sala} → {objetivo} (mapa de distancias; solo la primera vez)", flush=True)
    mapa = resolver_sala(nueva)
    destino.mkdir(parents=True, exist_ok=True)
    distancia.guardar(destino / "distancia.npz", mapa)
    (destino / "sala.json").write_text(json.dumps({k: getattr(nueva, k) for k in (
        "room_entrenamiento", "room_objetivo", "room_ancho", "room_alto")}))
    return nueva, mapa, False
