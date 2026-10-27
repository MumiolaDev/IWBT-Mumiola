"""Salas donde jugar una red ya entrenada: la de su corrida u otras que nunca vio (para medir si generaliza).

Cada sala necesita sus ids de room y su mapa de distancias a la salida (`ga/distancia.py`). El de la sala de
entrenamiento está en la corrida; los demás se arman la primera vez (unos segundos) y quedan en
corridas_redes/_salas/<dificultad>_<sala>_<objetivo>/, compartidos por todas las corridas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import almacen
from .config import Config

SALAS = [f"rStage{i:02d}" for i in range(1, 18)]
CACHE = almacen.CORRIDAS / "_salas"


def siguiente_sala(sala: str) -> str:
    """rStage03 → rStage04 (la salida normal de cada stage)."""
    m = re.fullmatch(r"rStage(\d+)", sala)
    if not m:
        raise ValueError(f"no sé cuál es la sala siguiente a {sala}: indica --objetivo")
    return f"rStage{int(m[1]) + 1:02d}"


def esperas(n: int, espera_max: int) -> list[int]:
    """`n` esperas repartidas entre 0 y `espera_max` (la primera, 0): las mismas en cada prueba, para comparar."""
    return [int(round(e)) for e in np.linspace(0, espera_max, n)] if n > 1 else [0]


def config_en_sala(carpeta: Path, cfg: Config, sala: str | None = None,
                   objetivo: str | None = None) -> tuple[Config, dict, bool]:
    """(config para jugar en `sala`, mapa de distancias, si es la sala de entrenamiento)."""
    from ga import distancia

    sala = sala or cfg.start_room
    objetivo = objetivo or (cfg.sala_objetivo if sala == cfg.start_room else siguiente_sala(sala))
    if sala == cfg.start_room and objetivo == cfg.sala_objetivo:
        return cfg, distancia.cargar(carpeta / "distancia.npz"), True
    nueva = Config.from_dict(cfg.to_dict())
    nueva.start_room, nueva.sala_objetivo, nueva.zonas_prohibidas = sala, objetivo, "auto"
    destino = CACHE / f"{cfg.difficulty}_{sala}_{objetivo}"
    if (destino / "sala.json").exists() and (destino / "distancia.npz").exists():
        for k, v in json.loads((destino / "sala.json").read_text()).items():
            setattr(nueva, k, v)
        return nueva, distancia.cargar(destino / "distancia.npz"), False
    from .entrenar import resolver_sala
    print(f"preparando {sala} → {objetivo} (mapa de distancias; solo la primera vez)", flush=True)
    mapa = resolver_sala(nueva)
    destino.mkdir(parents=True, exist_ok=True)
    distancia.guardar(destino / "distancia.npz", mapa)
    (destino / "sala.json").write_text(json.dumps({k: getattr(nueva, k) for k in (
        "room_entrenamiento", "room_objetivo", "room_ancho", "room_alto")}))
    return nueva, mapa, False
