"""Traza compacta de una partida (un registro por frame del jugador), la que se guarda en disco y con la que se
dibujan poblaciones enteras sin volver a emularlas."""

from __future__ import annotations

import numpy as np

SIN_TECLAS, TECLA_R = 254, 255  # valores de `accion` que no son una acción de ACTIONS (espera y reintento)

TRAZA_DTYPE = np.dtype([
    ("room", "<i2"), ("alive", "u1"), ("x", "<f4"), ("y", "<f4"),
    ("sprite", "<i2"), ("image", "u1"), ("xscale", "i1"),
    ("gen", "<u2"),     # GA: gen que se jugaba en ese frame
    ("accion", "u1"),   # teclas de ese frame: índice de ACTIONS, SIN_TECLAS o TECLA_R
])


def compactar(traza: np.ndarray, gen: int = 0, accion: int = SIN_TECLAS) -> np.ndarray:
    """De `emulador.TRACE_DTYPE` (lo que manda el emulador) a `TRAZA_DTYPE`."""
    out = np.zeros(len(traza), TRAZA_DTYPE)
    out["gen"] = gen
    out["accion"] = accion
    out["room"] = traza["room"]
    out["alive"] = traza["alive"]
    out["x"] = traza["x"]
    out["y"] = traza["y"]
    out["sprite"] = traza["sprite"]
    out["image"] = np.floor(traza["image"]).astype(np.int64) % 256
    out["xscale"] = np.sign(traza["xscale"]).astype(np.int8)
    return out


def leer_save(emu) -> bytes:
    """Sala y posición guardadas en el archivo de partida del juego (sus primeros 9 bytes): cambian al guardar."""
    archivos = [p for p in sorted((emu.workdir / "save").glob("save[0-9]*")) if p.suffix == ""]
    return archivos[0].read_bytes()[:9] if archivos else b""
