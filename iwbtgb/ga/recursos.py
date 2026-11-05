"""Recursos gráficos para los videos: mapas completos de las rooms y sprites del jugador.

Quedan en caché en <carpeta>/recursos/ (mapas como PNG, sprites como PNG RGBA + JSON con el origen de cada uno).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import gm7
from ..emulador import arrancar
from ..rutas import JUEGO

VISTA_W, VISTA_H = 800, 600
SPRITES_JUGADOR = ("sprPlayerIdle", "sprPlayerRunning", "sprPlayerJump", "sprPlayerFall")


def extraer_sprites(destino: Path, exe: Path = JUEGO) -> dict:
    """Extrae del .exe los sprites del jugador. Devuelve {índice: {nombre, xorig, yorig, frames}}."""
    destino.mkdir(parents=True, exist_ok=True)
    indice = {}
    for idx, s in gm7.sprites(exe, SPRITES_JUGADOR).items():
        archivos = []
        for k, rgba in enumerate(s["frames"]):
            archivo = f"{s['nombre']}_{k}.png"
            Image.fromarray(rgba).save(destino / archivo)
            archivos.append(archivo)
        indice[idx] = {"nombre": s["nombre"], "xorig": s["xorig"], "yorig": s["yorig"], "frames": archivos}
    (destino / "sprites.json").write_text(json.dumps(indice, indent=2))
    return indice


def cargar_sprites(destino: Path) -> dict[int, dict]:
    if not (destino / "sprites.json").exists():
        extraer_sprites(destino)
    indice = json.loads((destino / "sprites.json").read_text())
    out = {}
    for idx, s in indice.items():
        s = dict(s)
        s["imagenes"] = [Image.open(destino / f).convert("RGBA") for f in s["frames"]]
        out[int(idx)] = s
    return out


def capturar_mapas(destino: Path, cfg, rooms: set[int]) -> dict[int, dict]:
    """Imagen completa de cada room (todas sus pantallas), sin el jugador y antes de que salten las trampas. `cfg`:
    algo con difficulty, start_room y seed_juego (la config de una corrida)."""
    destino.mkdir(parents=True, exist_ok=True)
    meta_ruta = destino / "mapas.json"
    meta = {int(k): v for k, v in json.loads(meta_ruta.read_text()).items()} if meta_ruta.exists() else {}
    faltan = [r for r in sorted(rooms) if r not in meta or not (destino / f"mapa_{r}.png").exists()]
    if faltan:
        emu = arrancar(cfg.difficulty, cfg.start_room, cfg.seed_juego)
        try:
            for room in faltan:
                emu.load(0)
                if emu.info().id != room:
                    emu.exec(f"room_goto({room})")
                    emu.exec("with (player) { x = playerStart.x + 17; y = playerStart.y + 23; }")
                    emu.step(frames=1, obs=False)
                info = emu.info()
                emu.exec("with (camera) instance_destroy(); with (player) visible = false;")
                # La vista mide 800x600 aunque camera la mueva en pasos de 800x608: se recorre de a 800x600
                ancho, alto = info.width, info.height
                lienzo = np.zeros((alto, ancho, 3), np.uint8)
                for vy in range(0, alto, VISTA_H):
                    for vx in range(0, ancho, VISTA_W):
                        emu.exec(f"view_xview[0] = {vx}; view_yview[0] = {vy};")
                        img = emu.step(frames=1, obs=True).obs
                        h, w = min(img.shape[0], alto - vy), min(img.shape[1], ancho - vx)
                        lienzo[vy:vy + h, vx:vx + w] = img[:h, :w]
                Image.fromarray(lienzo).save(destino / f"mapa_{room}.png")
                meta[room] = {"nombre": info.name, "ancho": info.width, "alto": info.height}
        finally:
            emu.close()
        meta_ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {r: {**meta[r], "imagen": Image.open(destino / f"mapa_{r}.png").convert("RGB")} for r in rooms}
