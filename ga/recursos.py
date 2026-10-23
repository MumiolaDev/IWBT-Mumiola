"""Recursos gráficos para los videos: mapas completos de las rooms y sprites del jugador.

Se guardan en caché en corridas/<nombre>/recursos/ (mapas como PNG, sprites como PNG RGBA + JSON con el
origen de cada uno).
"""

from __future__ import annotations

import importlib.util
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from iwbtgb_rl import IWannaBeTheGBEnv
from iwbtgb_rl.client import DEFAULT_GAME

from .nucleo import Config

VISTA_W, VISTA_H = 800, 600

ROOT = Path(__file__).resolve().parent.parent
SPRITES_JUGADOR = ("sprPlayerIdle", "sprPlayerRunning", "sprPlayerJump", "sprPlayerFall")


def _exportador():
    spec = importlib.util.spec_from_file_location("export_gml", ROOT / "tools" / "export_gml.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extraer_sprites(destino: Path, exe: Path = DEFAULT_GAME) -> dict:
    """Extrae del .exe (GM7) los sprites del jugador. Devuelve {índice: {nombre, xorig, yorig, frames}}."""
    ex = _exportador()
    r = ex.Reader(ex.load_gamedata(exe.read_bytes()), 0)
    r.skip(24)
    r.expect(700, "extensiones")
    if r.u32():
        raise ValueError("extensiones GM7 no soportadas")
    r.expect(400, "sonidos")
    for _ in range(r.u32()):
        if r.u32():
            r.bytes(); r.u32(); r.u32(); r.bytes(); r.bytes()
            if r.u32():
                r.bytes()
            r.u32(); r.skip(16); r.u32()

    destino.mkdir(parents=True, exist_ok=True)
    indice = {}
    r.expect(400, "sprites")
    for idx in range(r.u32()):
        if not r.u32():
            continue
        nombre = ex.text(r.bytes())
        r.expect(542, "sprite")
        campos = [r.i32() for _ in range(13)]
        xorig, yorig = campos[11], campos[12]
        imagenes = []
        for _ in range(r.u32()):
            r.expect(540, "imagen")
            if r.i32() == -1:
                imagenes.append(None)
                continue
            w, h = r.u32(), r.u32()
            datos = zlib.decompress(r.bytes())
            imagenes.append((w, h, datos))
        if nombre not in SPRITES_JUGADOR:
            continue
        frames = []
        for k, img in enumerate(imagenes):
            if img is None:
                continue
            w, h, datos = img
            bgr0 = np.frombuffer(datos, np.uint8)[: w * h * 4].reshape(h, w, 4)
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = bgr0[..., 2], bgr0[..., 1], bgr0[..., 0]
            clave = rgba[h - 1, 0, :3].copy()           # color clave: pixel inferior izquierdo
            rgba[..., 3] = np.where((rgba[..., :3] == clave).all(-1), 0, 255)
            archivo = f"{nombre}_{k}.png"
            Image.fromarray(rgba).save(destino / archivo)
            frames.append(archivo)
        indice[idx] = {"nombre": nombre, "xorig": xorig, "yorig": yorig, "frames": frames}
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


def capturar_mapas(destino: Path, cfg: Config, rooms: set[int]) -> dict[int, dict]:
    """Imagen completa de cada room (todas sus pantallas), sin el jugador y antes de que salten las trampas."""
    destino.mkdir(parents=True, exist_ok=True)
    meta_ruta = destino / "mapas.json"
    meta = {int(k): v for k, v in json.loads(meta_ruta.read_text()).items()} if meta_ruta.exists() else {}
    faltan = [r for r in sorted(rooms) if r not in meta or not (destino / f"mapa_{r}.png").exists()]
    if faltan:
        env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                               seed=cfg.seed_juego)
        emu = env.emu
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
            env.close()
        meta_ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {r: {**meta[r], "imagen": Image.open(destino / f"mapa_{r}.png").convert("RGB")} for r in rooms}
