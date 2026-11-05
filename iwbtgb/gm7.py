"""Lector del ejecutable GameMaker 7.0 (el mismo formato que vendor/OpenGMK/gm8exe/src/gm7.rs; ver docs/emulador.md).

    uv run python -m iwbtgb.gm7 game/i_wanna_be_the_GB.exe decompiled/gml

exporta todo el código GML a texto UTF-8: scripts/, objects/ (todos los eventos y acciones), timelines/ y rooms/
(código de creación de la room y de cada instancia). `sprites` lee las imágenes de algunos sprites.
"""

from __future__ import annotations

import re
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HEADER_OFFSET = 1980000
EVENT_NAMES = ["create", "destroy", "alarm", "step", "collision", "keyboard", "mouse", "other", "draw",
               "keypress", "keyrelease"]


def reverse_swap_table(seed: int) -> list[int]:
    a, b = 6 + seed % 250, seed // 250
    table = list(range(256))
    for i in range(1, 10001):
        j = 1 + (i * a + b) % 254
        table[j], table[j + 1] = table[j + 1], table[j]
    rev = [0] * 256
    for i in range(1, 256):
        rev[table[i]] = i
    return rev


class Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.d, self.p = data, pos

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.d, self.p)
        self.p += 4
        return v

    def i32(self) -> int:
        (v,) = struct.unpack_from("<i", self.d, self.p)
        self.p += 4
        return v

    def skip(self, n: int) -> None:
        self.p += n

    def bytes(self) -> bytes:
        n = self.u32()
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def expect(self, value: int, what: str) -> None:
        got = self.u32()
        if got != value:
            raise ValueError(f"{what}: se esperaba versión {value}, vino {got} (offset {self.p - 4})")


def text(b: bytes) -> str:
    return b.decode("shift_jis", "replace").replace("\r\n", "\n")


def load_gamedata(exe: bytes) -> bytes:
    r = Reader(exe, HEADER_OFFSET)
    if r.u32() != 1234321 or r.u32() != 700:
        raise ValueError("no es un ejecutable GameMaker 7.0")
    r.u32()                       # debug
    r.expect(702, "settings")
    r.skip(4 * 22)                # settings hasta la barra de carga
    if r.u32():                   # barra de carga: imágenes de fondo y frente
        for _ in range(2):
            if r.u32():
                r.bytes()
    if r.u32():                   # imagen de carga propia
        r.bytes()
    r.skip(4 * 8)
    r.bytes()                     # nombre de la DLL
    r.bytes()                     # DLL
    data = zlib.decompress(r.bytes())

    r = Reader(data)
    bill, fred = r.u32(), r.u32()
    r.skip(4 * bill)
    seed = r.u32()
    r.skip(4 * fred)
    rev = reverse_swap_table(seed)
    start = r.p
    out = bytearray([data[start]])
    out += bytes((rev[b] - (start + 1 + i)) & 0xFF for i, b in enumerate(data[start + 1:]))
    return bytes(out)


def read_actions(r: Reader) -> list[dict]:
    r.expect(400, "evento")
    actions = []
    for _ in range(r.u32()):
        r.expect(440, "acción")
        lib, aid, kind, _rel, question, _applies, exe_type = (r.u32() for _ in range(7))
        fn, code = r.bytes(), r.bytes()
        argc = r.u32()
        r.expect(8, "tipos de argumentos")
        r.skip(4 * 8)
        applies_to, relative = r.i32(), r.u32()
        r.expect(8, "argumentos")
        args = [r.bytes() for _ in range(8)]
        negate = r.u32()
        actions.append(dict(lib=lib, id=aid, kind=kind, question=question, exe_type=exe_type, fn=fn,
                            code=code, args=args[:argc], applies_to=applies_to, relative=relative, negate=negate))
    return actions


def format_action(a: dict) -> str:
    target = {-1: "self", -2: "other"}.get(a["applies_to"], f"objeto {a['applies_to']}")
    if a["lib"] == 1 and a["id"] == 603:   # "execute code"
        return f"// ejecutar código (aplica a: {target})\n{text(a['args'][0])}\n"
    desc = f"lib {a['lib']} acción {a['id']}"
    if a["fn"]:
        desc += f" -> {text(a['fn'])}"
    args = ", ".join(repr(text(x)) for x in a["args"])
    return f"// acción D&D: {desc}({args}) [aplica a: {target}{', relativa' if a['relative'] else ''}" \
           f"{', NOT' if a['negate'] else ''}]\n"


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name) or "_"


def hasta_sprites(exe_path: Path) -> Reader:
    """Lector posicionado en la cantidad de sprites (después de la cabecera, las extensiones y los sonidos)."""
    r = Reader(load_gamedata(exe_path.read_bytes()), 0)
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
    r.expect(400, "sprites")
    return r


def color_clave(bgr0: np.ndarray) -> np.ndarray:
    """Color transparente de una imagen (h, w, 4) BGR + 4.º byte: el pixel inferior izquierdo, salvo que la fila de
    abajo sea de relleno (4.º byte en 0 en toda ella): entonces, el color más común entre los píxeles con byte 0."""
    if (bgr0[-1, :, 3] == 0).all():
        fondo = bgr0[bgr0[..., 3] == 0][:, :3]
        return np.array(Counter(map(tuple, fondo.tolist())).most_common(1)[0][0], np.uint8)
    return bgr0[-1, 0, :3]


def sprites(exe_path: Path, nombres) -> dict[int, dict]:
    """{índice: {nombre, xorig, yorig, frames: [RGBA (h, w, 4)]}} de los sprites con esos nombres."""
    r = hasta_sprites(exe_path)
    out = {}
    for idx in range(r.u32()):
        if not r.u32():
            continue
        nombre = text(r.bytes())
        r.expect(542, "sprite")
        campos = [r.i32() for _ in range(13)]
        frames = []
        for _ in range(r.u32()):
            r.expect(540, "imagen")
            if r.i32() == -1:
                continue
            w, h = r.u32(), r.u32()
            datos = r.bytes()
            if nombre not in nombres:
                continue
            bgr0 = np.frombuffer(zlib.decompress(datos), np.uint8)[: w * h * 4].reshape(h, w, 4)
            rgba = np.empty((h, w, 4), np.uint8)
            rgba[..., :3] = bgr0[..., 2::-1]
            rgba[..., 3] = np.where((bgr0[..., :3] == color_clave(bgr0)).all(-1), 0, 255)
            frames.append(rgba)
        if nombre in nombres:
            out[idx] = {"nombre": nombre, "xorig": campos[11], "yorig": campos[12], "frames": frames}
    return out


def export(exe_path: Path, out: Path) -> dict[str, int]:
    r = hasta_sprites(exe_path)
    rev12345 = reverse_swap_table(12345)

    def image():
        r.expect(540, "imagen")
        if r.i32() != -1:
            r.u32(); r.u32(); r.bytes()

    sprite_names = []
    for _ in range(r.u32()):
        if not r.u32():
            sprite_names.append(None)
            continue
        sprite_names.append(text(r.bytes()))
        r.expect(542, "sprite")
        r.skip(4 * 13)
        for _ in range(r.u32()):
            image()

    r.expect(400, "fondos")
    for _ in range(r.u32()):
        if r.u32():
            r.bytes(); r.expect(543, "fondo"); r.skip(4 * 5)
            if r.u32():
                image()

    r.expect(420, "paths")
    for _ in range(r.u32()):
        if r.u32():
            r.bytes(); r.expect(530, "path"); r.skip(12); r.skip(24 * r.u32())

    counts = {"scripts": 0, "objects": 0, "rooms": 0}
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    r.expect(400, "scripts")
    for _ in range(r.u32()):
        if not r.u32():
            continue
        name = text(r.bytes())
        r.expect(400, "script")
        blob = r.bytes()
        source = b""
        if blob:
            plain = bytes(rev12345[b] for b in zlib.decompress(blob))
            source = Reader(plain).bytes()
        (out / "scripts" / f"{safe_name(name)}.gml").write_text(text(source), encoding="utf-8")
        counts["scripts"] += 1

    r.expect(540, "fuentes")
    for _ in range(r.u32()):
        if r.u32():
            r.bytes(); r.expect(540, "fuente"); r.bytes(); r.skip(4 * 5); r.skip(4 * 0x600); r.skip(8); r.bytes()

    r.expect(500, "timelines")
    timelines = []
    for _ in range(r.u32()):
        if not r.u32():
            continue
        name = text(r.bytes())
        r.expect(500, "timeline")
        moments = []
        for _ in range(r.u32()):
            moments.append((r.u32(), read_actions(r)))
        timelines.append((name, moments))

    (out / "objects").mkdir(parents=True, exist_ok=True)
    r.expect(400, "objetos")
    object_names = []
    objects = []
    for _ in range(r.u32()):
        if not r.u32():
            object_names.append(None)
            continue
        name = text(r.bytes())
        object_names.append(name)
        r.expect(430, "objeto")
        sprite, solid, visible, depth, persistent, parent, mask = (r.i32() for _ in range(7))
        events = []
        for ev in range(r.u32() + 1):
            while (sub := r.i32()) != -1:
                events.append((ev, sub, read_actions(r)))
        objects.append((name, sprite, solid, visible, depth, persistent, parent, mask, events))

    def obj_name(i: int) -> str:
        return object_names[i] if 0 <= i < len(object_names) and object_names[i] else str(i)

    for name, sprite, solid, visible, depth, persistent, parent, mask, events in objects:
        sprite_name = sprite_names[sprite] if 0 <= sprite < len(sprite_names) else "ninguno"
        lines = [f"// objeto {name}: sprite={sprite_name} solid={solid} visible={visible} depth={depth} "
                 f"persistent={persistent} parent={obj_name(parent) if parent >= 0 else 'ninguno'}\n"]
        for ev, sub, actions in events:
            sub_desc = obj_name(sub) if ev == 4 else str(sub)
            lines.append(f"\n// ===== evento {EVENT_NAMES[ev]} ({sub_desc}) =====\n")
            lines += [format_action(a) for a in actions]
        (out / "objects" / f"{safe_name(name)}.gml").write_text("".join(lines), encoding="utf-8")
        counts["objects"] += 1

    (out / "timelines").mkdir(parents=True, exist_ok=True)
    for name, moments in timelines:
        lines = []
        for step, actions in moments:
            lines.append(f"\n// ===== momento {step} =====\n")
            lines += [format_action(a) for a in actions]
        (out / "timelines" / f"{safe_name(name)}.gml").write_text("".join(lines), encoding="utf-8")

    (out / "rooms").mkdir(parents=True, exist_ok=True)
    r.expect(420, "rooms")
    for _ in range(r.u32()):
        if not r.u32():
            continue
        name = text(r.bytes())
        r.expect(541, "room")
        r.bytes()
        r.skip(4 * 6)
        lines = [f"// ===== código de creación de la room {name} =====\n{text(r.bytes())}\n"]
        r.expect(8, "fondos de room")
        r.skip(4 * 10 * 8)
        r.u32()
        r.expect(8, "vistas")
        r.skip(4 * 14 * 8)
        for _ in range(r.u32()):
            x, y, obj, iid = r.i32(), r.i32(), r.i32(), r.i32()
            code = r.bytes()
            if code:
                lines.append(f"\n// ===== instancia {iid} de {obj_name(obj)} en ({x}, {y}) =====\n{text(code)}\n")
        r.skip(36 * r.u32())
        (out / "rooms" / f"{safe_name(name)}.gml").write_text("".join(lines), encoding="utf-8")
        counts["rooms"] += 1
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(export(Path(sys.argv[1]), Path(sys.argv[2])))
