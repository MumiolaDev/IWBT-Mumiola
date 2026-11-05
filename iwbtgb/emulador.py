"""Cliente del modo `--rl-socket` de gm8emulator (protocolo en vendor/OpenGMK/gm8emulator/src/game/rl.rs y en
docs/emulador.md) y arranque del juego hasta el comienzo de una sala."""

from __future__ import annotations

import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rutas import EMULADOR, JUEGO

OP_STEP, OP_SAVE, OP_LOAD, OP_OBS, OP_QUERY, OP_INFO, OP_EXEC, OP_TRACK, OP_QUIT = 1, 2, 3, 4, 5, 6, 7, 8, 255
OP_GRID_CONFIG, OP_GRID, OP_RAYS_CONFIG = 9, 10, 11
RAYO_NADA = 0xFFFF                # distancia de un rayo que no vio ese canal
STEP_GRID = 32
STATUS_OK, STATUS_ENDED, STATUS_ERROR = 0, 1, 2

# Códigos de tecla virtual de Windows, los que usa GameMaker
VK = {
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "shift": 0x10, "enter": 0x0D, "space": 0x20, "esc": 0x1B,
    "f2": 0x71, "f12": 0x7B,
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
}

# Combinaciones de teclas: movimiento × salto (Shift) × disparo (Z)
ACTIONS: list[tuple[str, ...]] = [
    (), ("left",), ("right",), ("shift",), ("left", "shift"), ("right", "shift"),
    ("z",), ("left", "z"), ("right", "z"), ("shift", "z"), ("left", "shift", "z"), ("right", "shift", "z"),
]

# Portales de dificultad (instancias de warpStart) en rSelectStage
DIFFICULTY_WARPS = {"medium": (96, 384), "hard": (672, 384), "veryhard": (64, 224), "impossible": (704, 224)}
SALAS = [f"rStage{i:02d}" for i in range(1, 18)]
SLOT_INICIO = 0    # savestate del comienzo de la sala que deja `arrancar`

# Un registro por frame del objeto seguido con `track()`
TRACE_DTYPE = np.dtype([
    ("room", "<i4"), ("alive", "u1"), ("x", "<f4"), ("y", "<f4"),
    ("sprite", "<i4"), ("image", "<f4"), ("xscale", "<f4"),
])


def limpiar_huerfanos(minutos: float = 10.0) -> None:
    """Borra las carpetas iwbtgb_* de emuladores que ya no existen (un proceso muerto sin close() las deja, y /tmp
    está en RAM). Huérfana = su socket no acepta conexiones, o no tiene socket y tiene más de `minutos`."""
    for d in Path(tempfile.gettempdir()).glob("iwbtgb_*"):
        sock = d / "rl.sock"
        try:
            if sock.exists():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(str(sock))
                continue  # hay un emulador escuchando
            if time.time() - d.stat().st_mtime < minutos * 60:
                continue  # recién creada: su emulador puede estar arrancando
        except ConnectionRefusedError:
            pass
        except OSError:
            continue
        shutil.rmtree(d, ignore_errors=True)


class EmulatorError(RuntimeError):
    pass


@dataclass
class StepResult:
    status: int
    room: int
    frame: int
    obs: np.ndarray | None
    trace: np.ndarray | None = None
    grid: Grid | None = None

    @property
    def ended(self) -> bool:
        return self.status == STATUS_ENDED


@dataclass
class Grid:
    """Vista semántica alrededor del objeto de `track()` (ver `grid_config`)."""
    room: int
    centro: tuple[int, int]      # píxel de la room en el centro de la grilla
    vivo: bool                   # existe el objeto seguido (si no, la grilla queda donde se lo vio por última vez)
    hspeed: float
    vspeed: float
    campos: tuple[float, ...]    # variables pedidas del objeto seguido (NaN si no son números)
    canales: np.ndarray          # uint8 (canales, alto, ancho): fracción de cada celda cubierta, 0-255
    # con `rays_config`: px de cada rayo al primer píxel de cada canal y, en la última columna, al borde de la
    # pantalla (RAYO_NADA = no lo alcanzó); desde el centro de la caja del jugador, que mira hacia `xscale`
    rayos: np.ndarray | None = None     # uint16 (rayos, canales + 1)
    origen: tuple[int, int] = (0, 0)
    xscale: float = 0.0


@dataclass
class Instance:
    id: int
    x: float
    y: float
    hspeed: float
    vspeed: float


@dataclass
class RoomInfo:
    id: int
    width: int
    height: int
    speed: int
    name: str
    frame: int


class Emulator:
    """Una instancia oculta del juego, controlada frame a frame.

    Cada instancia trabaja sobre su propia copia del .exe en un directorio temporal, porque el juego guarda sus
    partidas junto al ejecutable y varias instancias en paralelo se pisarían.
    """

    def __init__(self, game: str | os.PathLike = JUEGO, emulator: str | os.PathLike = EMULADOR, seed: int = 0,
                 show_window: bool = False, verbose: bool = False, startup_timeout: float = 60.0):
        limpiar_huerfanos()
        self.workdir = Path(tempfile.mkdtemp(prefix="iwbtgb_"))
        self._archivos: dict[int, dict[str, bytes]] = {}
        self._grid = (0, 0, 0, 0)  # canales, alto, ancho, campos (de grid_config)
        self._rayos = 0            # rayos (de rays_config)
        self.file = None
        exe = self.workdir / Path(game).name
        shutil.copy2(game, exe)
        self.socket_path = self.workdir / "rl.sock"

        env = dict(os.environ)
        if show_window:
            env["OPENGMK_RL_SHOW"] = "1"
        self.proc = subprocess.Popen(
            [str(emulator), str(exe), "--rl-socket", str(self.socket_path), "--rl-seed", str(seed)],
            cwd=self.workdir, env=env,
            stdout=None if verbose else subprocess.DEVNULL, stderr=None if verbose else subprocess.DEVNULL,
        )

        deadline = time.monotonic() + startup_timeout
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while True:
            if self.proc.poll() is not None:
                self._cleanup()
                raise EmulatorError(f"el emulador terminó al arrancar (código {self.proc.returncode})")
            try:
                self.sock.connect(str(self.socket_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    self.close()
                    raise EmulatorError("el emulador no abrió el socket a tiempo")
                time.sleep(0.05)
        self.file = self.sock.makefile("rwb", buffering=1 << 20)

    # -- protocolo ---------------------------------------------------------

    def _send(self, data: bytes) -> None:
        self.file.write(data)
        self.file.flush()

    def _recv(self, n: int) -> bytes:
        data = self.file.read(n)
        if data is None or len(data) != n:
            raise EmulatorError("conexión cerrada por el emulador")
        return data

    def _unpack(self, fmt: str):
        return struct.unpack(fmt, self._recv(struct.calcsize(fmt)))

    def _recv_str(self) -> bytes:
        (n,) = self._unpack("<H")
        return self._recv(n)

    def _recv_obs(self) -> np.ndarray:
        w, h = self._unpack("<II")
        return np.frombuffer(self._recv(w * h * 3), dtype=np.uint8).reshape(h, w, 3)

    # -- API ---------------------------------------------------------------

    def step(self, keys=(), frames: int = 1, obs: bool = True, skip_draw: bool = False, trace: bool = False,
             no_draw: bool = False, stop_on_gone: bool = False, grid: bool = False) -> StepResult:
        """Mantiene presionadas `keys` (nombres de VK o códigos) durante `frames` frames.

        `skip_draw` dibuja solo el último frame (los eventos Draw intermedios no corren) y `no_draw`, ninguno.
        `trace` devuelve un registro por frame (`TRACE_DTYPE`) del objeto elegido con `track()`; `stop_on_gone` corta
        el paso en cuanto ese objeto deja de existir; `grid` devuelve la vista semántica tras el último frame.
        """
        codes = sorted({VK[k] if isinstance(k, str) else int(k) for k in keys})
        flags = int(obs) | (int(skip_draw) << 1) | (int(trace) << 2) | (int(no_draw) << 3) | (int(stop_on_gone) << 4)
        if grid:
            flags |= STEP_GRID
        self._send(struct.pack(f"<BB{len(codes)}BIB", OP_STEP, len(codes), *codes, frames, flags))
        (status,) = self._unpack("<B")
        error = self._recv_str().decode("utf-8", "replace") if status == STATUS_ERROR else None
        # Tras un error el emulador manda igual el resto de la respuesta: se lee entera para no desfasar el protocolo
        room, frame = self._unpack("<iQ")
        image = self._recv_obs() if obs else None
        records = None
        if trace:
            (n,) = self._unpack("<I")
            records = np.frombuffer(self._recv(n * TRACE_DTYPE.itemsize), dtype=TRACE_DTYPE)
        vista = self._recv_grid() if grid else None
        if error is not None:
            raise EmulatorError(f"error de ejecución GML: {error}")
        return StepResult(status, room, frame, image, records, vista)

    def grid_config(self, celda: int, ancho: int, alto: int, muestras: int,
                    canales: list[tuple[list[str], list[str], bool]], campos: list[str] = ()) -> list[str]:
        """Configura la vista semántica: grilla de `ancho`×`alto` celdas de `celda` px centrada en el objeto de
        `track()`. Cada canal es (objetos, excluidos, solo_visibles): una celda vale la fracción de sus
        `muestras`×`muestras` puntos que cae dentro de la máscara de colisión de alguna instancia de esos objetos
        o sus hijos, salvo hijos de los excluidos. `campos` son variables del objeto seguido que se devuelven
        (p. ej. "djump"). Devuelve los nombres de objetos que no existen en el juego."""
        def nombres(lista) -> bytes:
            out = struct.pack("<B", len(lista))
            for n in lista:
                b = n.encode("shift_jis")
                out += struct.pack("<H", len(b)) + b
            return out
        datos = struct.pack("<BBHHB", OP_GRID_CONFIG, celda, ancho, alto, muestras) + struct.pack("<B", len(canales))
        for objetos, excluidos, solo_visibles in canales:
            datos += struct.pack("<B", int(solo_visibles)) + nombres(objetos) + nombres(excluidos)
        datos += nombres(list(campos))
        self._send(datos)
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError("no se pudo configurar la grilla")
        (n,) = self._unpack("<H")
        self._grid = (len(canales), alto, ancho, len(campos))
        self._rayos = 0  # el emulador los olvida: hay que configurarlos de nuevo
        return [self._recv_str().decode("shift_jis", "replace") for _ in range(n)]

    def rays_config(self, n: int, paso: int, banderas: list[int]) -> None:
        """Rayos desde el centro de la caja del jugador hasta el borde de la pantalla (la vista 0, o la room sin
        vistas), con los canales de `grid_config` (llamarla después). `banderas`: una por canal, 1 = corta el rayo,
        2 = se ve (1 | 2 = se ve y lo corta). Cada `paso` px se mira el canal en ese punto. El rayo i apunta a
        i·2π/n, en sentido antihorario en la pantalla y empezando hacia la derecha. n = 0 los apaga."""
        self._send(struct.pack(f"<BHBB{len(banderas)}B", OP_RAYS_CONFIG, n, paso, len(banderas), *banderas))
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError("no se pudieron configurar los rayos (¿una bandera por canal de la grilla?)")
        self._rayos = n

    def grid(self) -> Grid:
        """Vista semántica del frame actual."""
        self._send(struct.pack("<B", OP_GRID))
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError("no se pudo leer la grilla")
        return self._recv_grid()

    def _recv_grid(self) -> Grid:
        c, h, w, k = self._grid
        room, cx, cy, vivo, hs, vs = self._unpack("<iiiBdd")
        campos = self._unpack(f"<{k}d") if k else ()
        canales = np.frombuffer(self._recv(c * h * w), dtype=np.uint8).reshape(c, h, w)
        if not self._rayos:
            return Grid(room, (cx, cy), bool(vivo), hs, vs, tuple(campos), canales)
        ox, oy, xscale = self._unpack("<iif")
        rayos = np.frombuffer(self._recv(self._rayos * (c + 1) * 2), dtype="<u2").reshape(self._rayos, c + 1)
        return Grid(room, (cx, cy), bool(vivo), hs, vs, tuple(campos), canales, rayos, (ox, oy), xscale)

    def track(self, object_name: str = "player") -> None:
        """Elige el objeto cuya primera instancia describen las trazas de `step(trace=True)`."""
        name = object_name.encode("shift_jis")
        self._send(struct.pack("<BH", OP_TRACK, len(name)) + name)
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError(f"no existe el objeto {object_name}")

    def save(self, slot: int = 0) -> None:
        """Savestate del emulador. Incluye los archivos de partida que escribe el juego (`save/`), que el
        emulador no guarda: sin ellos, cargar un savestate y apretar R llevaría al save de otra partida."""
        self._send(struct.pack("<BI", OP_SAVE, slot))
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError("no se pudo guardar el savestate")
        self._archivos[slot] = self.archivos()

    def load(self, slot: int = 0) -> None:
        self._send(struct.pack("<BI", OP_LOAD, slot))
        if self._unpack("<B")[0] != STATUS_OK:
            raise EmulatorError(f"no existe el savestate {slot}")
        if slot in self._archivos:
            self.restaurar_archivos(self._archivos[slot])

    # Las capturas .bmp que hace el juego al guardar solo se muestran en el menú: no se copian
    def archivos(self) -> dict[str, bytes]:
        carpeta = self.workdir / "save"
        if not carpeta.is_dir():
            return {}
        return {p.name: p.read_bytes() for p in carpeta.iterdir() if p.is_file() and p.suffix != ".bmp"}

    def restaurar_archivos(self, archivos: dict[str, bytes]) -> None:
        carpeta = self.workdir / "save"
        actuales = self.archivos()
        for nombre in actuales.keys() - archivos.keys():
            (carpeta / nombre).unlink()
        for nombre, datos in archivos.items():
            if actuales.get(nombre) != datos:
                carpeta.mkdir(exist_ok=True)
                (carpeta / nombre).write_bytes(datos)

    def observe(self) -> np.ndarray:
        self._send(struct.pack("<B", OP_OBS))
        return self._recv_obs()

    def query(self, object_name: str) -> list[Instance]:
        """Instancias activas de un objeto (incluye sus hijos), p. ej. `query("player")`."""
        name = object_name.encode("shift_jis")
        self._send(struct.pack("<BH", OP_QUERY, len(name)) + name)
        (n,) = self._unpack("<I")
        return [Instance(*self._unpack("<idddd")) for _ in range(n)]

    def info(self) -> RoomInfo:
        self._send(struct.pack("<B", OP_INFO))
        room, w, h, speed = self._unpack("<iIII")
        name = self._recv_str().decode("shift_jis", "replace")
        (frame,) = self._unpack("<Q")
        return RoomInfo(room, w, h, speed, name, frame)

    def exec(self, code: str) -> float | str:
        """Ejecuta GML como código de creación de room y devuelve el valor de su `return`, p. ej.
        `emu.exec("return global.death")` o `emu.exec("room_goto(rStage01)")` (el cambio de room es inmediato)."""
        data = code.encode("shift_jis")
        self._send(struct.pack("<BI", OP_EXEC, len(data)) + data)
        (status,) = self._unpack("<B")
        if status != STATUS_OK:
            raise EmulatorError(f"error en GML: {self._recv_str().decode('utf-8', 'replace')}")
        (kind,) = self._unpack("<B")
        if kind == 0:
            return self._unpack("<d")[0]
        return self._recv_str().decode("shift_jis", "replace")

    def close(self) -> None:
        try:
            self._send(struct.pack("<B", OP_QUIT))
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self._cleanup()

    def _cleanup(self) -> None:
        for recurso in (self.file, self.sock):
            try:
                if recurso is not None:
                    recurso.close()
            except Exception:
                pass
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def arrancar(difficulty: str = "medium", start_room: str | None = None, seed: int = 0,
             show_window: bool = False) -> Emulator:
    """Un emulador al comienzo de una sala, con el savestate SLOT_INICIO ahí: título → menú → sala de dificultades →
    portal de `difficulty` → rStage01 y, con `start_room`, `room_goto` a esa sala."""
    if difficulty not in DIFFICULTY_WARPS:
        raise ValueError(f"la dificultad debe ser una de {list(DIFFICULTY_WARPS)}")
    emu = Emulator(seed=seed, show_window=show_window)
    try:
        emu.step(frames=60, obs=False)
        for _ in range(50):
            if emu.info().name == "rSelectStage" and emu.query("player"):
                break
            emu.step(["shift"], frames=2, obs=False)
            emu.step(frames=20, obs=False)
        else:
            raise RuntimeError("no se llegó a la sala de selección de dificultad")

        wx, wy = DIFFICULTY_WARPS[difficulty]
        emu.exec(f"with (player) {{ x = {wx + 16}; y = {wy + 16}; }}")
        for _ in range(100):
            emu.step(frames=5, obs=False)
            if emu.info().name.startswith("rStage") and emu.query("player"):
                break
        else:
            raise RuntimeError("el portal de dificultad no llevó a un nivel")

        if start_room is not None:
            # `player` es persistente: room_goto lo deja en las coordenadas de la room anterior, que pueden quedar
            # dentro de un muro. Se lo lleva al playerStart de la room nueva
            emu.exec(f"room_goto({start_room})")
            emu.step(frames=1, obs=False)
            emu.exec("with (player) { x = playerStart.x + 17; y = playerStart.y + 23; }")
            if not emu.exec("with (player) return place_free(x, y); return 0"):
                raise RuntimeError(f"el jugador quedó atascado al iniciar en {start_room}")
        emu.step(frames=30, obs=True)
        emu.save(SLOT_INICIO)
    except BaseException:
        emu.close()
        raise
    return emu
