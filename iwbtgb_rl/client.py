"""Cliente del modo `--rl-socket` de gm8emulator (ver vendor/OpenGMK/gm8emulator/src/game/rl.rs)."""

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

ROOT = Path(__file__).resolve().parent.parent
# OPENGMK_EMULATOR: otro binario del emulador (p. ej. un build experimental en otra carpeta)
DEFAULT_EMULATOR = Path(os.environ.get("OPENGMK_EMULATOR")
                        or ROOT / "vendor" / "OpenGMK" / "target" / "release" / "gm8emulator")
DEFAULT_GAME = ROOT / "game" / "i_wanna_be_the_GB.exe"

OP_STEP, OP_SAVE, OP_LOAD, OP_OBS, OP_QUERY, OP_INFO, OP_EXEC, OP_TRACK, OP_QUIT = 1, 2, 3, 4, 5, 6, 7, 8, 255
STATUS_OK, STATUS_ENDED, STATUS_ERROR = 0, 1, 2

# Códigos de tecla virtual de Windows, los que usa GameMaker
VK = {
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "shift": 0x10, "enter": 0x0D, "space": 0x20, "esc": 0x1B,
    "f2": 0x71, "f12": 0x7B,
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
}


# Un registro por frame del objeto seguido con `track()` (ver rl.rs)
TRACE_DTYPE = np.dtype([
    ("room", "<i4"), ("alive", "u1"), ("x", "<f4"), ("y", "<f4"),
    ("sprite", "<i4"), ("image", "<f4"), ("xscale", "<f4"),
])


class EmulatorError(RuntimeError):
    pass


@dataclass
class StepResult:
    status: int
    room: int
    frame: int
    obs: np.ndarray | None
    trace: np.ndarray | None = None

    @property
    def ended(self) -> bool:
        return self.status == STATUS_ENDED


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
    """Lanza una instancia oculta del juego y la controla frame a frame.

    Cada instancia trabaja sobre su propia copia del .exe en un directorio temporal, porque el juego
    guarda sus partidas junto al ejecutable y varias instancias en paralelo se pisarían.
    """

    def __init__(
        self,
        game: str | os.PathLike = DEFAULT_GAME,
        emulator: str | os.PathLike = DEFAULT_EMULATOR,
        seed: int = 0,
        show_window: bool = False,
        verbose: bool = False,
        startup_timeout: float = 60.0,
    ):
        self.workdir = Path(tempfile.mkdtemp(prefix="iwbtgb_"))
        self._archivos: dict[int, dict[str, bytes]] = {}
        exe = self.workdir / Path(game).name
        shutil.copy2(game, exe)
        self.socket_path = self.workdir / "rl.sock"

        env = dict(os.environ)
        if show_window:
            env["OPENGMK_RL_SHOW"] = "1"
        self.proc = subprocess.Popen(
            [str(emulator), str(exe), "--rl-socket", str(self.socket_path), "--rl-seed", str(seed)],
            cwd=self.workdir,
            env=env,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
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

    def step(
        self,
        keys=(),
        frames: int = 1,
        obs: bool = True,
        skip_draw: bool = False,
        trace: bool = False,
        no_draw: bool = False,
        stop_on_gone: bool = False,
    ) -> StepResult:
        """Mantiene presionadas `keys` (nombres de VK o códigos) durante `frames` frames.

        Con `skip_draw` solo se dibuja el último frame: bastante más rápido, pero los eventos Draw
        de los frames intermedios no se ejecutan (si el juego tiene lógica ahí, cambia su comportamiento).
        `no_draw` no dibuja ninguno. `trace` devuelve un registro por frame (`TRACE_DTYPE`) del objeto
        elegido con `track()`, y `stop_on_gone` corta el paso en cuanto ese objeto deja de existir.
        """
        codes = sorted({VK[k] if isinstance(k, str) else int(k) for k in keys})
        flags = int(obs) | (int(skip_draw) << 1) | (int(trace) << 2) | (int(no_draw) << 3) | (int(stop_on_gone) << 4)
        self._send(struct.pack(f"<BB{len(codes)}BIB", OP_STEP, len(codes), *codes, frames, flags))
        (status,) = self._unpack("<B")
        if status == STATUS_ERROR:
            msg = self._recv_str().decode("utf-8", "replace")
            raise EmulatorError(f"error de ejecución GML: {msg}")
        room, frame = self._unpack("<iQ")
        image = self._recv_obs() if obs else None
        records = None
        if trace:
            (n,) = self._unpack("<I")
            records = np.frombuffer(self._recv(n * TRACE_DTYPE.itemsize), dtype=TRACE_DTYPE)
        return StepResult(status, room, frame, image, records)

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
        """Ejecuta GML como código de creación de room y devuelve el valor de su `return`.

        Útil para leer o fijar estado del juego: `emu.exec("return global.death")`,
        `emu.exec("room_goto(rStage01)")`. Un cambio de room solicitado se aplica de inmediato.
        """
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
        try:
            self.sock.close()
        except Exception:
            pass
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
