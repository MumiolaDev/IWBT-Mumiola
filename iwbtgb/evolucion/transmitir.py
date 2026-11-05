"""Transmitir en vivo (YouTube, Twitch u otro servidor RTMP) el entrenamiento de una corrida.

    uv run python -m iwbtgb.rayos transmitir --corrida rayos01                       # al servidor configurado
    uv run python -m iwbtgb.rayos transmitir --corrida auto                          # la que se esté entrenando
    uv run python -m iwbtgb.rayos transmitir --corrida rayos01 --destino prueba.flv  # a un archivo, para probar

Lo que se ve es el entrenamiento en vivo de `visualizar entreno` (`--quien alternar`: una sala con toda la población
y otra con la mejor red); en una corrida de un nivel, la mejor red jugándolo. Pasa sola a cada generación nueva y,
con `--corrida auto`, a la corrida de la línea que se esté entrenando cuando cambia.

El servidor y la clave van en ~/.config/iwbtgb/transmision.json, fuera del repositorio:

    {"url": "rtmp://a.rtmp.youtube.com/live2", "clave": "xxxx-xxxx-xxxx-xxxx"}

(o en la variable de entorno IWBTGB_RTMP, con la dirección completa). Los cuadros salen a un ritmo fijo de 25 por
segundo: si el juego va más rápido, espera; si se atrasa, repite el último cuadro (el stream no se corta). Si se cae
la conexión, vuelve a conectarse sola. Corre con tope de memoria, como los en vivo, para dejarlo horas.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

from ..medios import (FONDO, H, REPORTE_CADA, W, con_tope_de_memoria, fuente, memoria_libre_mb, memoria_mb,
                      sigterm_como_ctrl_c)
from ..qt import en_curso
from . import almacen

CONFIG = Path.home() / ".config" / "iwbtgb" / "transmision.json"
FPS = 25
ROJO = (230, 60, 60)


def destino_configurado() -> str | None:
    if os.environ.get("IWBTGB_RTMP"):
        return os.environ["IWBTGB_RTMP"]
    try:
        c = json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return None
    url, clave = c.get("url", "").rstrip("/"), c.get("clave", "")
    return f"{url}/{clave}" if url and clave else (url or None)


def oculto(destino: str) -> str:
    """El destino sin la clave (para los mensajes)."""
    return destino.rsplit("/", 1)[0] + "/…" if destino.startswith("rtmp") else destino


class Transmision:
    """Envía cuadros a ffmpeg a 25 por segundo exactos desde un hilo propio. `escribir` pone el cuadro en una cola
    (y espera si va adelantado); si no llega uno a tiempo, se repite el último. Reconecta si ffmpeg termina."""

    def __init__(self, destino: str, alto: int = 720, bitrate: int = 3000):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg no está instalado")
        self.destino, self.alto, self.bitrate = destino, alto, bitrate
        self.cola: queue.Queue[bytes] = queue.Queue(maxsize=FPS)
        self.ultimo = Image.new("RGB", (W, H), FONDO).tobytes()
        self.proc: subprocess.Popen | None = None
        self.activa = True
        self.inicio = time.monotonic()
        self.enviados = self.repetidos = self.reconexiones = 0
        self.f_vivo = fuente(22, True)
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _abrir(self) -> subprocess.Popen:
        archivo = not self.destino.startswith(("rtmp://", "rtmps://", "srt://"))
        salida = ["-f", "flv"] if self.destino.startswith("rtmp") or self.destino.endswith(".flv") else []
        cmd = ["ffmpeg", "-loglevel", "error", "-y" if archivo else "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-re", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",   # YouTube exige audio;
               # -re: en tiempo real, como el video (si no, el silencio se adelanta)
               "-vf", f"scale=-2:{self.alto}", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
               "-b:v", f"{self.bitrate}k", "-maxrate", f"{self.bitrate}k", "-bufsize", f"{2 * self.bitrate}k",
               "-g", str(2 * FPS), "-keyint_min", str(2 * FPS), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "96k", *salida, self.destino]
        if archivo:
            Path(self.destino).parent.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _bucle(self) -> None:
        siguiente = time.monotonic()
        espera_reconexion = 2.0
        while self.activa:
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    self.reconexiones += 1
                    print(f"transmisión: ffmpeg terminó (código {self.proc.returncode}); reconectando en "
                          f"{espera_reconexion:.0f} s", flush=True)
                    time.sleep(espera_reconexion)
                    espera_reconexion = min(60.0, 2 * espera_reconexion)
                self.proc = self._abrir()
                siguiente = time.monotonic()
            try:
                self.ultimo = self.cola.get_nowait()
                self.enviados += 1
            except queue.Empty:
                self.repetidos += 1
            try:
                self.proc.stdin.write(self.ultimo)
                if self.enviados % (60 * FPS) == 0:
                    espera_reconexion = 2.0          # un minuto sin cortes: la próxima reconexión, rápida
            except (BrokenPipeError, OSError):
                time.sleep(0.2)                      # ffmpeg está terminando: la vuelta siguiente lo reabre
                continue
            siguiente += 1 / FPS
            pausa = siguiente - time.monotonic()
            if pausa > 0:
                time.sleep(pausa)
            elif pausa < -1:                         # muy atrasado (p. ej. tras reconectar): se retoma el ritmo
                siguiente = time.monotonic()

    def escribir(self, img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        x = W - 24
        d.text((x, 30), "EN VIVO", fill=ROJO, font=self.f_vivo, anchor="rm")
        r = 8
        cx = x - d.textlength("EN VIVO", font=self.f_vivo) - 16
        d.ellipse((cx - r, 30 - r, cx + r, 30 + r), fill=ROJO)
        self.cola.put(img.tobytes())

    def cerrar(self) -> None:
        self.activa = False
        self.hilo.join(timeout=5)
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def corrida_activa(linea) -> str | None:
    """La corrida de la línea que se está entrenando (la más reciente si hay varias)."""
    vivas = [c for c in en_curso(linea.paquete) if linea.dir_corrida(c).exists()]
    if not vivas:
        return None
    return max(vivas, key=lambda c: (linea.dir_corrida(c) / "metricas.csv").stat().st_mtime
               if (linea.dir_corrida(c) / "metricas.csv").exists() else 0)


def transmitir(linea, corrida: str, destino: str, quien: str = "alternar", alto: int = 720, bitrate: int = 3000,
               velocidad: int = 1) -> None:
    from .visualizar import entreno, vivo_red
    sigterm_como_ctrl_c()
    t = Transmision(destino, alto, bitrate)
    print(f"transmitiendo a {oculto(destino)} ({alto}p, {bitrate} kb/s)", flush=True)
    estado = {"reporte": time.monotonic(), "actual": None}

    def mostrar(img: Image.Image) -> None:
        ahora = time.monotonic()
        if ahora - estado["reporte"] >= REPORTE_CADA:
            estado["reporte"] = ahora
            print(f"transmisión: {(ahora - t.inicio) / 3600:.1f} h · {estado['actual']} · {t.enviados} cuadros, "
                  f"{t.repetidos} repetidos, {t.reconexiones} reconexiones · memoria {memoria_mb():.0f} MB · libre "
                  f"{memoria_libre_mb() / 1024:.1f} GB", flush=True)
        t.escribir(img)

    def elegir() -> str | None:
        return corrida_activa(linea) if corrida == "auto" else corrida

    def cambio() -> bool:
        return corrida == "auto" and elegir() not in (None, estado["actual"])

    try:
        while True:
            actual = elegir()
            if actual is None or not almacen.generaciones_guardadas(linea.dir_corrida(actual)):
                time.sleep(10)
                continue
            estado["actual"] = actual
            cfg = linea.cargar_config(linea.dir_corrida(actual))
            print(f"transmisión: {actual} ({cfg.entorno})", flush=True)
            if cfg.entorno == "gimnasio":
                entreno(linea, actual, quien, mostrar, velocidad, parar=cambio)
            else:
                vivo_red(linea, actual, None, velocidad, mostrar=mostrar, parar=cambio)
    except KeyboardInterrupt:
        pass
    finally:
        t.cerrar()
        print(f"transmisión terminada: {t.enviados} cuadros ({t.repetidos} repetidos)", flush=True)


def main(linea, argv=None) -> None:
    p = argparse.ArgumentParser(prog=f"python -m {linea.paquete} transmitir", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", required=True, help="nombre de una corrida, o «auto»")
    p.add_argument("--destino", default=None, help="rtmp://… o un archivo (.flv, .mp4); por defecto, el configurado")
    p.add_argument("--quien", choices=["mejor", "todas", "alternar"], default="alternar")
    p.add_argument("--alto", type=int, default=720, help="resolución del stream (720 o 1080)")
    p.add_argument("--bitrate", type=int, default=3000, help="kb/s de video")
    p.add_argument("--velocidad", type=int, default=1)
    a = p.parse_args(argv)
    destino = a.destino or destino_configurado()
    if not destino:
        raise SystemExit(f"falta el destino: --destino rtmp://… o un archivo, o configura {CONFIG} con "
                         '{"url": "rtmp://a.rtmp.youtube.com/live2", "clave": "…"}')
    if not destino.startswith(("rtmp://", "rtmps://", "srt://")):
        destino = str(Path(destino).resolve())
    con_tope_de_memoria()
    transmitir(linea, a.corrida, destino, a.quien, a.alto, a.bitrate, a.velocidad)
