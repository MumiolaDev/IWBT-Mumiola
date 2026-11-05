"""Videos, ventanas en vivo, fuentes, colores y gráficos comunes a todas las líneas."""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 50                           # del juego
FONDO = (26, 26, 25)
TEXTO = (240, 239, 236)
TEXTO_2 = (195, 194, 183)
TEXTO_3 = (130, 129, 122)
GRIS = TEXTO_3
AZUL = (57, 135, 229)
NARANJA = (217, 89, 38)
AQUA = (25, 158, 112)
ROJO = (230, 103, 103)
DORADO = (237, 161, 0)

LIMITE_VIVO = "1500M"   # tope de memoria de un en vivo (Python, emulador y ffplay), vía systemd-run
LIMITE_FFPLAY = 800     # MB: sobre esto se reinicia la ventana
REPORTE_CADA = 600      # s entre líneas de estado de un en vivo


def fuente(tam: int, negrita: bool = False, mono: bool = False):
    rutas = (
        ["/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf" if negrita else "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"]
        if mono else
        ["/usr/share/fonts/gnu-free/FreeSansBold.otf" if negrita else "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
         "/usr/share/fonts/gnu-free/FreeSans.otf"]
    )
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except OSError:
            continue
    return ImageFont.load_default(tam)


def plural(n: int, palabra: str) -> str:
    return f"{n} {palabra}" if n == 1 else f"{n} {palabra}s"


def tiempo(frames: int) -> str:
    s = frames / FPS
    return f"{int(s // 60)}:{s % 60:05.2f}"


def hexa(color) -> str:
    return "#%02x%02x%02x" % color


def memoria_mb(pid: int | str = "self") -> float:
    """RSS de un proceso en MB (0 si ya no existe)."""
    try:
        for linea in open(f"/proc/{pid}/status"):
            if linea.startswith("VmRSS:"):
                return int(linea.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def memoria_libre_mb() -> float:
    for linea in open("/proc/meminfo"):
        if linea.startswith("MemAvailable:"):
            return int(linea.split()[1]) / 1024
    return 0.0


def _morir_con_el_padre() -> None:
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class Video:
    """Envía cuadros RGB a ffmpeg (H.264, yuv420p)."""

    def __init__(self, ruta: Path, fps: int = FPS, ancho: int = W, alto: int = H):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg no está instalado")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta = ruta
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{ancho}x{alto}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(ruta)],
            stdin=subprocess.PIPE,
        )

    def escribir(self, img: Image.Image) -> None:
        self.proc.stdin.write(img.tobytes())

    def cerrar(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"video: {self.ruta}")


class Ventana:
    """Muestra cuadros RGB en una ventana de ffplay."""

    def __init__(self, titulo: str, fps: int = FPS, ancho: int = 1280):
        if shutil.which("ffplay") is None:
            raise SystemExit("ffplay no está instalado (viene con ffmpeg)")
        # -autoexit y PDEATHSIG: si el proceso de Python muere (aunque sea con SIGKILL), ffplay no queda huérfano
        self.proc = subprocess.Popen(
            ["ffplay", "-loglevel", "error", "-nostats", "-autoexit", "-an", "-sn", "-window_title", titulo,
             "-x", str(ancho), "-y", str(ancho * H // W), "-fflags", "nobuffer", "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size", f"{W}x{H}", "-framerate", str(fps), "-i", "-"],
            stdin=subprocess.PIPE, preexec_fn=_morir_con_el_padre,
        )

    def escribir(self, img: Image.Image) -> bool:
        """False si el usuario cerró la ventana."""
        try:
            self.proc.stdin.write(img.tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def abierta(self) -> bool:
        return self.proc.poll() is None

    def memoria_mb(self) -> float:
        return memoria_mb(self.proc.pid)

    def cerrar(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Cerrada(Exception):
    """El usuario cerró la ventana del en vivo."""


def _terminar(*_) -> None:
    raise KeyboardInterrupt


def sigterm_como_ctrl_c() -> None:
    """SIGTERM (p. ej. el kill del scope de memoria) termina como Ctrl+C: así se cierran ffplay y el emulador."""
    signal.signal(signal.SIGTERM, _terminar)


def tope() -> str:
    return LIMITE_VIVO if os.environ.get("IWBTGB_VIVO_SCOPE") else "ninguno"


def con_tope_de_memoria() -> None:
    """Vuelve a ejecutar el comando dentro de un scope de systemd con MemoryMax (sin swap): si algo crece sin control,
    el kernel mata solo a este grupo de procesos en vez de congelar el computador. Sin systemd-run sigue igual."""
    principal = getattr(sys.modules["__main__"], "__spec__", None)
    if os.environ.get("IWBTGB_VIVO_SCOPE") or shutil.which("systemd-run") is None or principal is None:
        return
    os.environ["IWBTGB_VIVO_SCOPE"] = "1"
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={LIMITE_VIVO}", "-p",
           "MemorySwapMax=0", sys.executable, "-m", principal.name.removesuffix(".__main__"), *sys.argv[1:]]
    try:
        os.execvp(cmd[0], cmd)
    except OSError:
        pass


class EnVivo:
    """Ventana de ffplay para dejarla horas: si ffplay pasa de LIMITE_FFPLAY se reinicia, y cada REPORTE_CADA s se
    escribe una línea con la memoria y `detalle()`."""

    def __init__(self, titulo: str, fps: int = FPS, detalle=lambda: ""):
        sigterm_como_ctrl_c()
        self.titulo, self.fps, self.detalle = titulo, fps, detalle
        self.ventana = Ventana(titulo, fps)
        self.inicio = self.reporte = time.monotonic()
        print(f"en vivo: {titulo} (tope de memoria {tope()})", flush=True)

    def abierta(self) -> bool:
        return self.ventana.abierta()

    def revisar(self) -> None:
        ahora = time.monotonic()
        if ahora - self.reporte >= REPORTE_CADA:
            self.reporte = ahora
            detalle = self.detalle()
            print(f"en vivo: {(ahora - self.inicio) / 3600:.1f} h{' · ' + detalle if detalle else ''} · memoria "
                  f"{memoria_mb():.0f} MB + ffplay {self.ventana.memoria_mb():.0f} MB · libre en el sistema "
                  f"{memoria_libre_mb() / 1024:.1f} GB", flush=True)
        if self.ventana.memoria_mb() > LIMITE_FFPLAY:
            print(f"en vivo: ffplay usa {self.ventana.memoria_mb():.0f} MB; se reinicia la ventana", flush=True)
            self.ventana.cerrar()
            self.ventana = Ventana(self.titulo, self.fps)

    def mostrar(self, img: Image.Image) -> None:
        self.revisar()
        if not self.ventana.escribir(img):
            raise Cerrada

    def cerrar(self) -> None:
        self.ventana.cerrar()


def en_ventana(titulo: str, jugar, fps: int = FPS, detalle=lambda: "") -> None:
    """Muestra en vivo los cuadros que `jugar(mostrar)` le pasa a `mostrar`, hasta que se cierre la ventana."""
    v = EnVivo(titulo, fps, detalle)
    try:
        jugar(v.mostrar)
    except (KeyboardInterrupt, Cerrada):
        pass
    finally:
        v.cerrar()


def mini_grafico(draw: ImageDraw.ImageDraw, caja, metricas: list[dict], hasta_gen: int, f_chica) -> None:
    """Fitness máximo (azul) y medio (naranja) por generación hasta `hasta_gen`."""
    x0, y0, x1, y1 = caja
    filas = [m for m in metricas if m["gen"] <= hasta_gen]
    if not filas:
        return
    gmax = max(m["gen"] for m in metricas) or 1
    vmax = max(m["fitness_max"] for m in metricas) or 1
    vmin = min(0.0, min(m["fitness_media"] for m in metricas))
    for k in range(5):
        yy = y1 - (y1 - y0) * k / 4
        draw.line((x0, yy, x1, yy), fill=(45, 45, 43), width=1)
        draw.text((x0 - 8, yy), f"{vmin + (vmax - vmin) * k / 4:.0f}", fill=TEXTO_3, font=f_chica, anchor="rm")

    def pts(clave):
        return [(x0 + (x1 - x0) * m["gen"] / gmax, y1 - (y1 - y0) * (m[clave] - vmin) / (vmax - vmin or 1))
                for m in filas]

    for clave, color in (("fitness_media", NARANJA), ("fitness_max", AZUL)):
        p = pts(clave)
        if len(p) > 1:
            draw.line(p, fill=color, width=3, joint="curve")
        draw.ellipse((p[-1][0] - 5, p[-1][1] - 5, p[-1][0] + 5, p[-1][1] + 5), fill=color)
    draw.text((x0, y1 + 10), "0", fill=TEXTO_3, font=f_chica, anchor="lt")
    draw.text((x1, y1 + 10), f"gen {gmax:.0f}", fill=TEXTO_3, font=f_chica, anchor="rt")


def estilo_graficos():
    """matplotlib (sin pantalla) con el estilo oscuro de los videos; devuelve pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": hexa(FONDO), "axes.facecolor": hexa(FONDO), "savefig.facecolor": hexa(FONDO),
        "text.color": hexa(TEXTO), "axes.labelcolor": hexa(TEXTO_2), "xtick.color": hexa(TEXTO_2),
        "ytick.color": hexa(TEXTO_2), "axes.edgecolor": "#3a3a38", "grid.color": "#2e2e2c",
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 13,
        "lines.linewidth": 2, "font.family": "sans-serif",
        "font.sans-serif": ["Adwaita Sans", "FreeSans", "DejaVu Sans"],
        "xtick.minor.visible": False, "ytick.minor.visible": False, "xtick.top": False, "ytick.right": False,
    })
    return plt


def guardar_grafico(plt, fig, ruta: Path, salidas: list) -> None:
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    salidas.append(ruta)
    print(f"gráfico: {ruta}", flush=True)
