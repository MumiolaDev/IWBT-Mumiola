"""Piezas comunes de las apps (PySide6): procesos lanzados por la app, controles y arranque."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox

from .emulador import SALAS
from .rutas import ROOT

DIFICULTADES = ["medium", "hard", "veryhard", "impossible"]


class Proceso(QObject):
    """`python -m <paquete> <args>` en su propio grupo de procesos (así se pueden cerrar también sus emuladores).
    `args[0]` es el comando (entrenar, visualizar…) y, en los entrenamientos, `args[2]` el nombre de la corrida."""
    # Se emiten desde el hilo lector: se conectan a métodos de la ventana para que Qt los pase al hilo de la UI
    linea = Signal(object, str)
    termino = Signal(object, int)

    def __init__(self, etiqueta: str, paquete: str, args: list[str]):
        super().__init__()
        self.etiqueta, self.paquete, self.args = etiqueta, paquete, args
        self.popen: subprocess.Popen | None = None

    def iniciar(self) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.popen = subprocess.Popen([sys.executable, "-m", self.paquete, *self.args], cwd=ROOT, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                                      start_new_session=True)
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self) -> None:
        for linea in self.popen.stdout:
            self.linea.emit(self, linea.rstrip("\n"))
        codigo = self.popen.wait()
        self._matar_grupo()  # emuladores o ffplay que hayan quedado huérfanos
        self.termino.emit(self, codigo)

    def vivo(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def detener(self) -> None:
        """Ctrl+C al proceso principal (termina la generación limpiamente) y, si no cierra, se mata el grupo."""
        if self.vivo():
            os.kill(self.popen.pid, signal.SIGINT)
            QTimer.singleShot(10_000, self._matar_grupo)

    def _matar_grupo(self) -> None:
        try:
            os.killpg(self.popen.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def en_curso(paquete: str, comando: str = "entrenar") -> dict[str, int]:
    """{corrida: pid} de los `python -m <paquete> <comando> --nombre X` que están corriendo (lanzados desde una app o
    desde la terminal; no el `uv run` que los envuelve ni sus workers)."""
    vivos = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (len(args) > 4 and args[1] == b"-m" and args[2] == paquete.encode() and args[3] == comando.encode()
                and b"--nombre" in args):
            i = args.index(b"--nombre")
            if i + 1 < len(args):
                vivos[args[i + 1].decode(errors="replace")] = int(d.name)
    return vivos


def spin(minimo: int, maximo: int, valor: int, especial: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(minimo, maximo)
    s.setValue(valor)
    if especial:
        s.setSpecialValueText(especial)  # se muestra en lugar del mínimo
    return s


def doble(minimo: float, maximo: float, valor: float, decimales: int, sufijo: str = "",
          paso: float | None = None) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimo, maximo)
    s.setDecimals(decimales)
    s.setValue(valor)
    if paso:
        s.setSingleStep(paso)
    if sufijo:
        s.setSuffix(sufijo)
    return s


def combo_salas(primera: str | None = None) -> QComboBox:
    c = QComboBox()
    if primera:
        c.addItem(primera)
    c.addItems(SALAS)
    return c


def ejecutar(crear_ventana, nombre: str, icono: Path) -> None:
    """Arranca una app. `nombre` (p. ej. "iwbtgb-ga") es el app_id de Wayland (lo asocia a su acceso directo del menú
    ~/.local/share/applications/<nombre>.desktop) y el del registro ~/.cache/<nombre>.log, adonde va la salida cuando
    se abre sin terminal."""
    log = Path.home() / ".cache" / f"{nombre}.log"
    if not sys.stdout or not sys.stdout.isatty():
        log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(log, "a", buffering=1)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")  # avisos inofensivos de Wayland
    app = QApplication(sys.argv)
    app.setDesktopFileName(nombre)
    app.setWindowIcon(QIcon(str(icono)))

    def error(tipo, valor, tb):  # sin terminal, un error inesperado se muestra en una ventana
        print("".join(traceback.format_exception(tipo, valor, tb)), flush=True)
        QMessageBox.critical(None, "Error", f"{valor}\n\nDetalle en {log}")

    sys.excepthook = error
    v = crear_ventana()
    v.show()
    sys.exit(app.exec())
