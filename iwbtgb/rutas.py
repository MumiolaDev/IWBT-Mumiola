"""Rutas del repositorio: juego, emulador y carpetas de corridas (enlaces al disco de datos)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUEGO = ROOT / "game" / "i_wanna_be_the_GB.exe"
# OPENGMK_EMULATOR: otro binario, p. ej. el respaldado para repetir corridas anteriores a la corrección de las púas
EMULADOR = Path(os.environ.get("OPENGMK_EMULATOR")
                or ROOT / "vendor" / "OpenGMK" / "target" / "release" / "gm8emulator")
CORRIDAS_GA = ROOT / "corridas"
CORRIDAS_REDES = ROOT / "corridas_redes"
CORRIDAS_RAYOS = ROOT / "corridas_rayos"
