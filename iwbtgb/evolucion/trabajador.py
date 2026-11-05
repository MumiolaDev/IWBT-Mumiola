"""Workers del entrenamiento: un jugador (emulador + red) por proceso."""

from __future__ import annotations

import importlib

from ..workers import cerrar_al_salir

_jugador = None


def iniciar_worker(paquete: str, cfg, mapa: dict) -> None:
    """`paquete`: el de la línea (su `LINEA` dice qué jugador crear)."""
    global _jugador
    linea = importlib.import_module(paquete).LINEA
    _jugador = linea.jugador_gimnasio(cfg) if cfg.entorno == "gimnasio" else linea.jugador_nivel(cfg, mapa)
    cerrar_al_salir(_jugador)


def evaluar_en_worker(args):
    """(i, genoma, escenarios, cfg) → (i, Evaluacion), con la config de la generación (el currículo la cambia)."""
    i, genoma, escenarios, cfg = args
    _jugador.cfg = cfg
    if hasattr(_jugador, "politica"):
        _jugador.politica.cfg = cfg
    return i, _jugador.evaluar(genoma, escenarios)
