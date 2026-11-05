"""La CNN de `redes` jugando: en un nivel (`JugadorGrilla`) o en el gimnasio (`PoliticaGrilla`). Lo demás (partidas,
reintentos, fitness) es de `evolucion`."""

from __future__ import annotations

import numpy as np

from ..emulador import ACTIONS
from ..evolucion.episodio import Jugador
from ..evolucion.gimnasio import Gimnasio, PoliticaRed
from .modelo import Red
from .vista import Vista, extras


def red_para(cfg) -> Red:
    vc = cfg.config_vista
    return Red((vc.n_canales * cfg.cuadros, vc.alto, vc.ancho), 2 + len(vc.campos), len(ACTIONS),
               cfg.filtros1, cfg.filtros2, cfg.oculta)


def decidir(red: Red, cfg, vista: Vista, g, cuadros: list) -> tuple[int, np.ndarray, list]:
    """(acción, puntuación de cada acción, cuadros): la grilla de `g` más las de las decisiones anteriores."""
    parches = red.parches(vista.completar(g)).astype(np.float32) / 255.0
    cuadros = (cuadros + [parches])[-cfg.cuadros:]
    while len(cuadros) < cfg.cuadros:
        cuadros.insert(0, parches)
    salidas = red.salidas_parches(np.concatenate(cuadros, axis=1), extras(g))
    return int(np.argmax(salidas)), salidas, cuadros


class JugadorGrilla(Jugador):
    def crear_vista(self) -> Vista:
        return Vista(self.emu, self.cfg.config_vista)

    def crear_red(self) -> Red:
        return red_para(self.cfg)

    def decidir(self, g, cuadros: list):
        return decidir(self.red, self.cfg, self.vista, g, cuadros)


def gimnasio(cfg) -> Gimnasio:
    return Gimnasio(lambda emu: Vista(emu, cfg.config_vista), max_salas=max(8, cfg.escenarios))


class PoliticaGrilla(PoliticaRed):
    def __init__(self, cfg, gim: Gimnasio | None = None):
        super().__init__(cfg, red_para(cfg))
        self.vista = gim.vista if gim is not None else None
        self.cuadros: list[np.ndarray] = []

    def reiniciar(self) -> None:
        self.cuadros = []

    def decidir(self, g):
        accion, self.salidas, self.cuadros = decidir(self.red, self.cfg, self.vista, g, self.cuadros)
        return accion, self.cfg.frames_por_decision
