"""La red de rayos jugando: en un nivel (`JugadorRayos`) o en el gimnasio (`PoliticaRayos`). Lo demás (partidas,
reintentos, fitness) es de `evolucion`."""

from __future__ import annotations

import numpy as np

from ..emulador import ACTIONS
from ..evolucion.episodio import Jugador
from ..evolucion.gimnasio import Gimnasio, PoliticaRed
from . import observacion
from .modelo import RedDensa


def red_para(cfg) -> RedDensa:
    return RedDensa(observacion.n_entradas(cfg), cfg.lista_capas, len(ACTIONS), cfg.activacion)


class JugadorRayos(Jugador):
    def crear_vista(self):
        observacion.configurar(self.emu, self.cfg)
        self.direccion = (0.0, 0.0)
        self._dir = None

    def crear_red(self) -> RedDensa:
        return red_para(self.cfg)

    def decidir(self, g, cuadros: list):
        if self._dir is None:
            self._dir = observacion.direccion_nivel(self.dist)
        self.direccion = self._dir.en(*g.centro)
        self.entrada, cuadros = observacion.entrada(self.cfg, g, cuadros, self.direccion)
        salidas = self.red.salidas(self.entrada)
        return int(np.argmax(salidas)), salidas, cuadros


def gimnasio(cfg) -> Gimnasio:
    return Gimnasio(lambda emu: observacion.configurar(emu, cfg), max_salas=max(8, cfg.escenarios))


class PoliticaRayos(PoliticaRed):
    """Decide en el gimnasio; la dirección del nivel sale del mapa de distancias de la sala armada en `gim`."""

    def __init__(self, cfg, gim: Gimnasio | None = None):
        super().__init__(cfg, red_para(cfg))
        self.gim = gim
        self.cuadros: list[np.ndarray] = []
        self.entrada: np.ndarray | None = None      # la última, para dibujarla
        self.direccion = (0.0, 0.0)
        self._dist = None
        self._dir: observacion.Direccion | None = None

    def reiniciar(self) -> None:
        self.cuadros = []
        if self.gim is not None and self.gim.dist is not self._dist:
            self._dist = self.gim.dist
            self._dir = observacion.direccion_gimnasio(self.gim.dist)

    def decidir(self, g):
        self.direccion = self._dir.en(*g.centro) if self._dir is not None else (0.0, 0.0)
        self.entrada, self.cuadros = observacion.entrada(self.cfg, g, self.cuadros, self.direccion)
        self.salidas = self.red.salidas(self.entrada)
        return int(np.argmax(self.salidas)), self.cfg.frames_por_decision
