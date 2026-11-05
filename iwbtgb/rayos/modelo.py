"""La red de `rayos`: densa (perceptrón multicapa) en numpy, con el genoma de semillas de `evolucion.genoma`.

    rayos (cercanías) + variables del jugador → capa oculta 1 → … → capa oculta N → una salida por acción (la mayor)
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..evolucion.genoma import Pesos


class RedDensa(Pesos):
    def __init__(self, n_entrada: int, capas: list[int], n_acciones: int, activacion: str = "relu"):
        if activacion not in ("relu", "tanh"):
            raise ValueError(f"activación desconocida: {activacion}")
        self.n_entrada, self.capas, self.n_acciones, self.activacion = n_entrada, list(capas), n_acciones, activacion
        self.ganancia = 2.0 if activacion == "relu" else 1.0      # pesos iniciales: He (relu) o Xavier (tanh)
        dims = [n_entrada, *self.capas, n_acciones]
        self.formas = OrderedDict()
        for k in range(len(dims) - 1):
            self.formas[f"w{k + 1}"] = (dims[k], dims[k + 1])
            self.formas[f"b{k + 1}"] = (dims[k + 1],)
        self.n_capas = len(dims) - 1
        self.n_pesos = sum(int(np.prod(f)) for f in self.formas.values())
        self.p: dict[str, np.ndarray] = {}
        self.registrar = False                  # guardar las activaciones de cada capa en `act` (para dibujarlas)
        self.act: dict = {}

    def salidas(self, x: np.ndarray) -> np.ndarray:
        """Vector de entrada float32 (n_entrada,) → una puntuación por acción."""
        p = self.p
        entrada = x
        ocultas = []
        for k in range(1, self.n_capas):
            x = x @ p[f"w{k}"] + p[f"b{k}"]
            x = np.maximum(x, 0) if self.activacion == "relu" else np.tanh(x)
            ocultas.append(x)
        out = x @ p[f"w{self.n_capas}"] + p[f"b{self.n_capas}"]
        if self.registrar:
            self.act = {"entrada": entrada, "ocultas": ocultas, "salida": out}
        return out
