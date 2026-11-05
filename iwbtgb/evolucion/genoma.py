"""Genoma de una red: su linaje, como en *Deep Neuroevolution* (Such et al., 2017).

`(semilla inicial, (semilla, sigma), (semilla, sigma), ...)`: los pesos son los iniciales (He o Xavier) más la suma de
las mutaciones, cada una ruido gaussiano generado con su semilla. Es exacto y ocupa unos bytes.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

Genoma = tuple


def genoma_de(lista) -> Genoma:
    """Genoma guardado en JSON ([semilla, [s, sigma], ...]) → tupla."""
    return tuple([int(lista[0])] + [(int(s), float(g)) for s, g in lista[1:]]) if len(lista) else ()


class Cache:
    """LRU de pesos por genoma (en cada worker): un hijo es su padre más una mutación."""

    def __init__(self, n: int = 64):
        self.n = n
        self.d: OrderedDict = OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v) -> None:
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.n:
            self.d.popitem(last=False)


class Pesos:
    """Todos los pesos de una red en un solo vector (para mutarlos). Las subclases definen `formas` (OrderedDict
    nombre → forma; los que empiezan con "w" son pesos, los demás sesgos), `n_pesos` y `ganancia` (2 = He, para
    ReLU; 1 = Xavier, para tanh)."""
    formas: OrderedDict
    n_pesos: int
    ganancia = 2.0

    def iniciales(self, semilla: int) -> np.ndarray:
        rng = np.random.default_rng(semilla)
        partes = []
        for nombre, forma in self.formas.items():
            if nombre.startswith("w"):
                partes.append((rng.standard_normal(forma, np.float32) * np.sqrt(self.ganancia / forma[0])).ravel())
            else:
                partes.append(np.zeros(int(np.prod(forma)), np.float32))
        return np.concatenate(partes)

    def ruido(self, semilla: int) -> np.ndarray:
        return np.random.default_rng(semilla).standard_normal(self.n_pesos, np.float32)

    def pesos(self, genoma: Genoma, cache: Cache | None = None) -> np.ndarray:
        """Pesos de un genoma, desde el ancestro más cercano que esté en la caché."""
        # el prefijo más largo que ya está en la caché (iterativo: los linajes tienen miles de mutaciones)
        k, v = len(genoma), None
        while cache is not None and k > 0 and (v := cache.get(genoma[:k])) is None:
            k -= 1
        if v is None:
            k, v = 1, self.iniciales(int(genoma[0]))
        else:
            v = v.copy()
        for semilla, sigma in genoma[k:]:
            v += np.float32(sigma) * self.ruido(int(semilla))
        if cache is not None:
            cache.put(genoma, v)
        return v

    def cargar(self, v: np.ndarray) -> None:
        """Deja los pesos en `self.p` (nombre → arreglo con su forma, vistas de `v`)."""
        i = 0
        for nombre, forma in self.formas.items():
            n = int(np.prod(forma))
            self.p[nombre] = v[i:i + n].reshape(forma)
            i += n
