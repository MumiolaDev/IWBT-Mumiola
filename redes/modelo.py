"""La red: una CNN chica en numpy (sin GPU) con todos sus pesos en un solo vector, para mutarlos.

    grilla (canales·cuadros, 48, 64) → conv 4×4 paso 4 → ReLU → conv 3×3 paso 2 → ReLU → aplanar
    + variables del jugador → densa → ReLU → una salida por acción (se elige la mayor)

Genoma: la red no se guarda peso a peso sino como su linaje, igual que en *Deep Neuroevolution* (Such et al.,
2017): `(semilla inicial, (semilla, sigma), (semilla, sigma), ...)`. Los pesos son los iniciales (He) más la suma
de las mutaciones, cada una ruido gaussiano generado con su semilla. Es exacto y ocupa unos bytes.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

Genoma = tuple  # (semilla_inicial, (semilla, sigma), ...)


class Red:
    def __init__(self, entrada: tuple[int, int, int], n_extra: int, n_acciones: int,
                 filtros1: int = 8, filtros2: int = 16, oculta: int = 64):
        c, h, w = entrada
        if h % 4 or w % 4:
            raise ValueError("el alto y el ancho de la grilla deben ser múltiplos de 4")
        self.entrada, self.n_extra, self.n_acciones = entrada, n_extra, n_acciones
        self.h1, self.w1 = h // 4, w // 4
        self.h2, self.w2 = (self.h1 + 1) // 2, (self.w1 + 1) // 2
        plano = self.h2 * self.w2 * filtros2 + n_extra
        self.formas = OrderedDict([
            ("w1", (c * 16, filtros1)), ("b1", (filtros1,)),
            ("w2", (filtros1 * 9, filtros2)), ("b2", (filtros2,)),
            ("w3", (plano, oculta)), ("b3", (oculta,)),
            ("w4", (oculta, n_acciones)), ("b4", (n_acciones,)),
        ])
        self.n_pesos = sum(int(np.prod(f)) for f in self.formas.values())
        self.p: dict[str, np.ndarray] = {}
        self.registrar = False                  # guardar las activaciones de cada capa en `act` (para dibujarlas)
        self.act: dict[str, np.ndarray] = {}

    # -- pesos ------------------------------------------------------------------------------------------

    def iniciales(self, semilla: int) -> np.ndarray:
        """He normal para los pesos, ceros para los sesgos."""
        rng = np.random.default_rng(semilla)
        partes = []
        for nombre, forma in self.formas.items():
            if nombre.startswith("w"):
                partes.append((rng.standard_normal(forma, np.float32) * np.sqrt(2.0 / forma[0])).ravel())
            else:
                partes.append(np.zeros(int(np.prod(forma)), np.float32))
        return np.concatenate(partes)

    def ruido(self, semilla: int) -> np.ndarray:
        return np.random.default_rng(semilla).standard_normal(self.n_pesos, np.float32)

    def pesos(self, genoma: Genoma, cache: "Cache | None" = None) -> np.ndarray:
        """Pesos de un genoma (con caché: un hijo es su padre más una mutación)."""
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
        i = 0
        for nombre, forma in self.formas.items():
            n = int(np.prod(forma))
            self.p[nombre] = v[i:i + n].reshape(forma)
            i += n

    # -- inferencia -------------------------------------------------------------------------------------

    def parches(self, grilla: np.ndarray) -> np.ndarray:
        """Grilla (c, h, w) → parches 4×4 de la primera conv, (h1·w1, c·16). Los parches de varias grillas
        apiladas son los de cada una concatenados (axis=1): así cada grilla se recorta una sola vez."""
        c = grilla.shape[0]
        return grilla.reshape(c, self.h1, 4, self.w1, 4).transpose(1, 3, 0, 2, 4).reshape(self.h1 * self.w1, c * 16)

    def salidas(self, grilla: np.ndarray, extra: np.ndarray) -> np.ndarray:
        """grilla float32 (c, h, w) en [0, 1], extra float32 (n_extra,) → una puntuación por acción."""
        return self.salidas_parches(self.parches(grilla), extra)

    def salidas_parches(self, x: np.ndarray, extra: np.ndarray) -> np.ndarray:
        p = self.p
        entrada = x
        # conv 4×4 paso 4: los parches no se solapan, así que es un producto de matrices
        x = c1 = np.maximum(x @ p["w1"] + p["b1"], 0).reshape(self.h1, self.w1, -1)
        # conv 3×3 paso 2 con relleno 1
        x = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        v = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(0, 1))[::2, ::2]  # (h2, w2, f1, 3, 3)
        x = c2 = np.maximum(v.reshape(self.h2 * self.w2, -1) @ p["w2"] + p["b2"], 0)
        x = np.concatenate([x.ravel(), extra])
        x = oculta = np.maximum(x @ p["w3"] + p["b3"], 0)
        out = x @ p["w4"] + p["b4"]
        if self.registrar:
            # cobertura media de cada canal de entrada (columnas de los parches: canal × 16 píxeles)
            self.act = {"entrada": entrada.reshape(entrada.shape[0], -1, 16).mean(axis=(0, 2)),
                        "c1": c1.mean(axis=(0, 1)), "c2": c2.mean(axis=0), "extra": np.asarray(extra),
                        "oculta": oculta, "salida": out}
        return out

    def accion(self, grilla: np.ndarray, extra: np.ndarray) -> int:
        return int(np.argmax(self.salidas(grilla, extra)))


class Cache:
    """LRU de pesos por genoma (en cada worker)."""

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
