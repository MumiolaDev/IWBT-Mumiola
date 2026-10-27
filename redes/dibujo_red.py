"""Dibujo clásico de la red para los videos: capas en columnas de círculos y conexiones como líneas coloreadas.

La red real es una CNN (grilla → 2 capas de filtros → neuronas ocultas → acciones); para que se pueda dibujar como
la red "de libro", cada filtro de una capa convolucional es un nodo y el peso entre dos nodos es el promedio de sus
pesos (sobre los píxeles del filtro y, en la entrada, sobre las grillas apiladas). Así:

    vista (un nodo por tipo de objeto: sólido, púas…)  →  capa 1 (filtros)  →  capa 2 (filtros) + velocidad y doble
    salto  →  neuronas ocultas  →  una salida por acción

Las líneas son fijas para una red (naranja = peso positivo, azul = negativo, más opaca = más fuerte; de las capas
grandes se dibujan las más fuertes). Los círculos se encienden con la activación de cada nodo en ese momento
(promedio sobre la grilla en las capas de filtros); en la salida, la probabilidad de cada acción, con la elegida
marcada. Si hay más de 32 neuronas ocultas se muestran las 32 de más peso hacia la salida.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ga.visualizar import fuente

from .modelo import Red
from .vista import ConfigVista

FONDO = (26, 26, 25)
POSITIVO = (237, 140, 40)
NEGATIVO = (70, 140, 235)
APAGADO = (58, 58, 55)
ENCENDIDO = (255, 226, 140)
TEXTO_2 = (195, 194, 183)
MAX_LINEAS = 160                   # por par de capas (las más fuertes)
MAX_OCULTAS = 32                   # con más neuronas ocultas se muestran las que más pesan en la decisión
CAMPOS = {"djump": "doble salto"}


def _mezcla(a, b, t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0, 1))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class DibujoRed:
    def __init__(self, red: Red, vista: ConfigVista, caja: tuple[int, int, int, int], etiquetas_salida=None):
        self.red, self.caja = red, caja
        self.nc = vista.n_canales
        self.cuadros = red.entrada[0] // self.nc
        self.nombres_entrada = [c.nombre for c in vista.canales] + ["fuera"]
        self.nombres_extra = ["vel. x", "vel. y"] + [CAMPOS.get(c, c) for c in vista.campos]
        self.etiquetas_salida = etiquetas_salida
        self.f, self.f_tit = fuente(13), fuente(14, True)
        self.fondo: Image.Image | None = None
        x0, y0, x1, y1 = caja
        ancho, top, bajo = x1 - x0, 22, y1 - y0 - 4
        cols = [x0 + 92 + k * (ancho - 92 - 58) / 4 for k in range(5)]

        def columna(n: int, x: float, desde: float = top, hasta: float = bajo) -> list[tuple[float, float]]:
            paso = (hasta - desde) / max(n, 1)
            return [(x, y0 + desde + paso * (i + 0.5)) for i in range(n)]

        f1, f2 = red.formas["b1"][0], red.formas["b2"][0]
        n_extra, n_sal = red.n_extra, red.n_acciones
        self.n_ocultas = red.formas["b3"][0]
        self.ocultas = np.arange(self.n_ocultas)        # las que se dibujan (se eligen en `cargar`)
        oculta = min(self.n_ocultas, MAX_OCULTAS)
        alto_c2 = (bajo - top) * f2 / (f2 + n_extra + 1)       # un hueco entre los filtros y las variables
        self.pos = {
            "entrada": columna(self.nc, cols[0]), "c1": columna(f1, cols[1]),
            "c2": columna(f2, cols[2], top, top + alto_c2),
            "extra": columna(n_extra, cols[2], top + alto_c2 + (bajo - top - alto_c2) / (n_extra + 1), bajo),
            "oculta": columna(oculta, cols[3]), "salida": columna(n_sal, cols[4]),
        }
        self.radio = {k: float(np.clip((v[1][1] - v[0][1]) * 0.36 if len(v) > 1 else 6, 2, 7))
                      for k, v in self.pos.items()}
        self.cols = cols
        if red.p:           # si la red todavía no tiene pesos, se arma en el primer `dibujar`
            self.cargar()

    def pesos_capas(self) -> list[tuple[str, str, np.ndarray]]:
        """(capa de origen, capa de destino, pesos (n_origen, n_destino)) promediados por nodo."""
        p, r = self.red.p, self.red
        f1, f2 = r.formas["b1"][0], r.formas["b2"][0]
        w1 = p["w1"].reshape(self.cuadros, self.nc, 16, f1).mean(axis=(0, 2))
        w2 = p["w2"].reshape(f1, 9, f2).mean(axis=1)
        n_conv = r.h2 * r.w2 * f2
        w3 = p["w3"][:n_conv].reshape(r.h2 * r.w2, f2, -1).mean(axis=0)
        w3x = p["w3"][n_conv:]
        o = self.ocultas
        return [("entrada", "c1", w1), ("c1", "c2", w2), ("c2", "oculta", w3[:, o]), ("extra", "oculta", w3x[:, o]),
                ("oculta", "salida", p["w4"][o])]

    def cargar(self) -> None:
        """Arma la parte fija (líneas, títulos y rótulos) con los pesos actuales de la red."""
        x0, y0, x1, y1 = self.caja
        if self.n_ocultas > MAX_OCULTAS:
            peso = np.abs(self.red.p["w4"]).sum(axis=1)
            self.ocultas = np.sort(np.argsort(-peso)[:MAX_OCULTAS])
        capa = Image.new("RGBA", (x1 - x0, y1 - y0), FONDO + (255,))
        lineas = Image.new("RGBA", capa.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lineas)
        for origen, destino, w in self.pesos_capas():
            m = float(np.abs(w).max()) or 1.0
            orden = np.argsort(-np.abs(w), axis=None)[:MAX_LINEAS]
            for idx in orden[::-1]:                       # las fuertes encima
                i, j = np.unravel_index(idx, w.shape)
                v = float(w[i, j]) / m
                (ax, ay), (bx, by) = self.pos[origen][i], self.pos[destino][j]
                color = (POSITIVO if v > 0 else NEGATIVO) + (int(25 + 200 * abs(v) ** 1.5),)
                d.line([(ax - x0, ay - y0), (bx - x0, by - y0)], fill=color, width=1)
        capa = Image.alpha_composite(capa, lineas).convert("RGB")
        d = ImageDraw.Draw(capa)
        ocultas = "ocultas" if self.n_ocultas <= MAX_OCULTAS else f"{MAX_OCULTAS} de {self.n_ocultas}"
        for x, titulo in zip(self.cols, ("vista", "capa 1", "capa 2", ocultas, "acción")):
            d.text((x - x0, 2), titulo, fill=TEXTO_2, font=self.f_tit, anchor="ma")
        for nombres, clave in ((self.nombres_entrada, "entrada"), (self.nombres_extra, "extra")):
            for nombre, (x, y) in zip(nombres, self.pos[clave]):
                d.text((x - x0 - self.radio[clave] - 5, y - y0), nombre, fill=TEXTO_2, font=self.f, anchor="rm")
        if self.etiquetas_salida:
            for nombre, (x, y) in zip(self.etiquetas_salida, self.pos["salida"]):
                d.text((x - x0 + self.radio["salida"] + 5, y - y0), nombre, fill=TEXTO_2, font=self.f, anchor="lm")
        self.fondo = capa

    def dibujar(self, img: Image.Image, accion: int | None) -> None:
        """Pega la red en `img` con las activaciones del último paso (`red.act`)."""
        if self.fondo is None:
            if not self.red.p:
                return
            self.cargar()
        x0, y0 = self.caja[:2]
        img.paste(self.fondo, (x0, y0))
        d = ImageDraw.Draw(img)
        act = self.red.act
        for clave, puntos in self.pos.items():
            v = act.get(clave)
            if clave == "oculta" and v is not None:
                v = v[self.ocultas]
            if clave == "salida" and v is not None:
                z = np.exp(v - v.max())
                v = z / z.sum()
            r = self.radio[clave]
            m = float(np.abs(v).max()) if v is not None and len(v) else 0.0
            for i, (x, y) in enumerate(puntos):
                a = float(v[i]) / m if m > 0 else 0.0
                color = _mezcla(APAGADO, ENCENDIDO if a >= 0 else NEGATIVO, abs(a))
                borde = (255, 255, 255) if clave == "salida" and i == accion else (20, 20, 20)
                d.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=borde,
                          width=2 if clave == "salida" and i == accion else 1)
