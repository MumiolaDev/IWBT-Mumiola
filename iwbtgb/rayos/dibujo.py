"""Lo que se dibuja de la red de rayos: los rayos sobre el juego y, con `--red`, la red por dentro.

La red por dentro, como la red "de libro" (capas en columnas de círculos, conexiones como líneas):

    lo que ve (un nodo por tipo de cosa: sólido, mortal…, y uno por variable del jugador)  →  capas ocultas  →
    una salida por acción

La entrada real son las cercanías de cada rayo y cada tipo (64 rayos × 7 tipos × cuadros): cada tipo es un nodo y el
peso desde él es el promedio sobre los rayos y los cuadros; su brillo es la cercanía media de ese tipo en ese momento.
Con más de 32 neuronas en una capa se muestran las 32 que más pesan hacia la capa siguiente.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ..evolucion.pantallas import APAGADO, ENCENDIDO, MAX_OCULTAS, NEGATIVO, lineas_de_pesos, mezcla
from ..medios import TEXTO_2, fuente
from . import observacion
from .modelo import RedDensa


class DibujoRedDensa:
    def __init__(self, red: RedDensa, cfg, caja: tuple[int, int, int, int], etiquetas_salida=None):
        self.red, self.cfg, self.caja = red, cfg, caja
        self.nombres_entrada = list(observacion.NOMBRES)
        self.nombres_extra = observacion.nombres_extras(cfg)
        self.colores_entrada = list(observacion.COLORES)
        self.etiquetas_salida = etiquetas_salida
        self.f, self.f_tit = fuente(13), fuente(14, True)
        self.fondo: Image.Image | None = None
        x0, y0, x1, y1 = caja
        ancho, top, bajo = x1 - x0, 22, y1 - y0 - 4
        n_cols = len(red.capas) + 2
        cols = [x0 + 92 + k * (ancho - 92 - 58) / (n_cols - 1) for k in range(n_cols)]

        def columna(n: int, x: float, desde: float = top, hasta: float = bajo) -> list[tuple[float, float]]:
            paso = (hasta - desde) / max(n, 1)
            return [(x, y0 + desde + paso * (i + 0.5)) for i in range(n)]

        n_ent, n_ext = len(self.nombres_entrada), len(self.nombres_extra)
        alto_ent = (bajo - top) * n_ent / (n_ent + n_ext + 1)        # un hueco entre los rayos y las variables
        self.pos = {"entrada": columna(n_ent, cols[0], top, top + alto_ent),
                    "extra": columna(n_ext, cols[0], top + alto_ent + (bajo - top - alto_ent) / (n_ext + 1), bajo)}
        self.visibles = [np.arange(n) for n in red.capas]            # neuronas dibujadas de cada capa oculta
        for k, n in enumerate(red.capas):
            self.pos[f"o{k}"] = columna(min(n, MAX_OCULTAS), cols[k + 1])
        self.pos["salida"] = columna(red.n_acciones, cols[-1])
        self.radio = {k: float(np.clip((v[1][1] - v[0][1]) * 0.36 if len(v) > 1 else 6, 2, 7))
                      for k, v in self.pos.items()}
        self.cols = cols
        if red.p:           # si la red todavía no tiene pesos, se arma en el primer `dibujar`
            self.cargar()

    def pesos_capas(self) -> list[tuple[str, str, np.ndarray]]:
        """(capa de origen, capa de destino, pesos (n_origen, n_destino)) entre los nodos dibujados."""
        p, red, cfg = self.red.p, self.red, self.cfg
        n_rayos_total = cfg.n_rayos * observacion.POR_RAYO * cfg.cuadros
        w1 = p["w1"]
        entrada = w1[:n_rayos_total].reshape(cfg.cuadros, cfg.n_rayos, observacion.POR_RAYO, -1).mean(axis=(0, 1))
        extra = w1[n_rayos_total:]
        v0 = self.visibles[0]
        capas = [("entrada", "o0", entrada[:, v0]), ("extra", "o0", extra[:, v0])]
        for k in range(1, len(red.capas)):
            capas.append((f"o{k - 1}", f"o{k}", p[f"w{k + 1}"][np.ix_(self.visibles[k - 1], self.visibles[k])]))
        ultima = len(red.capas)
        capas.append((f"o{ultima - 1}", "salida", p[f"w{ultima + 1}"][self.visibles[-1]]))
        return capas

    def cargar(self) -> None:
        """Arma la parte fija (líneas, títulos y rótulos) con los pesos actuales de la red."""
        x0, y0, x1, y1 = self.caja
        p = self.red.p
        for k, n in enumerate(self.red.capas):   # las que más pesan hacia la capa siguiente
            if n > MAX_OCULTAS:
                peso = np.abs(p[f"w{k + 2}"]).sum(axis=1)
                self.visibles[k] = np.sort(np.argsort(-peso)[:MAX_OCULTAS])
        capa = lineas_de_pesos(self.caja, self.pos, self.pesos_capas())
        d = ImageDraw.Draw(capa)
        titulos = ["ve"] + [f"capa {k + 1}" + ("" if n <= MAX_OCULTAS else f" ({MAX_OCULTAS}/{n})")
                            for k, n in enumerate(self.red.capas)] + ["acción"]
        for x, titulo in zip(self.cols, titulos):
            d.text((x - x0, 2), titulo, fill=TEXTO_2, font=self.f_tit, anchor="ma")
        for nombre, color, (x, y) in zip(self.nombres_entrada, self.colores_entrada, self.pos["entrada"]):
            d.text((x - x0 - self.radio["entrada"] - 5, y - y0), nombre, fill=color, font=self.f, anchor="rm")
        for nombre, (x, y) in zip(self.nombres_extra, self.pos["extra"]):
            d.text((x - x0 - self.radio["extra"] - 5, y - y0), nombre, fill=TEXTO_2, font=self.f, anchor="rm")
        if self.etiquetas_salida:
            for nombre, (x, y) in zip(self.etiquetas_salida, self.pos["salida"]):
                d.text((x - x0 + self.radio["salida"] + 5, y - y0), nombre, fill=TEXTO_2, font=self.f, anchor="lm")
        self.fondo = capa
        self._pesos = p.get("w1")

    def dibujar(self, img: Image.Image, accion: int | None) -> None:
        """Pega la red en `img` con las activaciones del último paso (`red.act`)."""
        if self.fondo is None or self._pesos is not self.red.p.get("w1"):   # red nueva: se redibuja
            if not self.red.p:
                return
            self.cargar()
        x0, y0 = self.caja[:2]
        img.paste(self.fondo, (x0, y0))
        d = ImageDraw.Draw(img)
        act = self.red.act
        valores = {}
        x = act.get("entrada")
        if x is not None:
            cfg = self.cfg
            n = cfg.n_rayos * observacion.POR_RAYO
            actual = x[(cfg.cuadros - 1) * n:cfg.cuadros * n].reshape(cfg.n_rayos, observacion.POR_RAYO)
            valores["entrada"] = actual.mean(axis=0)
            valores["extra"] = x[cfg.cuadros * n:]
            for k, o in enumerate(act.get("ocultas", [])):
                valores[f"o{k}"] = o[self.visibles[k]]
            v = act["salida"]
            z = np.exp(v - v.max())
            valores["salida"] = z / z.sum()
        for clave, puntos in self.pos.items():
            v = valores.get(clave)
            r = self.radio[clave]
            m = float(np.abs(v).max()) if v is not None and len(v) else 0.0
            for i, (px, py) in enumerate(puntos):
                a = float(v[i]) / m if m > 0 else 0.0
                color = mezcla(APAGADO, ENCENDIDO if a >= 0 else NEGATIVO, abs(a))
                elegida = clave == "salida" and i == accion
                d.ellipse((px - r, py - r, px + r, py + r), fill=color,
                          outline=(255, 255, 255) if elegida else (20, 20, 20), width=2 if elegida else 1)


# -- los rayos sobre el juego ----------------------------------------------------------------------------------

def rayos_sobre_juego(img: Image.Image, res, vista: tuple[float, float] = (0.0, 0.0)) -> None:
    """Los rayos de `res.grid` sobre el juego de `img` (a 1280×960 desde (24, 80)): una línea hasta lo que corta
    cada rayo (sólido o borde), con el color de lo primero que ve, y un punto en cada cosa que cruza. `vista`:
    esquina de la pantalla en la room."""
    from ..emulador import RAYO_NADA
    from ..evolucion.pantallas import ESCALA, JUEGO_X, JUEGO_Y
    g = res.grid
    if g is None or g.rayos is None or not g.vivo:
        return
    d = ImageDraw.Draw(img)
    ox, oy = g.origen[0] - vista[0], g.origen[1] - vista[1]

    def en_pantalla(x: float, y: float) -> tuple[float, float]:
        return JUEGO_X + x * ESCALA, JUEGO_Y + y * ESCALA

    p0 = en_pantalla(ox, oy)
    n = len(g.rayos)
    canales = len(observacion.CANALES)
    for i, r in enumerate(g.rayos):
        a = i * 2 * math.pi / n
        dx, dy = math.cos(a), -math.sin(a)
        fin = r[observacion.SOLIDO] if r[observacion.SOLIDO] != RAYO_NADA else r[-1]
        if fin == RAYO_NADA:
            continue
        vistos = sorted((int(r[c]), c) for c in range(canales) if r[c] != RAYO_NADA)
        primero = vistos[0][1] if vistos else canales          # nada antes del borde: el color del borde
        d.line([p0, en_pantalla(ox + dx * fin, oy + dy * fin)], fill=observacion.COLORES[primero], width=1)
        for dist, c in vistos:
            if c == observacion.SOLIDO:
                continue
            x, y = en_pantalla(ox + dx * dist, oy + dy * dist)
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=observacion.COLORES[c])


def leyenda(d: ImageDraw.ImageDraw, x: int, y: int) -> None:
    f = fuente(16)
    titulo = "Rayos (el color de lo primero que ve):"
    d.text((x, y), titulo, fill=TEXTO_2, font=f)
    x += d.textlength(titulo, font=f) + 12
    for nombre, color in zip(observacion.NOMBRES, observacion.COLORES):
        d.text((x, y), nombre, fill=color, font=f)
        x += d.textlength(nombre, font=f) + 14
