"""Lo que ve la red: rayos desde el jugador y unas pocas variables suyas.

Cada rayo sale del centro de la caja del jugador y llega hasta el borde de la pantalla, salvo que lo corte antes un
sólido. Por cada tipo de cosa (los mismos de la vista de `redes`: sólido, plataforma, mortal, save, salida, agua) da
la distancia a lo primero de ese tipo que cruza, y además la distancia al borde de la pantalla si no lo cortó nada.
Así ve una púa delante de un muro, y también el muro. Lo calcula el emulador (`RAYS_CONFIG` en `rl.rs`) contra las
máscaras de colisión: una púa es un triángulo, no un cuadrado.

La red recibe cada distancia como *cercanía*, `1 − d / ALCANCE` (0 = no hay, 1 = pegado), rayo por rayo, más:
velocidad, si le queda el doble salto, hacia dónde avanza el nivel (el camino más corto a la salida desde donde está)
y hacia qué lado mira.
"""

from __future__ import annotations

import numpy as np

from ..distancia import DY, PASO
from ..emulador import RAYO_NADA
from ..redes.vista import CANALES_IWBTG

CANALES = CANALES_IWBTG                       # sólido, plataforma, mortal, save, salida, agua
NOMBRES = tuple(c.nombre for c in CANALES) + ("borde",)
COLORES = tuple(c.color for c in CANALES) + ((110, 60, 140),)
SOLIDO = 0
BANDERAS = tuple(3 if i == SOLIDO else 2 for i in range(len(CANALES)))   # el sólido corta el rayo; todo se ve
POR_RAYO = len(NOMBRES)                       # valores por rayo
ALCANCE = 1000.0                              # px (la diagonal de la pantalla de 800×600)
RADIO_DIRECCION = 3                           # celdas del mapa de distancias que mira para la dirección del nivel


def configurar(emu, cfg) -> list[str]:
    """Deja el emulador devolviendo los rayos en cada paso con `grid=True` (sin la grilla de `redes`, que cuesta).
    Devuelve los objetos que este juego no tiene (su canal queda vacío)."""
    emu.track("player")
    faltan = emu.grid_config(8, 0, 0, 1, [(list(c.objetos), list(c.excluidos), c.solo_visibles) for c in CANALES],
                             ["djump"])
    emu.rays_config(cfg.n_rayos, cfg.paso_px, list(BANDERAS))
    return faltan


def cercanias(g) -> np.ndarray:
    """(rayos, POR_RAYO) float32 en [0, 1]: 0 = ese tipo no está en el rayo, 1 = pegado al jugador."""
    d = g.rayos.astype(np.float32)
    c = np.clip(1.0 - d / ALCANCE, 0.0, 1.0)
    c[g.rayos == RAYO_NADA] = 0.0
    return c


def nombres_extras(cfg) -> list[str]:
    n = []
    if cfg.ver_velocidad:
        n += ["vel →", "vel ↓"]
    if cfg.ver_doble_salto:
        n += ["doble salto"]
    if cfg.ver_direccion:
        n += ["camino →", "camino ↑"]
    if cfg.ver_mirando:
        n += ["mira →"]
    return n


def extras(cfg, g, direccion: tuple[float, float]) -> np.ndarray:
    """Las variables del jugador, en escala ~[-1, 1]."""
    v = []
    if cfg.ver_velocidad:
        v += [g.hspeed / 3.0, g.vspeed / 9.0]
    if cfg.ver_doble_salto:
        dj = g.campos[0] if g.campos else 0.0
        v += [0.0 if np.isnan(dj) else float(dj)]
    if cfg.ver_direccion:
        v += [direccion[0], -direccion[1]]      # hacia arriba = positivo
    if cfg.ver_mirando:
        v += [float(np.sign(g.xscale))]
    return np.array(v, np.float32)


def n_entradas(cfg) -> int:
    return cfg.n_rayos * POR_RAYO * cfg.cuadros + len(nombres_extras(cfg))


def entrada(cfg, g, cuadros: list, direccion: tuple[float, float]) -> tuple[np.ndarray, list]:
    """Vector de entrada de la red y los cuadros actualizados (los rayos de las últimas `cuadros` decisiones)."""
    r = cercanias(g).ravel()
    cuadros = (cuadros + [r])[-cfg.cuadros:]
    while len(cuadros) < cfg.cuadros:
        cuadros.insert(0, r)
    return np.concatenate(cuadros + [extras(cfg, g, direccion)]), cuadros


# -- dirección del nivel ------------------------------------------------------------------------------------

def tabla_direccion(dist: np.ndarray, radio: int = RADIO_DIRECCION) -> np.ndarray:
    """(alto, ancho, 2) float32: desde cada celda del mapa de distancias a la salida, el vector unitario (x, y de la
    pantalla) hacia la celda más cercana a la salida dentro de `radio` celdas; (0, 0) donde no hay camino o ya está
    en lo más cercano."""
    h, w = dist.shape
    d = np.where(np.isfinite(dist), dist, np.inf).astype(np.float64)
    mejor = d.copy()
    vec = np.zeros((h, w, 2), np.float32)
    pad = np.pad(d, radio, constant_values=np.inf)
    for oy in range(-radio, radio + 1):
        for ox in range(-radio, radio + 1):
            if (ox, oy) == (0, 0) or ox * ox + oy * oy > radio * radio:
                continue
            vecino = pad[radio + oy:radio + oy + h, radio + ox:radio + ox + w]
            mas_cerca = vecino < mejor - 1e-6
            mejor = np.where(mas_cerca, vecino, mejor)
            vec[mas_cerca] = (ox, oy)
    largo = np.linalg.norm(vec, axis=2, keepdims=True)
    vec = np.where(largo > 0, vec / np.maximum(largo, 1e-9), 0.0).astype(np.float32)
    vec[~np.isfinite(d)] = 0.0
    return vec


class Direccion:
    """Dirección del nivel en un punto de la sala, a partir de su mapa de distancias a la salida."""

    def __init__(self, dist: np.ndarray, celda: int, dy: float = 0.0, redondear: bool = False):
        self.tabla = tabla_direccion(dist)
        self.celda, self.dy, self.redondear = celda, dy, redondear

    def en(self, x: float, y: float) -> tuple[float, float]:
        h, w, _ = self.tabla.shape
        fy = (y - self.dy) / self.celda
        j = int(np.clip(round(fy) if self.redondear else fy // 1, 0, h - 1))
        i = int(np.clip(x // self.celda, 0, w - 1))
        v = self.tabla[j, i]
        return float(v[0]), float(v[1])


def direccion_gimnasio(dist: np.ndarray) -> Direccion:
    return Direccion(dist, 16)                          # `Gimnasio.avance_en`: celdas de 16 px


def direccion_nivel(dist: np.ndarray) -> Direccion:
    return Direccion(dist, PASO, DY, redondear=True)    # `episodio.Jugador.avance_en`
