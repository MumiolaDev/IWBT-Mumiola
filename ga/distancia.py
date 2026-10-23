"""Mapa de distancia a la salida: la "noción de dónde está la salida" que tiene el agente.

La room se divide en una grilla de `PASO` px. Para cada celda se prueba si el jugador cabe ahí
(`place_meeting` con su máscara contra `block`, sin contar las plataformas, que se mueven) y si tocaría algo que mata (`playerKiller`). Desde las
celdas junto a la salida se calcula la distancia más corta caminando por el espacio libre (Dijkstra con
8 vecinos), así que mide el camino por el laberinto y no la línea recta. Pasar por celdas que matan cuesta
`COSTO_PELIGRO` veces más (no se prohíbe, porque a veces el camino real pasa saltando sobre pinchos).

Es una heurística: ignora la gravedad y el alcance del salto, igual que un jugador que mira el mapa.

La salida es el `roomChanger` (o un warp habilitado) cuyo `roomTo` es la sala objetivo.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np

PASO = 16
COSTO_PELIGRO = 8.0
# Punto de prueba dentro de cada celda: con y = 16·j + 7 el jugador parado sobre un piso alineado a 16 px
# (y ≈ 16·j + 7.4) cae en una celda libre.
DY = 7
LIBRE, MURO, PELIGRO = 0, 1, 2


def capturar_grilla(emu, ancho: int, alto: int) -> np.ndarray:
    """Grilla (alto/PASO, ancho/PASO) con LIBRE, MURO o PELIGRO, probando la máscara del jugador."""
    nx, ny = (ancho + PASO - 1) // PASO, (alto + PASO - 1) // PASO
    codigo = f"""
        var s, i, j, cx, cy; s = "";
        instance_deactivate_object(platform);  // las plataformas se mueven: no son muros fijos
        with (player) {{
            for (j = 0; j < {ny}; j += 1) {{
                for (i = 0; i < {nx}; i += 1) {{
                    cx = i * {PASO} + {PASO // 2}; cy = j * {PASO} + {DY};
                    if (place_meeting(cx, cy, block)) s += "1";
                    else if (place_meeting(cx, cy, playerKiller)) s += "2";
                    else s += "0";
                }}
            }}
        }}
        instance_activate_object(platform);
        return s;
    """
    s = emu.exec(codigo)
    if not isinstance(s, str) or len(s) != nx * ny:
        raise RuntimeError("no se pudo leer la grilla de colisiones (¿no hay player en la room?)")
    return np.frombuffer(s.encode(), np.uint8).reshape(ny, nx) - ord("0")


def buscar_salidas(emu, room_objetivo: int) -> list[tuple[float, float, float, float]]:
    """Cajas (izq, arriba, der, abajo) de los cambiadores de room que llevan a la sala objetivo."""
    codigo = f"""
        var s; s = "";
        with (roomChanger) {{
            if (roomTo == {room_objetivo}) {{
                if (object_index == roomChanger || object_index == WarioRoomChanger) s += string(bbox_left) + "," + string(bbox_top) + "," + string(bbox_right) + "," + string(bbox_bottom) + ";";
                else if (onCollide) s += string(bbox_left) + "," + string(bbox_top) + "," + string(bbox_right) + "," + string(bbox_bottom) + ";";
            }}
        }}
        return s;
    """
    s = emu.exec(codigo)
    return [tuple(float(v) for v in caja.split(",")) for caja in str(s).split(";") if caja]


def celdas_meta(grilla: np.ndarray, salidas, margen: float = 40.0) -> np.ndarray:
    """Celdas libres a menos de `margen` px de alguna salida (recortadas a la room)."""
    ny, nx = grilla.shape
    cy, cx = (np.mgrid[0:ny, 0:nx] * PASO).astype(float)
    cy += DY
    cx += PASO // 2
    meta = np.zeros_like(grilla, bool)
    for (x0, y0, x1, y1) in salidas:
        dx = np.maximum(0, np.maximum(x0 - cx, cx - x1))
        dy = np.maximum(0, np.maximum(y0 - cy, cy - y1))
        meta |= np.hypot(dx, dy) <= margen
    meta &= grilla == LIBRE
    if not meta.any():  # salida rodeada de cosas: la celda libre más cercana
        x0, y0, x1, y1 = salidas[0]
        d = np.hypot(cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
        d[grilla != LIBRE] = np.inf
        meta.flat[int(np.argmin(d))] = True
    return meta


def dijkstra(grilla: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Distancia en px desde cada celda transitable a la meta (inf si no hay camino)."""
    ny, nx = grilla.shape
    costo = np.where(grilla == MURO, np.inf, np.where(grilla == PELIGRO, COSTO_PELIGRO, 1.0))
    dist = np.full(grilla.shape, np.inf)
    cola = []
    for j, i in zip(*np.nonzero(meta)):
        dist[j, i] = 0.0
        cola.append((0.0, int(j), int(i)))
    heapq.heapify(cola)
    vecinos = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    while cola:
        d, j, i = heapq.heappop(cola)
        if d > dist[j, i]:
            continue
        for dj, di, paso in vecinos:
            a, b = j + dj, i + di
            if 0 <= a < ny and 0 <= b < nx and costo[a, b] != np.inf:
                # Las diagonales no cortan esquinas de muro
                if dj and di and (costo[j + dj, i] == np.inf or costo[j, i + di] == np.inf):
                    continue
                nd = d + paso * 0.5 * (costo[j, i] + costo[a, b]) * PASO
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(cola, (nd, a, b))
    return dist


def rellenar(dist: np.ndarray) -> np.ndarray:
    """Las celdas sin distancia (muros, zonas cerradas) toman la de la celda vecina más cercana con valor,
    para que una posición pegada a un muro no quede sin puntaje."""
    out = dist.copy()
    for _ in range(64):
        falta = ~np.isfinite(out)
        if not falta.any():
            break
        p = np.pad(out, 1, constant_values=np.inf)
        vec = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]) + PASO
        out[falta] = vec.min(0)[falta]
    out[~np.isfinite(out)] = np.nanmax(out[np.isfinite(out)])
    return out


def camino(dist: np.ndarray, inicio: tuple[int, int]) -> list[tuple[int, int]]:
    """Celdas (j, i) del camino más corto desde `inicio` bajando por el gradiente de distancia."""
    ny, nx = dist.shape
    j, i = inicio
    ruta = [(j, i)]
    for _ in range(ny * nx):
        if dist[j, i] == 0:
            break
        mejor = min(((dist[a, b], a, b) for a in range(max(0, j - 1), min(ny, j + 2))
                     for b in range(max(0, i - 1), min(nx, i + 2))), key=lambda t: t[0])
        if mejor[0] >= dist[j, i]:
            break
        j, i = mejor[1], mejor[2]
        ruta.append((j, i))
    return ruta


def construir(emu, room_objetivo: int, ancho: int, alto: int, inicio_xy: tuple[float, float]) -> dict:
    grilla = capturar_grilla(emu, ancho, alto)
    salidas = buscar_salidas(emu, room_objetivo)
    if not salidas:
        raise RuntimeError("la room de entrenamiento no tiene ninguna salida hacia la sala objetivo")
    dist_cruda = dijkstra(grilla, celdas_meta(grilla, salidas))
    ij = celda_de(inicio_xy[0], inicio_xy[1], grilla.shape)
    if not np.isfinite(dist_cruda[ij]):  # el inicio cayó en una celda ocupada: la libre más cercana
        jj, ii = np.nonzero(np.isfinite(dist_cruda))
        k = int(np.argmin(np.hypot(jj - ij[0], ii - ij[1])))
        if np.hypot(jj[k] - ij[0], ii[k] - ij[1]) > 3:
            raise RuntimeError("no hay camino libre desde el inicio hasta la salida")
        ij = (int(jj[k]), int(ii[k]))
    dist = rellenar(dist_cruda)
    ruta = camino(dist_cruda, ij)
    # Pantallas de 800x608 que atraviesa el camino más corto (las "útiles")
    pantallas = sorted({(i * PASO // 800, (j * PASO + DY) // 608) for j, i in ruta})
    return {
        "grilla": grilla, "dist": dist.astype(np.float32), "salidas": np.array(salidas, np.float32),
        "ruta": np.array(ruta, np.int32), "d0": np.float32(dist[ij]), "inicio": np.array(inicio_xy, np.float32),
        "pantallas_ruta": np.array(pantallas, np.int32).reshape(-1, 2),
    }


def celda_de(x: float, y: float, forma) -> tuple[int, int]:
    ny, nx = forma
    return (int(np.clip(y // PASO, 0, ny - 1)), int(np.clip(x // PASO, 0, nx - 1)))


def guardar(ruta: Path, mapa: dict) -> None:
    np.savez_compressed(ruta, **mapa)


def cargar(ruta: Path) -> dict:
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}
