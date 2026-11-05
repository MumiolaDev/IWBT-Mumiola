"""Go-Explore sobre 'I wanna be the GB': un archivo de celdas en vez de una población.

    uv run python -m iwbtgb.ga explorar --nombre ge01 --workers 4 --iteraciones 2000
    uv run python -m iwbtgb.ga explorar --nombre ge01 --continuar --iteraciones 5000

Una celda es un lugar de la sala (cuadro de `celda_px` px) más si el jugador sube, baja o está quieto en
vertical. Para cada celda el archivo guarda el camino más corto (en frames) con que se llegó a ella. Cada
iteración elige celdas, vuelve a ellas (savestate o repitiendo su camino desde el comienzo de la sala) y desde
ahí prueba ráfagas de teclas al azar; toda celda nueva, o alcanzada con un camino más corto, entra al archivo.
Morir solo termina la ráfaga: lo ganado antes de morir queda.

Elección de celdas: peso = (1/√(1+veces elegida) + peso_vista/√(1+veces vista)) × e^(beta·(progreso − mejor)),
con el progreso medido como en la recompensa `salida` (distancia por el laberinto a la salida).

Cada `guardar_cada` iteraciones se guarda el archivo (`archivo.pkl`, para `--continuar`), el mapa de celdas
(`graficos/mapa_archivo.png`) y, como `gen_NNNN.npz` (NNNN = iteración), las `muestra` celdas más avanzadas con
su partida completa desde el comienzo de la sala: `iwbtgb.ga visualizar` hace con ellas sus videos y gráficos. La
config del juego queda en `config.json` (la misma `Config` del algoritmo genético) y la de Go-Explore en
`explorar.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import signal
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import opciones
from ..emulador import ACTIONS
from . import almacen
from .entrenar import resumen
from .nucleo import (TRAZA_DTYPE, Config, Genoma, compactar, distancias, emulador, frames_prohibidos, mapa_distancia,
                     puntuar, resolver_rooms, resolver_zonas)


@dataclass
class ConfigExplorar:
    iteraciones: int = 1000
    workers: int = 4
    tareas_por_worker: int = 4         # celdas que explora cada worker por iteración
    exploraciones: int = 4             # ráfagas desde cada celda elegida
    frames_exploracion: int = 100      # largo máximo de cada ráfaga (2 s)
    celda_px: int = 16
    celda_salto: bool = True           # la celda distingue si al jugador le queda el doble salto
    celda_plataforma: bool = True      # y dónde está la plataforma móvil más cercana (en cuadros de 48 px)
    radio_plataforma: int = 160
    frames_plataformas: int = 30000    # cuántos frames se precalcula el movimiento de las plataformas
    beta: float = 15.0                 # preferencia por las celdas más cerca de la salida (0 = ninguna)
    peso_vista: float = 0.5
    cache: int = 16                    # savestates por worker (~11 MB cada uno)
    guardar_cada: int = 25
    muestra: int = 30                  # celdas más avanzadas que se guardan como "generación"
    seed: int = 1


CAMPOS_EXTRA = ["archivo_celdas", "nuevas", "mejoradas", "exploraciones_total", "frames_repeticion", "llegadas"]


@dataclass
class Nodo:
    """Trozo de camino: los genes y la traza desde el estado del nodo `padre` hasta una celda. Los nodos no
    cambian; una celda que mejora apunta a un nodo nuevo."""
    padre: int
    acciones: np.ndarray
    frames: np.ndarray
    traza: np.ndarray            # TRAZA_DTYPE; `gen` es el índice dentro de este trozo
    total: int                   # frames desde el comienzo de la sala
    fin: tuple = (None, 0.0, True, False)  # estado al final: (y, dy, doble salto, Shift apretado)


@dataclass
class Celda:
    nodo: int
    total: int
    progreso: float
    x: float
    y: float
    iteracion: int               # en que se descubrió
    elegida: int = 0
    vista: int = 0


@dataclass
class Archivo:
    nodos: dict = field(default_factory=dict)
    celdas: dict = field(default_factory=dict)
    siguiente: int = 1
    iteracion: int = 0
    exploraciones: int = 0
    llegadas: int = 0

    def cadena(self, nodo: int) -> list[int]:
        """Ids de los nodos desde el comienzo (sin la raíz 0) hasta `nodo`."""
        ids = []
        while nodo != 0:
            ids.append(nodo)
            nodo = self.nodos[nodo].padre
        return ids[::-1]


RAIZ = Nodo(0, np.zeros(0, np.uint8), np.zeros(0, np.uint8), np.zeros(0, TRAZA_DTYPE), 0)


# -- caché de savestates por worker (el proceso principal lleva una copia exacta para repartir las tareas) --

class Cache:
    def __init__(self, n: int):
        self.slots: OrderedDict[int, int] = OrderedDict()
        self.libres = list(range(n, 0, -1))   # el slot 0 es el comienzo de la sala

    def mas_profundo(self, ids: list[int]) -> int:
        for k in range(len(ids) - 1, -1, -1):
            if ids[k] in self.slots:
                return k
        return -1

    def _poner(self, nodo: int) -> int:
        if nodo not in self.slots:
            self.slots[nodo] = self.libres.pop() if self.libres else self.slots.popitem(last=False)[1]
        self.slots.move_to_end(nodo)
        return self.slots[nodo]

    def ir(self, ids: list[int], escala: int = -1) -> tuple[int, int, int]:
        """Registra la vuelta a `ids[-1]`, guardando de paso `ids[escala]` (un ancestro que comparten muchas
        celdas) si queda después del nodo cacheado. Devuelve (índice del nodo cacheado desde el que se parte o
        -1, slot donde queda el destino, slot de la escala o -1)."""
        k = self.mas_profundo(ids)
        if k >= 0:
            self.slots.move_to_end(ids[k])
        slot_escala = self._poner(ids[escala]) if k < escala < len(ids) - 1 else -1
        if not ids:
            return k, 0, -1
        return k, self._poner(ids[-1]), slot_escala


# -- worker ----------------------------------------------------------------------------------------------

_plataformas: np.ndarray | None = None  # en cada worker: posiciones de las plataformas móviles por frame


def _worker(conexion, cfg: Config, cfg_e: ConfigExplorar, plataformas: np.ndarray | None) -> None:
    global _plataformas
    _plataformas = plataformas
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # el principal decide cuándo cerrar
    emu = emulador(cfg)
    emu.track("player")
    cache = Cache(cfg_e.cache)
    try:
        while (tareas := conexion.recv()) is not None:
            try:
                conexion.send([_tarea(emu, cfg, cfg_e, cache, *t) for t in tareas])
            except Exception:
                conexion.send(RuntimeError("worker:\n" + traceback.format_exc()))
                raise
    finally:
        emu.close()


def _tarea(emu, cfg: Config, cfg_e: ConfigExplorar, cache: Cache, ids: list[int], escala: int, segmentos: list,
           fin: tuple, total: int, semilla: int):
    k, slot, slot_escala = cache.ir(ids, escala)
    emu.load(cache.slots[ids[k]] if k >= 0 else 0)
    repeticion = 0
    for j in range(k + 1, len(segmentos)):
        acciones, frames = segmentos[j]
        for a, n in zip(acciones.tolist(), frames.tolist()):
            emu.step(ACTIONS[a], frames=n, obs=False, no_draw=True)
            repeticion += n
        if j == escala and slot_escala >= 0:
            emu.save(slot_escala)
    if ids and k < len(ids) - 1:
        emu.save(slot)
    rng = np.random.default_rng(semilla)
    salidas = []
    for m in range(cfg_e.exploraciones):
        if m > 0:
            emu.load(slot)
        salidas.append(_rafaga(emu, cfg, cfg_e, rng, fin, total))
    return salidas, repeticion


CON_SHIFT = np.array(["shift" in teclas for teclas in ACTIONS] + [False] * (256 - len(ACTIONS)))


def doble_salto(traza: np.ndarray, fin: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Deduce de la traza, frame a frame, `player.djump` (si le queda el doble salto) y el dy. Se gasta al
    apretar Shift en el aire y se recupera al quedar quieto en vertical (aterrizar). Es una aproximación (un
    golpe contra el techo también deja dy = 0): solo sirve para distinguir celdas, los caminos son exactos."""
    y0, dy_prev, dj, shift = fin
    y = traza["y"].astype(np.float64)
    dy = np.round(y - np.concatenate([[y[0] if y0 is None else y0], y[:-1]]), 3) if len(y) else y
    sh = CON_SHIFT[traza["accion"]]
    djs = np.empty(len(traza), bool)
    for i in range(len(traza)):
        if sh[i] and not shift and dy_prev != 0:
            dj = False
        if dy[i] == 0:
            dj = True
        djs[i], shift, dy_prev = dj, sh[i], dy[i]
    return djs, dy


def _rafaga(emu, cfg: Config, cfg_e: ConfigExplorar, rng, estado0: tuple, t0: int):
    """Teclas al azar hasta `frames_exploracion` frames, la muerte, irse de la sala o llegar a la salida, desde
    un nodo que terminó en el estado `estado0` (`Nodo.fin`). Devuelve los genes, la traza (solo frames con el
    jugador vivo y en un lugar válido), si llegó, la primera vez que pasó por cada celda: [(celda, frame)], y el
    doble salto deducido por frame."""
    acciones, frames, trozos = [], [], []
    total, llego = 0, False
    while total < cfg_e.frames_exploracion:
        a = int(rng.integers(len(ACTIONS)))
        n = min(int(rng.integers(1, cfg.max_frames_gen + 1)), cfg_e.frames_exploracion - total)
        r = emu.step(ACTIONS[a], frames=n, obs=False, trace=True, no_draw=True, stop_on_gone=True)
        t = compactar(r.trace, len(acciones), a)
        acciones.append(a)
        frames.append(n)
        corte, fin = len(t), len(t) < n or r.ended
        meta = np.nonzero(t["room"] == cfg.room_objetivo)[0]
        if len(meta):
            corte, fin, llego = int(meta[0]) + 1, True, True
        fuera = frames_prohibidos(t, cfg)
        if len(fuera) and fuera[0] < corte:
            corte, fin, llego = int(fuera[0]), True, False
        muerto = np.nonzero(t["alive"][:corte] == 0)[0]
        if len(muerto):
            corte, fin, llego = int(muerto[0]), True, False
        trozos.append(t[:corte])
        total += corte
        if fin:
            break
    traza = np.concatenate(trozos) if trozos else np.zeros(0, TRAZA_DTYPE)
    djs, dys = doble_salto(traza, estado0)
    en_sala = np.nonzero(traza["room"] == cfg.room_entrenamiento)[0]
    vistas = {}
    if len(en_sala):
        x = np.clip(traza["x"][en_sala], 0, cfg.room_ancho - 1).astype(np.int64) // cfg_e.celda_px
        cy = np.clip(traza["y"][en_sala], 0, cfg.room_alto - 1).astype(np.int64) // cfg_e.celda_px
        dy = np.sign(dys[en_sala]).astype(np.int64)
        dj = djs[en_sala].astype(np.int64) if cfg_e.celda_salto else np.zeros(len(en_sala), np.int64)
        claves = [x.tolist(), cy.tolist(), dy.tolist(), dj.tolist()]
        if cfg_e.celda_plataforma and _plataformas is not None:
            px, py = plataforma_cercana(traza["x"][en_sala], traza["y"][en_sala], t0 + en_sala + 1, cfg_e)
            claves += [px.tolist(), py.tolist()]
        for f, celda in zip(en_sala.tolist(), zip(*claves)):
            vistas.setdefault(celda, f)
    return (np.array(acciones, np.uint8), np.array(frames, np.uint8), traza, llego, list(vistas.items()),
            djs, dys)


def plataforma_cercana(x: np.ndarray, y: np.ndarray, t: np.ndarray, cfg_e: ConfigExplorar):
    """Posición relativa (cuadros de 48 px) de la plataforma móvil más cercana al jugador en cada frame, si hay
    una a menos de `radio_plataforma` px; si no, (9, 9). Las plataformas se mueven solas (no dependen del
    jugador), así que su posición sale de la tabla precalculada por frame desde el comienzo de la sala."""
    fuera = np.full(len(x), 9, np.int64)
    validos = t < len(_plataformas)
    if not validos.any():
        return fuera, fuera.copy()
    pos = _plataformas[np.minimum(t, len(_plataformas) - 1)]          # (frames, plataformas, 2)
    dx = pos[..., 0] - x[:, None]
    dy = pos[..., 1] - y[:, None]
    dist = np.where(np.isnan(dx), np.inf, np.maximum(np.abs(dx), np.abs(dy)))
    j = np.argmin(dist, axis=1)
    k = np.arange(len(x))
    cerca = validos & (dist[k, j] <= cfg_e.radio_plataforma)
    bx = np.where(cerca, np.floor(np.nan_to_num(dx[k, j]) / 48), 9).astype(np.int64)
    by = np.where(cerca, np.floor(np.nan_to_num(dy[k, j]) / 48), 9).astype(np.int64)
    return bx, by


def tabla_plataformas(carpeta: Path, cfg: Config, frames: int) -> np.ndarray:
    """Posición (x, y) de cada plataforma móvil en cada frame desde el comienzo de la sala, sin teclas (NaN si
    no existe). Se calcula una vez y queda en plataformas.npy."""
    ruta = carpeta / "plataformas.npy"
    if ruta.exists():
        return np.load(ruta)
    print(f"precalculando las plataformas móviles ({frames} frames)...", flush=True)
    emu = emulador(cfg)
    try:
        emu.load(0)
        columnas: dict[int, int] = {}
        filas = []
        for _ in range(frames):
            fila = {}
            for i in emu.query("movingPlatform"):
                fila[columnas.setdefault(i.id, len(columnas))] = (i.x, i.y)
            filas.append(fila)
            emu.step((), frames=1, obs=False, no_draw=True)
    finally:
        emu.close()
    tabla = np.full((frames, max(1, len(columnas)), 2), np.nan, np.float32)
    for t, fila in enumerate(filas):
        for c, xy in fila.items():
            tabla[t, c] = xy
    np.save(ruta, tabla)
    return tabla


# -- proceso principal ------------------------------------------------------------------------------------

def progreso_de(cfg: Config, x: float, y: float) -> float:
    d0 = float(mapa_distancia(cfg)["d0"])
    d = float(distancias(cfg, np.array([x], np.float64), np.array([y], np.float64))[0])
    return 1.0 - d / d0 if np.isfinite(d) else -1.0


def recortar(acciones: np.ndarray, frames: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Los genes que cubren los primeros `n` frames (el último, acortado)."""
    fin = np.cumsum(frames.astype(np.int64))
    j = int(np.searchsorted(fin, n))
    fr = frames[: j + 1].copy()
    fr[j] = n - (int(fin[j - 1]) if j > 0 else 0)
    return acciones[: j + 1].copy(), fr


def incorporar(arch: Archivo, cfg: Config, padre: int, rafaga) -> tuple[int, int]:
    """Agrega al archivo lo que encontró una ráfaga desde el nodo `padre`. Devuelve (celdas nuevas, mejoradas)."""
    acciones, frames, traza, llego, vistas, djs, dys = rafaga
    base = arch.nodos[padre].total
    nuevas = mejoradas = 0
    arch.exploraciones += 1

    def nodo(n: int) -> int:
        acc, fr = recortar(acciones, frames, n)
        i = arch.siguiente
        arch.siguiente += 1
        fin = (float(traza["y"][n - 1]), float(dys[n - 1]), bool(djs[n - 1]), bool(CON_SHIFT[traza["accion"][n - 1]]))
        arch.nodos[i] = Nodo(padre, acc, fr, traza[:n].copy(), base + n, fin)
        return i

    for celda, f in vistas:
        total = base + f + 1
        c = arch.celdas.get(celda)
        if c is None:
            x, y = float(traza["x"][f]), float(traza["y"][f])
            arch.celdas[celda] = Celda(nodo(f + 1), total, progreso_de(cfg, x, y), x, y, arch.iteracion, vista=1)
            nuevas += 1
            continue
        c.vista += 1
        if total < c.total:
            c.nodo, c.total = nodo(f + 1), total
            mejoradas += 1
    if llego:
        total = base + len(traza)
        c = arch.celdas.get("meta")
        if c is None or total < c.total:
            arch.celdas["meta"] = Celda(nodo(len(traza)), total, 1.0, float(traza["x"][-1]),
                                        float(traza["y"][-1]), arch.iteracion)
            arch.llegadas += 1
    return nuevas, mejoradas


def elegir(arch: Archivo, cfg_e: ConfigExplorar, rng, n: int) -> list:
    claves = [k for k in arch.celdas if k != "meta"]
    if not claves:
        return [None] * n                              # al principio: todo desde el comienzo de la sala
    c = [arch.celdas[k] for k in claves]
    prog = np.array([x.progreso for x in c])
    w = (1 / np.sqrt(1 + np.array([x.elegida for x in c]))
         + cfg_e.peso_vista / np.sqrt(1 + np.array([x.vista for x in c])))
    w *= np.exp(cfg_e.beta * (prog - prog.max()))
    idx = rng.choice(len(claves), size=n, p=w / w.sum())
    for i in idx:
        c[i].elegida += 1
    return [claves[i] for i in idx]


def repartir(arch: Archivo, elegidas: list, caches: list[Cache], por_worker: int) -> list[list]:
    """Cada celda va al worker que tenga en caché el nodo más cercano a ella (menos frames que repetir), con
    `por_worker` tareas cada uno. Las cachés del principal imitan exactamente las de los workers."""
    tareas = [[] for _ in caches]
    for clave in elegidas:
        nodo = arch.celdas[clave].nodo if clave is not None else 0
        ids = arch.cadena(nodo)
        libres = [w for w in range(len(caches)) if len(tareas[w]) < por_worker]

        def costo(w):
            k = caches[w].mas_profundo(ids)
            return arch.nodos[nodo].total - (arch.nodos[ids[k]].total if k >= 0 else 0), len(tareas[w])

        w = min(libres, key=costo)
        # Escala: el ancestro más profundo que queda al menos 3 s antes de la celda (lo comparten sus vecinas)
        tope = arch.nodos[nodo].total - 150
        escala = max((j for j, i in enumerate(ids) if arch.nodos[i].total <= tope), default=-1)
        caches[w].ir(ids, escala)
        segmentos = [(arch.nodos[i].acciones, arch.nodos[i].frames) for i in ids]
        tareas[w].append((nodo, (ids, escala, segmentos, arch.nodos[nodo].fin, arch.nodos[nodo].total)))
    return tareas


def partida(arch: Archivo, nodo: int) -> tuple[Genoma, np.ndarray]:
    """Genes y traza completos desde el comienzo de la sala hasta `nodo`."""
    acc, fr, trozos, g = [], [], [], 0
    for i in arch.cadena(nodo):
        n = arch.nodos[i]
        t = n.traza.copy()
        t["gen"] += g
        acc.append(n.acciones)
        fr.append(n.frames)
        trozos.append(t)
        g += len(n.acciones)
    if not trozos:
        return Genoma(np.zeros(0, np.uint8), np.zeros(0, np.uint8)), np.zeros(0, TRAZA_DTYPE)
    return Genoma(np.concatenate(acc), np.concatenate(fr)), np.concatenate(trozos)


def muestra(arch: Archivo, cfg: Config, n: int) -> list:
    """Las `n` celdas más avanzadas (la llegada primero), como Resultados con su partida completa."""
    orden = sorted(arch.celdas.values(), key=lambda c: (-c.progreso, c.total))[:n]
    pob = []
    for c in orden:
        genoma, traza = partida(arch, c.nodo)
        pob.append(puntuar(cfg, genoma, traza, False, origen="archivo"))
    return pob


def podar(arch: Archivo) -> None:
    """Borra los nodos que ya no están en el camino de ninguna celda."""
    vivos = {0}
    for c in arch.celdas.values():
        n = c.nodo
        while n not in vivos:
            vivos.add(n)
            n = arch.nodos[n].padre
    arch.nodos = {i: n for i, n in arch.nodos.items() if i in vivos}


_escena = None


def mapa_archivo(carpeta: Path, cfg: Config, arch: Archivo, mejor: np.ndarray) -> Path:
    """Las celdas del archivo sobre el mapa de la sala, coloreadas por la iteración en que se descubrieron, y
    el camino de la celda más avanzada."""
    import matplotlib
    from PIL import Image, ImageDraw
    from .visualizar import Escena
    global _escena
    if _escena is None:
        _escena = Escena(carpeta, cfg)
    esc = _escena
    img = esc.mapa.copy().convert("RGBA")
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dc = ImageDraw.Draw(capa)
    k = esc.escala
    px = json.loads((carpeta / "explorar.json").read_text())["celda_px"]
    primera = {}
    for clave, c in arch.celdas.items():
        if clave != "meta":
            xy = clave[:2]
            primera[xy] = min(primera.get(xy, c.iteracion), c.iteracion)
    tope = max(arch.iteracion, 1)
    cmap = matplotlib.colormaps["viridis"]
    for (cx, cy), it in primera.items():
        r, g, b, _ = cmap(it / tope)
        dc.rectangle((cx * px * k, cy * px * k, (cx + 1) * px * k - 1, (cy + 1) * px * k - 1),
                     fill=(int(r * 255), int(g * 255), int(b * 255), 150))
    oscuro = Image.new("RGBA", img.size, (0, 0, 0, 110))
    img = Image.alpha_composite(Image.alpha_composite(img, oscuro), capa)
    dr = ImageDraw.Draw(img)
    puntos = [(float(r["x"]) * k, (float(r["y"]) - 8) * k) for r in mejor if r["room"] == cfg.room_entrenamiento]
    if len(puntos) > 1:
        dr.line(puntos, fill=(255, 255, 255, 255), width=3)
    destino = carpeta / "graficos"
    destino.mkdir(exist_ok=True)
    ruta = destino / "mapa_archivo.png"
    img.convert("RGB").save(ruta)
    return ruta


def guardar(carpeta: Path, cfg: Config, arch: Archivo, rng) -> None:
    podar(arch)
    tmp = carpeta / "archivo.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump({"archivo": arch, "rng": rng.bit_generator.state}, f, protocol=5)
    tmp.replace(carpeta / "archivo.pkl")
    pob = muestra(arch, cfg, json.loads((carpeta / "explorar.json").read_text())["muestra"])
    if pob:
        almacen.guardar_generacion(carpeta, arch.iteracion, pob)
        mapa_archivo(carpeta, cfg, arch, pob[0].traza)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m iwbtgb.ga explorar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True)
    p.add_argument("--continuar", action="store_true", help="seguir desde el último archivo guardado")
    for nombre in ("start_room", "sala_objetivo", "difficulty", "zonas_prohibidas"):
        p.add_argument(f"--{nombre.replace('_', '-')}", default=None)
    p.add_argument("--max-frames-gen", type=int, default=None)
    opciones.agregar(p, ConfigExplorar)
    args = p.parse_args(argv)

    carpeta = almacen.dir_corrida(args.nombre)
    if args.continuar:
        cfg = almacen.cargar_config(carpeta)
        d = json.loads((carpeta / "explorar.json").read_text())
        d.setdefault("celda_salto", False)  # archivos de antes del doble salto en la celda
        d.setdefault("celda_plataforma", False)  # y de antes de las plataformas
        cfg_e = ConfigExplorar(**d)
        with (carpeta / "archivo.pkl").open("rb") as f:
            estado = pickle.load(f)
        arch = estado["archivo"]
        rng = np.random.default_rng()
        rng.bit_generator.state = estado["rng"]
    else:
        if carpeta.exists():
            sys.exit(f"{carpeta} ya existe: usa --continuar u otro nombre")
        cfg = Config(nombre=args.nombre, resets=0, prob_desde_save=0.0, save_points="", max_segundos=600.0,
                     seleccion="archivo")
        for nombre in ("start_room", "sala_objetivo", "difficulty", "zonas_prohibidas", "max_frames_gen"):
            if getattr(args, nombre) is not None:
                setattr(cfg, nombre, getattr(args, nombre))
        cfg_e = ConfigExplorar()
        arch = Archivo(nodos={0: RAIZ})
        rng = None
    for k, v in opciones.dadas(args, ConfigExplorar).items():
        setattr(cfg_e, k, v)
    if rng is None:
        rng = np.random.default_rng(cfg_e.seed)
    cfg.workers, cfg.poblacion = cfg_e.workers, cfg_e.muestra
    if cfg.room_entrenamiento < 0:
        resolver_rooms(cfg)
    resolver_zonas(cfg)
    carpeta.mkdir(parents=True, exist_ok=True)
    plataformas = tabla_plataformas(carpeta, cfg, cfg_e.frames_plataformas) if cfg_e.celda_plataforma else None
    almacen.guardar_config(carpeta, cfg)
    (carpeta / "explorar.json").write_text(json.dumps(asdict(cfg_e), indent=2))
    print(f"Go-Explore {carpeta}  workers {cfg_e.workers}  iteración {arch.iteracion}  celdas {len(arch.celdas)}  "
          f"room {cfg.room_entrenamiento} -> {cfg.sala_objetivo}  zonas prohibidas: {cfg.zonas_prohibidas or 'ninguna'}",
          flush=True)

    ctx = mp.get_context("spawn")
    procesos, conexiones = [], []
    for _ in range(cfg_e.workers):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=_worker, args=(b, cfg, cfg_e, plataformas), daemon=True)
        pr.start()
        b.close()  # si el worker muere, recv() da EOFError en vez de esperar para siempre
        procesos.append(pr)
        conexiones.append(a)
    caches = [Cache(cfg_e.cache) for _ in procesos]
    campos = almacen.CAMPOS_METRICAS + CAMPOS_EXTRA
    try:
        while arch.iteracion < cfg_e.iteraciones:
            t0 = time.time()
            arch.iteracion += 1
            elegidas = elegir(arch, cfg_e, rng, cfg_e.workers * cfg_e.tareas_por_worker)
            tareas = repartir(arch, elegidas, caches, cfg_e.tareas_por_worker)
            for con, ts in zip(conexiones, tareas):
                con.send([(*datos, int(rng.integers(2**31))) for _, datos in ts])
            nuevas = mejoradas = repeticion = 0
            for con, ts in zip(conexiones, tareas):
                respuesta = con.recv()
                if isinstance(respuesta, Exception):
                    raise respuesta
                for (nodo, _), (rafagas, rep) in zip(ts, respuesta):
                    repeticion += rep
                    for r in rafagas:
                        n, m = incorporar(arch, cfg, nodo, r)
                        nuevas += n
                        mejoradas += m
            pob = muestra(arch, cfg, cfg_e.muestra)
            fila = resumen(arch.iteracion, pob, cfg_e.workers * cfg_e.tareas_por_worker * cfg_e.exploraciones,
                           time.time() - t0, cfg)
            fila.update(archivo_celdas=len(arch.celdas), nuevas=nuevas, mejoradas=mejoradas,
                        exploraciones_total=arch.exploraciones, frames_repeticion=repeticion,
                        llegadas=arch.llegadas)
            almacen.agregar_metricas(carpeta, fila, campos)
            print(f"iter {arch.iteracion:5d} | celdas {len(arch.celdas):6d} (+{nuevas:3d}, mejor {mejoradas:3d}) | "
                  f"camino máx {fila['progreso_max']:5.1f} % | llegó {arch.llegadas} | "
                  f"repetidos {repeticion:6d} frames | {fila['segundos_reloj']:5.1f} s", flush=True)
            if arch.iteracion % cfg_e.guardar_cada == 0 or arch.iteracion == cfg_e.iteraciones:
                guardar(carpeta, cfg, arch, rng)
    except KeyboardInterrupt:
        # Lo que alcanzó a incorporarse de la iteración a medias queda (son caminos válidos)
        print("detenido; guardando el archivo (para seguir: --continuar)", flush=True)
        guardar(carpeta, cfg, arch, rng)
        raise
    finally:
        for con in conexiones:
            try:
                con.send(None)
            except (BrokenPipeError, OSError):
                pass
        for pr in procesos:
            pr.join(timeout=10)
            if pr.is_alive():
                pr.terminate()
    print("listo", flush=True)
