"""Algoritmo genético sobre secuencias de entradas para 'I wanna be the GB'.

Un individuo es una secuencia de genes `(acción, frames)`: mantener una de las 12 combinaciones de teclas de
`iwbtgb_rl.ACTIONS` durante 1..`max_frames_gen` frames. El juego es determinista desde el savestate inicial,
así que el genoma reproduce siempre la misma partida (como un TAS) y la traza guardada es exactamente lo que
se ve al repetirlo.

Si el agente sobrevive más de lo que cubre su genoma, se le agregan genes al azar (con la semilla del
individuo) y el genoma guardado incluye esos genes.

Hay dos recompensas (`Config.recompensa`):
- "exploracion" (experimento 1): pantallas y celdas nuevas de la room.
- "salida" (experimento 2): qué tan cerca de la salida queda el agente, medido por el camino del laberinto
  (ver `ga/distancia.py`), más un premio por cada pantalla del camino hacia la salida.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from iwbtgb_rl import ACTIONS, IWannaBeTheGBEnv

FPS = 50
ROOT = Path(__file__).resolve().parent.parent
SCREEN_W, SCREEN_H = 800, 608  # el objeto camera mueve la vista en pasos de 800x608


@dataclass
class Config:
    # Experimento
    nombre: str = "rStage01"
    start_room: str | None = None      # None = inicio normal del juego (rStage01)
    difficulty: str = "medium"
    seed_juego: int = 0
    seed: int = 1
    # Episodio
    max_segundos: float = 60.0         # 1 minuto de juego (3000 frames a 50 FPS)
    sala_objetivo: str = "rStage02"    # llegar aquí completa el episodio
    salir_es_muerte: bool = True       # irse a otra room (p. ej. rStage07 por el techo) cuenta como morir
    # Rectángulos "x0,y0,x1,y1;..." (px) de la room de entrenamiento donde entrar cuenta como morir: zonas sin
    # salida que el agente no puede saber que son inútiles. "auto" = las de ZONAS_AUTO para la sala; "" = ninguna
    zonas_prohibidas: str = "auto"
    # Recompensa (fitness). Solo cuenta lo que pasa en la room de entrenamiento.
    recompensa: str = "salida"         # "salida" (exp. 2) o "exploracion" (exp. 1)
    r_cercania: float = 1000.0         # salida: × fracción del camino recorrida al final (o al morir)
    r_avance: float = 300.0            # salida: × la mayor fracción del camino alcanzada en el episodio
    r_objetivo: float = 1000.0         # por llegar a la sala objetivo
    r_tiempo: float = 5.0              # por cada segundo que sobra al llegar (premia llegar antes)
    r_pantalla: float = 300.0          # por pantalla nueva (en "salida", solo las del camino a la salida)
    r_celda: float = 0.0               # por cada celda de 32x32 px visitada por primera vez
    celda: int = 32
    castigo_muerte: float = 50.0       # salida: 5 % del camino. Morir avanzando sigue conviniendo
    castigo_tiempo: float = 0.0        # exploracion: por segundo hasta el último progreso
    # Se resuelven al arrancar (ids de room de GameMaker)
    room_entrenamiento: int = -1
    room_objetivo: int = -1
    room_ancho: int = 0
    room_alto: int = 0
    mapa_distancia: str = ""           # ruta (relativa al repo) del mapa de distancias a la salida
    # Genoma
    max_frames_gen: int = 20
    genes_iniciales: int = 64
    # Algoritmo genético
    poblacion: int = 50
    generaciones: int = 300
    seleccion: str = "nichos"          # "nichos": los mejores de lugares distintos; "torneo" (exp. 1)
    padres: int = 10                   # nichos: cuántos de los mejores generan la siguiente generación
    nicho_px: int = 64                 # nichos: dos agentes que terminan en el mismo cuadro de 64 px son "iguales"
    elite: int = 10                    # pasan sin cambios a la siguiente generación
    torneo: int = 3
    prob_cruce: float = 0.2
    prob_mut_global: float = 0.1       # prob. de que un hijo además mute en un punto al azar de toda su partida
    # Tipo de mutación de cada hijo (se normalizan): ensayo y error justo antes de morir/estancarse,
    # cambios en un momento al azar de la partida, o "rama nueva" (desde un punto, teclas al azar)
    prob_local: float = 0.5
    prob_punto: float = 0.3
    prob_rama: float = 0.2
    ventana_muerte_s: float = 2.0      # largo de la zona donde caen las mutaciones
    mut_ventana_min: int = 1
    mut_ventana_max: int = 6
    # Ejecución
    workers: int = max(1, min(12, (os.cpu_count() or 2) - 2))

    @property
    def max_frames(self) -> int:
        return int(round(self.max_segundos * FPS))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Genoma:
    acciones: np.ndarray   # uint8
    frames: np.ndarray     # uint8

    def __len__(self) -> int:
        return len(self.acciones)

    def copia(self) -> "Genoma":
        return Genoma(self.acciones.copy(), self.frames.copy())

    def inicios(self) -> np.ndarray:
        """Frame en que empieza cada gen."""
        return np.concatenate([[0], np.cumsum(self.frames.astype(np.int64))[:-1]])


def genes_al_azar(rng: np.random.Generator, n: int, cfg: Config) -> Genoma:
    return Genoma(
        rng.integers(0, len(ACTIONS), n, dtype=np.uint8),
        rng.integers(1, cfg.max_frames_gen + 1, n, dtype=np.uint8),
    )


def concatenar(a: Genoma, b: Genoma) -> Genoma:
    return Genoma(np.concatenate([a.acciones, b.acciones]), np.concatenate([a.frames, b.frames]))


@dataclass(eq=False)
class Resultado:
    genoma: Genoma
    fitness: float
    murio: bool
    frames: int                 # frames jugados
    llego: int                  # 1 = llegó a la sala objetivo, 0 = no
    pantallas: int              # pantallas distintas visitadas de la room de entrenamiento (incluida la inicial)
    celdas: int
    frame_progreso: int         # frame de la última celda nueva (desempate: antes es mejor)
    traza: np.ndarray = field(repr=False)  # TRACE_DTYPE, un registro por frame
    padres: tuple = ()
    origen: str = "azar"
    progreso: float = 0.0       # salida: fracción del camino recorrida al final (1 = en la salida)
    progreso_max: float = 0.0   # salida: la mayor fracción alcanzada

    def clave(self) -> tuple:
        return (self.fitness, -self.frame_progreso)


# -- evaluación (en procesos aparte, un emulador por proceso) ----------------------------------------

_env: IWannaBeTheGBEnv | None = None
_cfg: Config | None = None


def iniciar_worker(cfg: Config) -> None:
    global _env, _cfg
    _cfg = cfg
    _env = IWannaBeTheGBEnv(
        difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state", seed=cfg.seed_juego
    )
    _env.emu.track("player")


MAX_SLOTS = 6  # savestates por familia (~11 MB cada uno)


def evaluar_en_worker(args) -> list[Resultado]:
    """Evalúa una familia: los hijos de un mismo padre parten del savestate del padre donde se separan."""
    padre, hijos = args
    return evaluar_familia(_env, _cfg, padre, hijos)


# Zonas sin salida conocidas, por sala de entrenamiento
ZONAS_AUTO = {
    "rStage01": "800,0,1600,608",  # pantalla de arriba a la derecha: sala de portales que abren los jefes
}


def resolver_zonas(cfg: Config) -> None:
    """Reemplaza "auto" por los rectángulos de la sala, para que config.json diga exactamente qué se usó."""
    if cfg.zonas_prohibidas == "auto":
        cfg.zonas_prohibidas = ZONAS_AUTO.get(cfg.start_room or "rStage01", "")


@lru_cache(maxsize=8)
def _rectangulos(texto: str) -> tuple:
    return tuple(tuple(float(v) for v in z.split(",")) for z in texto.split(";") if z.strip())


def zonas(cfg: Config) -> tuple:
    return () if cfg.zonas_prohibidas == "auto" else _rectangulos(cfg.zonas_prohibidas)


def frames_prohibidos(traza: np.ndarray, cfg: Config) -> np.ndarray:
    """Frames en que el jugador se fue a otra room que no es la objetivo o entró a una zona prohibida."""
    malo = np.zeros(len(traza), bool)
    if cfg.salir_es_muerte:
        malo |= (traza["room"] != cfg.room_objetivo) & (traza["room"] != cfg.room_entrenamiento)
    en_sala = traza["room"] == cfg.room_entrenamiento
    for x0, y0, x1, y1 in zonas(cfg):
        malo |= en_sala & (traza["x"] >= x0) & (traza["x"] < x1) & (traza["y"] >= y0) & (traza["y"] < y1)
    return np.nonzero(malo)[0]


def jugar(emu, cfg: Config, genoma: Genoma, rng, desde: int = 0, frame0: int = 0, guardar: dict | None = None,
          hasta: int | None = None):
    """Ejecuta los genes desde el índice `desde` (estando el juego en el frame `frame0`).

    `guardar` = {índice de gen: slot}: guarda un savestate justo antes de ejecutar ese gen y anota en
    `puntos` el frame y cuántos trozos de traza había. `hasta` detiene la ejecución antes de ese gen.
    Devuelve (genoma, trozos de traza, murió, puntos).
    """
    max_frames = cfg.max_frames
    trozos, total, i, murio = [], frame0, desde, False
    puntos = {}
    meta = cfg.room_objetivo
    while total < max_frames:
        if hasta is not None and i >= hasta:
            break
        if guardar and i in guardar:
            emu.save(guardar[i])
            puntos[i] = (guardar[i], total, len(trozos))
        if i >= len(genoma):
            genoma = concatenar(genoma, genes_al_azar(rng, 64, cfg))
        frames = min(int(genoma.frames[i]), max_frames - total)
        r = emu.step(ACTIONS[genoma.acciones[i]], frames=frames, obs=False, trace=True, no_draw=True,
                     stop_on_gone=True)
        traza = compactar(r.trace)
        i += 1
        llego = np.nonzero(traza["room"] == meta)[0]
        fuera = frames_prohibidos(traza, cfg)
        if len(fuera) and (not len(llego) or fuera[0] < llego[0]):
            # Se fue a otra room o entró a una zona prohibida: termina como si hubiera muerto justo antes
            trozos.append(traza[: fuera[0]])
            total += fuera[0]
            murio = True
            break
        if len(llego):
            trozos.append(traza[: llego[0] + 1])
            total += llego[0] + 1
            break
        trozos.append(traza)
        total += len(traza)
        if len(traza) == 0 or not traza[-1]["alive"] or r.ended:
            murio = True
            break
    return genoma, trozos, murio, puntos


def evaluar(env: IWannaBeTheGBEnv, cfg: Config, genoma: Genoma, semilla: int, padres=(), origen="azar") -> Resultado:
    """Evalúa un genoma desde el inicio de la sala."""
    env.emu.load(0)
    genoma, trozos, murio, _ = jugar(env.emu, cfg, genoma, np.random.default_rng(semilla))
    return puntuar(cfg, genoma, unir(trozos), murio, padres, origen)


def primer_gen_distinto(a: Genoma, b: Genoma) -> int:
    n = min(len(a), len(b))
    distinto = np.nonzero((a.acciones[:n] != b.acciones[:n]) | (a.frames[:n] != b.frames[:n]))[0]
    return int(distinto[0]) if len(distinto) else n


def evaluar_familia(env: IWannaBeTheGBEnv, cfg: Config, padre: Genoma | None, hijos: list) -> list[Resultado]:
    """`hijos` = [(genoma, semilla, padres, origen)]. Sin padre, todos se juegan desde el inicio."""
    emu = env.emu
    ks = [primer_gen_distinto(padre, h[0]) if padre is not None else 0 for h in hijos]
    candidatos = sorted({k for k in ks if k > 0})
    if len(candidatos) > MAX_SLOTS:
        candidatos = sorted({candidatos[int(round(q))] for q in np.linspace(0, len(candidatos) - 1, MAX_SLOTS)})
    puntos, prefijo = {}, []
    if candidatos:
        emu.load(0)
        guardar = {k: slot + 1 for slot, k in enumerate(candidatos)}
        _, prefijo, _, puntos = jugar(emu, cfg, padre, np.random.default_rng(0), guardar=guardar,
                                      hasta=candidatos[-1] + 1)
    resultados = []
    for (genoma, semilla, padres, origen), k in zip(hijos, ks):
        usables = [j for j in puntos if j <= k]
        rng = np.random.default_rng(semilla)
        if usables:
            j = max(usables)
            slot, frame0, n_trozos = puntos[j]
            emu.load(slot)
            genoma, trozos, murio, _ = jugar(emu, cfg, genoma, rng, desde=j, frame0=frame0)
            trozos = prefijo[:n_trozos] + trozos
        else:
            emu.load(0)
            genoma, trozos, murio, _ = jugar(emu, cfg, genoma, rng)
        resultados.append(puntuar(cfg, genoma, unir(trozos), murio, padres, origen))
    return resultados


def unir(trozos: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(trozos) if trozos else np.zeros(0, TRAZA_DTYPE)


# Traza compacta (la que se guarda en disco)
TRAZA_DTYPE = np.dtype([
    ("room", "<i2"), ("alive", "u1"), ("x", "<f4"), ("y", "<f4"),
    ("sprite", "<i2"), ("image", "u1"), ("xscale", "i1"),
])


def compactar(traza: np.ndarray) -> np.ndarray:
    out = np.zeros(len(traza), TRAZA_DTYPE)
    out["room"] = traza["room"]
    out["alive"] = traza["alive"]
    out["x"] = traza["x"]
    out["y"] = traza["y"]
    out["sprite"] = traza["sprite"]
    out["image"] = np.floor(traza["image"]).astype(np.int64) % 256
    out["xscale"] = np.sign(traza["xscale"]).astype(np.int8)
    return out


def puntuar(cfg: Config, genoma: Genoma, traza: np.ndarray, murio: bool, padres=(), origen="azar") -> Resultado:
    if cfg.recompensa == "salida":
        return puntuar_salida(cfg, genoma, traza, murio, padres, origen)
    en_sala = np.nonzero((traza["alive"] == 1) & (traza["room"] == cfg.room_entrenamiento))[0]
    # Fuera de los bordes (p. ej. dentro del hueco del techo) cuenta como el borde: no es una pantalla nueva
    x = np.clip(np.floor(traza["x"][en_sala]).astype(np.int64), 0, cfg.room_ancho - 1)
    y = np.clip(np.floor(traza["y"][en_sala]).astype(np.int64), 0, cfg.room_alto - 1)
    celdas = (x // cfg.celda) * 10_000 + (y // cfg.celda)
    n_celdas = len(np.unique(celdas))
    n_pantallas = len(np.unique((x // SCREEN_W) * 100 + (y // SCREEN_H)))
    llego = len(traza) > 0 and int(traza["room"][-1]) == cfg.room_objetivo
    if llego:
        frame_progreso = len(traza) - 1
    elif len(celdas):
        _, primeras = np.unique(celdas, return_index=True)
        frame_progreso = int(en_sala[primeras.max()])
    else:
        frame_progreso = 0
    fitness = (
        cfg.r_pantalla * max(0, n_pantallas - 1)
        + cfg.r_celda * n_celdas
        - (cfg.castigo_muerte if murio else 0.0)
        - cfg.castigo_tiempo * frame_progreso / FPS
    )
    if llego:
        fitness += cfg.r_objetivo + cfg.r_tiempo * (cfg.max_frames - len(traza)) / FPS
    return Resultado(genoma, float(fitness), murio, len(traza), int(llego), n_pantallas, n_celdas, frame_progreso,
                     traza, tuple(padres), origen)


_mapas: dict[str, dict] = {}


def mapa_distancia(cfg: Config) -> dict:
    if cfg.mapa_distancia not in _mapas:
        from . import distancia
        _mapas[cfg.mapa_distancia] = distancia.cargar(ROOT / cfg.mapa_distancia)
    return _mapas[cfg.mapa_distancia]


def distancias(cfg: Config, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distancia (px, por el laberinto) desde cada posición hasta la salida."""
    from .distancia import DY, PASO
    dist = mapa_distancia(cfg)["dist"]
    ny, nx = dist.shape
    # La grilla prueba y = 16·j + DY: se redondea a la celda cuyo punto de prueba queda más cerca
    j = np.clip(np.round((y - DY) / PASO).astype(np.int64), 0, ny - 1)
    i = np.clip((x // PASO).astype(np.int64), 0, nx - 1)
    return dist[j, i]


def puntuar_salida(cfg: Config, genoma: Genoma, traza: np.ndarray, murio: bool, padres=(), origen="azar") -> Resultado:
    """Fitness por cercanía a la salida: r_cercania × progreso final + r_avance × mejor progreso
    + r_pantalla por pantalla del camino alcanzada − castigo por morir (+ premio si llega)."""
    mapa = mapa_distancia(cfg)
    d0 = float(mapa["d0"])
    en_sala = np.nonzero((traza["alive"] == 1) & (traza["room"] == cfg.room_entrenamiento))[0]
    x = np.clip(traza["x"][en_sala].astype(np.float64), 0, cfg.room_ancho - 1)
    y = np.clip(traza["y"][en_sala].astype(np.float64), 0, cfg.room_alto - 1)
    celdas = (x.astype(np.int64) // cfg.celda) * 10_000 + (y.astype(np.int64) // cfg.celda)
    n_celdas = len(np.unique(celdas))
    pantallas = set(zip((x // SCREEN_W).astype(int).tolist(), (y // SCREEN_H).astype(int).tolist()))
    ruta = {tuple(p) for p in mapa["pantallas_ruta"].tolist()}
    llego = len(traza) > 0 and int(traza["room"][-1]) == cfg.room_objetivo
    if len(en_sala):
        d = distancias(cfg, x, y)
        progreso = 1.0 - float(d[-1]) / d0
        k = int(np.argmin(d))
        progreso_max = 1.0 - float(d[k]) / d0
        frame_progreso = int(en_sala[k])
    else:
        progreso = progreso_max = 0.0
        frame_progreso = 0
    if llego:
        progreso = progreso_max = 1.0
        frame_progreso = len(traza) - 1
    fitness = (
        cfg.r_cercania * progreso
        + cfg.r_avance * progreso_max
        + cfg.r_pantalla * max(0, len(pantallas & ruta) - 1)
        + cfg.r_celda * n_celdas
        - (cfg.castigo_muerte if murio else 0.0)
    )
    if llego:
        fitness += cfg.r_objetivo + cfg.r_tiempo * (cfg.max_frames - len(traza)) / FPS
    return Resultado(genoma, float(fitness), murio, len(traza), int(llego), len(pantallas), n_celdas, frame_progreso,
                     traza, tuple(padres), origen, progreso, progreso_max)


def posicion_final(r: Resultado, cfg: Config) -> tuple[float, float]:
    """Último lugar donde estuvo vivo dentro de la sala (donde murió, o donde terminó el tiempo)."""
    t = r.traza
    vivo = np.nonzero((t["alive"] == 1) & (t["room"] == cfg.room_entrenamiento))[0]
    if not len(vivo):
        return (0.0, 0.0)
    reg = t[vivo[-1]]
    return float(reg["x"]), float(reg["y"])


def nicho(r: Resultado, cfg: Config) -> tuple:
    x, y = posicion_final(r, cfg)
    return (int(r.llego), int(x // cfg.nicho_px), int(y // cfg.nicho_px))


def resolver_rooms(cfg: Config) -> None:
    """Llena `room_entrenamiento` y `room_objetivo` con los ids de GameMaker y, con la recompensa
    "salida", construye el mapa de distancias a la salida."""
    env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                           seed=cfg.seed_juego)
    try:
        info = env.emu.info()
        cfg.room_entrenamiento, cfg.room_ancho, cfg.room_alto = info.id, info.width, info.height
        cfg.room_objetivo = int(env.emu.exec(f"return {cfg.sala_objetivo}"))
        if cfg.recompensa == "salida":
            from . import distancia
            jugador = env.emu.query("player")[0]
            mapa = distancia.construir(env.emu, cfg.room_objetivo, info.width, info.height, (jugador.x, jugador.y))
            ruta = ROOT / "corridas" / cfg.nombre / "distancia.npz"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            distancia.guardar(ruta, mapa)
            cfg.mapa_distancia = str(ruta.relative_to(ROOT))
    finally:
        env.close()


# -- operadores genéticos -----------------------------------------------------------------------------

def torneo(rng: np.random.Generator, pob: list[Resultado], k: int) -> int:
    """Índice del mejor de `k` individuos tomados al azar."""
    idx = rng.choice(len(pob), size=min(k, len(pob)), replace=False)
    return int(max(idx, key=lambda i: pob[i].clave()))


def cortar_en_frame(g: Genoma, frame: int) -> tuple[Genoma, Genoma]:
    """Parte el genoma en el frame dado (dividiendo el gen que lo cruza)."""
    inicios = g.inicios()
    if frame >= int(g.frames.astype(np.int64).sum()):
        return g.copia(), Genoma(g.acciones[:0], g.frames[:0])
    i = int(np.searchsorted(inicios, frame, side="right")) - 1
    if i < 0:
        return Genoma(g.acciones[:0], g.frames[:0]), g.copia()
    dentro = frame - int(inicios[i])
    a = Genoma(g.acciones[:i].copy(), g.frames[:i].copy())
    b = Genoma(g.acciones[i:].copy(), g.frames[i:].copy())
    if dentro > 0:
        a = concatenar(a, Genoma(b.acciones[:1].copy(), np.array([dentro], np.uint8)))
        b.frames[0] -= dentro
        if b.frames[0] == 0:
            b = Genoma(b.acciones[1:], b.frames[1:])
    return a, b


def cruzar(rng: np.random.Generator, a: Resultado, b: Resultado) -> Genoma:
    """Cruce en un punto del tiempo: la partida de A hasta el frame t y desde ahí las entradas de B."""
    limite = max(1, min(a.frames, b.frames))
    t = int(rng.integers(1, limite + 1))
    pre, _ = cortar_en_frame(a.genoma, t)
    _, post = cortar_en_frame(b.genoma, t)
    return concatenar(pre, post)


def genes_usados(g: Genoma, frames: int) -> int:
    """Cuántos genes alcanzan a ejecutarse en `frames` frames."""
    return min(len(g), int(np.searchsorted(np.cumsum(g.frames.astype(np.int64)), frames)) + 1)


def otra_accion(rng: np.random.Generator, actual: int) -> int:
    a = int(rng.integers(0, len(ACTIONS) - 1))
    return a + (a >= actual)


def mutar(rng: np.random.Generator, padre: Resultado, cfg: Config) -> Genoma:
    """Elige un tipo de mutación: ensayo y error justo antes de donde el padre murió o se estancó (local),
    cambios en un momento al azar de su partida (punto) o una rama nueva con teclas al azar desde un punto."""
    g = padre.genoma.copia()
    acciones, frames = list(g.acciones), list(g.frames)
    jugados = max(1, padre.frames)

    pesos = np.array([cfg.prob_local, cfg.prob_punto, cfg.prob_rama], float)
    modo = int(rng.choice(3, p=pesos / pesos.sum())) if pesos.sum() > 0 else 0
    if modo == 2:
        t = int(rng.integers(max(1, int(0.2 * jugados)), jugados + 1))
        pre, _ = cortar_en_frame(g, t)
        return concatenar(pre, genes_al_azar(rng, cfg.genes_iniciales, cfg))

    # 1) a veces, además, un cambio en cualquier punto de la partida (explora rutas alternativas)
    if rng.random() < cfg.prob_mut_global and len(acciones) > 1:
        i = int(rng.integers(0, genes_usados(g, jugados)))
        acciones[i] = otra_accion(rng, int(acciones[i]))

    # 2) mutaciones concentradas en una ventana: antes del final (muerte) o del mejor momento (local),
    #    o en un momento al azar (punto)
    if modo == 1:
        fin = int(rng.integers(1, jugados + 1))
    else:
        fin = padre.frames if padre.murio else padre.frame_progreso + 1
    ventana = int(cfg.ventana_muerte_s * FPS * rng.uniform(0.5, 1.5))
    t0 = max(0, fin - ventana)
    inicios = np.concatenate([[0], np.cumsum(np.array(frames, np.int64))[:-1]]) if frames else np.zeros(0, np.int64)
    candidatos = [i for i in range(len(acciones)) if t0 <= inicios[i] < max(fin, 1)]
    if not candidatos:
        candidatos = [max(0, len(acciones) - 1)]
    n = int(rng.integers(cfg.mut_ventana_min, cfg.mut_ventana_max + 1))
    for _ in range(n):
        i = int(rng.choice(candidatos))
        if i >= len(acciones):
            continue
        op = rng.random()
        if op < 0.45:                                    # otra acción
            acciones[i] = otra_accion(rng, int(acciones[i]))
        elif op < 0.75:                                  # otra duración
            delta = int(rng.choice([-1, 1]) * rng.integers(1, 7))
            frames[i] = int(np.clip(int(frames[i]) + delta, 1, cfg.max_frames_gen))
        elif op < 0.9:                                   # insertar un gen
            acciones.insert(i, rng.integers(0, len(ACTIONS)))
            frames.insert(i, rng.integers(1, cfg.max_frames_gen + 1))
            candidatos = [c + (c >= i) for c in candidatos]
        elif len(acciones) > 1:                          # borrar un gen
            del acciones[i], frames[i]
            candidatos = [c - (c > i) for c in candidatos if c != i] or [max(0, i - 1)]
    return Genoma(np.array(acciones, np.uint8), np.array(frames, np.uint8))
