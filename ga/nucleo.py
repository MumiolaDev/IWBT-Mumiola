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
from dataclasses import asdict, dataclass, field, replace
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
    # Reintentos: al morir (o entrar a una zona prohibida, o irse de la sala) el agente aprieta R como un jugador
    # y el juego lo devuelve a su último save. Con `resets` reintentos usados, la siguiente muerte lo descalifica
    resets: int = 3
    frames_reset: int = 25             # lo que tarda en apretar R después de morir (0,5 s)
    save_points: str = "auto"          # "x,y;..." de la sala; "auto" = los del juego, salvo los de zonas prohibidas
    # Recompensa (fitness). Solo cuenta lo que pasa en la room de entrenamiento.
    recompensa: str = "salida"         # "salida" (exp. 2) o "exploracion" (exp. 1)
    r_cercania: float = 1000.0         # salida: × fracción del camino recorrida al final (o al morir)
    r_avance: float = 300.0            # salida: × la mayor fracción del camino alcanzada en el episodio
    r_objetivo: float = 1000.0         # por llegar a la sala objetivo
    r_tiempo: float = 5.0              # por cada segundo que sobra al llegar (premia llegar antes)
    r_pantalla: float = 300.0          # por pantalla nueva (en "salida", solo las del camino a la salida)
    r_celda: float = 0.0               # por cada celda de 32x32 px visitada por primera vez
    celda: int = 32
    castigo_muerte: float = 50.0       # al quedar descalificado (sin reintentos). Morir avanzando sigue conviniendo
    castigo_reset: float = 10.0        # por cada reintento usado
    r_save: float = 150.0              # por cada save point distinto activado por primera vez (metas cortas)
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
    muertes: int = 0            # reintentos usados (sin contar la muerte que lo descalifica)
    saves: int = 0              # save points distintos activados

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


SIN_TECLAS, TECLA_R = 254, 255  # valores de `accion` en la traza que no son genes (espera y reintento)


@dataclass
class Estado:
    """Lo que arrastra una partida además del emulador; se copia junto a cada savestate de familia."""
    muertes: int = 0             # reintentos usados
    saves: tuple = ()            # save points activados (índices de `save_points(cfg)`), en orden
    ultimo_save: bytes = b""     # sala y posición del archivo de partida (cambia al guardar)


def leer_save(emu) -> bytes:
    """Sala y posición guardadas en el archivo de partida del juego (sus primeros 9 bytes)."""
    archivos = [p for p in sorted((emu.workdir / "save").glob("save[0-9]*")) if p.suffix == ""]
    return archivos[0].read_bytes()[:9] if archivos else b""


@lru_cache(maxsize=8)
def _puntos(texto: str) -> tuple:
    return tuple(tuple(float(v) for v in z.split(",")) for z in texto.split(";") if z.strip())


def save_points(cfg: Config) -> tuple:
    return () if cfg.save_points == "auto" else _puntos(cfg.save_points)


def save_activado(emu, cfg: Config, guardado: bytes) -> int:
    """Índice del save point que se acaba de activar, o -1 si no es uno de los de la sala."""
    puntos = save_points(cfg)
    b = guardado
    if not puntos or len(b) < 9 or b[0] * 10000 + b[1] * 100 + b[2] != cfg.room_entrenamiento:
        return -1
    # El save al que se le disparó queda con image_index >= 1 durante ~60 frames
    s = emu.exec('with (savePoint) if (image_index >= 1) return string(x) + "," + string(y); return "";')
    if s:
        x, y = (float(v) for v in str(s).split(","))
        tope = 1.0
    else:  # no debería pasar: el más cercano a donde quedó guardado el jugador
        x, y = b[3] * 10000 + b[4] * 100 + b[5], b[6] * 10000 + b[7] * 100 + b[8]
        tope = 400.0
    d = [np.hypot(px - x, py - y) for px, py in puntos]
    k = int(np.argmin(d))
    return k if d[k] <= tope else -1


def muerto(base: np.ndarray, gen: int, acciones) -> np.ndarray:
    """Registros de frames sin jugador: copias del último registro vivo, marcadas como muerto."""
    out = np.repeat(base[None], len(acciones))
    out["alive"] = 0
    out["gen"] = gen
    out["accion"] = acciones
    return out


def ultimo_vivo(trozos: list[np.ndarray], cfg: Config) -> np.ndarray:
    for t in reversed(trozos):
        k = np.nonzero((t["alive"] == 1) & (t["room"] == cfg.room_entrenamiento))[0]
        if len(k):
            return t[k[-1]].copy()
    base = np.zeros((), TRAZA_DTYPE)
    base["room"] = cfg.room_entrenamiento
    return base


def reaparecer(emu, cfg: Config, base: np.ndarray, gen: int, restantes: int) -> np.ndarray:
    """El jugador tarda `frames_reset` en apretar R y el juego lo devuelve a su último save. Devuelve la traza
    de esos frames (como muerto en `base` hasta que reaparece)."""
    acciones = []
    espera = min(cfg.frames_reset, restantes)
    if espera > 0:
        emu.step((), frames=espera, obs=False, no_draw=True)
        acciones += [SIN_TECLAS] * espera
    if len(acciones) < restantes:
        emu.step(("r",), frames=1, obs=False, no_draw=True)
        acciones.append(TECLA_R)
    trozos = [muerto(base, gen, acciones)]
    n = len(acciones)
    while n < restantes and n < len(acciones) + 100:  # el juego pasa por rInit y vuelve a la sala del save
        r = emu.step((), frames=1, obs=False, trace=True, no_draw=True)
        t = compactar(r.trace, gen, SIN_TECLAS)
        n += 1
        if len(t) and t[0]["room"] == cfg.room_entrenamiento and t[0]["alive"]:
            trozos.append(t)
            break
        trozos.append(muerto(base, gen, [SIN_TECLAS]))
    return np.concatenate(trozos)


def jugar(emu, cfg: Config, genoma: Genoma, rng, desde: int = 0, frame0: int = 0, guardar: dict | None = None,
          hasta: int | None = None, estado: Estado | None = None):
    """Ejecuta los genes desde el índice `desde` (estando el juego en el frame `frame0`).

    `guardar` = {índice de gen: slot}: guarda un savestate justo antes de ejecutar ese gen y anota en
    `puntos` el frame, cuántos trozos de traza había y el estado. `hasta` detiene la ejecución antes de ese gen.
    Al morir usa un reintento (R → último save) mientras le queden.
    Devuelve (genoma, trozos de traza, descalificado, puntos, estado).
    """
    max_frames = cfg.max_frames
    trozos, total, i, murio = [], frame0, desde, False
    puntos = {}
    estado = replace(estado) if estado is not None else Estado(ultimo_save=leer_save(emu))
    meta = cfg.room_objetivo
    while total < max_frames:
        if hasta is not None and i >= hasta:
            break
        if guardar and i in guardar:
            emu.save(guardar[i])
            puntos[i] = (guardar[i], total, len(trozos), replace(estado))
        if i >= len(genoma):
            genoma = concatenar(genoma, genes_al_azar(rng, 64, cfg))
        a = int(genoma.acciones[i])
        frames = min(int(genoma.frames[i]), max_frames - total)
        r = emu.step(ACTIONS[a], frames=frames, obs=False, trace=True, no_draw=True, stop_on_gone=True)
        traza = compactar(r.trace, i, a)
        i += 1
        llego = np.nonzero(traza["room"] == meta)[0]
        fuera = frames_prohibidos(traza, cfg)
        cae = False
        if len(fuera) and (not len(llego) or fuera[0] < llego[0]):
            # Se fue a otra room o entró a una zona prohibida: cuenta como morir justo antes
            cae = True
            if estado.muertes >= cfg.resets:
                trozos.append(traza[: fuera[0]])
                total += fuera[0]
                murio = True
                break
            # El resto del gen igual se jugó (el emulador avanzó): queda como muerto en el último lugar válido
            base = ultimo_vivo(trozos + [traza[: fuera[0]]], cfg)
            traza = np.concatenate([traza[: fuera[0]], muerto(base, i - 1, traza["accion"][fuera[0]:])])
            llego = llego[:0]
        if len(llego):
            trozos.append(traza[: llego[0] + 1])
            total += llego[0] + 1
            break
        trozos.append(traza)
        total += len(traza)
        if len(traza) == 0 or not traza[-1]["alive"] or r.ended:
            cae = True
        guardado = leer_save(emu)
        if guardado != estado.ultimo_save:
            estado.ultimo_save = guardado
            k = save_activado(emu, cfg, guardado)
            if k >= 0 and k not in estado.saves:
                estado.saves += (k,)
        if cae:
            if r.ended or estado.muertes >= cfg.resets:
                murio = True
                break
            estado.muertes += 1
            if total < max_frames:
                vuelta = reaparecer(emu, cfg, ultimo_vivo(trozos, cfg), i - 1, max_frames - total)
                trozos.append(vuelta)
                total += len(vuelta)
    return genoma, trozos, murio, puntos, estado


def evaluar(env: IWannaBeTheGBEnv, cfg: Config, genoma: Genoma, semilla: int, padres=(), origen="azar") -> Resultado:
    """Evalúa un genoma desde el inicio de la sala."""
    env.emu.load(0)
    genoma, trozos, murio, _, estado = jugar(env.emu, cfg, genoma, np.random.default_rng(semilla))
    return puntuar(cfg, genoma, unir(trozos), murio, padres, origen, estado)


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
        _, prefijo, _, puntos, _ = jugar(emu, cfg, padre, np.random.default_rng(0), guardar=guardar,
                                         hasta=candidatos[-1] + 1)
    resultados = []
    for (genoma, semilla, padres, origen), k in zip(hijos, ks):
        usables = [j for j in puntos if j <= k]
        rng = np.random.default_rng(semilla)
        if usables:
            j = max(usables)
            slot, frame0, n_trozos, estado0 = puntos[j]
            emu.load(slot)
            genoma, trozos, murio, _, estado = jugar(emu, cfg, genoma, rng, desde=j, frame0=frame0, estado=estado0)
            trozos = prefijo[:n_trozos] + trozos
        else:
            emu.load(0)
            genoma, trozos, murio, _, estado = jugar(emu, cfg, genoma, rng)
        resultados.append(puntuar(cfg, genoma, unir(trozos), murio, padres, origen, estado))
    return resultados


def unir(trozos: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(trozos) if trozos else np.zeros(0, TRAZA_DTYPE)


# Traza compacta (la que se guarda en disco)
TRAZA_DTYPE = np.dtype([
    ("room", "<i2"), ("alive", "u1"), ("x", "<f4"), ("y", "<f4"),
    ("sprite", "<i2"), ("image", "u1"), ("xscale", "i1"),
    ("gen", "<u2"),     # gen que se jugaba en ese frame
    ("accion", "u1"),   # teclas de ese frame: índice de ACTIONS, SIN_TECLAS o TECLA_R
])


def compactar(traza: np.ndarray, gen: int = 0, accion: int = 254) -> np.ndarray:
    out = np.zeros(len(traza), TRAZA_DTYPE)
    out["gen"] = gen
    out["accion"] = accion
    out["room"] = traza["room"]
    out["alive"] = traza["alive"]
    out["x"] = traza["x"]
    out["y"] = traza["y"]
    out["sprite"] = traza["sprite"]
    out["image"] = np.floor(traza["image"]).astype(np.int64) % 256
    out["xscale"] = np.sign(traza["xscale"]).astype(np.int8)
    return out


def completar_traza(traza: np.ndarray, genoma: Genoma) -> np.ndarray:
    """Traza de una corrida anterior a los reintentos (sin `gen` ni `accion`): se deducen del genoma."""
    out = np.zeros(len(traza), TRAZA_DTYPE)
    for k in traza.dtype.names:
        out[k] = traza[k]
    if len(genoma):
        g = np.searchsorted(np.cumsum(genoma.frames.astype(np.int64)), np.arange(len(traza)), side="right")
        g = np.minimum(g, len(genoma) - 1)
        out["gen"] = g
        out["accion"] = genoma.acciones[g]
    return out


def puntuar(cfg: Config, genoma: Genoma, traza: np.ndarray, murio: bool, padres=(), origen="azar",
            estado: Estado | None = None) -> Resultado:
    estado = estado or Estado()
    if cfg.recompensa == "salida":
        return puntuar_salida(cfg, genoma, traza, murio, padres, origen, estado)
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
        + cfg.r_save * len(estado.saves)
        - cfg.castigo_reset * estado.muertes
        - (cfg.castigo_muerte if murio else 0.0)
        - cfg.castigo_tiempo * frame_progreso / FPS
    )
    if llego:
        fitness += cfg.r_objetivo + cfg.r_tiempo * (cfg.max_frames - len(traza)) / FPS
    return Resultado(genoma, float(fitness), murio, len(traza), int(llego), n_pantallas, n_celdas, frame_progreso,
                     traza, tuple(padres), origen, muertes=estado.muertes, saves=len(estado.saves))


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


def puntuar_salida(cfg: Config, genoma: Genoma, traza: np.ndarray, murio: bool, padres=(), origen="azar",
                   estado: Estado | None = None) -> Resultado:
    """Fitness por cercanía a la salida: r_cercania × progreso final + r_avance × mejor progreso
    + r_pantalla por pantalla del camino alcanzada + r_save por save point nuevo − castigos por reintento y por
    quedar descalificado (+ premio si llega)."""
    estado = estado or Estado()
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
        # Cercanía "al final": con reintentos, el mejor final de vida (donde murió o donde lo pilló el tiempo),
        # no el save al que volvió. Con una sola vida es la posición final
        fin_vida = np.nonzero(np.diff(en_sala) != 1)[0]
        progreso = 1.0 - float(min(d[-1], d[fin_vida].min() if len(fin_vida) else np.inf)) / d0
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
        + cfg.r_save * len(estado.saves)
        - cfg.castigo_reset * estado.muertes
        - (cfg.castigo_muerte if murio else 0.0)
    )
    if llego:
        fitness += cfg.r_objetivo + cfg.r_tiempo * (cfg.max_frames - len(traza)) / FPS
    return Resultado(genoma, float(fitness), murio, len(traza), int(llego), len(pantallas), n_celdas, frame_progreso,
                     traza, tuple(padres), origen, progreso, progreso_max, estado.muertes, len(estado.saves))


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


def resolver_saves(cfg: Config) -> None:
    """Reemplaza "auto" por los save points de la sala (los que existen en esta dificultad), salvo los que
    caen en zonas prohibidas."""
    if cfg.save_points != "auto":
        return
    env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                           seed=cfg.seed_juego)
    try:
        s = env.emu.exec('var s; s = ""; with (savePoint) s += string(x) + "," + string(y) + ";"; return s;')
    finally:
        env.close()
    puntos = [p for p in _puntos(str(s))
              if not any(x0 <= p[0] < x1 and y0 <= p[1] < y1 for x0, y0, x1, y1 in zonas(cfg))]
    cfg.save_points = ";".join(f"{x:g},{y:g}" for x, y in puntos)


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


def gen_en_frame(r: Resultado, t: int) -> int:
    """Índice del gen que se jugaba en el frame t (con reintentos no coincide con la suma de duraciones)."""
    if len(r.traza) == 0:
        return 0
    return int(r.traza["gen"][min(max(t, 0), len(r.traza) - 1)])


def muertes_en(r: Resultado) -> list[int]:
    """Frames en que murió (o se cortó la partida por descalificación)."""
    vivo = r.traza["alive"].astype(np.int8)
    fallas = (np.nonzero((vivo[:-1] == 1) & (vivo[1:] == 0))[0] + 1).tolist()
    if r.murio and (not fallas or fallas[-1] < r.frames - 1):
        fallas.append(r.frames)
    return fallas


def cruzar(rng: np.random.Generator, a: Resultado, b: Resultado) -> Genoma:
    """Cruce en un punto del tiempo: los genes de A hasta el frame t y desde ahí los de B."""
    limite = max(1, min(a.frames, b.frames))
    t = int(rng.integers(1, limite + 1))
    ga, gb = gen_en_frame(a, t), gen_en_frame(b, t)
    return concatenar(Genoma(a.genoma.acciones[:ga].copy(), a.genoma.frames[:ga].copy()),
                      Genoma(b.genoma.acciones[gb:].copy(), b.genoma.frames[gb:].copy()))


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
        k = gen_en_frame(padre, t)
        return concatenar(Genoma(g.acciones[:k], g.frames[:k]), genes_al_azar(rng, cfg.genes_iniciales, cfg))

    # 1) a veces, además, un cambio en cualquier punto de la partida (explora rutas alternativas)
    if rng.random() < cfg.prob_mut_global and len(acciones) > 1:
        i = int(rng.integers(0, min(len(acciones), gen_en_frame(padre, jugados) + 1)))
        acciones[i] = otra_accion(rng, int(acciones[i]))

    # 2) mutaciones concentradas en una ventana: antes de una de sus muertes o del mejor momento (local),
    #    o en un momento al azar (punto)
    fallas = muertes_en(padre)
    if modo == 1:
        fin = int(rng.integers(1, jugados + 1))
    else:
        fin = int(rng.choice(fallas)) if fallas else padre.frame_progreso + 1
    ventana = int(cfg.ventana_muerte_s * FPS * rng.uniform(0.5, 1.5))
    t0 = max(0, fin - ventana)
    en_ventana = np.unique(padre.traza["gen"][t0:max(fin, 1)]) if len(padre.traza) else []
    candidatos = [int(i) for i in en_ventana if i < len(acciones)]
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
