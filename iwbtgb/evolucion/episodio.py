"""Una red jugando un nivel: escenarios, avance hacia la salida, reintentos y fitness."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..distancia import DY, PASO
from ..emulador import ACTIONS, SLOT_INICIO, arrancar
from ..traza import SIN_TECLAS, TECLA_R, compactar, leer_save
from .genoma import Cache, Genoma

QUIETO_PX = 0.5      # desplazamiento entre decisiones bajo el cual cuenta como quieto
SLOT_MUERTE = 99     # savestate del momento de morir, mientras se muestra la muerte en un video (`reaparecer`)


@dataclass
class Escenario:
    espera: int = 0              # frames sin control antes de que juegue la red
    semilla: int = 0             # para las esperas al reaparecer (`espera_reintento`); en el gimnasio, la sala
    dificultad: float = 0.0      # en el gimnasio


@dataclass
class Partida:
    """Cómo le fue en un escenario."""
    avance: float                # mejor fracción del camino a la salida (0 = comienzo, 1 = salida)
    frame_avance: int            # frame en que lo alcanzó
    frames: int
    fin: str                     # "llegó", "murió", "salió" (otra sala o zona prohibida), "estancado" o "tiempo"
    final: tuple[float, float]   # último lugar donde estuvo vivo en la sala (con reintentos: el de su mejor avance)
    muertes: int = 0             # con reintentos: veces que apretó R (murió, salió o se atascó)
    quieto: float = 0.0          # segundos con el control sin desplazarse (frente a una pared, también)
    saves: int = 0               # save points nuevos activados
    celdas: int = 0              # celdas distintas (`celda_explorar`) por las que pasó vivo en la sala
    acciones: np.ndarray | None = None   # índice de ACTIONS por decisión (para repetir la partida)
    recorrido: np.ndarray | None = None  # (x, y) por decisión
    traza: np.ndarray | None = None      # registro por frame del jugador (`traza.TRAZA_DTYPE`)


@dataclass
class Evaluacion:
    fitness: float
    partidas: list[Partida] = field(default_factory=list)

    @property
    def avance(self) -> float:
        return float(np.mean([p.avance for p in self.partidas]))

    @property
    def llego(self) -> bool:
        return any(p.fin == "llegó" for p in self.partidas)


def plan_teclas(teclas: tuple[str, ...], frames: int, sostenido: bool, soltar: bool, g) -> list[tuple[tuple, int]]:
    """Cómo emular una decisión: [(teclas, frames), ...]. En el juego el salto sale al *apretar* la tecla: si la red
    sigue pidiendo salto con la tecla ya apretada y el jugador no está subiendo (en el suelo, o cayendo con el doble
    salto disponible), con `soltar` el primer frame va sin salto para que el siguiente lo apriete de nuevo. Mientras
    sube se mantiene (soltar antes corta el salto)."""
    if soltar and sostenido and "shift" in teclas and g is not None and g.vspeed >= 0:
        sin = tuple(t for t in teclas if t != "shift")
        return [(sin, 1)] + ([(teclas, frames - 1)] if frames > 1 else [])
    return [(teclas, frames)]


def desglose(cfg, p: Partida) -> dict[str, float]:
    """Puntaje de una partida por término (el fitness de una red es el promedio de la suma en sus escenarios)."""
    return {
        "avance": cfg.r_avance * p.avance,
        "llegó": cfg.r_objetivo if p.fin == "llegó" else 0.0,
        "rapidez": -cfg.r_rapidez * p.frame_avance / cfg.max_frames,
        "saves": cfg.r_save * p.saves,
        "exploración": cfg.r_explorar * p.celdas,
        "quieto": -cfg.castigo_quieto * p.quieto,
        "muertes": -cfg.castigo_muerte * (p.muertes + (p.fin in ("murió", "salió", "estancado"))),
    }


class Jugador:
    """Emulador + lo que ve la red + mapa de distancias + red: juega escenarios de un nivel con los pesos de un
    genoma. Cada línea implementa `crear_vista`, `crear_red` y `decidir`."""

    def __init__(self, cfg, mapa: dict):
        self.cfg = cfg
        self.emu = arrancar(cfg.difficulty, cfg.start_room, cfg.seed_juego)
        self.vista = self.crear_vista()
        self.red = self.crear_red()
        self.cache = Cache(64)
        self.dist = mapa["dist"]
        self.d0 = float(mapa["d0"])
        self.zonas = cfg.zonas()
        self.saves = self.puntos_save() if cfg.r_save else []
        self.muertes, self.activados, self.celdas = 0, set(), set()
        self._traza: list | None = None
        self.saltar_muerte = True

    def close(self) -> None:
        self.emu.close()

    # -- lo que cambia con la forma de ver ----------------------------------------------------------------

    def crear_vista(self):
        """Configura en el emulador lo que ve la red (y el objeto seguido); devuelve lo que use `decidir`."""
        raise NotImplementedError

    def crear_red(self):
        raise NotImplementedError

    def decidir(self, g, cuadros: list) -> tuple[int, np.ndarray, list]:
        """(acción, puntuación de cada acción, cuadros) con la grilla o los rayos `g`; `cuadros` son las
        observaciones de las decisiones anteriores (se vacía al reaparecer)."""
        raise NotImplementedError

    # -- partida ------------------------------------------------------------------------------------------

    def avance_en(self, x: float, y: float) -> float:
        ny, nx = self.dist.shape
        j = int(np.clip(round((y - DY) / PASO), 0, ny - 1))
        i = int(np.clip(x // PASO, 0, nx - 1))
        return 1.0 - float(self.dist[j, i]) / self.d0

    def puntos_save(self) -> list[tuple[float, float]]:
        """Save points que dan premio: los de la sala salvo los del comienzo y los de zonas prohibidas."""
        texto = str(self.emu.exec('var s; s = ""; with (savePoint) s += string(x) + "," + string(y) + ";"; '
                                  'return s;'))
        puntos = []
        for z in texto.split(";"):
            if z:
                x, y = (float(v) for v in z.split(","))
                if self.avance_en(x + 16, y + 16) >= 0.05 and not self.en_zona(x, y):
                    puntos.append((x, y))
        return puntos

    def en_zona(self, x: float, y: float) -> bool:
        return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in self.zonas)

    def save_activado(self) -> int:
        """Índice en `self.saves` del save point al que se acaba de disparar (-1 si no da premio)."""
        s = self.emu.exec('with (savePoint) if (image_index >= 1) return string(x) + "," + string(y); return "";')
        if s:
            x, y = (float(v) for v in str(s).split(","))
            for i, (sx, sy) in enumerate(self.saves):
                if abs(sx - x) < 1 and abs(sy - y) < 1:
                    return i
        return -1

    def jugar(self, genoma: Genoma, esc: Escenario, grabar: bool = False, al_frame=None) -> Partida:
        """`al_frame(res, accion, salidas)` se llama tras cada decisión (para videos), con la puntuación que la red
        le dio a cada acción; entonces se dibujan los frames. Mientras reaparece, con `accion` = None.

        Con `reintentos`, morir, salir de la sala (o entrar a una zona prohibida) o pasar `paciencia` sin avanzar no
        terminan la partida: aprieta R, el juego lo devuelve al último save y sigue hasta que se acaba el tiempo."""
        cfg, emu = self.cfg, self.emu
        self.red.cargar(self.red.pesos(genoma, self.cache))
        emu.load(SLOT_INICIO)
        dibujar = al_frame is not None
        k = cfg.frames_por_decision
        rng = np.random.default_rng((esc.semilla, esc.espera))
        self._traza = [] if grabar else None
        res = self.paso((), esc.espera + 1, dibujar, SIN_TECLAS)
        cuadros: list[np.ndarray] = []
        mejor, frame_mejor, frames = -np.inf, 0, esc.espera + 1
        sin_mejorar = 0
        paciencia = int(cfg.paciencia * 50)
        acciones, recorrido = [], []
        final = lugar_mejor = res.grid.centro
        self.muertes, self.activados, self.quieto = 0, set(), 0     # a la vista para los videos (quieto en frames)
        self.celdas = set()
        celda = cfg.celda_explorar
        antes = None                                                 # posición en la decisión anterior
        sostenido = False                                            # la tecla de salto quedó apretada
        guardado = leer_save(emu) if self.saves else b""
        fin = "tiempo"
        while True:
            g = res.grid
            if res.ended:
                fin = "murió"
                break
            caida = None
            if not g.vivo:
                caida = "murió"
            elif g.room != cfg.room_entrenamiento:
                if g.room == cfg.room_objetivo:
                    fin = "llegó"
                    break
                caida = "salió"
            elif self.en_zona(*g.centro):
                caida = "salió"
            else:
                x, y = g.centro
                # Quieto = no se desplazó desde la decisión anterior (no se mira hspeed: una plataforma móvil mueve al
                # jugador sin cambiarle la velocidad)
                if antes is not None and abs(x - antes[0]) < QUIETO_PX and abs(y - antes[1]) < QUIETO_PX:
                    self.quieto += k
                antes = (x, y)
                final = (float(x), float(y))
                if cfg.r_explorar:
                    self.celdas.add((int(x // celda), int(y // celda)))
                a = self.avance_en(x, y)
                if a > mejor + 1e-3:
                    mejor, frame_mejor, sin_mejorar, lugar_mejor = a, frames, 0, final
                elif sin_mejorar >= paciencia:
                    caida = "estancado"
            if caida:
                if not cfg.reintentos:
                    fin = caida
                    break
                self.muertes += 1
                res, frames = self.reaparecer(caida, frames, rng, al_frame)
                if res is None:
                    break
                cuadros, sin_mejorar, antes, sostenido = [], 0, None, False
                continue
            if frames >= cfg.max_frames:
                break
            accion, salidas, cuadros = self.decidir(g, cuadros)
            if grabar:
                acciones.append(accion)
                recorrido.append((x, y))
            plan = plan_teclas(ACTIONS[accion], k, sostenido, cfg.soltar_salto, g)
            for i, (teclas, n) in enumerate(plan):
                if i and not res.grid.vivo:
                    break
                res = self.paso(teclas, n, dibujar, ACTIONS.index(teclas), stop_on_gone=True)
            sostenido = "shift" in plan[-1][0]
            frames += k
            sin_mejorar += k
            if self.saves:
                nuevo = leer_save(emu)
                if nuevo != guardado:
                    guardado = nuevo
                    i = self.save_activado()
                    if i >= 0:
                        self.activados.add(i)
            if dibujar:
                al_frame(res, accion, salidas)
        if fin == "llegó":
            mejor, frame_mejor = 1.0, frames
        return Partida(max(mejor, 0.0), frame_mejor, frames, fin, lugar_mejor if cfg.reintentos else final,
                       self.muertes, self.quieto / 50, len(self.activados), len(self.celdas),
                       np.array(acciones, np.uint8) if grabar else None,
                       np.array(recorrido, np.float32).reshape(-1, 2) if grabar else None,
                       np.concatenate(self._traza) if grabar and self._traza else None)

    def paso(self, teclas, frames: int, dibujar: bool, cod: int, stop_on_gone: bool = False):
        """Un paso del emulador con la grilla; si se está grabando, guarda la traza del jugador (con ella se dibujan
        todas las redes de una generación juntas)."""
        res = self.emu.step(teclas, frames=frames, obs=dibujar, no_draw=not dibujar, grid=True,
                            trace=self._traza is not None, stop_on_gone=stop_on_gone)
        if self._traza is not None:
            self._traza.append(compactar(res.trace, 0, cod))
        return res

    def muerto(self, frames: int) -> None:
        """Frames de muerto antes de apretar R, sin emularlos: R reinicia la sala, y son los más caros (la sangre:
        ~11 ms por frame contra ~1,4 vivo). Solo cuentan en el reloj y en la traza."""
        if self._traza:
            reg = self._traza[-1][-1:].copy()
            reg["alive"], reg["accion"] = 0, SIN_TECLAS
            self._traza.append(np.repeat(reg, frames))

    def reaparecer(self, caida: str, frames: int, rng, al_frame):
        """Espera `frames_reset` (si murió), aprieta R y espera a estar de nuevo vivo en la sala (y, con
        `espera_reintento`, unos frames al azar sin control). Devuelve (último paso, frames), o (None, frames) si se
        acabó el tiempo o no reapareció.

        Los frames de muerto no se emulan (`saltar_muerte`, ver `muerto`). En un video sí se muestran, pero desde un
        savestate al que después se vuelve: la sangre usa random() y, si no, el azar del juego quedaría distinto que
        al entrenar (lo usan jefes y enemigos de otros niveles)."""
        cfg, emu, k = self.cfg, self.emu, self.cfg.frames_por_decision
        dibujar = al_frame is not None
        espera_r = int(rng.integers(0, cfg.espera_reintento + 1)) if cfg.espera_reintento else 0
        plan = [((), cfg.frames_reset if caida == "murió" else 0), (("r",), k)]
        for teclas, n in plan:
            mostrar_muerte = not teclas and self.saltar_muerte and dibujar and n > 0
            if mostrar_muerte:
                emu.save(SLOT_MUERTE)
            while n > 0 and frames < cfg.max_frames:
                if not teclas and self.saltar_muerte:
                    self.muerto(min(k, n))
                    if dibujar:
                        res = emu.step((), frames=min(k, n), obs=True, grid=True)
                else:
                    res = self.paso(teclas, min(k, n), dibujar, TECLA_R if teclas else SIN_TECLAS)
                frames, n = frames + min(k, n), n - k
                if dibujar:
                    al_frame(res, None, None)
            if mostrar_muerte:
                emu.load(SLOT_MUERTE)
            if n > 0:   # se acabó el tiempo
                return None, frames
        for _ in range(150 // k):  # el juego pasa por rInit y vuelve a la sala del save
            if frames >= cfg.max_frames:
                return None, frames
            res = self.paso((), k, dibujar, SIN_TECLAS)
            frames += k
            if dibujar:
                al_frame(res, None, None)
            if res.grid.vivo and res.grid.room == cfg.room_entrenamiento:
                break
        else:
            return None, frames
        while espera_r > 0 and frames < cfg.max_frames:
            res = self.paso((), min(k, espera_r), dibujar, SIN_TECLAS)
            frames, espera_r = frames + min(k, espera_r), espera_r - k
            if dibujar:
                al_frame(res, None, None)
        return res, frames

    def evaluar(self, genoma: Genoma, escenarios: list[Escenario]) -> Evaluacion:
        # El primer escenario se graba (acciones y traza): con eso se puede repetir y dibujar cualquier red
        partidas = [self.jugar(genoma, e, grabar=k == 0) for k, e in enumerate(escenarios)]
        puntos = [sum(desglose(self.cfg, p).values()) for p in partidas]
        return Evaluacion(float(np.mean(puntos)), partidas)
