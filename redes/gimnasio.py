"""Gimnasio: salas generadas al azar dentro del juego real, para que una red practique situaciones variadas (saltos,
púas, pozos, muros, techos, plataformas) antes de enfrentar los niveles del GB, que no ve hasta la prueba.

Las salas se arman en `rStageBace`, una sala del juego casi vacía (jugador, cámara y controladores), creando con GML
las mismas instancias que usan los niveles (`block`, `spikeUp`…, `movingPlatform`, `warp`) y los tiles de ladrillo de
rStage01: la física y las colisiones son las del juego, sin diferencias de simulación. Una sala ocupa una pantalla
(25 × 19 celdas de 32 px, lo que muestra la cámara), con el jugador a la izquierda y la salida a la derecha. Con
~120 instancias en vez de las ~970 de rStage01, el emulador corre ~6 veces más rápido.

Cada sala sale de una semilla y una dificultad (0 a 1): la dificultad habilita obstáculos (muros que piden doble
salto, techos de púas, islas, plataformas móviles) y agranda pozos, escalones y filas de púas. Los obstáculos respetan
la física del jugador (salto 8,5 → ~90 px; doble salto 7 → ~150 px en total; 3 px por frame), pero no se garantiza
que toda sala tenga solución: `resolver` lo comprueba con una búsqueda (archivo de celdas con savestates, como
Go-Explore) y, si la encuentra, devuelve las acciones.

    uv run python -m redes.gimnasio_ver ...   (muestras, en vivo y videos)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from iwbtgb_rl import ACTIONS, IWannaBeTheGBEnv

from .vista import ConfigVista, Vista, extras

TILE = 32
COLS, FILAS = 25, 19                # una pantalla: la cámara del juego avanza de a 800×608
VACIO, BLOQUE, PUA_ARRIBA, PUA_ABAJO, PUA_IZQ, PUA_DER = range(6)
PUAS = (PUA_ARRIBA, PUA_ABAJO, PUA_IZQ, PUA_DER)
OBJETO = {BLOQUE: "block", PUA_ARRIBA: "spikeUp", PUA_ABAJO: "spikeDown", PUA_IZQ: "spikeLeft",
          PUA_DER: "spikeRight"}
SUELO_MIN, SUELO_MAX = 7, 17        # fila del borde superior del suelo
PROFUNDIDAD_TILES = 1000001
# Caja de colisión del jugador respecto de (x, y) (medida en el juego)
CAJA = (-5, -12.2, 5, 7.8)

# Obstáculos: (dificultad desde la que aparecen, peso)
TIPOS = {
    "plano": (0.0, 0.6), "escalón": (0.0, 1.5), "púas": (0.0, 1.5), "pozo": (0.0, 1.5),
    "muro": (0.3, 1.0), "techo": (0.4, 0.8), "islas": (0.5, 1.0), "móvil": (0.6, 0.8),
}


@dataclass
class Sala:
    semilla: int
    dificultad: float
    celdas: np.ndarray                                  # (FILAS, COLS) uint8
    inicio: tuple[int, int]                             # celda (col, fila) donde aparece el jugador
    meta: tuple[int, int]                               # celda de la salida
    moviles: list[tuple[float, float, float]] = field(default_factory=list)   # plataformas móviles (x, y, hspeed)
    obstaculos: list[str] = field(default_factory=list)

    @property
    def nombre(self) -> str:
        return f"semilla {self.semilla} · dificultad {self.dificultad:.2f}"


# -- generador --------------------------------------------------------------------------------------------

class _Generador:
    def __init__(self, semilla: int, dificultad: float):
        self.rng = np.random.default_rng(semilla)
        self.d = float(np.clip(dificultad, 0.0, 1.0))
        g = self.g = np.zeros((FILAS, COLS), np.uint8)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = BLOQUE
        self.col = 1
        self.h = int(self.rng.integers(13, SUELO_MAX + 1))
        self.moviles: list[tuple[float, float, float]] = []
        self.obstaculos: list[str] = []

    def suelo(self, c: int, h: int) -> None:
        self.g[h:FILAS - 1, c] = BLOQUE

    def plano(self, n: int) -> None:
        for _ in range(n):
            self.suelo(self.col, self.h)
            self.col += 1

    def entero(self, a: int, b: int) -> int:
        """Entero en [a, b]."""
        return int(self.rng.integers(a, b + 1))

    def libres(self) -> int:
        return COLS - 4 - self.col      # las 3 últimas columnas interiores son la llegada

    # Cada obstáculo devuelve False si no cabe (no toca nada)
    def escalon(self) -> bool:
        if self.libres() < 2:
            return False
        if self.rng.random() < 0.6:
            alto = self.entero(1, 2 if self.d < 0.3 else (3 if self.d < 0.6 else 4))
            h = max(self.h - alto, SUELO_MIN)
        else:
            h = min(self.h + self.entero(1, 4), SUELO_MAX)
        if h == self.h:
            return False
        self.h = h
        self.plano(2)
        return True

    def puas(self) -> bool:
        n = self.entero(1, 1 + round(2 * self.d))
        if self.libres() < n + 2:
            return False
        self.plano(1)
        for _ in range(n):
            self.suelo(self.col, self.h)
            self.g[self.h - 1, self.col] = PUA_ARRIBA
            self.col += 1
        self.plano(1)
        return True

    def pozo(self) -> bool:
        ancho = self.entero(1, 2 + round(3 * self.d))
        if self.libres() < ancho + 3:
            return False
        self.plano(1)
        for _ in range(ancho):
            self.g[FILAS - 2, self.col] = PUA_ARRIBA
            self.col += 1
        bajar = self.entero(-1 if self.d > 0.3 and ancho < 4 else 0, 2)
        self.h = int(np.clip(self.h + bajar, SUELO_MIN, SUELO_MAX))
        self.plano(2)
        return True

    def muro(self) -> bool:
        ancho = self.entero(1, 2)
        alto = 3 if self.d < 0.6 else self.entero(3, 4)
        if self.libres() < ancho + 2 or self.h - alto < 3:
            return False
        self.plano(1)
        for _ in range(ancho):
            self.suelo(self.col, self.h - alto)
            self.col += 1
        self.plano(1)
        return True

    def techo(self) -> bool:
        """Púas colgando sobre un tramo: con 4 celdas de alto pasa un salto simple pero no el doble."""
        n = self.entero(3, 5)
        if self.libres() < n + 1 or self.h < 11:
            return False
        con_pua = self.d >= 0.7 and n >= 4
        for i in range(n):
            self.suelo(self.col, self.h)
            self.g[self.h - 5, self.col] = PUA_ABAJO
            self.g[1:self.h - 5, self.col] = BLOQUE
            if con_pua and i == n // 2:
                self.g[self.h - 1, self.col] = PUA_ARRIBA
            self.col += 1
        self.plano(1)
        return True

    def islas(self) -> bool:
        """Un pozo de púas ancho con bloques sueltos en el medio (saltos de a lo más 3 celdas)."""
        ancho = self.entero(5, 8)
        if self.libres() < ancho + 3:
            return False
        self.plano(1)
        inicio, fin = self.col, self.col + ancho
        c = inicio
        while c < fin:
            self.g[FILAS - 2, c] = PUA_ARRIBA
            c += 1
        c = inicio + self.entero(1, 2)
        while c < fin - 1:
            fila = int(np.clip(self.h + self.entero(-1, 1), SUELO_MIN, SUELO_MAX - 1))
            self.g[fila, c] = BLOQUE
            c += self.entero(3, 4)
        self.col = fin
        self.plano(2)
        return True

    def movil(self) -> bool:
        """Un pozo de púas con una plataforma móvil que va y viene entre los bordes."""
        ancho = self.entero(5, 7)
        if self.libres() < ancho + 3 or self.h > SUELO_MAX - 1:
            return False
        self.plano(1)
        for i in range(ancho):
            self.g[FILAS - 2, self.col + i] = PUA_ARRIBA
        velocidad = float(self.rng.choice([1.5, 2.0, 2.5]))
        self.moviles.append((self.col * TILE + 2, self.h * TILE, velocidad))
        self.col += ancho
        self.plano(2)
        return True

    def generar(self, semilla: int) -> Sala:
        self.plano(3)
        inicio = (2, self.h - 1)
        metodos = {"plano": lambda: (self.plano(self.entero(1, 2)), True)[1], "escalón": self.escalon,
                   "púas": self.puas, "pozo": self.pozo, "muro": self.muro, "techo": self.techo,
                   "islas": self.islas, "móvil": self.movil}
        tipos = [t for t, (dmin, _) in TIPOS.items() if self.d >= dmin]
        pesos = np.array([TIPOS[t][1] for t in tipos])
        intentos = 0
        while self.libres() > 0 and intentos < 50:
            intentos += 1
            tipo = tipos[int(self.rng.choice(len(tipos), p=pesos / pesos.sum()))]
            if metodos[tipo]():
                if tipo != "plano":
                    self.obstaculos.append(tipo)
        while self.col < COLS - 1:
            self.plano(1)
        return Sala(semilla, self.d, self.g, inicio, (COLS - 3, self.h - 1), self.moviles, self.obstaculos)


def generar(semilla: int, dificultad: float) -> Sala:
    return _Generador(semilla, dificultad).generar(semilla)


# -- en el emulador ---------------------------------------------------------------------------------------

def gml_sala(sala: Sala) -> str:
    """Código GML que arma la sala en rStageBace (instancias + tiles de ladrillo) y ubica al jugador."""
    g = sala.celdas
    partes = []
    for f in range(FILAS):
        for c in range(COLS):
            t = int(g[f, c])
            if t == VACIO:
                continue
            x, y = c * TILE, f * TILE
            partes.append(f"instance_create({x},{y},{OBJETO[t]});")
            if t == BLOQUE:
                # el ladrillo con borde (0, 32) donde el bloque tiene aire encima; el liso (32, 32) dentro
                izq = 0 if f == 0 or g[f - 1, c] != BLOQUE else 32
                partes.append(f"tile_add(0,{izq},32,32,32,{x},{y},{PROFUNDIDAD_TILES});")
    for x, y, v in sala.moviles:
        partes.append(f"with (instance_create({x},{y},movingPlatform)) hspeed = {v};")
    mx, my = sala.meta
    partes.append(f"with (instance_create({mx * TILE},{my * TILE},warp)) onCollide = false;")
    ix, iy = sala.inicio
    partes.append(f"with (player) {{ x = {ix * TILE + 17}; y = {iy * TILE + 23}; hspeed = 0; vspeed = 0; "
                  f"djump = true; }}")
    return " ".join(partes)


def distancias(sala: Sala) -> np.ndarray:
    """Pasos (celdas de 16 px, 4 vecinos) desde cada celda libre hasta la salida; inf donde no se llega."""
    libre = ~np.isin(sala.celdas, (BLOQUE, *PUAS))
    libre = np.repeat(np.repeat(libre, 2, axis=0), 2, axis=1)
    h, w = libre.shape
    d = np.full((h, w), np.inf)
    mx, my = sala.meta
    cola = deque()
    for j in (2 * my, 2 * my + 1):
        for i in (2 * mx, 2 * mx + 1):
            d[j, i] = 0
            cola.append((j, i))
    while cola:
        j, i = cola.popleft()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = j + dj, i + di
            if 0 <= a < h and 0 <= b < w and libre[a, b] and d[a, b] == np.inf:
                d[a, b] = d[j, i] + 1
                cola.append((a, b))
    return d


@dataclass
class Intento:
    fin: str                     # "llegó", "murió" o "tiempo"
    frames: int
    avance: float                # mejor fracción del camino a la salida (1 = llegó)
    recorrido: np.ndarray        # (x, y) por frame, vivo
    frame_avance: int = 0        # frame en que alcanzó su mejor avance
    traza: np.ndarray | None = None  # con `grabar`: registro por frame del jugador (`ga.nucleo.TRAZA_DTYPE`)
    acciones: list[tuple[int, int]] = field(default_factory=list)   # (acción, frames)

    @property
    def llego(self) -> bool:
        return self.fin == "llegó"


class Gimnasio:
    """Un emulador en rStageBace que arma salas y hace jugar a una política en ellas. Guarda hasta 64 salas armadas
    (un savestate cada una, slots 10–73): al entrenar, todas las redes de una generación juegan las mismas."""
    BASE, ARCHIVO = 1, 100               # slots de savestate (el archivo de `resolver` usa de 100 en adelante)
    PRIMERA_SALA, MAX_SALAS = 10, 64

    def __init__(self, vista: ConfigVista | None = None):
        self.env = IWannaBeTheGBEnv(difficulty="medium", start_room="rStageBace", obs_type="state")
        self.emu = emu = self.env.emu
        emu.exec("background_index[0] = backgroundIWBTG; background_visible[0] = true; "
                 "background_htiled[0] = true; background_vtiled[0] = true;")
        self.vista = Vista(emu, vista)
        emu.step((), frames=1, obs=False, no_draw=True)
        emu.save(self.BASE)
        self.sala: Sala | None = None
        self.SALA = self.PRIMERA_SALA
        self.dist = np.zeros((1, 1))
        self.d0 = 1.0
        self._armadas: dict[tuple, tuple] = {}

    def close(self) -> None:
        self.env.close()

    def construir(self, sala: Sala) -> None:
        """Deja lista la sala para `jugar` (la arma, o reutiliza la que ya estaba armada)."""
        clave = (sala.semilla, round(sala.dificultad, 6))
        if clave not in self._armadas:
            if len(self._armadas) >= self.MAX_SALAS:
                self._armadas.clear()
            slot = self.PRIMERA_SALA + len(self._armadas)
            emu = self.emu
            emu.load(self.BASE)
            emu.exec(gml_sala(sala))
            emu.step((), frames=1, obs=False, no_draw=True)
            emu.save(slot)
            dist = distancias(sala)
            ix, iy = sala.inicio
            self._armadas[clave] = (slot, sala, dist, max(float(dist[2 * iy + 1, 2 * ix + 1]), 1.0))
        self.SALA, self.sala, self.dist, self.d0 = self._armadas[clave]

    def avance_en(self, x: float, y: float) -> float:
        h, w = self.dist.shape
        d = self.dist[int(np.clip(y // 16, 0, h - 1)), int(np.clip(x // 16, 0, w - 1))]
        return 0.0 if not np.isfinite(d) else float(np.clip(1.0 - d / self.d0, 0.0, 1.0))

    def en_meta(self, x: float, y: float) -> bool:
        mx, my = self.sala.meta[0] * TILE, self.sala.meta[1] * TILE
        return (x + CAJA[2] >= mx and x + CAJA[0] < mx + TILE and y + CAJA[3] >= my and y + CAJA[1] < my + TILE)

    def paso(self, teclas: tuple[str, ...], frames: int, dibujar: bool, grilla: bool):
        return self.emu.step(teclas, frames=frames, obs=dibujar, no_draw=not dibujar, grid=grilla, trace=True,
                             stop_on_gone=True)

    def jugar(self, politica: "Politica", max_frames: int = 1000, al_frame=None, paciencia: int = 0,
              cada: int = 2, grabar: bool = False) -> Intento:
        """Un intento en la sala construida, hasta llegar, morir, `max_frames` o `paciencia` frames sin mejorar el
        avance (0 = sin límite; termina "estancado").

        Para videos, `al_frame(res, accion, salidas)` recibe un cuadro cada `cada` frames de juego (2 = tiempo real
        a 25 cuadros/s; 2·N = N veces más rápido), sea cual sea el ritmo de la política: las decisiones largas se
        emulan en trozos con las mismas teclas (el emulador las mantiene apretadas: da lo mismo que un paso largo) y
        el emulador solo dibuja en los pasos que se muestran. Con `grabar`, el intento trae la traza por frame (para
        dibujar después a toda la población junta)."""
        from ga.nucleo import SIN_TECLAS, compactar
        dibujar = al_frame is not None
        trozos_traza: list[np.ndarray] = []
        self.emu.load(self.SALA)
        politica.reiniciar()
        estado = {"frames": 0, "mejor": 0.0, "fin": None, "frame_mejor": 0}
        recorrido, acciones = [], []

        def revisar(res, cod: int) -> None:
            if grabar:
                trozos_traza.append(compactar(res.trace, 0, cod))
            vivos = res.trace[res.trace["alive"] == 1]
            for r in vivos:
                recorrido.append((float(r["x"]), float(r["y"])))
                a = self.avance_en(r["x"], r["y"])
                if a > estado["mejor"] + 1e-3:
                    estado["mejor"], estado["frame_mejor"] = a, estado["frames"]
                if self.en_meta(r["x"], r["y"]):
                    estado["fin"], estado["mejor"], estado["frame_mejor"] = "llegó", 1.0, estado["frames"]
                    return
            if res.ended or len(vivos) < len(res.trace) or not self.emu_vivo(res):
                estado["fin"] = "murió"
            elif estado["frames"] >= max_frames:
                estado["fin"] = "tiempo"
            elif paciencia and estado["frames"] - estado["frame_mejor"] >= paciencia:
                estado["fin"] = "estancado"

        res = self.paso((), 1, dibujar, politica.usa_vista)
        estado["frames"] = 1
        revisar(res, SIN_TECLAS)
        if dibujar:
            al_frame(res, None, None)
        pendientes = 0                  # frames de juego desde el último cuadro mostrado
        sostenido = False               # la tecla de salto quedó apretada en el paso anterior
        while estado["fin"] is None:
            accion, n = politica.decidir(res.grid)
            if accion is None:          # una solución grabada que se terminó
                estado["fin"] = "tiempo"
                break
            acciones.append((accion, n))
            plan = plan_teclas(ACTIONS[accion], n, sostenido, politica.soltar_salto, res.grid)
            sostenido = "shift" in plan[-1][0]
            for teclas, m in plan:
                trozos = [m] if not dibujar else [min(cada, m - i) for i in range(0, m, cada)]
                for t in trozos:
                    toca = dibujar and pendientes + t >= cada
                    res = self.paso(teclas, t, toca, politica.usa_vista)
                    estado["frames"] += t
                    pendientes = pendientes + t - (cada if toca else 0)
                    revisar(res, ACTIONS.index(teclas))
                    if toca:
                        al_frame(res, accion, politica.salidas)
                    if estado["fin"]:
                        break
                if estado["fin"]:
                    break
        return Intento(estado["fin"], estado["frames"], estado["mejor"],
                       np.array(recorrido, np.float32).reshape(-1, 2), frame_avance=estado["frame_mejor"],
                       acciones=acciones, traza=np.concatenate(trozos_traza) if grabar else None)

    @staticmethod
    def emu_vivo(res) -> bool:
        return res.grid.vivo if res.grid is not None else bool(len(res.trace) and res.trace["alive"][-1])


def plan_teclas(teclas: tuple[str, ...], frames: int, sostenido: bool, soltar: bool, g) -> list[tuple[tuple, int]]:
    """Cómo emular una decisión: [(teclas, frames), ...]. En el juego el salto sale al *apretar* la tecla: si la red
    sigue pidiendo salto con la tecla ya apretada y el jugador no está subiendo (en el suelo, o cayendo con el doble
    salto disponible), con `soltar` el primer frame va sin salto para que el siguiente lo apriete de nuevo. Mientras
    sube se mantiene (soltar antes corta el salto)."""
    if soltar and sostenido and "shift" in teclas and g is not None and g.vspeed >= 0:
        sin = tuple(t for t in teclas if t != "shift")
        return [(sin, 1)] + ([(teclas, frames - 1)] if frames > 1 else [])
    return [(teclas, frames)]


# -- políticas --------------------------------------------------------------------------------------------

class Politica:
    nombre = ""
    usa_vista = False
    soltar_salto = False
    salidas: np.ndarray | None = None

    def reiniciar(self) -> None:
        pass

    def decidir(self, g) -> tuple[int | None, int]:
        raise NotImplementedError


class PoliticaAzar(Politica):
    """Mantiene una acción al azar unos frames y a veces cambia (más vistoso que cambiar en cada frame)."""
    nombre = "acciones al azar"

    def __init__(self, semilla: int = 0):
        self.rng = np.random.default_rng(semilla)
        self.accion = 0

    def decidir(self, g):
        if self.rng.random() < 0.25:
            self.accion = int(self.rng.integers(len(ACTIONS)))
        return self.accion, 2


class PoliticaGrabada(Politica):
    """Repite una lista de (acción, frames), p. ej. la solución que encontró `resolver`."""
    nombre = "solución de la búsqueda"

    def __init__(self, acciones: list[tuple[int, int]]):
        self.acciones = acciones
        self.i = 0

    def reiniciar(self) -> None:
        self.i = 0

    def decidir(self, g):
        if self.i >= len(self.acciones):
            return None, 0
        a, n = self.acciones[self.i]
        self.i += 1
        return a, n


class PoliticaCNN(Politica):
    """Una red de `redes` (con la configuración de su corrida) decidiendo con su vista."""
    usa_vista = True

    def __init__(self, cfg):
        from .episodio import red_para
        self.cfg = cfg
        self.red = red_para(cfg)
        self.soltar_salto = bool(cfg.soltar_salto)
        self.vista: Vista | None = None
        self.cuadros: list[np.ndarray] = []

    @property
    def config_vista(self) -> ConfigVista:
        return self.cfg.config_vista

    def reiniciar(self) -> None:
        self.cuadros = []

    def decidir(self, g):
        parches = self.red.parches(self.vista.completar(g)).astype(np.float32) / 255.0
        self.cuadros = (self.cuadros + [parches])[-self.cfg.cuadros:]
        while len(self.cuadros) < self.cfg.cuadros:
            self.cuadros.insert(0, parches)
        self.salidas = self.red.salidas_parches(np.concatenate(self.cuadros, axis=1), extras(g))
        return int(np.argmax(self.salidas)), self.cfg.frames_por_decision


class PoliticaRed(PoliticaCNN):
    """La mejor red de una corrida (la que llega más lejos), jugando con su vista."""

    def __init__(self, corrida: str, gen: int = -1):
        from . import almacen
        self.corrida = corrida
        self.carpeta = almacen.dir_corrida(corrida)
        super().__init__(almacen.cargar_config(self.carpeta))
        self.gen = -1
        self.cargar(gen)

    def cargar(self, gen: int = -1) -> bool:
        """Carga la red de la generación (la última guardada con -1); True si cambió."""
        from .visualizar import red_de
        g, genoma, self.fitness = red_de(self.carpeta, gen)
        if g == self.gen:
            return False
        self.gen = g
        self.red.cargar(self.red.pesos(genoma))
        self.nombre = f"{self.corrida} · gen {g}"
        return True


# -- entrenar en el gimnasio (lo usan los workers de `redes.entrenar` con entorno = "gimnasio") ---------------

class JugadorGimnasio:
    """Como `episodio.Jugador`, pero cada escenario es una sala del gimnasio (`Escenario.semilla` y `.dificultad`):
    devuelve las mismas `Partida`/`Evaluacion`, así el resto del entrenamiento no cambia. Sin reintentos: la partida
    termina al llegar, morir, en `max_segundos` o tras `paciencia` sin avanzar."""

    def __init__(self, cfg):
        from .modelo import Cache
        self.cfg = cfg
        self.politica = PoliticaCNN(cfg)
        self.gim = Gimnasio(cfg.config_vista)
        self.politica.vista = self.gim.vista
        self.cache = Cache(64)

    def close(self) -> None:
        self.gim.close()

    def evaluar(self, genoma, escenarios):
        from ga.nucleo import TRAZA_DTYPE
        from .episodio import Evaluacion, Partida, desglose
        cfg, red = self.cfg, self.politica.red
        red.cargar(red.pesos(genoma, self.cache))
        partidas = []
        for k, e in enumerate(escenarios):
            self.gim.construir(generar(e.semilla, e.dificultad))
            it = self.gim.jugar(self.politica, cfg.max_frames, paciencia=int(cfg.paciencia * 50), grabar=True)
            final = tuple(float(v) for v in it.recorrido[-1]) if len(it.recorrido) else (0.0, 0.0)
            # la traza de cada sala (para ver a toda la población junta en el en vivo del entrenamiento)
            partidas.append(Partida(it.avance, it.frame_avance, it.frames, it.fin, final,
                                    acciones=np.array([a for a, _ in it.acciones], np.uint8) if k == 0 else None,
                                    traza=it.traza if it.traza is not None else np.zeros(0, TRAZA_DTYPE)))
        puntos = [sum(desglose(cfg, p).values()) for p in partidas]
        return Evaluacion(float(np.mean(puntos)), partidas)


# -- ¿tiene solución? -------------------------------------------------------------------------------------

def resolver(gim: Gimnasio, presupuesto: int = 600, semilla: int = 0) -> list[tuple[int, int]] | None:
    """Busca cómo llegar a la salida de la sala construida: un archivo de celdas de 32 px con un savestate cada una;
    se elige una celda (mucho más a menudo las cercanas a la salida, y las poco visitadas), se carga y se prueban
    unas acciones al azar; cada celda nueva entra al archivo. Cargar un savestate cuesta ~17 ms: conviene probar
    varias acciones seguidas por cada carga. Devuelve las acciones desde el comienzo de la sala (para
    `PoliticaGrabada`), o None si no la encontró en `presupuesto` iteraciones."""
    emu, rng = gim.emu, np.random.default_rng(semilla)
    emu.load(gim.SALA)
    res = gim.paso((), 1, False, False)
    r = res.trace[-1]
    # celda -> [slot, acciones, visitas, avance]
    archivo = {(int(r["x"]) // 32, int(r["y"]) // 32): [gim.ARCHIVO, [], 0, gim.avance_en(r["x"], r["y"])]}
    emu.save(gim.ARCHIVO)
    for _ in range(presupuesto):
        celdas = list(archivo.values())
        peso = np.array([np.exp(6 * c[3]) / np.sqrt(1 + c[2]) for c in celdas])
        elegida = celdas[int(rng.choice(len(celdas), p=peso / peso.sum()))]
        elegida[2] += 1
        emu.load(elegida[0])
        acciones = list(elegida[1])
        for _ in range(int(rng.integers(2, 13))):
            a, n = int(rng.integers(len(ACTIONS))), int(rng.choice([2, 4, 6, 8, 12, 16]))
            res = gim.paso(ACTIONS[a], n, False, False)
            acciones.append((a, n))
            vivos = res.trace[res.trace["alive"] == 1]
            if any(gim.en_meta(v["x"], v["y"]) for v in vivos):
                return acciones
            if res.ended or len(vivos) < len(res.trace) or not len(vivos):
                break
            v = vivos[-1]
            clave = (int(v["x"]) // 32, int(v["y"]) // 32)
            if clave not in archivo:
                slot = gim.ARCHIVO + len(archivo)
                emu.save(slot)
                archivo[clave] = [slot, list(acciones), 0, gim.avance_en(v["x"], v["y"])]
    return None
