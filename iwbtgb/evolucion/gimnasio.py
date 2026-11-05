"""Gimnasio: salas generadas al azar dentro del juego real, para que una red practique situaciones variadas (saltos,
púas, pozos, muros, techos, plataformas) antes de enfrentar los niveles del GB, que no ve hasta la prueba.

Las salas se arman en `rStageBace`, una sala del juego casi vacía (jugador, cámara y controladores), creando con GML
las mismas instancias que usan los niveles (`block`, `spikeUp`…, `movingPlatform`, `warp`) y los tiles de ladrillo de
rStage01: la física y las colisiones son las del juego, sin diferencias de simulación. Una sala ocupa una pantalla
(25 × 19 celdas de 32 px, lo que muestra la cámara), con el jugador a la izquierda y la salida a la derecha. Con
~120 instancias en vez de las ~970 de rStage01, el emulador corre ~6 veces más rápido. La salida es una abertura en
la pared derecha, como las salidas de los niveles.

Cada sala sale de una semilla y una dificultad (0 a 1), y opcionalmente con trampas (púas que saltan, caen o se
deslizan y cerezas, con los disparadores del juego): la dificultad habilita obstáculos (muros que piden doble
salto, techos de púas, islas, plataformas móviles) y agranda pozos, escalones y filas de púas. Los obstáculos respetan
la física del jugador (salto 8,5 → ~90 px; doble salto 7 → ~150 px en total; 3 px por frame), pero no se garantiza
que toda sala tenga solución: `resolver` lo comprueba con una búsqueda (archivo de celdas con savestates, como
Go-Explore) y, si la encuentra, devuelve las acciones.

**Cambios al generador**: siempre detrás de una regla nueva con valor "apagado" (como `salida_borde`, `trampas` y
`trampas_min`), que se guarda en cada generación: cambiar el catálogo cambia el azar de todas las salas y rompería el
en vivo de las generaciones guardadas.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

from ..emulador import ACTIONS, arrancar
from ..traza import SIN_TECLAS, TRAZA_DTYPE, compactar
from .episodio import Evaluacion, Partida, desglose, plan_teclas
from .genoma import Cache

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
# Trampas (con `trampas`): las piezas del juego, un disparador invisible `triggerNN` y una púa o cereza
# `spikeTriggerNN`/`cherryTriggerNN` que sale con `direction` y `trapSpeed` cuando el jugador lo toca.
# Cada trampa de una sala usa otro número (todos los disparadores de un número se activan juntos).
TIPOS_TRAMPA = {"púa que salta": (0.2, 0.8), "púa que cae": (0.3, 0.8), "púa lateral": (0.4, 0.7)}
LARGO_TRAMPA = 8                     # columnas que puede ocupar una trampa (la púa lateral: hasta 7 + 1)
NUMEROS_TRAMPA = {
    "spikeTrigger_arriba": [1, 2, 5, 6, 15],     # sprSpikeUp
    "spikeTrigger_abajo": [7, 8, 16],             # sprSpike (apunta hacia abajo)
    "spikeTrigger_izq": [11, 12],                 # sprSpikeLeft
    "cherryTrigger": [3, 10, 17],
}


@dataclass
class Sala:
    semilla: int
    dificultad: float
    celdas: np.ndarray                                  # (FILAS, COLS) uint8
    inicio: tuple[int, int]                             # celda (col, fila) donde aparece el jugador
    meta: tuple[int, int]                               # celda de la salida (la de abajo, si es una abertura)
    moviles: list[tuple[float, float, float]] = field(default_factory=list)   # plataformas móviles (x, y, hspeed)
    obstaculos: list[str] = field(default_factory=list)
    salida: list[tuple[int, int]] = field(default_factory=list)   # celdas de la salida (vacío: solo `meta`)
    # trampas: (objeto, x, y, GML para la instancia), p. ej. ("spikeTrigger01", 448, 544, "direction = 90; ...")
    trampas: list[tuple[str, float, float, str]] = field(default_factory=list)

    @property
    def celdas_salida(self) -> list[tuple[int, int]]:
        return self.salida or [self.meta]

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
        self.trampas: list[tuple[str, float, float, str]] = []
        self.usados: set[int] = set()       # números de disparador ya usados en esta sala
        self.salida_borde = True
        self.con_trampas = False
        self.trampas_min = 0                # con trampas: al menos tantas por sala (0 = las que salgan al azar)

    def numero(self, tipo: str) -> int | None:
        libres = [n for n in NUMEROS_TRAMPA[tipo] if n not in self.usados]
        if not libres:
            return None
        n = libres[int(self.rng.integers(len(libres)))]
        self.usados.add(n)
        return n

    def disparador(self, n: int, c: int, fila: int, alto: int) -> None:
        """Disparador invisible `triggerNN` de una celda de ancho y `alto` celdas, desde (c, fila) hacia abajo."""
        self.trampas.append((f"trigger{n:02d}", c * TILE, fila * TILE,
                             f"image_yscale = {alto}; triggered = false;"))

    def pua_que_salta(self) -> bool:
        """Un pozo de púas en el que una de las del fondo sale disparada hacia arriba cuando el jugador llega al
        borde (como la del pozo de rStage01): hay que esperar a que pase y después saltar."""
        ancho = self.entero(2, 3)
        if self.libres() < ancho + 3 or self.h - 4 < 1:
            return False
        n = self.numero("spikeTrigger_arriba")
        if n is None:
            return False
        self.plano(1)
        self.disparador(n, self.col - 1, self.h - 4, 4)
        trampa = self.col + self.entero(0, ancho - 1)
        for i in range(ancho):
            c = self.col + i
            if c == trampa:
                self.trampas.append((f"spikeTrigger{n:02d}", c * TILE, (FILAS - 2) * TILE,
                                     f"direction = 90; trapSpeed = {self.entero(8, 12)};"))
            else:
                self.g[FILAS - 2, c] = PUA_ARRIBA
        self.col += ancho
        self.plano(2)
        return True

    def pua_que_cae(self) -> bool:
        """Una púa (o una cereza) colgando sobre el camino cae cuando el jugador pasa un poco antes de ella: hay que
        frenar, dejarla caer y seguir."""
        largo = self.entero(4, 6)
        if self.libres() < largo + 1 or self.h < 9:
            return False
        cereza = self.rng.random() < 0.35
        n = self.numero("cherryTrigger" if cereza else "spikeTrigger_abajo")
        if n is None:
            return False
        inicio = self.col
        self.plano(largo)
        fila = max(1, self.h - self.entero(5, 7))
        velocidad = self.entero(6, 9) if cereza else self.entero(8, 12)
        # El disparador va tantas columnas antes como camina el jugador (3 px/frame) mientras la púa baja hasta su
        # cabeza: quien sigue derecho llega justo debajo; hay que frenar o esperar a que pase
        caida = (self.h - fila - 1) * TILE - 20
        antes = int(np.clip(round(3 * caida / velocidad / TILE), 1, largo - 2))
        c = inicio + antes + self.entero(0, largo - 1 - antes)
        if cereza:   # el origen de la cereza es su centro
            self.trampas.append((f"cherryTrigger{n:02d}", c * TILE + 16, fila * TILE + 16,
                                 f"direction = 270; trapSpeed = {velocidad};"))
        else:
            self.trampas.append((f"spikeTrigger{n:02d}", c * TILE, fila * TILE,
                                 f"direction = 270; trapSpeed = {velocidad};"))
        self.disparador(n, c - antes, fila, self.h - fila)
        self.plano(1)
        return True

    def pua_lateral(self) -> bool:
        """Al final de un pasillo, una púa que se desliza por el suelo hacia el jugador cuando entra: hay que
        saltarla."""
        largo = self.entero(5, 7)
        if self.libres() < largo + 1 or self.h - 4 < 1:
            return False
        n = self.numero("spikeTrigger_izq")
        if n is None:
            return False
        inicio = self.col
        self.plano(largo)
        self.trampas.append((f"spikeTrigger{n:02d}", (inicio + largo - 1) * TILE, (self.h - 1) * TILE,
                             f"direction = 180; trapSpeed = {self.entero(4, 8)};"))
        self.disparador(n, inicio + 1, self.h - 4, 4)
        self.plano(1)
        return True

    def trampa_forzada(self) -> bool:
        """Una trampa cualquiera (según sus pesos); prueba los otros tipos si no cabe."""
        tipos = list(TIPOS_TRAMPA)
        pesos = np.array([TIPOS_TRAMPA[t][1] for t in tipos])
        metodos = {"púa que salta": self.pua_que_salta, "púa que cae": self.pua_que_cae,
                   "púa lateral": self.pua_lateral}
        for i in self.rng.choice(len(tipos), len(tipos), replace=False, p=pesos / pesos.sum()):
            if metodos[tipos[int(i)]]():
                self.obstaculos.append(tipos[int(i)])
                return True
        return False

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
                   "islas": self.islas, "móvil": self.movil, "púa que salta": self.pua_que_salta,
                   "púa que cae": self.pua_que_cae, "púa lateral": self.pua_lateral}
        # sin trampas, la misma lista de siempre: las salas viejas salen idénticas
        catalogo = {**TIPOS, **TIPOS_TRAMPA} if self.con_trampas else TIPOS
        tipos = [t for t, (dmin, _) in catalogo.items() if self.d >= dmin]
        pesos = np.array([catalogo[t][1] for t in tipos])
        # `trampas_min`: columnas al azar desde las que, si aún faltan trampas, la siguiente pieza es una (de
        # cualquier tipo, aunque la dificultad sea baja). Cada trampa ocupa hasta 8 columnas.
        minimo = self.trampas_min if self.con_trampas else 0
        forzar = []
        if minimo:
            hasta = max(self.col, COLS - 4 - LARGO_TRAMPA * minimo)
            forzar = [int(c) + LARGO_TRAMPA * i
                      for i, c in enumerate(sorted(self.rng.integers(self.col, hasta + 1, minimo)))]
        intentos = 0
        while self.libres() > 0 and intentos < 50:
            intentos += 1
            puestas = sum(t in TIPOS_TRAMPA for t in self.obstaculos)
            if puestas < minimo and self.col >= forzar[puestas]:
                if self.trampa_forzada():
                    continue
            tipo = tipos[int(self.rng.choice(len(tipos), p=pesos / pesos.sum()))]
            if metodos[tipo]():
                if tipo != "plano":
                    self.obstaculos.append(tipo)
        while self.col < COLS - 1:
            self.plano(1)
        if not self.salida_borde:   # las primeras corridas: un portal a 2 celdas de la pared
            return Sala(semilla, self.d, self.g, inicio, (COLS - 3, self.h - 1), self.moviles, self.obstaculos,
                        trampas=self.trampas)
        # Como en los niveles: una abertura de 3 celdas en la pared derecha, a la altura del suelo final
        filas = range(self.h - 3, self.h)
        self.g[list(filas), COLS - 1] = VACIO
        return Sala(semilla, self.d, self.g, inicio, (COLS - 1, self.h - 1), self.moviles, self.obstaculos,
                    [(COLS - 1, f) for f in filas], self.trampas)


def generar(semilla: int, dificultad: float, salida_borde: bool = True, trampas: bool = True,
            trampas_min: int = 0) -> Sala:
    """La sala de esa semilla y dificultad. `salida_borde`: la salida es una abertura en la pared derecha (como en
    los niveles); False, un portal a 2 celdas de la pared. `trampas`: púas que saltan, caen o se deslizan y cerezas
    (desde dificultad 0,2). `trampas_min`: con trampas, al menos tantas en cada sala, a cualquier dificultad. Con cada
    regla apagada, el azar es el de antes de que existiera: las salas no cambian."""
    minimo = trampas_min if trampas else 0
    for intento in range(20):
        # si un obstáculo largo dejó sin lugar la trampa que faltaba, otra sala con un azar derivado de la semilla
        gen = _Generador(semilla if intento == 0 else [semilla, intento], dificultad)
        gen.salida_borde, gen.con_trampas, gen.trampas_min = salida_borde, trampas, trampas_min
        sala = gen.generar(semilla)
        if sum(o in TIPOS_TRAMPA for o in sala.obstaculos) >= minimo:
            break
    return sala


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
    for objeto, x, y, codigo in sala.trampas:
        partes.append(f"with (instance_create({x},{y},{objeto})) {{ {codigo} }}")
    for mx, my in sala.celdas_salida:   # el portal solo se ve: llegar se detecta en Python (`en_meta`)
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
    cola = deque()
    for mx, my in sala.celdas_salida:
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
    traza: np.ndarray | None = None  # con `grabar`: registro por frame del jugador (`traza.TRAZA_DTYPE`)
    acciones: list[tuple[int, int]] = field(default_factory=list)   # (acción, frames)

    @property
    def llego(self) -> bool:
        return self.fin == "llegó"


class Gimnasio:
    """Un emulador en rStageBace que arma salas y hace jugar a una política en ellas. Guarda hasta `max_salas` salas
    armadas (un savestate de ~11 MB cada una, desde el slot 10): al entrenar, todas las redes de una generación juegan
    las mismas. Volver a armar una sala da lo mismo, así que el tope no cambia resultados.

    `configurar(emu)`, si se da, prepara lo que ve la red (la grilla o los rayos) y devuelve lo que la política
    necesite de eso (queda en `vista`)."""
    BASE, ARCHIVO = 1, 100               # slots de savestate (el archivo de `resolver` usa de 100 en adelante)
    PRIMERA_SALA = 10

    def __init__(self, configurar=None, max_salas: int = 8):
        self.emu = emu = arrancar("medium", "rStageBace")
        emu.exec("background_index[0] = backgroundIWBTG; background_visible[0] = true; "
                 "background_htiled[0] = true; background_vtiled[0] = true;")
        emu.track("player")
        self.vista = configurar(emu) if configurar is not None else None
        emu.step((), frames=1, obs=False, no_draw=True)
        emu.save(self.BASE)
        self.max_salas = max_salas
        self.sala: Sala | None = None
        self.SALA = self.PRIMERA_SALA
        self.dist = np.zeros((1, 1))
        self.d0 = 1.0
        self._armadas: dict[tuple, tuple] = {}

    def close(self) -> None:
        self.emu.close()

    def construir(self, sala: Sala) -> None:
        """Deja lista la sala para `jugar` (la arma, o reutiliza la que ya estaba armada)."""
        clave = (sala.semilla, round(sala.dificultad, 6), sala.celdas.tobytes(), tuple(sala.celdas_salida),
                 tuple(sala.trampas))
        if clave not in self._armadas:
            if len(self._armadas) >= self.max_salas:
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
        """La caja del jugador toca alguna celda de la salida."""
        for c, f in self.sala.celdas_salida:
            mx, my = c * TILE, f * TILE
            if x + CAJA[2] >= mx and x + CAJA[0] < mx + TILE and y + CAJA[3] >= my and y + CAJA[1] < my + TILE:
                return True
        return False

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


class PoliticaRed(Politica):
    """Una red decidiendo con lo que ve (cada línea implementa `decidir`). Con `carpeta` (la de una corrida),
    `cargar` pone la mejor red de una generación."""
    usa_vista = True

    def __init__(self, cfg, red):
        self.cfg, self.red = cfg, red
        self.soltar_salto = bool(cfg.soltar_salto)
        self.carpeta = None
        self.gen, self.fitness = -1, 0.0

    def cargar(self, gen: int = -1) -> bool:
        """Carga la mejor red (la que llega más lejos) de la generación (la última con -1); True si cambió."""
        from .almacen import red_de
        g, genoma, self.fitness = red_de(self.carpeta, gen)
        if g == self.gen:
            return False
        self.gen = g
        self.red.cargar(self.red.pesos(genoma))
        self.nombre = f"{self.carpeta.name} · gen {g}"
        return True


# -- entrenar en el gimnasio ------------------------------------------------------------------------------------

class JugadorGimnasio:
    """Como `episodio.Jugador`, pero cada escenario es una sala del gimnasio (`Escenario.semilla` y `.dificultad`):
    devuelve las mismas `Partida`/`Evaluacion`. Sin reintentos: la partida termina al llegar, morir, en
    `max_segundos` o tras `paciencia` sin avanzar."""

    def __init__(self, cfg, gim: Gimnasio, politica: PoliticaRed):
        self.cfg, self.gim, self.politica = cfg, gim, politica
        self.cache = Cache(64)

    def close(self) -> None:
        self.gim.close()

    def evaluar(self, genoma, escenarios) -> Evaluacion:
        cfg, red = self.cfg, self.politica.red
        red.cargar(red.pesos(genoma, self.cache))
        partidas = []
        for k, e in enumerate(escenarios):
            self.gim.construir(generar(e.semilla, e.dificultad, bool(cfg.gim_salida_borde), bool(cfg.gim_trampas),
                                       cfg.gim_trampas_min))
            it = self.gim.jugar(self.politica, cfg.max_frames, paciencia=int(cfg.paciencia * 50), grabar=True)
            final = tuple(float(v) for v in it.recorrido[-1]) if len(it.recorrido) else (0.0, 0.0)
            # la traza de cada sala: para ver a toda la población junta en el en vivo del entrenamiento
            partidas.append(Partida(it.avance, it.frame_avance, it.frames, it.fin, final,
                                    acciones=np.array([a for a, _ in it.acciones], np.uint8) if k == 0 else None,
                                    traza=it.traza if it.traza is not None else np.zeros(0, TRAZA_DTYPE)))
        puntos = [sum(desglose(cfg, p).values()) for p in partidas]
        return Evaluacion(float(np.mean(puntos)), partidas)


# -- ¿tiene solución? -------------------------------------------------------------------------------------

def resolver(gim: Gimnasio, presupuesto: int = 600, semilla: int = 0,
             guardados: int = 32) -> list[tuple[int, int]] | None:
    """Busca cómo llegar a la salida de la sala construida: un archivo de celdas de 32 px; se elige una celda (mucho
    más a menudo las cercanas a la salida, y las poco visitadas), se vuelve a ella y se prueban unas acciones al
    azar; cada celda nueva entra al archivo. Devuelve las acciones desde el comienzo de la sala (para
    `PoliticaGrabada`), o None si no la encontró en `presupuesto` iteraciones.

    Solo las `guardados` celdas usadas más recientemente tienen savestate (~11 MB cada uno; una sala puede dejar más
    de 100 celdas): a las demás se vuelve desde su ancestro más cercano que lo tenga, repitiendo el resto de sus
    acciones, que da el mismo estado."""
    emu, rng = gim.emu, np.random.default_rng(semilla)
    slots: OrderedDict[tuple, int] = OrderedDict()               # celda -> slot, de la menos a la más reciente
    libres = list(range(gim.ARCHIVO + guardados - 1, gim.ARCHIVO - 1, -1))

    def guardar(celda: tuple) -> None:
        slot = libres.pop() if libres else slots.popitem(last=False)[1]
        emu.save(slot)
        slots[celda] = slot

    def volver(celda: tuple) -> None:
        ancestro = celda
        while ancestro is not None and ancestro not in slots:
            ancestro = archivo[ancestro][3]
        if ancestro is None:                                     # ni la raíz: desde el comienzo de la sala
            emu.load(gim.SALA)
            gim.paso((), 1, False, False)
            hechas = 0
        else:
            slots.move_to_end(ancestro)
            emu.load(slots[ancestro])
            if ancestro == celda:
                return
            hechas = len(archivo[ancestro][0])
        for a, n in archivo[celda][0][hechas:]:
            gim.paso(ACTIONS[a], n, False, False)
        guardar(celda)

    emu.load(gim.SALA)
    res = gim.paso((), 1, False, False)
    r = res.trace[-1]
    raiz = (int(r["x"]) // 32, int(r["y"]) // 32)
    archivo = {raiz: [[], 0, gim.avance_en(r["x"], r["y"]), None]}   # celda -> [acciones, visitas, avance, padre]
    guardar(raiz)
    for _ in range(presupuesto):
        celdas = list(archivo.items())
        peso = np.array([np.exp(6 * c[2]) / np.sqrt(1 + c[1]) for _, c in celdas])
        celda, elegida = celdas[int(rng.choice(len(celdas), p=peso / peso.sum()))]
        elegida[1] += 1
        volver(celda)
        acciones = list(elegida[0])
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
                archivo[clave] = [list(acciones), 0, gim.avance_en(v["x"], v["y"]), celda]
                guardar(clave)
    return None
