"""Los cuadros de los videos y en vivo de las redes: el juego grande a la izquierda y un panel a la derecha.

- `PantallaNivel`: una red en un nivel (tiempo, avance, puntaje en curso con su desglose, generación, gráfico de
  fitness o la red por dentro, preferencia por acción).
- `PantallaGimnasio`: una política en salas del gimnasio (sala, obstáculos, historial de salas superadas).
- `PantallaPoblacion`: toda una generación en una sala, la mejor emulada y las demás como fantasmas de sus trazas.

Lo propio de cada línea (los rayos dibujados encima del juego y su leyenda) llega como `encima(img, res, vista)` y
`leyenda(draw, x, y)`. Todo va a 25 cuadros/s, uno cada 2·velocidad frames de juego (`Ritmo`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..emulador import ACTIONS
from ..ga import recursos
from ..ga.visualizar import Escena
from ..medios import (AQUA, AZUL, DORADO, FONDO, GRIS, H, NARANJA, ROJO, TEXTO, TEXTO_2, W, fuente, mini_grafico)
from . import almacen
from .episodio import desglose

FPS_VIDEO = 25
CAJA_PANEL = (1340, 560, W - 24, 846)   # el gráfico de fitness o la red por dentro, a la derecha del juego
JUEGO_X, JUEGO_Y, ESCALA = 24, 80, 1.6  # el juego (800×600) se muestra a 1280×960 desde (24, 80)
FINES = {"llegó": "¡LLEGÓ!", "murió": "MURIÓ", "salió": "SALIÓ DE LA SALA", "estancado": "SE ESTANCÓ",
         "tiempo": "SE ACABÓ EL TIEMPO", "sin solución": "SIN SOLUCIÓN"}

# La red por dentro (`--red`): naranja = peso positivo, azul = negativo; de cada par de capas, las más fuertes
POSITIVO = (237, 140, 40)
NEGATIVO = (70, 140, 235)
APAGADO = (58, 58, 55)
ENCENDIDO = (255, 226, 140)
MAX_LINEAS = 160                   # por par de capas
MAX_OCULTAS = 32                   # con más neuronas en una capa se muestran las que más pesan hacia la siguiente


def mezcla(a, b, t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0, 1))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def lineas_de_pesos(caja, pos: dict, capas) -> Image.Image:
    """Fondo de la red por dentro: una línea por peso entre nodos (`capas` = [(origen, destino, pesos)]), las
    `MAX_LINEAS` más fuertes de cada par, más opacas cuanto más fuertes."""
    x0, y0, x1, y1 = caja
    capa = Image.new("RGBA", (x1 - x0, y1 - y0), FONDO + (255,))
    lineas = Image.new("RGBA", capa.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lineas)
    for origen, destino, w in capas:
        if not w.size:
            continue
        m = float(np.abs(w).max()) or 1.0
        orden = np.argsort(-np.abs(w), axis=None)[:MAX_LINEAS]
        for idx in orden[::-1]:                       # las fuertes encima
            i, j = np.unravel_index(idx, w.shape)
            v = float(w[i, j]) / m
            (ax, ay), (bx, by) = pos[origen][i], pos[destino][j]
            color = (POSITIVO if v > 0 else NEGATIVO) + (int(25 + 200 * abs(v) ** 1.5),)
            d.line([(ax - x0, ay - y0), (bx - x0, by - y0)], fill=color, width=1)
    return Image.alpha_composite(capa, lineas).convert("RGB")


def nombre_accion(teclas: tuple[str, ...]) -> str:
    """Etiqueta corta: ← →, ↑ = salto, Z = disparo."""
    return "".join({"left": "←", "right": "→", "shift": "↑", "z": "Z"}[t] for t in teclas) or "nada"


ETIQUETAS = [nombre_accion(t) for t in ACTIONS]


def grafico_fitness(d: ImageDraw.ImageDraw, caja, metricas: list[dict], gen: int) -> None:
    """Fitness máximo y medio por generación (el de los videos del GA), con su título, hasta la generación `gen`."""
    x0, y0, x1, y1 = caja
    f_med, f_chica = fuente(24), fuente(18)
    d.text((x0, y0), "Fitness por generación", fill=TEXTO_2, font=f_med)
    lx = x0 + 280
    for color, nombre in ((AZUL, "máx"), (NARANJA, "media")):
        d.line((lx, y0 + 14, lx + 20, y0 + 14), fill=color, width=4)
        d.text((lx + 26, y0 + 14), nombre, fill=TEXTO_2, font=f_chica, anchor="lm")
        lx += 26 + d.textlength(nombre, font=f_chica) + 24
    if metricas:
        mini_grafico(d, (x0 + 50, y0 + 46, x1 - 16, y1 - 26), metricas, gen if gen >= 0 else 10**9, f_chica)


def grafico_salas(d: ImageDraw.ImageDraw, caja, metricas: list[dict], hasta_gen: int, f_chica) -> None:
    """% de salas superadas por generación: el promedio de la población (verde) y la mejor red (dorado)."""
    x0, y0, x1, y1 = caja
    filas = [m for m in metricas if m["gen"] <= hasta_gen and "salas_pct" in m]
    d.text((x0 - 50, y0 - 34), "Salas superadas por generación (%)", fill=TEXTO_2, font=f_chica)
    if not filas:
        return
    gmax = max(m["gen"] for m in metricas) or 1
    for k in range(5):
        yy = y1 - (y1 - y0) * k / 4
        d.line((x0, yy, x1, yy), fill=(45, 45, 43), width=1)
        d.text((x0 - 8, yy), f"{25 * k}", fill=GRIS, font=f_chica, anchor="rm")
    n_salas = max(1, max(m.get("salas_mejor", 0) for m in metricas))

    def punto(m, v):
        return x0 + (x1 - x0) * m["gen"] / gmax, y1 - (y1 - y0) * min(v, 100) / 100

    for color, valor in ((AQUA, lambda m: m["salas_pct"]), (DORADO, lambda m: 100 * m["salas_mejor"] / n_salas)):
        p = [punto(m, valor(m)) for m in filas]
        if len(p) > 1:
            d.line(p, fill=color, width=3)
        d.ellipse((p[-1][0] - 5, p[-1][1] - 5, p[-1][0] + 5, p[-1][1] + 5), fill=color)
    d.text((x1, y1 + 8), f"gen {gmax:.0f}", fill=GRIS, font=f_chica, anchor="rt")


class Ritmo:
    """Cuántos cuadros de video tocan tras cada paso de la partida, según los frames de juego que pasaron (el
    contador del emulador): uno cada `cada` frames, sea cual sea el ritmo de la red (una que decide cada 5 frames da
    dos o tres copias del mismo cuadro)."""

    def __init__(self, cada: int = 2):
        self.cada, self.ultimo, self.resto = cada, None, 0

    def veces(self, res) -> int:
        if self.ultimo is None:
            self.ultimo = res.frame - self.cada
        self.resto += res.frame - self.ultimo
        self.ultimo = res.frame
        n, self.resto = divmod(self.resto, self.cada)
        return n


def barras_de_preferencia(d: ImageDraw.ImageDraw, x: int, salidas: np.ndarray, accion: int | None, f_barra,
                          rotulos: bool = True) -> None:
    """Preferencia por cada acción (softmax de las salidas de la red; la elegida en dorado)."""
    by = H - 190
    bw = (W - 40 - x - 11 * 6) // 12
    z = salidas - salidas.max()
    p = np.exp(z) / np.exp(z).sum()
    for k, v in enumerate(p):
        bx = x + k * (bw + 6)
        d.rectangle((bx, by + 110 - int(110 * v), bx + bw, by + 110), fill=DORADO if k == accion else AZUL)
        if rotulos or k == accion:
            d.text((bx + bw / 2, by + 120), ETIQUETAS[k], fill=TEXTO if k == accion else TEXTO_2, font=f_barra,
                   anchor="ma")


class PantallaNivel:
    """Una red jugando un nivel (`episodio.Jugador`): el juego grande a la izquierda y, a la derecha, tiempo, avance,
    el puntaje de la partida con su desglose, el fitness de la red y de su generación, el gráfico de fitness (o, con
    `con_red`, la red por dentro) y la preferencia por acción."""

    def __init__(self, linea, cfg, jugador, carpeta: Path, con_red: bool = False):
        self.linea, self.cfg, self.jugador, self.carpeta, self.con_red = linea, cfg, jugador, carpeta, con_red
        self.f_titulo, self.f_med, self.f_chica = fuente(30, True), fuente(24), fuente(20)
        self.f_grande, self.f_barra = fuente(40, True, True), fuente(15)
        self.juego_w, self.juego_h = 1280, 960            # 800×600 × 1,6
        self.px = 24 + self.juego_w + 36                  # panel
        self.gen, self.fitness, self.media, self.maximo, self.metricas = -1, 0.0, 0.0, 0.0, []
        jugador.red.registrar = True
        self.dibujo = linea.dibujo(jugador.red, cfg, (self.px, 560, W - 24, 846), ETIQUETAS) if con_red else None
        self.nueva_partida()

    def nueva_partida(self) -> None:
        self.frames, self.mejor, self.ultimo_frame = 0, 0.0, None

    def red(self, gen: int, fitness: float) -> None:
        """La red que juega: su fitness, el promedio y el máximo de su generación y las métricas para el gráfico."""
        f = almacen.cargar_generacion(self.carpeta, gen, ("fitness",))["fitness"]
        self.gen, self.fitness = gen, fitness
        self.media, self.maximo = float(f.mean()), float(f.max())
        self.metricas = almacen.leer_metricas(self.carpeta)

    def _base(self, titulo: str, juego: np.ndarray | None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), FONDO)
        if juego is not None:
            img.paste(Image.fromarray(juego).resize((self.juego_w, self.juego_h), Image.NEAREST), (JUEGO_X, JUEGO_Y))
        d = ImageDraw.Draw(img)
        d.text((24, 26), titulo, fill=TEXTO, font=self.f_titulo)
        return img, d

    def _puntaje(self, d: ImageDraw.ImageDraw, y: int, partes: dict[str, float], final: bool) -> int:
        """Puntaje de la partida (grande) y su desglose; devuelve la y siguiente."""
        x = self.px
        d.text((x, y), "Puntaje de la partida" + ("" if final else " (en curso)"), fill=TEXTO_2, font=self.f_med)
        d.text((x, y + 28), f"{sum(partes.values()):+.1f}", fill=DORADO, font=fuente(56, True, True))
        y += 100
        for k, v in partes.items():
            if abs(v) < 0.05 and k in ("llegó", "rapidez", "saves", "exploración"):
                continue
            v = v if abs(v) >= 0.05 else 0.0  # sin "−0.0"
            d.text((x, y), k, fill=TEXTO_2, font=self.f_chica)
            d.text((x + 260, y), f"{v:+.1f}", fill=AQUA if v > 0 else (ROJO if v < 0 else TEXTO_2),
                   font=self.f_chica, anchor="ra")
            y += 26
        return y + 14

    def _generacion(self, d: ImageDraw.ImageDraw, y: int, grafico: bool = True) -> int:
        x = self.px
        d.text((x, y), f"Generación {self.gen}", fill=TEXTO_2, font=self.f_med)
        y += 34
        for etiqueta, v, color in (("fitness de esta red", self.fitness, DORADO), ("promedio", self.media, NARANJA),
                                   ("máximo", self.maximo, AZUL)):
            d.text((x, y), etiqueta, fill=TEXTO_2, font=self.f_chica)
            d.text((x + 260, y), f"{v:.1f}", fill=color, font=self.f_chica, anchor="ra")
            y += 26
        if grafico and len(self.metricas) > 1:
            y += 12
            mini_grafico(d, (x + 50, y + 10, W - 40, y + 150), self.metricas, self.gen, self.f_chica)
            y += 190
        return y

    def cuadro(self, res, accion: int | None, salidas: np.ndarray | None, titulo: str, detalle: str) -> Image.Image:
        cfg, j = self.cfg, self.jugador
        self.frames += res.frame - self.ultimo_frame if self.ultimo_frame is not None else cfg.frames_por_decision
        self.ultimo_frame = res.frame
        g = res.grid
        img, d = self._base(titulo, res.obs)
        if g.vivo and g.room == cfg.room_entrenamiento:
            self.mejor = max(self.mejor, j.avance_en(*g.centro))
        if self.linea.sobre_juego is not None and g.vivo:
            vista = tuple(float(v) for v in str(j.emu.exec(
                'return string(view_xview[0]) + "," + string(view_yview[0]);')).split(","))
            self.linea.sobre_juego(img, res, vista)
            if self.linea.leyenda is not None:
                self.linea.leyenda(d, JUEGO_X, JUEGO_Y + 966)
        x, s = self.px, self.frames / 50
        d.text((x, 80), "Tiempo", fill=TEXTO_2, font=self.f_med)
        d.text((x, 108), f"{int(s // 60)}:{s % 60:05.2f}", fill=TEXTO, font=self.f_grande)
        d.text((x + 280, 80), "Avance", fill=TEXTO_2, font=self.f_med)
        d.text((x + 280, 108), f"{100 * self.mejor:.1f} %", fill=TEXTO, font=self.f_grande)
        if detalle:
            d.text((x, 162), detalle, fill=TEXTO_2, font=self.f_chica)
        # puntaje en curso (sin el término de rapidez, que se sabe al final)
        partes = {"avance": cfg.r_avance * self.mejor, "saves": cfg.r_save * len(j.activados),
                  "muertes": -cfg.castigo_muerte * j.muertes}
        if cfg.castigo_quieto:
            partes["quieto"] = -cfg.castigo_quieto * j.quieto / 50
        if cfg.r_explorar:
            partes["exploración"] = cfg.r_explorar * len(j.celdas)
        y = self._puntaje(d, 200, partes, final=False)
        self._generacion(d, y, grafico=False)
        if self.dibujo is not None:
            self.dibujo.dibujar(img, accion)
        else:
            grafico_fitness(d, CAJA_PANEL, self.metricas, self.gen)
        d.text((x, H - 220), "Preferencia por cada acción (↑ salto, Z disparo)", fill=TEXTO_2, font=self.f_chica)
        if salidas is None:
            d.text((x, H - 140), "R · vuelve al último save", fill=DORADO, font=fuente(34, True), anchor="lm")
        else:
            barras_de_preferencia(d, x, salidas, accion, self.f_barra)
        return img

    def final(self, p, titulo: str, nota: str = "") -> Image.Image:
        img, d = self._base(titulo, self.jugador.emu.observe())
        texto = "TIEMPO" if p.fin == "tiempo" and self.cfg.reintentos else FINES[p.fin]
        d.text((JUEGO_X + self.juego_w // 2, JUEGO_Y + self.juego_h // 2), texto,
               fill=AQUA if p.fin == "llegó" else ROJO, font=fuente(80, True), anchor="mm", stroke_width=4,
               stroke_fill=(0, 0, 0))
        x = self.px
        d.text((x, 80), f"Avance {100 * p.avance:.1f} % en {p.frames / 50:.1f} s", fill=TEXTO, font=self.f_med)
        celdas = f" · {p.celdas} celdas" if self.cfg.r_explorar else ""
        d.text((x, 112), f"{p.muertes} muertes · {p.saves} saves · {p.quieto:.1f} s quieta{celdas}", fill=TEXTO_2,
               font=self.f_chica)
        y = self._puntaje(d, 160, desglose(self.cfg, p), final=True)
        y = self._generacion(d, y)
        if nota:
            d.text((x, y), nota, fill=TEXTO_2, font=self.f_chica)
        return img


class PantallaGimnasio:
    """Una política en salas del gimnasio: el juego grande a la izquierda y, al lado, la sala, el intento y el
    historial de salas superadas. Con una red, su gráfico de fitness (o la red por dentro) y la preferencia por
    acción."""

    def __init__(self, agente: str, politica=None, titulo: str = "Gimnasio", linea=None, con_red: bool = False):
        from .gimnasio import PoliticaRed
        self.agente, self.titulo, self.politica, self.linea = agente, titulo, politica, linea
        self.barras = isinstance(politica, PoliticaRed)
        self.dibujo = (linea.dibujo(politica.red, politica.cfg, CAJA_PANEL, ETIQUETAS)
                       if self.barras and con_red else None)
        self.juego_w, self.juego_h = 1280, 960
        self.px = 1340
        self.f_titulo, self.f_grande = fuente(30, True), fuente(44, True, True)
        self.f_med, self.f_chica, self.f_barra = fuente(24), fuente(20), fuente(15)
        self.historial: list[str] = []          # fin de cada sala jugada
        self.sala = None
        self.n = 0

    def nueva_sala(self, sala, n: int, encabezado: str = "", nota: str = "") -> None:
        """Arma una vez por sala la parte fija del cuadro (título, sala, historial, rótulos): dibujarla en cada
        cuadro no deja llegar a tiempo real. `encabezado` reemplaza a «Sala n» y `nota` va bajo los obstáculos."""
        self.sala, self.n = sala, n
        self.encabezado, self.nota = encabezado or f"Sala {n}", nota
        self.frame0, self.mejor = None, 0.0
        img, d = self._base(None)
        x = self.px
        y = self._sala(d, 80)
        d.text((x, y), "Tiempo", fill=TEXTO_2, font=self.f_med)
        d.text((x + 280, y), "Avance", fill=TEXTO_2, font=self.f_med)
        self.y_valores = y + 28
        self._historial(d, y + 100)
        carpeta = getattr(self.politica, "carpeta", None)
        if carpeta is not None and self.dibujo is None:
            grafico_fitness(d, CAJA_PANEL, almacen.leer_metricas(carpeta), getattr(self.politica, "gen", -1))
        if self.barras:
            d.text((x, H - 220), "Preferencia por cada acción (↑ salto, Z disparo)", fill=TEXTO_2,
                   font=self.f_chica)
            bw = (W - 40 - x - 11 * 6) // 12
            for k, etiqueta in enumerate(ETIQUETAS):
                d.text((x + k * (bw + 6) + bw / 2, H - 70), etiqueta, fill=TEXTO_2, font=self.f_barra, anchor="ma")
        if self.linea is not None and self.linea.leyenda is not None:
            self.linea.leyenda(d, JUEGO_X, JUEGO_Y + 966)
        self.fondo = img

    def _base(self, juego: np.ndarray | None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), FONDO)
        if juego is not None:
            img.paste(Image.fromarray(juego).resize((self.juego_w, self.juego_h), Image.NEAREST), (JUEGO_X, JUEGO_Y))
        d = ImageDraw.Draw(img)
        d.text((24, 26), f"{self.titulo} · {self.agente}", fill=TEXTO, font=self.f_titulo)
        return img, d

    def _sala(self, d: ImageDraw.ImageDraw, y: int) -> int:
        s, x = self.sala, self.px
        d.text((x, y), self.encabezado, fill=TEXTO, font=self.f_grande)
        d.text((x, y + 56), f"semilla {s.semilla} · dificultad {s.dificultad:.2f}", fill=TEXTO_2, font=self.f_chica)
        y += 92
        linea = ""
        for palabra in ("Obstáculos: " + (", ".join(s.obstaculos) or "terreno plano")).split(" "):
            if d.textlength(linea + palabra, font=self.f_chica) > W - x - 40:
                d.text((x, y), linea, fill=TEXTO_2, font=self.f_chica)
                y, linea = y + 26, ""
            linea += palabra + " "
        d.text((x, y), linea, fill=TEXTO_2, font=self.f_chica)
        if self.nota:
            y += 30
            d.text((x, y), self.nota, fill=DORADO, font=self.f_chica)
        return y + 50

    def _historial(self, d: ImageDraw.ImageDraw, y: int) -> int:
        x = self.px
        jugadas = [h for h in self.historial if h != "sin solución"]
        llego = sum(h == "llegó" for h in jugadas)
        pct = f" ({100 * llego / len(jugadas):.0f} %)" if jugadas else ""
        d.text((x, y), f"Superó {llego} de {len(jugadas)} salas{pct}", fill=TEXTO, font=self.f_med)
        y += 40
        lado, sep, por_fila = 22, 6, 18
        for i, h in enumerate(self.historial[-2 * por_fila:]):
            cx, cy = x + (i % por_fila) * (lado + sep), y + (i // por_fila) * (lado + sep)
            color = AQUA if h == "llegó" else (GRIS if h == "sin solución" else ROJO)
            d.rectangle((cx, cy, cx + lado, cy + lado), fill=color)
        y += 2 * (lado + sep) + 8
        d.text((x, y), "verde: llegó · rojo: murió o se acabó el tiempo", fill=TEXTO_2, font=fuente(16))
        return y + 40

    def cuadro(self, res, accion: int | None, salidas: np.ndarray | None, gim) -> Image.Image:
        if self.frame0 is None:
            self.frame0 = res.frame - len(res.trace)
        r = res.trace[res.trace["alive"] == 1]
        if len(r):
            self.mejor = max(self.mejor, gim.avance_en(r["x"][-1], r["y"][-1]))
        img = self.fondo.copy()
        if res.obs is not None:
            img.paste(Image.fromarray(res.obs).resize((self.juego_w, self.juego_h), Image.NEAREST), (JUEGO_X, JUEGO_Y))
        d = ImageDraw.Draw(img)
        x, s, y = self.px, (res.frame - self.frame0) / 50, self.y_valores
        d.text((x, y), f"{int(s // 60)}:{s % 60:05.2f}", fill=TEXTO, font=self.f_grande)
        d.text((x + 280, y), f"{100 * self.mejor:.0f} %", fill=TEXTO, font=self.f_grande)
        if self.dibujo is not None:
            self.dibujo.dibujar(img, accion)
        if salidas is not None and self.barras:
            barras_de_preferencia(d, x, salidas, accion, self.f_barra, rotulos=False)
        if self.linea is not None and self.linea.sobre_juego is not None:
            self.linea.sobre_juego(img, res, (0.0, 0.0))
        return img

    def final(self, fin: str, juego: np.ndarray | None) -> Image.Image:
        img, d = self._base(juego)
        d.text((JUEGO_X + self.juego_w // 2, JUEGO_Y + self.juego_h // 2), FINES[fin] if fin != "tiempo" else "TIEMPO",
               fill=AQUA if fin == "llegó" else ROJO, font=fuente(80, True), anchor="mm", stroke_width=4,
               stroke_fill=(0, 0, 0))
        y = self._sala(d, 80)
        self._historial(d, y + 100)
        return img


class Sprites(Escena):
    """Solo los sprites del jugador de `ga.visualizar.Escena`, escalados como el juego en pantalla."""

    def __init__(self, carpeta: Path, escala: float):
        self.sprites = recursos.cargar_sprites(carpeta)
        self.escala = escala
        self._cache = {}


class PantallaPoblacion:
    """Toda la población en una sala: la mejor red emulada en vivo (con las plataformas moviéndose) y las demás
    dibujadas encima como fantasmas, desde sus trazas del entrenamiento (todas empiezan la sala en el mismo frame)."""

    def __init__(self, linea, corrida: str):
        self.corrida = corrida
        self.spr = Sprites(linea.corridas / "_gimnasio" / "recursos", ESCALA)
        self.px = 1340
        self.f_titulo, self.f_grande = fuente(30, True), fuente(44, True, True)
        self.f_med, self.f_chica = fuente(24), fuente(20)

    def a_pantalla(self, x: float, y: float) -> tuple[int, int]:
        return int(JUEGO_X + x * ESCALA), int(JUEGO_Y + y * ESCALA)

    def nueva_sala(self, gen: int, k: int, sala, trazas: list[np.ndarray], fines: list[str], mejor: int,
                   anteriores: list[tuple[int, int]], metricas: list[dict], n_salas: int) -> None:
        self.trazas, self.fines, self.mejor = trazas, fines, mejor
        self.largos = np.array([len(t) for t in trazas])
        self.muertes = []          # (frame en que terminó, x, y en pantalla del último lugar donde estuvo vivo)
        for t, fin in zip(trazas, fines):
            vivo = np.nonzero(t["alive"])[0]
            if fin == "murió" and len(vivo):
                self.muertes.append((len(t), *self.a_pantalla(float(t["x"][vivo[-1]]), float(t["y"][vivo[-1]]))))
        self.muertes.sort()
        img = Image.new("RGB", (W, H), FONDO)
        d = ImageDraw.Draw(img)
        n = len(trazas)
        d.text((24, 26), f"Entrenamiento en vivo · {self.corrida} · las {n} redes de la generación {gen}",
               fill=TEXTO, font=self.f_titulo)
        x = self.px
        d.text((x, 80), f"Sala {k + 1} de {n_salas}", fill=TEXTO, font=self.f_grande)
        d.text((x, 136), f"generación {gen} · semilla {sala.semilla} · dificultad {sala.dificultad:.2f}",
               fill=TEXTO_2, font=self.f_chica)
        linea, y = "", 162
        for palabra in ("Obstáculos: " + (", ".join(sala.obstaculos) or "terreno plano")).split(" "):
            if d.textlength(linea + palabra, font=self.f_chica) > W - x - 30:
                d.text((x, y), linea, fill=TEXTO_2, font=self.f_chica)
                y, linea = y + 26, ""
            linea += palabra + " "
        d.text((x, y), linea, fill=TEXTO_2, font=self.f_chica)
        d.text((x, 216), "Tiempo", fill=TEXTO_2, font=self.f_med)
        self.y_tiempo = 244
        self.y_cuentas = 304
        for i, (rotulo, color) in enumerate((("en juego", TEXTO), ("llegaron", AQUA), ("murieron", ROJO),
                                             ("se estancaron o sin tiempo", TEXTO_2))):
            d.text((x + 90, self.y_cuentas + 30 * i + 4), rotulo, fill=color, font=self.f_chica)
        y = self.y_cuentas + 30 * 4 + 6
        if anteriores:   # las salas ya jugadas de esta generación, en una o dos líneas
            linea = "Antes: " + " · ".join(f"sala {j + 1} {ll}/{t}" for j, (ll, t) in enumerate(anteriores))
            for k in range(0, len(linea), 62):
                d.text((x, y), linea[k:k + 62], fill=TEXTO_2, font=fuente(17))
                y += 22
        grafico_fitness(d, (x, 500, W - 24, 700), metricas, gen)
        grafico_salas(d, (x + 50, 760, W - 40, 890), metricas, gen, self.f_chica)
        d.text((x, 930), "Dorado: la mejor red de la generación (la que se emula en vivo)", fill=DORADO,
               font=fuente(17))
        d.text((x, 954), "Translúcidas: las demás, desde lo que jugaron al entrenar", fill=TEXTO_2, font=fuente(17))
        d.text((x, 978), "Cruz roja: donde murió cada una", fill=ROJO, font=fuente(17))
        self.fondo = img

    def cuadro(self, obs: np.ndarray | None, f: int) -> Image.Image:
        """Cuadro del frame `f` de la sala (índice en las trazas)."""
        img = self.fondo.copy()
        if obs is not None:
            img.paste(Image.fromarray(obs).resize((1280, 960), Image.NEAREST), (JUEGO_X, JUEGO_Y))
        d = ImageDraw.Draw(img)
        for fin, cx, cy in self.muertes:
            if fin > f:
                break
            d.line((cx - 7, cy - 7, cx + 7, cy + 7), fill=ROJO, width=3)
            d.line((cx - 7, cy + 7, cx + 7, cy - 7), fill=ROJO, width=3)
        for i, t in enumerate(self.trazas):
            if i == self.mejor or f >= len(t) or not t["alive"][f]:
                continue
            self.spr.pegar(img, t[f], a_pantalla=self.a_pantalla, alpha=0.45)
        t = self.trazas[self.mejor]               # estela y aro de la mejor (su sprite lo dibuja el juego)
        hasta = min(f, len(t) - 1)
        estela = [self.a_pantalla(float(r["x"]), float(r["y"])) for r in t[max(0, hasta - 50):hasta + 1:2]
                  if r["alive"]]
        if len(estela) > 1:
            d.line(estela, fill=DORADO, width=3)
        if f < len(t) and t["alive"][f]:
            cx, cy = self.a_pantalla(float(t["x"][f]), float(t["y"][f]))
            d.ellipse((cx - 26, cy - 30, cx + 26, cy + 22), outline=DORADO, width=3)
        x, s = self.px, f / 50
        d.text((x, self.y_tiempo), f"{int(s // 60)}:{s % 60:05.2f}", fill=TEXTO, font=self.f_grande)
        en_juego = int((self.largos > f).sum())
        terminadas = [fin for fin, largo in zip(self.fines, self.largos) if largo <= f]
        cuentas = (en_juego, terminadas.count("llegó"), terminadas.count("murió"),
                   len(terminadas) - terminadas.count("llegó") - terminadas.count("murió"))
        for i, (v, color) in enumerate(zip(cuentas, (TEXTO, AQUA, ROJO, TEXTO_2))):
            d.text((x + 70, self.y_cuentas + 30 * i), str(v), fill=color, font=self.f_med, anchor="ra")
        return img
