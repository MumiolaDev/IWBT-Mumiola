"""Ver el gimnasio: muestras de salas, el entrenamiento en vivo, la prueba con salas nuevas (ventana) y videos.

    uv run python -m redes.gimnasio_ver muestras [--dificultad 0.5] [--semilla 0] [--salas 12] [--verificar]
    uv run python -m redes.gimnasio_ver entreno --agente gym02 --quien mejor|todas [--velocidad 1]
    uv run python -m redes.gimnasio_ver entreno-video --agente gym02 --quien todas [--gen -1]
    uv run python -m redes.gimnasio_ver vivo  --agente red07 [--dificultad 0.5] [--semilla 0] [--velocidad 1]
    uv run python -m redes.gimnasio_ver video --agente solucion [--dificultad -1] [--salas 10]

`entreno` (ventana) y `entreno-video`: las salas de una generación tal como las jugó la población al entrenar
(sus mismas semillas y dificultades). `--quien mejor`: la mejor red, con lo que ve por dentro y lo que le pasó al
entrenar; `--quien todas`: toda la población junta (la mejor emulada, las demás como fantasmas desde sus trazas).
En vivo repite las salas de la última generación y pasa a la nueva cuando aparece: se ve cómo aprende.

`vivo` y `video` (la prueba): salas nuevas, que la red nunca vio, una tras otra: sirven para ver si generaliza.

`muestras`: una imagen con 12 salas (4 × 3), con sus obstáculos; con `--verificar` busca la solución de cada una y
la dibuja (o marca que no la encontró). Queda en corridas_redes/_gimnasio/muestras/.

`--agente`: el nombre de una corrida (juega su mejor red, la que llega más lejos; en vivo toma la nueva cuando
aparece), `azar` (acciones al azar) o `solucion` (la búsqueda resuelve cada sala y se ve su solución). Las salas van
de `--semilla` en adelante; `--dificultad` entre 0 y 1, o -1 para una al azar en cada sala. Con `--verificar`, las
salas sin solución encontrada se saltan (siempre con `solucion`). El video queda en la carpeta de la corrida
(videos/gimnasio_*.mp4) o, con `azar`/`solucion`, en corridas_redes/_gimnasio/videos/.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ga import recursos
from ga.visualizar import (H, LIMITE_FFPLAY, REPORTE_CADA, W, Escena, Ventana, Video, _con_tope_de_memoria,
                           _terminar, fuente, memoria_libre_mb, memoria_mb)
from iwbtgb_rl import ACTIONS

from . import almacen
from .dibujo_red import DibujoRed
from .gimnasio import (Gimnasio, Politica, PoliticaAzar, PoliticaGrabada, PoliticaRed, Sala, generar,
                       resolver)
from .visualizar import AQUA, AZUL, DORADO, FONDO, GRIS, ROJO, TEXTO, TEXTO_2, nombre_accion

CARPETA = almacen.CORRIDAS / "_gimnasio"
FPS = 25                     # al dibujar, cada paso son 2 frames del juego (50 por segundo)
MAX_FRAMES = 1000            # 20 s por sala con `azar` y `solucion` (una red usa la duración y la paciencia de su
                             # corrida, como al entrenar)
PRESUPUESTO = 600            # iteraciones de la búsqueda por sala
FINES = {"llegó": "¡LLEGÓ!", "murió": "MURIÓ", "tiempo": "TIEMPO", "estancado": "SE ESTANCÓ",
         "sin solución": "SIN SOLUCIÓN"}


def dificultad_de(dificultad: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(0, 1)) if dificultad < 0 else dificultad


def preparar(agente: str, gen: int = -1) -> tuple[Gimnasio, Politica | None, str]:
    """Gimnasio (con la vista de la red, si es una corrida) y la política; None si la resuelve la búsqueda."""
    if agente == "azar":
        return Gimnasio(), PoliticaAzar(), "acciones al azar"
    if agente == "solucion":
        return Gimnasio(), None, "solución de la búsqueda"
    politica = PoliticaRed(agente, gen)
    politica.red.registrar = True
    gim = Gimnasio(politica.config_vista)
    politica.vista = gim.vista
    return gim, politica, politica.nombre


class Pantalla:
    """El juego grande a la izquierda (como los videos de redes) y al lado la sala, el intento y el historial."""

    def __init__(self, agente: str, politica: Politica | None = None, titulo: str = "Gimnasio"):
        self.agente, self.titulo = agente, titulo
        self.barras = isinstance(politica, PoliticaRed)    # preferencia por acción y dibujo de la red
        self.dibujo = (DibujoRed(politica.red, politica.config_vista, (1340, 560, W - 24, 846),
                                 [nombre_accion(t) for t in ACTIONS]) if self.barras else None)
        self.juego_w, self.juego_h = 1280, 960
        self.px = 1340
        self.f_titulo, self.f_grande = fuente(30, True), fuente(44, True, True)
        self.f_med, self.f_chica, self.f_barra = fuente(24), fuente(20), fuente(15)
        self.historial: list[str] = []          # fin de cada sala jugada
        self.sala: Sala | None = None
        self.n = 0
        self.frames, self.mejor = 0, 0.0

    def nueva_sala(self, sala: Sala, n: int, encabezado: str = "", nota: str = "") -> None:
        """Arma la parte fija del cuadro (título, sala, historial, rótulos) una vez por sala: dibujarla en cada
        cuadro costaba ~30 ms y el en vivo no llegaba a tiempo real. `encabezado` reemplaza a «Sala n» y `nota` va
        bajo los obstáculos."""
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
        if self.barras:
            d.text((x, H - 220), "Preferencia por cada acción (↑ salto, Z disparo)", fill=TEXTO_2,
                   font=self.f_chica)
            for k, teclas in enumerate(ACTIONS):
                bx, by, bw = self._barra(k)
                d.text((bx + bw / 2, by + 120), nombre_accion(teclas), fill=TEXTO_2, font=self.f_barra, anchor="ma")
        self.fondo = img

    def _barra(self, k: int) -> tuple[int, int, int]:
        bw = (W - 40 - self.px - 11 * 6) // 12
        return self.px + k * (bw + 6), H - 190, bw

    def _base(self, juego: np.ndarray | None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), FONDO)
        if juego is not None:
            img.paste(Image.fromarray(juego).resize((self.juego_w, self.juego_h), Image.NEAREST), (24, 80))
        d = ImageDraw.Draw(img)
        d.text((24, 26), f"{self.titulo} · {self.agente}", fill=TEXTO, font=self.f_titulo)
        return img, d

    def _sala(self, d: ImageDraw.ImageDraw, y: int) -> int:
        s, x = self.sala, self.px
        d.text((x, y), self.encabezado, fill=TEXTO, font=self.f_grande)
        d.text((x, y + 56), f"semilla {s.semilla} · dificultad {s.dificultad:.2f}", fill=TEXTO_2, font=self.f_chica)
        y += 92
        texto = ", ".join(s.obstaculos) or "terreno plano"
        linea = ""
        for palabra in ("Obstáculos: " + texto).split(" "):
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

    def cuadro(self, res, accion: int | None, salidas: np.ndarray | None, gim: Gimnasio) -> Image.Image:
        if self.frame0 is None:
            self.frame0 = res.frame - len(res.trace)
        r = res.trace[res.trace["alive"] == 1]
        if len(r):
            self.mejor = max(self.mejor, gim.avance_en(r["x"][-1], r["y"][-1]))
        img = self.fondo.copy()
        if res.obs is not None:
            img.paste(Image.fromarray(res.obs).resize((self.juego_w, self.juego_h), Image.NEAREST), (24, 80))
        d = ImageDraw.Draw(img)
        x, s, y = self.px, (res.frame - self.frame0) / 50, self.y_valores
        d.text((x, y), f"{int(s // 60)}:{s % 60:05.2f}", fill=TEXTO, font=self.f_grande)
        d.text((x + 280, y), f"{100 * self.mejor:.0f} %", fill=TEXTO, font=self.f_grande)
        if self.dibujo is not None:
            self.dibujo.dibujar(img, accion)
        if salidas is not None and self.barras:
            z = salidas - salidas.max()
            p = np.exp(z) / np.exp(z).sum()
            for k, v in enumerate(p):
                bx, by, bw = self._barra(k)
                d.rectangle((bx, by + 110 - int(110 * v), bx + bw, by + 110), fill=DORADO if k == accion else AZUL)
            if accion is not None:  # el rótulo de la elegida, resaltado (los demás están en el fondo)
                bx, by, bw = self._barra(accion)
                d.text((bx + bw / 2, by + 120), nombre_accion(ACTIONS[accion]), fill=TEXTO, font=self.f_barra,
                       anchor="ma")
        return img

    def final(self, fin: str, juego: np.ndarray | None) -> Image.Image:
        img, d = self._base(juego)
        d.text((24 + self.juego_w // 2, 80 + self.juego_h // 2), FINES[fin], fill=AQUA if fin == "llegó" else ROJO,
               font=fuente(80, True), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
        y = self._sala(d, 80)
        self._historial(d, y + 100)
        return img


class _Cerrada(Exception):
    pass


def jugar_salas(agente: str, dificultad: float, semilla: int, salas: int | None, verificar: bool, mostrar,
                recargar: bool = False, velocidad: int = 1, gen: int = -1) -> None:
    """Juega salas una tras otra (`salas` = None: sin fin) y pasa cada cuadro a `mostrar` (uno cada 2·`velocidad`
    frames de juego: a 25 cuadros/s, `velocidad` veces el tiempo real)."""
    gim, politica, nombre = preparar(agente, gen)
    pantalla = Pantalla(nombre, politica)
    if isinstance(politica, PoliticaRed):   # como al entrenar
        tope, paciencia = politica.cfg.max_frames, int(politica.cfg.paciencia * 50)
    else:
        tope, paciencia = MAX_FRAMES, 0
    rng = np.random.default_rng(semilla)
    try:
        n, s = 0, semilla
        while salas is None or n < salas:
            sala = generar(s, dificultad_de(dificultad, rng))
            s += 1
            gim.construir(sala)
            if politica is None or verificar:
                solucion = resolver(gim, PRESUPUESTO, sala.semilla)
                if solucion is None:
                    print(f"sala {sala.nombre}: sin solución encontrada, se salta", flush=True)
                    continue
            n += 1
            pol = PoliticaGrabada(solucion) if politica is None else politica
            if recargar and isinstance(pol, PoliticaRed) and pol.cargar(-1):
                pantalla.agente = pol.nombre
                pantalla.dibujo.cargar()
            pantalla.nueva_sala(sala, n)
            ultimo = {"obs": None}

            def al_frame(res, accion, salidas):
                ultimo["obs"] = res.obs
                mostrar(pantalla.cuadro(res, accion, salidas, gim))

            intento = gim.jugar(pol, tope, al_frame, paciencia=paciencia, cada=2 * velocidad)
            pantalla.historial.append(intento.fin)
            print(f"sala {n} ({sala.nombre}): {intento.fin} · avance {100 * intento.avance:.0f} % · "
                  f"{intento.frames / 50:.1f} s", flush=True)
            img = pantalla.final(intento.fin, ultimo["obs"])
            for _ in range(FPS):
                mostrar(img)
    finally:
        gim.close()


def vivo(agente: str, dificultad: float, semilla: int, verificar: bool, velocidad: int = 1) -> None:
    """Prueba en vivo: salas nuevas, que la red nunca vio (¿generaliza?)."""
    titulo = f"Gimnasio, salas nuevas · {agente}" + (f" (x{velocidad})" if velocidad > 1 else "")
    en_ventana(titulo, lambda mostrar: jugar_salas(agente, dificultad, semilla, None, verificar, mostrar,
                                                   recargar=True, velocidad=velocidad))


def en_ventana(titulo: str, jugar) -> None:
    """Muestra en una ventana (ffplay) los cuadros que `jugar(mostrar)` le pasa a `mostrar`, hasta que se cierre."""
    signal.signal(signal.SIGTERM, _terminar)
    estado = {"ventana": Ventana(titulo, fps=FPS), "reporte": time.monotonic()}
    inicio = time.monotonic()

    def mostrar(img: Image.Image) -> None:
        v = estado["ventana"]
        ahora = time.monotonic()
        if ahora - estado["reporte"] >= REPORTE_CADA:
            estado["reporte"] = ahora
            print(f"en vivo: {(ahora - inicio) / 3600:.1f} h · memoria {memoria_mb():.0f} MB + ffplay "
                  f"{v.memoria_mb():.0f} MB · libre {memoria_libre_mb() / 1024:.1f} GB", flush=True)
        if v.memoria_mb() > LIMITE_FFPLAY:
            v.cerrar()
            v = estado["ventana"] = Ventana(titulo, fps=FPS)
        if not v.escribir(img):
            raise _Cerrada

    try:
        jugar(mostrar)
    except (KeyboardInterrupt, _Cerrada):
        pass
    finally:
        estado["ventana"].cerrar()


def video(agente: str, dificultad: float, semilla: int, salas: int, verificar: bool, gen: int = -1) -> Path:
    dif = "azar" if dificultad < 0 else f"{dificultad:.2f}"
    if agente in ("azar", "solucion"):
        ruta = CARPETA / "videos" / f"{agente}_dif{dif}_s{semilla}.mp4"
    else:
        from .visualizar import red_de
        gen = red_de(almacen.dir_corrida(agente), gen)[0]
        ruta = almacen.dir_corrida(agente) / "videos" / f"gimnasio_gen{gen:04d}_dif{dif}_s{semilla}.mp4"
    v = Video(ruta, fps=FPS)
    try:
        jugar_salas(agente, dificultad, semilla, salas, verificar, v.escribir, gen=gen)
    finally:
        v.cerrar()
    return ruta


# -- el entrenamiento en vivo: las salas de una generación, como las jugó la población ------------------------

def generacion(carpeta: Path, gen: int) -> tuple[int, dict]:
    gens = almacen.generaciones_guardadas(carpeta)
    if not gens:
        raise SystemExit(f"{carpeta.name} todavía no tiene generaciones")
    gen = gens[-1] if gen < 0 else gen
    d = almacen.cargar_generacion(carpeta, gen)
    if "gimnasio" not in d:
        raise SystemExit(f"{carpeta.name} no entrena en el gimnasio")
    return gen, d


def salas_de(d: dict) -> list[Sala]:
    return [generar(int(s), float(f)) for s, f in zip(d["semillas"], d["dificultades"])]


def sin_trazas(corrida: str, g: int) -> SystemExit:
    return SystemExit(f"la generación {g} de {corrida} no guardó las trazas de sus salas: «todas las redes» necesita "
                      "generaciones entrenadas desde el 2026-10-27 (detén la corrida y continúala para que las "
                      "nuevas las guarden). «La mejor red» sí funciona")


def trazas_de(d: dict) -> list[list[np.ndarray]] | None:
    """[red][sala] → traza por frame de esa partida; None si la generación no las guardó (antes del 2026-10-28)."""
    largos = d["traza_largo"]
    if largos.ndim != 2 or not largos.sum():
        return None
    fin = np.cumsum(largos.ravel())
    trozos = np.split(d["trazas"], fin[:-1])
    n, s = largos.shape
    return [[trozos[i * s + k] for k in range(s)] for i in range(n)]


class _Sprites(Escena):
    """Solo los sprites del jugador de `ga.visualizar.Escena` (escalados como el juego en pantalla)."""

    def __init__(self, escala: float):
        self.sprites = recursos.cargar_sprites(CARPETA / "recursos")
        self.escala = escala
        self._cache = {}


class PantallaPoblacion:
    """Toda la población en una sala: la mejor red emulada en vivo (con las plataformas moviéndose) y las demás
    dibujadas encima como fantasmas, desde sus trazas del entrenamiento (todas empiezan la sala en el mismo frame)."""

    def __init__(self, corrida: str):
        self.corrida = corrida
        self.escala = 1.6
        self.spr = _Sprites(self.escala)
        self.px = 1340
        self.f_titulo, self.f_grande = fuente(30, True), fuente(44, True, True)
        self.f_med, self.f_chica = fuente(24), fuente(20)

    def a_pantalla(self, x: float, y: float) -> tuple[int, int]:
        return int(24 + x * self.escala), int(80 + y * self.escala)

    def nueva_sala(self, gen: int, k: int, sala: Sala, trazas: list[np.ndarray], fines: list[str], mejor: int,
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
        self.y_cuentas = 310
        for i, (rotulo, color) in enumerate((("en juego", TEXTO), ("llegaron", AQUA), ("murieron", ROJO),
                                             ("se estancaron o sin tiempo", TEXTO_2))):
            d.text((x + 90, self.y_cuentas + 34 * i + 4), rotulo, fill=color, font=self.f_chica)
        y = self.y_cuentas + 34 * 4 + 20
        if anteriores:
            d.text((x, y), "Salas anteriores de esta generación", fill=TEXTO_2, font=self.f_med)
            y += 34
            for j, (llegaron, total) in enumerate(anteriores):
                d.text((x, y), f"sala {j + 1}: llegaron {llegaron} de {total}", fill=TEXTO_2, font=self.f_chica)
                y += 26
        grafico_salas(d, (x + 50, 700, W - 40, 860), metricas, gen, self.f_chica)
        d.text((x, 920), "Dorado: la mejor red de la generación (la que se emula en vivo)", fill=DORADO,
               font=fuente(17))
        d.text((x, 946), "Translúcidas: las demás, desde lo que jugaron al entrenar", fill=TEXTO_2, font=fuente(17))
        d.text((x, 972), "Cruz roja: donde murió cada una", fill=ROJO, font=fuente(17))
        self.fondo = img

    def cuadro(self, obs: np.ndarray | None, f: int) -> Image.Image:
        """Cuadro del frame `f` de la sala (índice en las trazas)."""
        img = self.fondo.copy()
        if obs is not None:
            img.paste(Image.fromarray(obs).resize((1280, 960), Image.NEAREST), (24, 80))
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
            d.text((x + 70, self.y_cuentas + 34 * i), str(v), fill=color, font=self.f_med, anchor="ra")
        return img


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


def entreno(corrida: str, quien: str, mostrar, velocidad: int = 1, gen: int = -1, en_vivo: bool = True) -> None:
    """Las salas de una generación tal como las jugó la población: `quien` = "mejor" (la mejor red, con lo que ve
    por dentro) o "todas" (toda la población junta). En vivo (`gen` = -1) repite las salas de la última generación
    guardada y, cuando aparece una nueva, pasa a las suyas: se ve cómo aprende."""
    carpeta = almacen.dir_corrida(corrida)
    cfg = almacen.cargar_config(carpeta)
    g, d = generacion(carpeta, gen)
    if quien == "todas" and trazas_de(d) is None:
        raise sin_trazas(corrida, g)
    politica = PoliticaRed(corrida, g)
    politica.red.registrar = True
    gim = Gimnasio(politica.config_vista)
    politica.vista = gim.vista
    tope, paciencia, cada = cfg.max_frames, int(cfg.paciencia * 50), 2 * velocidad
    pantalla = (Pantalla(politica.nombre, politica, titulo="Entrenamiento en vivo") if quien == "mejor" else
                PantallaPoblacion(corrida))
    try:
        while True:
            if politica.cargar(g) and quien == "mejor":
                pantalla.agente = politica.nombre
                pantalla.dibujo.cargar()
            mejor = almacen.indice_mejor(d)
            salas = salas_de(d)
            trazas = trazas_de(d) if quien == "todas" else None
            if quien == "todas" and trazas is None:
                raise sin_trazas(corrida, g)
            metricas = almacen.leer_metricas(carpeta)
            anteriores: list[tuple[int, int]] = []
            for k, sala in enumerate(salas):
                gim.construir(sala)
                fines = [str(f) for f in d["fines"][:, k]]
                if quien == "mejor":
                    _sala_mejor(gim, politica, pantalla, sala, g, k, len(salas), fines[mejor],
                                float(d["avances"][mejor, k]), tope, paciencia, cada, mostrar)
                else:
                    pantalla.nueva_sala(g, k, sala, [t[k] for t in trazas], fines, mejor, anteriores, metricas,
                                        len(salas))
                    _sala_todas(gim, politica, pantalla, tope, paciencia, cada, mostrar)
                    anteriores.append((fines.count("llegó"), len(fines)))
                print(f"gen {g} sala {k + 1}/{len(salas)} ({sala.nombre}): llegaron {fines.count('llegó')} de "
                      f"{len(fines)} · la mejor: {fines[mejor]}", flush=True)
            if not en_vivo:
                break
            g, d = generacion(carpeta, gen)
    finally:
        gim.close()


def _sala_mejor(gim, politica, pantalla, sala, g, k, n_salas, fin, avance, tope, paciencia, cada, mostrar) -> None:
    pantalla.nueva_sala(sala, k + 1, encabezado=f"Sala {k + 1} de {n_salas}",
                        nota=f"Al entrenar (gen {g}): {fin} · avance {100 * avance:.0f} %")
    ultimo = {"obs": None}

    def al_frame(res, accion, salidas):
        ultimo["obs"] = res.obs
        mostrar(pantalla.cuadro(res, accion, salidas, gim))

    intento = gim.jugar(politica, tope, al_frame, paciencia=paciencia, cada=cada)
    if intento.fin != fin or abs(intento.avance - avance) > 1e-3:
        print(f"aviso: al repetirla terminó distinto que al entrenar ({intento.fin}, {100 * intento.avance:.0f} % "
              f"contra {fin}, {100 * avance:.0f} %)", flush=True)
    pantalla.historial.append(intento.fin)
    img = pantalla.final(intento.fin, ultimo["obs"])
    for _ in range(FPS):
        mostrar(img)


def _sala_todas(gim, politica, pantalla, tope, paciencia, cada, mostrar) -> None:
    estado = {"f0": None, "obs": None}

    def al_frame(res, accion, salidas):
        if estado["f0"] is None:
            estado["f0"] = res.frame - len(res.trace)
        estado["obs"] = res.obs
        mostrar(pantalla.cuadro(res.obs, res.frame - estado["f0"] - 1))

    intento = gim.jugar(politica, tope, al_frame, paciencia=paciencia, cada=cada)
    if intento.frames != pantalla.largos[pantalla.mejor]:
        print(f"aviso: la mejor duró {intento.frames} frames al repetirla y {pantalla.largos[pantalla.mejor]} al "
              "entrenar", flush=True)
    largo = int(pantalla.largos.max())
    f = intento.frames
    if f < largo:
        # La mejor terminó y otras siguen: se vuelve al comienzo de la sala, se esconde al jugador y se adelanta hasta
        # este frame, para que las plataformas sigan moviéndose bajo los fantasmas (sin el cartel de GAME OVER)
        gim.emu.load(gim.SALA)
        gim.emu.exec("with (player) { visible = false; frozen = true; } with (bow) visible = false;")
        gim.emu.step((), frames=f, obs=False, no_draw=True)
        while f < largo:
            res = gim.emu.step((), frames=cada, obs=True)
            f += cada
            mostrar(pantalla.cuadro(res.obs, min(f, largo) - 1))
    img = pantalla.cuadro(estado["obs"] if f <= intento.frames else res.obs, largo)
    for _ in range(FPS):
        mostrar(img)


def entreno_vivo(corrida: str, quien: str, velocidad: int = 1) -> None:
    if quien == "todas":   # antes de abrir la ventana
        g, d = generacion(almacen.dir_corrida(corrida), -1)
        if trazas_de(d) is None:
            raise sin_trazas(corrida, g)
    titulo = f"Entrenamiento en vivo · {corrida} · {'la mejor red' if quien == 'mejor' else 'todas las redes'}"
    en_ventana(titulo, lambda mostrar: entreno(corrida, quien, mostrar, velocidad))


def entreno_video(corrida: str, quien: str, gen: int = -1, velocidad: int = 1) -> Path:
    g, d = generacion(almacen.dir_corrida(corrida), gen)
    if quien == "todas" and trazas_de(d) is None:
        raise sin_trazas(corrida, g)
    ruta = almacen.dir_corrida(corrida) / "videos" / f"entrenamiento_gen{g:04d}_{quien}.mp4"
    v = Video(ruta, fps=FPS)
    try:
        entreno(corrida, quien, v.escribir, velocidad, g, en_vivo=False)
    finally:
        v.cerrar()
    return ruta


def muestras(dificultad: float, semilla: int, salas: int = 12, verificar: bool = False) -> Path:
    """Mosaico de 4 columnas con las salas (y la solución encontrada, con `verificar`)."""
    gim = Gimnasio()
    rng = np.random.default_rng(semilla)
    cols = 4
    filas = (salas + cols - 1) // cols
    ancho = W // cols
    alto = ancho * 3 // 4
    lienzo = Image.new("RGB", (W, filas * alto), FONDO)
    resueltas = 0
    try:
        for k in range(salas):
            sala = generar(semilla + k, dificultad_de(dificultad, rng))
            gim.construir(sala)
            gim.emu.load(gim.SALA)
            img = Image.fromarray(gim.emu.step((), frames=1, obs=True).obs).convert("RGB")
            d = ImageDraw.Draw(img)
            estado = ""
            if verificar:
                t = time.perf_counter()
                sol = resolver(gim, PRESUPUESTO, sala.semilla)
                if sol is not None:
                    resueltas += 1
                    it = gim.jugar(PoliticaGrabada(sol), 10_000)
                    d.line([tuple(p) for p in it.recorrido], fill=(255, 0, 200), width=4)
                    estado = f" · resuelta en {it.frames / 50:.1f} s"
                else:
                    estado = " · sin solución"
                print(f"{sala.nombre}{estado} (búsqueda {time.perf_counter() - t:.1f} s)", flush=True)
            img = img.resize((ancho, alto), Image.LANCZOS)
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, ancho, 48), fill=(0, 0, 0))
            d.text((8, 4), f"semilla {sala.semilla} · dif {sala.dificultad:.2f}{estado}",
                   fill=AQUA if "resuelta" in estado else (ROJO if "sin" in estado else TEXTO), font=fuente(18, True))
            d.text((8, 26), ", ".join(sala.obstaculos) or "terreno plano", fill=TEXTO_2, font=fuente(15))
            lienzo.paste(img, ((k % cols) * ancho, (k // cols) * alto))
    finally:
        gim.close()
    dif = "azar" if dificultad < 0 else f"{dificultad:.2f}"
    ruta = CARPETA / "muestras" / f"dif{dif}_s{semilla}{'_verificadas' if verificar else ''}.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(ruta)
    if verificar:
        print(f"con solución: {resueltas} de {salas}", flush=True)
    print(f"imagen: {ruta}", flush=True)
    return ruta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["muestras", "vivo", "video", "entreno", "entreno-video"])
    p.add_argument("--agente", default="solucion", help="nombre de una corrida, «azar» o «solucion»")
    p.add_argument("--quien", choices=["mejor", "todas"], default="mejor",
                   help="entreno: la mejor red o todas las de la generación")
    p.add_argument("--dificultad", type=float, default=0.5, help="0 a 1; -1 = al azar en cada sala")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--salas", type=int, default=12)
    p.add_argument("--verificar", action="store_true", help="saltar las salas sin solución encontrada")
    p.add_argument("--velocidad", type=int, default=1)
    p.add_argument("--gen", type=int, default=-1, help="video: generación de la red (-1 = la última)")
    a = p.parse_args()
    if a.modo == "muestras":
        muestras(a.dificultad, a.semilla, a.salas, a.verificar)
    elif a.modo == "vivo":
        _con_tope_de_memoria("redes.gimnasio_ver")
        vivo(a.agente, a.dificultad, a.semilla, a.verificar, a.velocidad)
    elif a.modo == "entreno":
        _con_tope_de_memoria("redes.gimnasio_ver")
        entreno_vivo(a.agente, a.quien, a.velocidad)
    elif a.modo == "entreno-video":
        entreno_video(a.agente, a.quien, a.gen, a.velocidad)
    else:
        video(a.agente, a.dificultad, a.semilla, a.salas, a.verificar, a.gen)


if __name__ == "__main__":
    main()
