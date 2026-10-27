"""Videos, gráficos y en vivo de una corrida de redes.

    uv run python -m redes.visualizar mejor      --corrida red01 [--gen -1] [--espera 0] [--sala rStage02]
    uv run python -m redes.visualizar generacion --corrida red05 [--gen -1] [--velocidad 1]
    uv run python -m redes.visualizar vivo       --corrida red05 [--velocidad 4]
    uv run python -m redes.visualizar vivo-red   --corrida red01 [--sala rStage02] [--velocidad 1]
    uv run python -m redes.visualizar graficos   --corrida red01

`mejor`: la mejor red de la generación (la que llega más lejos) jugando un escenario (`--espera` = frames sin
control al inicio): el juego grande y al lado el puntaje de la partida con su desglose, el fitness de la red y de su
generación (promedio y máximo), la red dibujada por dentro (`dibujo_red.py`) y la preferencia de la red por cada
acción. `--sala`
la pone en otra sala (que puede no haber visto nunca). Queda en corridas_redes/<nombre>/videos/.

`generacion` y `vivo`: todas las redes de una generación jugando juntas sobre el mapa (su primer escenario), con
el mismo dibujo que el GA; `vivo` es una ventana (ffplay) que repite la última generación guardada y salta a la
más nueva. Necesitan las trazas que se guardan desde el 2026-10-26 (corridas anteriores: `vivo-red`).

`vivo-red`: ventana con la mejor red jugando en tiempo real, con lo que ve, partida tras partida, cada una con una
espera al azar; toma la red nueva cuando aparece una mejor. Los dos en vivo corren con tope de memoria, como el del
GA, para dejarlos horas (stream).

`graficos`: PNG con avance, fitness y cómo terminan las partidas, en corridas_redes/<nombre>/graficos/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from ga.visualizar import (H, LIMITE_FFPLAY, LIMITE_VIVO, REPORTE_CADA, ROOT, W, Ventana, Video,
                           _con_tope_de_memoria, _terminar, cuadros_generacion, fuente, memoria_libre_mb, memoria_mb,
                           mini_grafico)
from iwbtgb_rl import ACTIONS

from . import almacen
from .config import Config
from .dibujo_red import DibujoRed
from .episodio import Escenario, Jugador, Partida, desglose
from .salas import config_en_sala

FONDO = (26, 26, 25)
TEXTO = (240, 239, 236)
TEXTO_2 = (195, 194, 183)
AZUL = (57, 135, 229)
NARANJA = (217, 89, 38)
DORADO = (237, 161, 0)
ROJO = (230, 103, 103)
AQUA = (25, 158, 112)
GRIS = (130, 129, 122)

FPS_VIDEO = 25          # videos y en vivo de una red: un cuadro cada 2 frames de juego (a 1×, tiempo real)


class Ritmo:
    """Cuántos cuadros de video tocan tras cada paso de la partida, según los frames de juego que pasaron (el
    contador del emulador): uno cada `cada` frames, sea cual sea el ritmo de la red. Una red que decide cada frame
    da un cuadro cada dos pasos (antes, uno por paso: el video iba a la mitad de la velocidad real); una que decide
    cada 5, dos o tres copias del mismo cuadro."""

    def __init__(self, cada: int = 2):
        self.cada, self.ultimo, self.resto = cada, None, 0

    def veces(self, res) -> int:
        if self.ultimo is None:
            self.ultimo = res.frame - self.cada
        self.resto += res.frame - self.ultimo
        self.ultimo = res.frame
        n, self.resto = divmod(self.resto, self.cada)
        return n


FINES = {"llegó": "¡LLEGÓ!", "murió": "MURIÓ", "salió": "SALIÓ DE LA SALA", "estancado": "SE ESTANCÓ",
         "tiempo": "SE ACABÓ EL TIEMPO"}


def nombre_accion(teclas: tuple[str, ...]) -> str:
    """Etiqueta corta: ← →, ↑ = salto, Z = disparo."""
    return "".join({"left": "←", "right": "→", "shift": "↑", "z": "Z"}[t] for t in teclas) or "nada"


def red_de(carpeta: Path, gen: int) -> tuple[int, tuple, float]:
    """(generación, genoma, fitness) de la mejor red de `gen` (la última con -1): la que llega más lejos."""
    gens = almacen.generaciones_guardadas(carpeta)
    gen = gens[-1] if gen < 0 else gen
    d = almacen.cargar_generacion(carpeta, gen)
    i = almacen.indice_mejor(d)
    if (carpeta / "redes" / "mejor.json").exists():
        try:
            info = json.loads((carpeta / "redes" / "mejor.json").read_text())
        except ValueError:  # se está escribiendo justo ahora
            info = {"gen": -1}
        if info["gen"] == gen and info["agente"] == i:
            return gen, genoma_de(info["genoma"]), float(d["fitness"][i])
    return gen, almacen.genomas(carpeta, gen)[i], float(d["fitness"][i])


def genoma_de(lista: list) -> tuple:
    return tuple([lista[0]] + [tuple(x) for x in lista[1:]])


class Pantalla:
    """Arma cada cuadro: el juego grande a la izquierda y, a la derecha, tiempo, avance, el puntaje de la partida
    con su desglose, el fitness de la red y de su generación, el gráfico de fitness y la preferencia por acción."""

    def __init__(self, cfg: Config, jugador: Jugador, carpeta: Path):
        self.cfg, self.jugador, self.carpeta = cfg, jugador, carpeta
        self.f_titulo, self.f_med, self.f_chica = fuente(30, True), fuente(24), fuente(20)
        self.f_grande, self.f_num = fuente(40, True, True), fuente(30, True, True)
        self.juego_w, self.juego_h = 1280, 960            # 800×600 × 1,6
        self.px = 24 + self.juego_w + 36                  # panel
        self.gen, self.fitness, self.media, self.maximo, self.metricas = -1, 0.0, 0.0, 0.0, []
        self.f_barra = fuente(15)
        jugador.red.registrar = True
        self.dibujo = DibujoRed(jugador.red, cfg.config_vista, (self.px, 560, W - 24, 846),
                                [nombre_accion(t) for t in ACTIONS])
        self._pesos = None
        self.nueva_partida()

    def nueva_partida(self) -> None:
        self.frames, self.mejor, self.ultimo_frame = 0, 0.0, None

    def red(self, gen: int, fitness: float) -> None:
        """La red que juega: su fitness, el promedio y el máximo de su generación y las métricas para el gráfico."""
        d = almacen.cargar_generacion(self.carpeta, gen)
        self.gen, self.fitness = gen, fitness
        self.media, self.maximo = float(d["fitness"].mean()), float(d["fitness"].max())
        self.metricas = almacen.leer_metricas(self.carpeta)

    def _base(self, titulo: str, juego: np.ndarray | None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), FONDO)
        if juego is not None:
            img.paste(Image.fromarray(juego).resize((self.juego_w, self.juego_h), Image.NEAREST), (24, 80))
        d = ImageDraw.Draw(img)
        d.text((24, 26), titulo, fill=TEXTO, font=self.f_titulo)
        return img, d

    def _puntaje(self, d: ImageDraw.ImageDraw, y: int, partes: dict[str, float], final: bool) -> int:
        """Puntaje de la partida (grande) y su desglose; devuelve la y siguiente."""
        x = self.px
        d.text((x, y), "Puntaje de la partida" + ("" if final else " (en curso)"), fill=TEXTO_2, font=self.f_med)
        total = sum(partes.values())
        d.text((x, y + 28), f"{total:+.1f}", fill=DORADO, font=fuente(56, True, True))
        y += 100
        nombres = {"avance": "avance", "llegó": "llegó", "rapidez": "rapidez", "saves": "saves", "quieto": "quieto",
                   "exploración": "exploración", "muertes": "muertes"}
        for k, v in partes.items():
            if abs(v) < 0.05 and k in ("llegó", "rapidez", "saves", "exploración"):
                continue
            v = v if abs(v) >= 0.05 else 0.0  # sin "−0.0"
            d.text((x, y), nombres[k], fill=TEXTO_2, font=self.f_chica)
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

    def cuadro(self, res, accion: int | None, salidas: np.ndarray | None, titulo: str,
               detalle: str) -> Image.Image:
        cfg, j = self.cfg, self.jugador
        self.frames += res.frame - self.ultimo_frame if self.ultimo_frame is not None else cfg.frames_por_decision
        self.ultimo_frame = res.frame
        g = res.grid
        img, d = self._base(titulo, res.obs)
        if g.vivo and g.room == cfg.room_entrenamiento:
            self.mejor = max(self.mejor, j.avance_en(*g.centro))
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
        if self._pesos is not j.red.p.get("w4"):     # red nueva (cada partida carga los pesos): se redibuja
            self._pesos = j.red.p.get("w4")
            self.dibujo.cargar()
        self.dibujo.dibujar(img, accion)
        # preferencia por cada acción (softmax de las salidas; la elegida en dorado); al reaparecer, R
        by = H - 190
        d.text((x, by - 30), "Preferencia por cada acción (↑ salto, Z disparo)", fill=TEXTO_2, font=self.f_chica)
        bw = (W - 40 - x - 11 * 6) // 12
        if salidas is None:
            d.text((x, by + 50), "R · vuelve al último save", fill=DORADO, font=fuente(34, True), anchor="lm")
            return img
        z = salidas - salidas.max()
        p = np.exp(z) / np.exp(z).sum()
        for k, (teclas, v) in enumerate(zip(ACTIONS, p)):
            bx = x + k * (bw + 6)
            alto = int(110 * v)
            d.rectangle((bx, by + 110 - alto, bx + bw, by + 110), fill=DORADO if k == accion else AZUL)
            d.text((bx + bw / 2, by + 120), nombre_accion(teclas), fill=TEXTO if k == accion else TEXTO_2,
                   font=self.f_barra, anchor="ma")
        return img

    def final(self, p: Partida, titulo: str, nota: str = "") -> Image.Image:
        img, d = self._base(titulo, self.jugador.emu.observe())
        texto = "TIEMPO" if p.fin == "tiempo" and self.cfg.reintentos else FINES[p.fin]
        d.text((24 + self.juego_w // 2, 80 + self.juego_h // 2), texto, fill=AQUA if p.fin == "llegó" else ROJO,
               font=fuente(80, True), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
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


def en_sala(corrida: str, sala: str | None, objetivo: str | None = None,
            max_segundos: float | None = None) -> tuple[Path, Config, Jugador, str]:
    """Jugador para la sala pedida (la de entrenamiento si es None) y el texto para el título."""
    carpeta = almacen.dir_corrida(corrida)
    cfg, mapa, entrenada = config_en_sala(carpeta, almacen.cargar_config(carpeta), sala, objetivo)
    if max_segundos:
        cfg.max_segundos = max_segundos
    donde = "" if entrenada else f" · en {cfg.start_room} (no la vio al entrenar)"
    return carpeta, cfg, Jugador(cfg, mapa), donde


def video_mejor(corrida: str, gen: int = -1, espera: int = 0, salida: Path | None = None, sala: str | None = None,
                objetivo: str | None = None, max_segundos: float | None = None) -> Path:
    carpeta, cfg, jugador, donde = en_sala(corrida, sala, objetivo, max_segundos)
    gen, genoma, fitness = red_de(carpeta, gen)
    nombre = f"gen_{gen:04d}_mejor" + (f"_{cfg.start_room}" if donde else "") + (f"_espera{espera}" if espera else "")
    salida = salida or carpeta / "videos" / f"{nombre}.mp4"
    video = Video(salida, fps=FPS_VIDEO)
    pantalla = Pantalla(cfg, jugador, carpeta)
    pantalla.red(gen, fitness)
    titulo = f"{corrida} · la red que llegó más lejos en la generación {gen}{donde}"
    detalle = f"Espera inicial {espera} frames" if espera else ""
    ritmo = Ritmo()

    def al_frame(res, a, s):
        n = ritmo.veces(res)
        if n:
            img = pantalla.cuadro(res, a, s, titulo, detalle)
            for _ in range(n):
                video.escribir(img)

    try:
        p = jugador.jugar(genoma, Escenario(espera), al_frame=al_frame)
        img = pantalla.final(p, titulo)
        for _ in range(2 * FPS_VIDEO):
            video.escribir(img)
    finally:
        video.cerrar()
        jugador.close()
    return salida


class _Cerrada(Exception):
    pass


def vivo_red(corrida: str, sala: str | None = None, velocidad: int = 1) -> None:
    """La mejor red (la que llega más lejos) jugando sin parar; toma la nueva entre partidas."""
    signal.signal(signal.SIGTERM, _terminar)
    carpeta, cfg, jugador, donde = en_sala(corrida, sala)
    pantalla = Pantalla(cfg, jugador, carpeta)
    fps = FPS_VIDEO
    titulo_ventana = f"{corrida} · red en vivo" + (f" (x{velocidad})" if velocidad > 1 else "")
    ventana = Ventana(titulo_ventana, fps=fps)
    print(f"en vivo: {corrida}{donde} (tope de memoria "
          f"{LIMITE_VIVO if os.environ.get('IWBTGB_VIVO_SCOPE') else 'ninguno'})", flush=True)
    inicio = ultimo_reporte = time.monotonic()
    gen, genoma, n = -1, None, 0
    historial: list[str] = []

    def mostrar(img: Image.Image) -> None:
        if not ventana.escribir(img):
            raise _Cerrada

    try:
        while ventana.abierta():
            ahora = time.monotonic()
            if ahora - ultimo_reporte >= REPORTE_CADA:
                ultimo_reporte = ahora
                print(f"en vivo: {(ahora - inicio) / 3600:.1f} h · gen {gen} · {n} partidas · memoria "
                      f"{memoria_mb():.0f} MB + ffplay {ventana.memoria_mb():.0f} MB · libre en el sistema "
                      f"{memoria_libre_mb() / 1024:.1f} GB", flush=True)
            if ventana.memoria_mb() > LIMITE_FFPLAY:
                print(f"en vivo: ffplay usa {ventana.memoria_mb():.0f} MB; se reinicia la ventana", flush=True)
                ventana.cerrar()
                ventana = Ventana(titulo_ventana, fps=fps)
            try:
                g, genoma_nuevo, fitness = red_de(carpeta, -1)
                if g != gen:
                    pantalla.red(g, fitness)
                    gen, genoma = g, genoma_nuevo
            except (OSError, ValueError, EOFError, KeyError, IndexError):
                if genoma is None:
                    time.sleep(1)
                    continue
            n += 1
            espera = 0 if n == 1 else random.randint(0, cfg.espera_max)
            pantalla.nueva_partida()
            titulo = f"{corrida} · la red que llegó más lejos en la generación {gen}{donde}"
            detalle = f"Partida {n}" + (f" · espera inicial {espera} frames" if espera else "")
            ritmo = Ritmo(2 * velocidad)

            def al_frame(res, a, s):
                n_cuadros = ritmo.veces(res)
                if n_cuadros:   # a más velocidad, se saltan cuadros (no se muestran más por segundo)
                    img = pantalla.cuadro(res, a, s, titulo, detalle)
                    for _ in range(min(n_cuadros, 3)):
                        mostrar(img)

            p = jugador.jugar(genoma, Escenario(espera, semilla=n), al_frame=al_frame)
            historial = (historial + [f"{100 * p.avance:.0f} %"])[-8:]
            img = pantalla.final(p, titulo, "Últimas partidas: " + "  ".join(historial))
            for _ in range(fps):
                mostrar(img)
    except (KeyboardInterrupt, _Cerrada):
        pass
    finally:
        ventana.cerrar()
        jugador.close()


# -- todas las redes de una generación juntas (con el dibujo del GA) ----------------------------------------

@dataclass
class _Red:
    """Una red con los campos que lee `ga.visualizar.cuadros_generacion` (su primer escenario)."""
    traza: np.ndarray
    fitness: float
    progreso: float
    saves: int
    muertes: int
    frames: int
    llego: bool
    murio: bool

    @property
    def progreso_max(self) -> float:
        return self.progreso

    def clave(self) -> tuple:  # orden de `cuadros_generacion`: la primera es "la mejor" (la que llega más lejos)
        return (self.progreso, self.fitness)


def _cfg_ga(carpeta: Path, cfg: Config) -> SimpleNamespace:
    """Lo que `ga.visualizar` lee de una config del GA, armado desde la de redes (mapa, salida y zonas)."""
    return SimpleNamespace(
        room_entrenamiento=cfg.room_entrenamiento, sala_objetivo=cfg.sala_objetivo, difficulty=cfg.difficulty,
        start_room=cfg.start_room, seed_juego=cfg.seed_juego, max_segundos=cfg.max_segundos,
        recompensa="salida", mapa_distancia=str((carpeta / "distancia.npz").relative_to(ROOT)),
        zonas_prohibidas=";".join(",".join(f"{v:g}" for v in z) for z in cfg.zonas()),
        save_points="sí" if cfg.r_save else "", resets=0)


def poblacion(cfg: Config, d: dict) -> list[_Red]:
    if "trazas" not in d:
        raise ValueError("esta generación no tiene trazas (se guardan desde el 2026-10-26)")
    fin = np.cumsum(d["traza_largo"])
    redes = []
    for i in range(len(d["fitness"])):
        f = str(d["fines"][i][0])
        redes.append(_Red(
            traza=d["trazas"][fin[i] - d["traza_largo"][i]:fin[i]], fitness=float(d["fitness"][i]),
            progreso=float(d["avances"][i][0]), saves=int(d["saves"][i][0]) if "saves" in d else 0,
            muertes=int(d["muertes"][i][0]) if "muertes" in d else 0, frames=int(d["traza_largo"][i]),
            llego=f == "llegó", murio=f in ("murió", "salió", "estancado") and not cfg.reintentos))
    return redes


def cuadros_todas(corrida: str, gen: int, velocidad: int = 1, en_vivo: bool = False):
    carpeta = almacen.dir_corrida(corrida)
    cfg = almacen.cargar_config(carpeta)
    gens = almacen.generaciones_guardadas(carpeta)
    gen = gens[gen] if gen < 0 else gen
    pob = poblacion(cfg, almacen.cargar_generacion(carpeta, gen))
    return cuadros_generacion(carpeta, _cfg_ga(carpeta, cfg), gen, pob, velocidad, en_vivo=en_vivo)


def video_generacion(corrida: str, gen: int = -1, velocidad: int = 1) -> Path:
    carpeta = almacen.dir_corrida(corrida)
    gens = almacen.generaciones_guardadas(carpeta)
    gen = gens[gen] if gen < 0 else gen
    video = Video(carpeta / "videos" / f"gen_{gen:04d}_todas.mp4")
    try:
        for img in cuadros_todas(corrida, gen, velocidad):
            video.escribir(img)
    finally:
        video.cerrar()
    return video.ruta


def vivo(corrida: str, velocidad: int = 4) -> None:
    """Como el en vivo del GA: todas las redes de la última generación guardada, y salta a la más nueva."""
    signal.signal(signal.SIGTERM, _terminar)
    carpeta = almacen.dir_corrida(corrida)
    titulo = f"{corrida} · todas las redes en vivo (x{velocidad})"
    ventana = Ventana(titulo)
    mostrada = -1
    inicio = ultimo_reporte = time.monotonic()
    print(f"en vivo: {corrida} (tope de memoria {LIMITE_VIVO if os.environ.get('IWBTGB_VIVO_SCOPE') else 'ninguno'})",
          flush=True)
    try:
        while ventana.abierta():
            ahora = time.monotonic()
            if ahora - ultimo_reporte >= REPORTE_CADA:
                ultimo_reporte = ahora
                print(f"en vivo: {(ahora - inicio) / 3600:.1f} h · gen {mostrada} · memoria {memoria_mb():.0f} MB "
                      f"+ ffplay {ventana.memoria_mb():.0f} MB · libre en el sistema "
                      f"{memoria_libre_mb() / 1024:.1f} GB", flush=True)
            if ventana.memoria_mb() > LIMITE_FFPLAY:
                print(f"en vivo: ffplay usa {ventana.memoria_mb():.0f} MB; se reinicia la ventana", flush=True)
                ventana.cerrar()
                ventana = Ventana(titulo)
            gens = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
            if not gens or gens[-1] == mostrada:
                time.sleep(1)
                continue
            mostrada = gens[-1]
            try:  # si se guarda justo mientras se lee, se reintenta en la próxima vuelta (los cuadros salen de a uno)
                cuadros = cuadros_todas(corrida, mostrada, velocidad, en_vivo=True)
            except (OSError, ValueError, EOFError, KeyError) as e:
                if isinstance(e, ValueError) and "trazas" in str(e):
                    raise SystemExit(f"{corrida}: {e}. Usa «vivo-red» (la mejor red jugando).")
                mostrada = -1
                time.sleep(1)
                continue
            for img in cuadros:
                if not ventana.escribir(img):
                    return
            del cuadros
    except KeyboardInterrupt:
        pass
    finally:
        ventana.cerrar()


def graficos(corrida: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = almacen.dir_corrida(corrida)
    m = almacen.leer_metricas(carpeta)
    if not m:
        raise SystemExit(f"{corrida} todavía no tiene generaciones")
    destino = carpeta / "graficos"
    destino.mkdir(exist_ok=True)
    hex_ = lambda c: "#%02x%02x%02x" % c
    plt.rcParams.update({
        "figure.facecolor": hex_(FONDO), "axes.facecolor": hex_(FONDO), "savefig.facecolor": hex_(FONDO),
        "text.color": hex_(TEXTO), "axes.labelcolor": hex_(TEXTO_2), "xtick.color": hex_(TEXTO_2),
        "ytick.color": hex_(TEXTO_2), "axes.edgecolor": "#3a3a38", "grid.color": "#2e2e2c",
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 13,
        "lines.linewidth": 2, "font.family": "sans-serif",
        "font.sans-serif": ["Adwaita Sans", "FreeSans", "DejaVu Sans"],
    })
    g = [r["gen"] for r in m]
    salidas = []

    def guardar(fig, nombre):
        ruta = destino / nombre
        fig.tight_layout()
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        salidas.append(ruta)
        print(f"gráfico: {ruta}", flush=True)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["avance_max"] for r in m], color=hex_(AZUL), label="lo más lejos en algún escenario")
    ax.plot(g, [r["avance_mejor"] for r in m], color=hex_(AQUA), label="la mejor red (promedio de escenarios)")
    ax.plot(g, [r["avance_media"] for r in m], color=hex_(NARANJA), label="media de la población")
    ax.set_title("Avance hacia la salida", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("% del camino")
    ax.set_ylim(0, max(5, max(r["avance_max"] for r in m) * 1.1))
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "avance.png")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["fitness_max"] for r in m], color=hex_(AZUL), label="máximo")
    ax.plot(g, [r["fitness_media"] for r in m], color=hex_(NARANJA), label="media")
    ax.set_title("Fitness por generación", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("fitness")
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "fitness.png")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    claves = [("murio_pct", "murió", ROJO), ("estancado_pct", "se estancó", GRIS), ("tiempo_pct", "tiempo", AZUL),
              ("salio_pct", "salió de la sala", DORADO)]
    ax.stackplot(g, *[[r[k] for r in m] for k, _, _ in claves], labels=[e for _, e, _ in claves],
                 colors=[hex_(c) for _, _, c in claves], alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(g, [r["nichos"] for r in m], color=hex_(TEXTO), linewidth=1.5, label="nichos (eje derecho)")
    ax2.set_ylabel("nichos")
    ax2.grid(False)
    ax.set_title("Cómo terminan las partidas", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("% de las partidas")
    ax.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", ncols=3)
    guardar(fig, "fines.png")

    if any("muertes_media" in r for r in m):
        con = [r for r in m if "muertes_media" in r]
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot([r["gen"] for r in con], [r["muertes_media"] for r in con], color=hex_(ROJO),
                label="muertes por partida (media)")
        ax.set_xlabel("generación")
        ax.set_ylabel("muertes")
        ax2 = ax.twinx()
        if any("quieto_pct" in r for r in con):
            ax.plot([r["gen"] for r in con], [r.get("quieto_pct", np.nan) / 10 for r in con], color=hex_(GRIS),
                    label="tiempo quieto (% / 10)")
        ax2.plot([r["gen"] for r in con], [r["saves_media"] for r in con], color=hex_(AQUA), label="saves (media)")
        ax2.plot([r["gen"] for r in con], [r["saves_max"] for r in con], color=hex_(DORADO), linewidth=1.5,
                 label="saves (máximo)")
        ax2.set_ylabel("saves")
        if any("celdas_media" in r for r in con):
            ax2.plot([r["gen"] for r in con], [r.get("celdas_media", np.nan) / 10 for r in con], color=hex_(NARANJA),
                     label="celdas exploradas (media / 10)")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
        ax.set_title("Muertes y saves", loc="left", fontsize=18)
        guardar(fig, "muertes_saves.png")
    return salidas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["mejor", "generacion", "vivo", "vivo-red", "graficos"])
    p.add_argument("--corrida", required=True)
    p.add_argument("--gen", type=int, default=-1)
    p.add_argument("--espera", type=int, default=0)
    p.add_argument("--sala", default=None, help="otra sala (rStage02 …); por defecto, la de entrenamiento")
    p.add_argument("--velocidad", type=int, default=None, help="por defecto 4 en «vivo» y 1 en lo demás")
    a = p.parse_args()
    if a.modo == "mejor":
        video_mejor(a.corrida, a.gen, a.espera, sala=a.sala)
    elif a.modo == "generacion":
        video_generacion(a.corrida, a.gen, a.velocidad or 1)
    elif a.modo == "vivo":
        _con_tope_de_memoria("redes.visualizar")
        vivo(a.corrida, a.velocidad or 4)
    elif a.modo == "vivo-red":
        _con_tope_de_memoria("redes.visualizar")
        vivo_red(a.corrida, a.sala, a.velocidad or 1)
    else:
        graficos(a.corrida)


if __name__ == "__main__":
    main()
