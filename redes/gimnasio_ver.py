"""Ver el gimnasio: muestras de salas, una política jugándolas en vivo (ventana) y videos.

    uv run python -m redes.gimnasio_ver muestras [--dificultad 0.5] [--semilla 0] [--salas 12] [--verificar]
    uv run python -m redes.gimnasio_ver vivo  --agente red07 [--dificultad 0.5] [--semilla 0] [--velocidad 1]
    uv run python -m redes.gimnasio_ver video --agente solucion [--dificultad -1] [--salas 10]

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

from ga.visualizar import (H, LIMITE_FFPLAY, REPORTE_CADA, W, Ventana, Video, _con_tope_de_memoria, _terminar,
                           fuente, memoria_libre_mb, memoria_mb)
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

    def __init__(self, agente: str, politica: Politica | None = None):
        self.agente = agente
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

    def nueva_sala(self, sala: Sala, n: int) -> None:
        """Arma la parte fija del cuadro (título, sala, historial, rótulos) una vez por sala: dibujarla en cada
        cuadro costaba ~30 ms y el en vivo no llegaba a tiempo real."""
        self.sala, self.n = sala, n
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
        d.text((24, 26), f"Gimnasio · {self.agente}", fill=TEXTO, font=self.f_titulo)
        return img, d

    def _sala(self, d: ImageDraw.ImageDraw, y: int) -> int:
        s, x = self.sala, self.px
        d.text((x, y), f"Sala {self.n}", fill=TEXTO, font=self.f_grande)
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
    signal.signal(signal.SIGTERM, _terminar)
    titulo = f"Gimnasio · {agente}" + (f" (x{velocidad})" if velocidad > 1 else "")
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
        jugar_salas(agente, dificultad, semilla, None, verificar, mostrar, recargar=True, velocidad=velocidad)
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
    p.add_argument("modo", choices=["muestras", "vivo", "video"])
    p.add_argument("--agente", default="solucion", help="nombre de una corrida, «azar» o «solucion»")
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
    else:
        video(a.agente, a.dificultad, a.semilla, a.salas, a.verificar, a.gen)


if __name__ == "__main__":
    main()
