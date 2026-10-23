"""Videos y gráficos de una corrida del algoritmo genético.

    uv run python -m ga.visualizar generacion --corrida rStage01 --gen 10   # todos los agentes de la gen 10, juntos
    uv run python -m ga.visualizar mejor      --corrida rStage01 --gen 10   # el mejor de la gen 10, partida real
    uv run python -m ga.visualizar evolucion  --corrida rStage01 --cada 5   # el mejor de cada generación, juntos
    uv run python -m ga.visualizar graficos   --corrida rStage01            # PNG con las métricas
    uv run python -m ga.visualizar todo       --corrida rStage01 --cada 10  # todo lo anterior
    uv run python -m ga.visualizar vivo       --corrida salida01            # ventana con la última generación

`--gen -1` es la última generación guardada. Los videos (1920x1080, 50 FPS, sin audio) quedan en
corridas/<nombre>/videos/ y los gráficos en corridas/<nombre>/graficos/. `--velocidad 2` acelera al doble.

`vivo` sirve mientras se entrena (en otra terminal): muestra en una ventana (ffplay) todos los agentes de la
última generación guardada, acelerados (`--velocidad`, 4 por defecto); al terminar salta a la más nueva,
saltándose las intermedias. Se cierra cerrando la ventana o con Ctrl+C.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iwbtgb_rl import ACTIONS, IWannaBeTheGBEnv

from . import almacen, recursos
from .nucleo import TECLA_R, FPS, ROOT, Config, Resultado, distancias, muertes_en, posicion_final, zonas

W, H = 1920, 1080
MAPA_W = 1440                      # el mapa ocupa la izquierda; a la derecha va el panel
PANEL_X = MAPA_W
FONDO = (26, 26, 25)               # superficie oscura
TEXTO = (240, 239, 236)
TEXTO_2 = (195, 194, 183)
TEXTO_3 = (130, 129, 122)
AZUL = (57, 135, 229)              # serie 1
NARANJA = (217, 89, 38)            # serie 2
AQUA = (25, 158, 112)
ROJO = (230, 103, 103)
DORADO = (237, 161, 0)


def fuente(tam: int, negrita: bool = False, mono: bool = False):
    rutas = (
        ["/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf" if negrita else "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"]
        if mono else
        ["/usr/share/fonts/gnu-free/FreeSansBold.otf" if negrita else "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
         "/usr/share/fonts/gnu-free/FreeSans.otf"]
    )
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except OSError:
            continue
    return ImageFont.load_default(tam)


def plural(n: int, palabra: str) -> str:
    return f"{n} {palabra}" if n == 1 else f"{n} {palabra}s"


def tiempo(frames: int) -> str:
    s = frames / FPS
    return f"{int(s // 60)}:{s % 60:05.2f}"


class Ventana:
    """Muestra frames RGB en una ventana de ffplay (en vez de grabarlos)."""

    def __init__(self, titulo: str, fps: int = FPS, ancho: int = 1280):
        if shutil.which("ffplay") is None:
            raise SystemExit("ffplay no está instalado (viene con ffmpeg)")
        self.proc = subprocess.Popen(
            ["ffplay", "-loglevel", "error", "-window_title", titulo, "-x", str(ancho), "-y", str(ancho * H // W),
             "-fflags", "nobuffer", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
             "-framerate", str(fps), "-i", "-"],
            stdin=subprocess.PIPE,
        )

    def escribir(self, img: Image.Image) -> bool:
        """False si el usuario cerró la ventana."""
        try:
            self.proc.stdin.write(img.tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def abierta(self) -> bool:
        return self.proc.poll() is None

    def cerrar(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()


class Video:
    """Envía frames RGB a ffmpeg (H.264, yuv420p)."""

    def __init__(self, ruta: Path, fps: int = FPS, ancho: int = W, alto: int = H):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg no está instalado")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta = ruta
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{ancho}x{alto}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(ruta)],
            stdin=subprocess.PIPE,
        )

    def escribir(self, img: Image.Image) -> None:
        self.proc.stdin.write(img.tobytes())

    def cerrar(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"video: {self.ruta}")


# -- recursos comunes ----------------------------------------------------------------------------------

def con_salida(cfg: Config) -> bool:
    return cfg.recompensa == "salida" and bool(cfg.mapa_distancia) and (ROOT / cfg.mapa_distancia).exists()


def marcar_salida(img: Image.Image, cfg: Config) -> Image.Image:
    """Dibuja la salida (recuadro aqua + "SALIDA") y las zonas prohibidas sobre la imagen completa de la room."""
    rects = zonas(cfg)
    if not con_salida(cfg) and not rects:
        return img
    from . import distancia
    img = img.copy()
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    for x0, y0, x1, y1 in rects:
        x1, y1 = min(img.width, x1) - 1, min(img.height, y1) - 1
        d.rectangle((x0, y0, x1, y1), fill=(220, 40, 40, 45), outline=(220, 40, 40, 200), width=4)
        d.text(((x0 + x1) / 2, y1 - 12), "ZONA PROHIBIDA (cuenta como muerte)", fill=(255, 90, 90, 255),
               font=fuente(18, True), anchor="mb", stroke_width=2, stroke_fill=(0, 0, 0, 255))
    salidas = distancia.cargar(ROOT / cfg.mapa_distancia)["salidas"] if con_salida(cfg) else []
    for x0, y0, x1, y1 in salidas:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width - 1, x1), min(img.height - 1, y1)
        d.rectangle((x0, y0, x1, y1), fill=(*AQUA, 110), outline=(*AQUA, 255), width=3)
        d.text((x1, y0 - 4), "SALIDA", fill=(*AQUA, 255), font=fuente(16, True), anchor="rb",
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB")


def final_muerte(r: Resultado, f: int) -> bool:
    """El último frame de una traza cortada por irse a otra room (cuenta como muerte, aunque esté vivo)."""
    return r.murio and f == len(r.traza) - 1 and bool(r.traza[f]["alive"])


def texto_resultado(r: Resultado, cfg: Config) -> str:
    saves = f" · {plural(r.saves, 'save')}" if cfg.save_points else ""
    if con_salida(cfg):
        return (f"fitness {r.fitness:.0f} · {100 * r.progreso:.0f} % del camino (máx {100 * r.progreso_max:.0f} %)"
                + saves)
    return f"fitness {r.fitness:.1f} · {plural(r.pantallas, 'pantalla')} · {r.celdas} celdas" + saves


def texto_fin(r: Resultado, cfg: Config) -> str:
    if r.llego:
        return "llegó a " + cfg.sala_objetivo
    if r.murio:
        return "descalificado (sin reintentos)" if cfg.resets else "murió"
    return f"sobrevivió {cfg.max_segundos:.0f} s" + (f" ({plural(r.muertes, 'reintento')})" if r.muertes else "")


class Escena:
    """Mapa de la room de entrenamiento escalado al área izquierda + sprites escalados igual."""

    def __init__(self, carpeta: Path, cfg: Config):
        self.cfg = cfg
        dir_rec = self.dir_rec = carpeta / "recursos"
        mapa = recursos.capturar_mapas(dir_rec, cfg, {cfg.room_entrenamiento})[cfg.room_entrenamiento]
        self.nombre_room = mapa["nombre"]
        img = marcar_salida(mapa["imagen"], cfg)
        self.escala = min(MAPA_W / img.width, H / img.height)
        self.mw, self.mh = int(img.width * self.escala), int(img.height * self.escala)
        self.ox, self.oy = (MAPA_W - self.mw) // 2, (H - self.mh) // 2
        self.mapa = img.resize((self.mw, self.mh), Image.LANCZOS)
        self.sprites = recursos.cargar_sprites(dir_rec)
        self._cache: dict = {}

    def a_pantalla(self, x: float, y: float) -> tuple[int, int]:
        return int(self.ox + x * self.escala), int(self.oy + y * self.escala)

    def fondo(self) -> Image.Image:
        img = Image.new("RGB", (W, H), FONDO)
        img.paste(self.mapa, (self.ox, self.oy))
        return img

    def sprite(self, sprite: int, image: int, xscale: int, alpha: float = 1.0, tinte=None, borde=None,
               escala: float | None = None):
        """Sprite escalado (RGBA) y su desplazamiento respecto de (x, y) en pantalla."""
        escala = self.escala if escala is None else escala
        clave = (sprite, image, xscale, alpha, tinte, borde, escala)
        if clave in self._cache:
            return self._cache[clave]
        s = self.sprites.get(sprite) or self.sprites[min(self.sprites)]
        img = s["imagenes"][image % len(s["imagenes"])]
        xo = s["xorig"]
        if xscale < 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            xo = img.width - xo
        arr = np.asarray(img).astype(np.float32)
        if tinte is not None:
            gris = arr[..., :3].mean(-1, keepdims=True) / 255.0
            arr[..., :3] = 0.35 * arr[..., :3] + 0.65 * gris * np.array(tinte, np.float32) * 1.4
        if borde is not None:
            # contorno de 1 px alrededor de los pixeles opacos
            a = arr[..., 3] > 0
            pad = np.pad(a, 1)
            vecino = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
            anillo = vecino & ~a
            arr[anillo] = [*borde, 255]
        arr[..., 3] *= alpha
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        e = max(escala, 0.5)
        tam = (max(1, round(out.width * e)), max(1, round(out.height * e)))
        out = out.resize(tam, Image.NEAREST)
        res = (out, round(xo * e), round(s["yorig"] * e))
        self._cache[clave] = res
        return res

    def pegar(self, lienzo: Image.Image, reg, a_pantalla=None, escala: float | None = None, **kw) -> None:
        spr, dx, dy = self.sprite(int(reg["sprite"]), int(reg["image"]), int(reg["xscale"]) or 1, escala=escala, **kw)
        px, py = (a_pantalla or self.a_pantalla)(float(reg["x"]), float(reg["y"]))
        lienzo.paste(spr, (px - dx, py - dy), spr)


class Camara:
    """Vista como la del juego: una pantalla de 800x600 (en pasos de 800x608) ampliada a 1440x1080."""

    VW, VH, PASO_Y = 800, 600, 608

    def __init__(self, esc: Escena):
        self.esc = esc
        self.z = MAPA_W / self.VW                      # 1.8
        self.mapa_full = marcar_salida(recursos.capturar_mapas(esc.dir_rec, esc.cfg, {esc.cfg.room_entrenamiento})[
            esc.cfg.room_entrenamiento]["imagen"], esc.cfg)
        self.nx = -(-self.mapa_full.width // self.VW)
        self.ny = -(-self.mapa_full.height // self.PASO_Y)
        self.bases: dict[tuple[int, int], Image.Image] = {}
        self.actual = (0, 0)
        self._candidata, self._cuenta = None, 0

    def pantalla_de(self, x: float, y: float) -> tuple[int, int]:
        return (int(min(max(x, 0), self.mapa_full.width - 1) // self.VW),
                int(min(max(y, 0), self.mapa_full.height - 1) // self.PASO_Y))

    def base(self, pantalla) -> Image.Image:
        if pantalla not in self.bases:
            sx, sy = pantalla
            vx, vy = sx * self.VW, sy * self.PASO_Y
            recorte = Image.new("RGB", (self.VW, self.VH), (0, 0, 0))
            recorte.paste(self.mapa_full.crop((vx, vy, vx + self.VW, vy + self.VH)), (0, 0))
            self.bases[pantalla] = recorte.resize((MAPA_W, H), Image.NEAREST)
        return self.bases[pantalla]

    def a_pantalla(self, x: float, y: float) -> tuple[int, int]:
        sx, sy = self.actual
        return int((x - sx * self.VW) * self.z), int((y - sy * self.PASO_Y) * self.z)

    def marcar_muerte(self, x: float, y: float, color=ROJO) -> None:
        p = self.pantalla_de(x, y)
        sx, sy = p
        d = ImageDraw.Draw(self.base(p))
        marca_muerte(d, int((x - sx * self.VW) * self.z), int((y - sy * self.PASO_Y) * self.z), r=9, color=color)

    def seguir(self, objetivo: tuple[int, int], vacia: bool) -> None:
        """Cambia de pantalla si el objetivo se mantiene 15 frames (o de inmediato si la actual quedó vacía)."""
        if objetivo == self.actual:
            self._candidata, self._cuenta = None, 0
            return
        if objetivo == self._candidata:
            self._cuenta += 1
        else:
            self._candidata, self._cuenta = objetivo, 1
        if vacia or self._cuenta >= 15:
            self.actual, self._candidata, self._cuenta = objetivo, None, 0


class Minimapa:
    """Mapa completo de la room, chico, para el panel."""

    def __init__(self, cam: Camara, ancho: int):
        self.cam = cam
        self.k = ancho / cam.mapa_full.width
        self.base = cam.mapa_full.resize((ancho, int(cam.mapa_full.height * self.k)), Image.LANCZOS)
        self.base = Image.blend(self.base, Image.new("RGB", self.base.size, FONDO), 0.35)

    def punto(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.k), int(y * self.k)

    def marcar_muerte(self, x: float, y: float) -> None:
        px, py = self.punto(x, y)
        ImageDraw.Draw(self.base).ellipse((px - 2, py - 2, px + 2, py + 2), fill=ROJO)

    def dibujar(self, lienzo: Image.Image, ox: int, oy: int, puntos, mejor=None) -> None:
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        sx, sy = self.cam.actual
        d.rectangle((sx * Camara.VW * self.k, sy * Camara.PASO_Y * self.k,
                     (sx * Camara.VW + Camara.VW) * self.k - 1, (sy * Camara.PASO_Y + Camara.VH) * self.k - 1),
                    outline=TEXTO, width=2)
        for x, y in puntos:
            px, py = self.punto(x, y)
            d.rectangle((px - 1, py - 1, px + 1, py + 1), fill=TEXTO)
        if mejor is not None:
            px, py = self.punto(*mejor)
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=DORADO)
        lienzo.paste(img, (ox, oy))


def marca_muerte(draw: ImageDraw.ImageDraw, x: int, y: int, r: int = 5, color=ROJO) -> None:
    draw.line((x - r, y - r, x + r, y + r), fill=color, width=3)
    draw.line((x - r, y + r, x + r, y - r), fill=color, width=3)


def mini_grafico(draw: ImageDraw.ImageDraw, caja, metricas: list[dict], hasta_gen: int, f_chica) -> None:
    """Fitness máximo y medio por generación hasta `hasta_gen` (línea azul = máx, naranja = media)."""
    x0, y0, x1, y1 = caja
    filas = [m for m in metricas if m["gen"] <= hasta_gen]
    if not filas:
        return
    gens = [m["gen"] for m in metricas]
    gmax = max(gens) or 1
    vmax = max(m["fitness_max"] for m in metricas) or 1
    vmin = min(0.0, min(m["fitness_media"] for m in metricas))
    for k in range(5):
        yy = y1 - (y1 - y0) * k / 4
        draw.line((x0, yy, x1, yy), fill=(45, 45, 43), width=1)
        draw.text((x0 - 8, yy), f"{vmin + (vmax - vmin) * k / 4:.0f}", fill=TEXTO_3, font=f_chica, anchor="rm")

    def pts(clave):
        return [(x0 + (x1 - x0) * m["gen"] / gmax, y1 - (y1 - y0) * (m[clave] - vmin) / (vmax - vmin or 1))
                for m in filas]

    for clave, color in (("fitness_media", NARANJA), ("fitness_max", AZUL)):
        p = pts(clave)
        if len(p) > 1:
            draw.line(p, fill=color, width=3, joint="curve")
        draw.ellipse((p[-1][0] - 5, p[-1][1] - 5, p[-1][0] + 5, p[-1][1] + 5), fill=color)
    draw.text((x0, y1 + 10), "0", fill=TEXTO_3, font=f_chica, anchor="lt")
    draw.text((x1, y1 + 10), f"gen {gmax:.0f}", fill=TEXTO_3, font=f_chica, anchor="rt")


def cargar(corrida: str, gen: int):
    carpeta = almacen.dir_corrida(corrida)
    gens = almacen.generaciones_guardadas(carpeta)
    if not gens:
        raise SystemExit(f"{carpeta} no tiene generaciones")
    if gen < 0:
        gen = gens[gen]
    return carpeta, almacen.cargar_config(carpeta), gen, almacen.cargar_generacion(carpeta, gen)


# -- video: todos los agentes de una generación ---------------------------------------------------------

def video_generacion(corrida: str, gen: int, velocidad: int = 1, salida: Path | None = None) -> Path:
    carpeta, cfg, gen, pob = cargar(corrida, gen)
    salida = salida or carpeta / "videos" / f"gen_{gen:04d}_todos.mp4"
    video = Video(salida)
    for img in cuadros_generacion(carpeta, cfg, gen, pob, velocidad):
        video.escribir(img)
    video.cerrar()
    return salida


def vivo(corrida: str, velocidad: int = 4) -> None:
    """Ventana que reproduce la última generación guardada y salta a la siguiente cuando aparece."""
    import time
    carpeta = almacen.dir_corrida(corrida)
    ventana = Ventana(f"{corrida} · en vivo (x{velocidad})")
    mostrada = -1
    try:
        while ventana.abierta():
            gens = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
            if not gens or gens[-1] == mostrada:
                time.sleep(1)
                continue
            mostrada = gens[-1]
            # Si se guarda justo mientras se lee, se reintenta en la próxima vuelta
            try:
                _, cfg, gen, pob = cargar(corrida, mostrada)
            except (OSError, ValueError, EOFError):
                mostrada = -1
                time.sleep(1)
                continue
            for img in cuadros_generacion(carpeta, cfg, gen, pob, velocidad, en_vivo=True):
                if not ventana.escribir(img):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        ventana.cerrar()


def cuadros_generacion(carpeta: Path, cfg: Config, gen: int, pob: list[Resultado], velocidad: int = 1,
                       en_vivo: bool = False):
    """Genera los cuadros (1920x1080) con todos los agentes de una generación jugando a la vez."""
    esc = Escena(carpeta, cfg)
    cam = Camara(esc)
    metricas = almacen.leer_metricas(carpeta)
    orden = sorted(range(len(pob)), key=lambda i: pob[i].clave(), reverse=True)
    mejor = orden[0]
    largo = max(len(r.traza) for r in pob)
    sala = cfg.room_entrenamiento

    f_titulo, f_grande, f_med, f_chica = fuente(54, True), fuente(40, True, True), fuente(24), fuente(18)
    f_num = fuente(30, True, True)
    x = PANEL_X + 36
    mm = Minimapa(cam, W - PANEL_X - 72)
    # Parte fija del panel
    panel = Image.new("RGB", (W - PANEL_X, H), FONDO)
    dp = ImageDraw.Draw(panel)
    px0 = 36
    dp.text((px0, 34), "Generación", fill=TEXTO_2, font=f_med)
    dp.text((px0, 60), f"{gen}", fill=TEXTO, font=f_titulo)
    dp.text((px0, 146), "Tiempo", fill=TEXTO_2, font=f_med)
    dp.text((px0, 240), "Vivos", fill=TEXTO_2, font=f_med)
    dp.text((px0 + 220, 240), "Muertes", fill=TEXTO_2, font=f_med)
    r = pob[mejor]
    dp.text((px0, 370), "Mejor de la generación", fill=DORADO, font=f_med)
    dp.text((px0, 400), texto_resultado(r, cfg), fill=TEXTO, font=f_chica)
    dp.text((px0, 426), f"{texto_fin(r, cfg)} · {tiempo(r.frames)}", fill=TEXTO, font=f_chica)
    dp.text((px0, 474), f"Mapa de {esc.nombre_room}", fill=TEXTO_2, font=f_med)
    mm_y = 508
    graf_y = mm_y + mm.base.height + 40
    dp.text((px0, graf_y), "Fitness por generación", fill=TEXTO_2, font=f_med)
    dp.line((px0 + 280, graf_y + 14, px0 + 300, graf_y + 14), fill=AZUL, width=4)
    dp.text((px0 + 306, graf_y + 14), "máx", fill=TEXTO_2, font=f_chica, anchor="lm")
    dp.line((px0 + 350, graf_y + 14, px0 + 370, graf_y + 14), fill=NARANJA, width=4)
    dp.text((px0 + 376, graf_y + 14), "media", fill=TEXTO_2, font=f_chica, anchor="lm")
    mini_grafico(dp, (px0 + 50, graf_y + 44, W - PANEL_X - 30, H - 50), metricas, gen, f_chica)

    vivo_antes = [True] * len(pob)
    llegado = [False] * len(pob)
    n_muertes = llegados = 0
    estela: list[tuple[float, float]] = []
    ultima = gen
    for f in list(range(0, largo, velocidad)) + [largo - 1] * FPS:
        # Quién está vivo en la sala y dónde
        vivos_reg, n_vivos = {}, 0
        for i in orden:
            t = pob[i].traza
            if f >= len(t):
                continue
            reg = t[f]
            if not reg["alive"] or final_muerte(pob[i], f):
                if vivo_antes[i] and f > 0:          # acaba de morir (con reintentos, reaparece después)
                    vivo_antes[i] = False
                    n_muertes += 1
                    ant = t[f - 1] if not reg["alive"] else reg
                    if ant["room"] == sala:
                        cam.marcar_muerte(float(ant["x"]), float(ant["y"]) - 8)
                        mm.marcar_muerte(float(ant["x"]), float(ant["y"]) - 8)
                continue
            vivo_antes[i] = True
            if f == len(t) - 1 and pob[i].llego and not llegado[i]:
                llegado[i] = True
                llegados += 1
            n_vivos += 1
            if reg["room"] == sala:
                vivos_reg[i] = reg
        # Cámara: la pantalla del mejor mientras vive; si no, la que tiene más agentes
        conteo: dict[tuple[int, int], int] = {}
        for reg in vivos_reg.values():
            p = cam.pantalla_de(float(reg["x"]), float(reg["y"]))
            conteo[p] = conteo.get(p, 0) + 1
        if mejor in vivos_reg:
            objetivo = cam.pantalla_de(float(vivos_reg[mejor]["x"]), float(vivos_reg[mejor]["y"]))
        elif conteo:
            objetivo = max(conteo, key=conteo.get)
        else:
            objetivo = cam.actual
        cam.seguir(objetivo, conteo.get(cam.actual, 0) == 0)

        img = Image.new("RGB", (W, H), FONDO)
        img.paste(cam.base(cam.actual), (0, 0))
        d = ImageDraw.Draw(img)
        sx, sy = cam.actual
        for i in orden[::-1]:                        # el mejor al final, encima de todos
            reg = vivos_reg.get(i)
            if reg is None or cam.pantalla_de(float(reg["x"]), float(reg["y"])) != cam.actual:
                continue
            if i == mejor:
                continue
            esc.pegar(img, reg, a_pantalla=cam.a_pantalla, escala=cam.z, alpha=0.55)
        if mejor in vivos_reg:
            reg = vivos_reg[mejor]
            estela.append((float(reg["x"]), float(reg["y"]) - 10))
            pts = [cam.a_pantalla(px, py) for px, py in estela[-100:]
                   if cam.pantalla_de(px, py + 10) == cam.actual]
            if len(pts) > 1:
                d.line(pts, fill=DORADO, width=3)
            if cam.pantalla_de(float(reg["x"]), float(reg["y"])) == cam.actual:
                esc.pegar(img, reg, a_pantalla=cam.a_pantalla, escala=cam.z, borde=DORADO)

        img.paste(panel, (PANEL_X, 0))
        d.text((x, 172), tiempo(min(f + 1, largo)), fill=TEXTO, font=f_grande)
        d.text((x, 266), f"{n_vivos} / {len(pob)}", fill=TEXTO, font=f_num)
        d.text((x + 220, 266), f"{n_muertes}", fill=ROJO, font=f_num)
        if llegados:
            d.text((x, 320), f"Llegaron a {cfg.sala_objetivo}: {llegados}", fill=AQUA, font=f_med)
        mm.dibujar(img, x, mm_y, [(float(r_["x"]), float(r_["y"])) for i, r_ in vivos_reg.items() if i != mejor],
                   (float(vivos_reg[mejor]["x"]), float(vivos_reg[mejor]["y"])) if mejor in vivos_reg else None)
        if en_vivo:
            if f % (FPS * velocidad) == 0:
                ultima = (almacen.generaciones_guardadas(carpeta) or [gen])[-1]
            d.rounded_rectangle((W - 236, 28, W - 36, 68), radius=10, fill=(150, 30, 30))
            d.text((W - 136, 48), f"EN VIVO · x{velocidad}", fill=TEXTO, font=f_chica, anchor="mm")
            if ultima > gen:
                d.text((W - 36, 88), f"entrenando: gen {ultima}", fill=TEXTO_2, font=f_chica, anchor="ra")
        yield img


# -- video: el mejor de una generación, jugando de verdad -------------------------------------------------

def repetir(cfg: Config, r: Resultado, extra: int = FPS):
    """Repite la partida en el emulador dibujando cada frame, con las teclas que quedaron en la traza (incluidos
    los reintentos con R). Entrega (frame RGB, registro, teclas) por frame y al final verifica que la
    repetición coincida con la traza del entrenamiento.
    """
    env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                           seed=cfg.seed_juego)
    emu = env.emu
    emu.track("player")
    emu.load(0)
    distintos = 0
    try:
        for esperado in r.traza:
            a = int(esperado["accion"])
            teclas = ACTIONS[a] if a < len(ACTIONS) else (("r",) if a == TECLA_R else ())
            res = emu.step(teclas, frames=1, obs=True, trace=True)
            reg = res.trace[0]
            if esperado["alive"] and (abs(float(reg["x"]) - float(esperado["x"])) > 1e-3
                                      or abs(float(reg["y"]) - float(esperado["y"])) > 1e-3):
                distintos += 1
            yield res.obs, esperado, teclas
        for _ in range(extra):                       # la muerte o la llegada, sin teclas
            yield emu.step((), frames=1, obs=True).obs, None, ()
    finally:
        env.close()
        if distintos:
            print(f"AVISO: {distintos} frames de la repetición no coinciden con la traza del entrenamiento")


def video_mejor(corrida: str, gen: int, salida: Path | None = None) -> Path:
    carpeta, cfg, gen, pob = cargar(corrida, gen)
    esc = Escena(carpeta, cfg)
    cam = Camara(esc)
    metricas = almacen.leer_metricas(carpeta)
    r = max(pob, key=Resultado.clave)
    n = len(r.traza)
    salida = salida or carpeta / "videos" / f"gen_{gen:04d}_mejor.mp4"
    video = Video(salida)
    sala = cfg.room_entrenamiento

    f_titulo, f_grande, f_med, f_chica = fuente(54, True), fuente(40, True, True), fuente(24), fuente(18)
    f_tecla = fuente(22, True)
    x = PANEL_X + 36
    mm = Minimapa(cam, W - PANEL_X - 72)
    panel = Image.new("RGB", (W - PANEL_X, H), FONDO)
    dp = ImageDraw.Draw(panel)
    px0 = 36
    dp.text((px0, 34), "Mejor agente · generación", fill=TEXTO_2, font=f_med)
    dp.text((px0, 60), f"{gen}", fill=DORADO, font=f_titulo)
    dp.text((px0, 146), "Tiempo", fill=TEXTO_2, font=f_med)
    salida_on = con_salida(cfg)
    dp.text((px0, 240), "Camino a la salida" if salida_on else "Celdas exploradas", fill=TEXTO_2, font=f_med)
    dp.text((px0, 330), texto_resultado(r, cfg), fill=TEXTO_2, font=f_chica)
    dp.text((px0, 352), texto_fin(r, cfg), fill=TEXTO_2, font=f_chica)
    vivo = r.traza["alive"].astype(np.int8)
    muertes_hasta = np.concatenate([[0], np.cumsum((vivo[:-1] == 1) & (vivo[1:] == 0))])
    dp.text((px0, 474), "Recorrido", fill=TEXTO_2, font=f_med)
    mm_y = 508
    graf_y = mm_y + mm.base.height + 40
    dp.text((px0, graf_y), "Fitness por generación", fill=TEXTO_2, font=f_med)
    dp.line((px0 + 280, graf_y + 14, px0 + 300, graf_y + 14), fill=AZUL, width=4)
    dp.text((px0 + 306, graf_y + 14), "máx", fill=TEXTO_2, font=f_chica, anchor="lm")
    dp.line((px0 + 350, graf_y + 14, px0 + 370, graf_y + 14), fill=NARANJA, width=4)
    dp.text((px0 + 376, graf_y + 14), "media", fill=TEXTO_2, font=f_chica, anchor="lm")
    mini_grafico(dp, (px0 + 50, graf_y + 44, W - PANEL_X - 30, H - 50), metricas, gen, f_chica)
    teclas_def = (("left", "←", 0, 56), ("right", "→", 66, 56), ("shift", "Salto", 132, 110), ("z", "Disparo", 252, 120))

    celdas_vistas: set = set()
    if salida_on:
        from . import distancia
        d0 = float(distancia.cargar(ROOT / cfg.mapa_distancia)["d0"])
        avance = 0.0
    recorrido: list[tuple[float, float]] = []
    ultimo = None
    for f, (obs, reg, teclas) in enumerate(repetir(cfg, r)):
        img = Image.new("RGB", (W, H), FONDO)
        img.paste(Image.fromarray(obs).resize((MAPA_W, H), Image.NEAREST), (0, 0))
        img.paste(panel, (PANEL_X, 0))
        d = ImageDraw.Draw(img)
        if reg is not None and reg["alive"] and reg["room"] == sala:
            celdas_vistas.add((int(reg["x"]) // cfg.celda, int(reg["y"]) // cfg.celda))
            ultimo = (float(reg["x"]), float(reg["y"]))
            recorrido.append(ultimo)
            cam.actual = cam.pantalla_de(*ultimo)
            if salida_on:
                avance = 1.0 - float(distancias(cfg, np.array([ultimo[0]]), np.array([ultimo[1]]))[0]) / d0
        d.text((x, 172), tiempo(min(f + 1, n)), fill=TEXTO, font=f_grande)
        if salida_on:
            if f >= n and r.llego:
                avance = 1.0
            d.text((x, 266), f"{100 * avance:.1f} %", fill=TEXTO, font=f_grande)
        else:
            d.text((x, 266), f"{len(celdas_vistas)}", fill=TEXTO, font=f_grande)
        for nombre, etiqueta, bx, bw in teclas_def:
            on = nombre in teclas
            d.rounded_rectangle((x + bx, 384, x + bx + bw, 430), radius=8, fill=AZUL if on else (45, 45, 43),
                                outline=AZUL if on else TEXTO_3)
            d.text((x + bx + bw / 2, 407), etiqueta, fill=TEXTO if on else TEXTO_3, font=f_tecla, anchor="mm")
        if cfg.resets:
            k = int(muertes_hasta[min(f, n - 1)])
            d.text((x, 444), f"Intento {min(k + 1, cfg.resets + 1)} de {cfg.resets + 1}"
                   + ("   · R" if "r" in teclas else ""), fill=TEXTO_2 if k == 0 else NARANJA, font=f_med)
        mm.dibujar(img, x, mm_y, [], ultimo)
        if len(recorrido) > 1:
            d.line([(x + px * mm.k, mm_y + py * mm.k) for px, py in recorrido[::2] + recorrido[-1:]],
                   fill=DORADO, width=2)
        if f >= n:
            estado = "¡LLEGÓ!" if r.llego else (("DESCALIFICADO" if cfg.resets else "MURIÓ") if r.murio
                                                else "SE ACABÓ EL TIEMPO")
            d.text((MAPA_W // 2, 80), estado, fill=AQUA if r.llego else ROJO, font=fuente(72, True),
                   anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
        video.escribir(img)
    video.cerrar()
    return salida


# -- video: el mejor de cada generación, juntos --------------------------------------------------------

def color_gen(t: float) -> tuple[int, int, int]:
    """De azul (primeras generaciones) a dorado (últimas)."""
    a, b = np.array(AZUL, float), np.array(DORADO, float)
    return tuple(int(v) for v in a + (b - a) * t)


def video_evolucion(corrida: str, cada: int = 1, velocidad: int = 1, salida: Path | None = None) -> Path:
    carpeta = almacen.dir_corrida(corrida)
    cfg = almacen.cargar_config(carpeta)
    gens = almacen.generaciones_guardadas(carpeta)
    elegidas = sorted(set(gens[::cada]) | {gens[-1]})
    mejores = []
    for g in elegidas:
        pob = almacen.cargar_generacion(carpeta, g)
        mejores.append((g, max(pob, key=Resultado.clave)))
    esc = Escena(carpeta, cfg)
    salida = salida or carpeta / "videos" / f"evolucion_cada{cada}.mp4"
    video = Video(salida)
    largo = max(len(r.traza) for _, r in mejores)
    base = esc.fondo()
    d_base = ImageDraw.Draw(base)
    f_titulo, f_med, f_chica, f_grande = fuente(48, True), fuente(26), fuente(20), fuente(40, True, True)
    g_max = max(elegidas) or 1
    vivo_antes = [True] * len(mejores)
    sala = cfg.room_entrenamiento
    # leyenda de color (estática)
    x = PANEL_X + 36
    d_base.text((x, 40), "El mejor de cada", fill=TEXTO_2, font=f_med)
    d_base.text((x, 72), "generación", fill=TEXTO, font=f_titulo)
    for k in range(300):
        d_base.line((x + k, 300, x + k, 330), fill=color_gen(k / 299))
    d_base.text((x, 340), "gen 0", fill=TEXTO_3, font=f_chica)
    d_base.text((x + 300, 340), f"gen {g_max}", fill=TEXTO_3, font=f_chica, anchor="ra")
    d_base.text((x, 400), f"{len(mejores)} generaciones", fill=TEXTO_2, font=f_med)
    d_base.text((x, 1000), f"{esc.nombre_room}", fill=TEXTO_3, font=f_chica)

    for f in list(range(0, largo, velocidad)) + [largo - 1] * FPS:
        img = base.copy()
        d = ImageDraw.Draw(img)
        for k, (g, r) in enumerate(mejores):
            t = r.traza
            if f >= len(t):
                continue
            reg = t[f]
            col = color_gen(g / g_max)
            if not reg["alive"] or final_muerte(r, f):
                if vivo_antes[k] and f > 0:
                    vivo_antes[k] = False
                    ant = t[f - 1] if not reg["alive"] else reg
                    px, py = esc.a_pantalla(float(ant["x"]), float(ant["y"]) - 8)
                    marca_muerte(d_base, px, py, color=col)
                continue
            vivo_antes[k] = True
            if reg["room"] != sala:
                continue
            esc.pegar(img, reg, tinte=col, borde=col if g == elegidas[-1] else None)
        d.text((x, 180), "Tiempo", fill=TEXTO_2, font=f_med)
        d.text((x, 212), tiempo(min(f + 1, largo)), fill=TEXTO, font=f_grande)
        video.escribir(img)
    video.cerrar()
    return salida


# -- gráficos -----------------------------------------------------------------------------------------

def graficos(corrida: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = almacen.dir_corrida(corrida)
    cfg = almacen.cargar_config(carpeta)
    m = almacen.leer_metricas(carpeta)
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
        "xtick.minor.visible": False, "ytick.minor.visible": False, "xtick.top": False, "ytick.right": False,
    })
    g = [r["gen"] for r in m]
    salidas = []

    def guardar(fig, nombre):
        ruta = destino / nombre
        fig.tight_layout()
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        salidas.append(ruta)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["fitness_max"] for r in m], color=hex_(AZUL), label="máximo")
    ax.plot(g, [r["fitness_media"] for r in m], color=hex_(NARANJA), label="media")
    ax.plot(g, [r["fitness_mediana"] for r in m], color=hex_(AQUA), label="mediana")
    ax.set_title("Fitness por generación", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("fitness")
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "fitness.png")

    salida_on = con_salida(cfg) and m and m[0].get("progreso_max") is not None
    if salida_on:
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot(g, [r["progreso_max"] for r in m], color=hex_(AZUL), label="lo más lejos que llegó alguien")
        ax.plot(g, [r["progreso_final_max"] for r in m], color=hex_(AQUA), label="mejor posición al final")
        ax.plot(g, [r["progreso_media"] for r in m], color=hex_(NARANJA), label="media al final")
        tope = max(r["progreso_max"] for r in m)
        if tope > 60:
            ax.axhline(100, color=hex_(TEXTO_3), linestyle="--", linewidth=1)
            ax.text(g[-1], 100, "salida ", va="bottom", ha="right", color=hex_(TEXTO_3))
            ax.set_ylim(top=105)
        else:  # lejos de la salida: escala ajustada para que se vea el avance
            ax.set_ylim(min(0, min(r["progreso_media"] for r in m)), max(5.0, tope * 1.35))
        ax.set_title("Camino recorrido hacia la salida (100 % = la salida)", loc="left", fontsize=18)
        ax.set_xlabel("generación")
        ax.set_ylabel("% del camino")
        ax.legend(frameon=False, loc="upper left")
        guardar(fig, "progreso.png")

        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot(g, [r["nichos"] for r in m], color=hex_(AZUL))
        ax.set_ylim(0, cfg.poblacion * 1.05)
        ax.set_title(f"Diversidad: lugares distintos donde terminan los {cfg.poblacion} agentes", loc="left",
                     fontsize=18)
        ax.set_xlabel("generación")
        ax.set_ylabel("nichos (cuadros de %d px)" % cfg.nicho_px)
        guardar(fig, "diversidad.png")
    else:
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot(g, [r["celdas_max"] for r in m], color=hex_(AZUL), label="máximo")
        ax.plot(g, [r["celdas_media"] for r in m], color=hex_(NARANJA), label="media")
        ax.set_title("Celdas de 32×32 px exploradas", loc="left", fontsize=18)
        ax.set_xlabel("generación")
        ax.set_ylabel("celdas")
        ax.legend(frameon=False, loc="upper left")
        guardar(fig, "exploracion.png")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["segundos_medio"] for r in m], color=hex_(AZUL), label="vida media")
    ax.plot(g, [r["segundos_mejor"] for r in m], color=hex_(NARANJA), label="mejor agente")
    ax.axhline(cfg.max_segundos, color=hex_(TEXTO_3), linestyle="--", linewidth=1)
    ax.text(g[0], cfg.max_segundos, f" límite de {cfg.max_segundos:.0f} s", va="bottom", color=hex_(TEXTO_3))
    ax.set_title("Segundos de juego antes de morir", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("segundos")
    ax.legend(frameon=False, loc="center left")
    guardar(fig, "supervivencia.png")

    # Mapa de muertes: dónde murió cada agente (todas sus muertes, con reintentos), todas las generaciones
    esc = Escena(carpeta, cfg)
    img = esc.mapa.copy().convert("RGBA")
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dc = ImageDraw.Draw(capa)
    gens = almacen.generaciones_guardadas(carpeta)
    for gi in gens:
        for r in almacen.cargar_generacion(carpeta, gi):
            if r.origen == "elite" or len(r.traza) < 2:
                continue
            for f in muertes_en(r):
                reg = r.traza[min(f, len(r.traza)) - 1]
                if reg["room"] != cfg.room_entrenamiento:
                    continue
                px, py = float(reg["x"]) * esc.escala, (float(reg["y"]) - 8) * esc.escala
                dc.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 70, 60, 110))
    oscuro = Image.new("RGBA", img.size, (0, 0, 0, 140))
    img = Image.alpha_composite(Image.alpha_composite(img, oscuro), capa)
    ruta = destino / "mapa_muertes.png"
    img.convert("RGB").save(ruta)
    salidas.append(ruta)

    if con_salida(cfg):
        salidas.append(mapa_distancia(carpeta, cfg, esc))
    for s in salidas:
        print(f"gráfico: {s}")
    return salidas


def mapa_distancia(carpeta: Path, cfg: Config, esc: Escena) -> Path:
    """La "noción de la salida" que tiene el agente: color = distancia por el laberinto, línea = camino corto."""
    import matplotlib
    from . import distancia
    mapa = distancia.cargar(ROOT / cfg.mapa_distancia)
    d = mapa["dist"] / mapa["dist"].max()
    rgb = (matplotlib.colormaps["viridis"](1 - d)[..., :3] * 255).astype(np.uint8)
    rgb[mapa["grilla"] == distancia.MURO] = (30, 30, 30)
    calor = Image.fromarray(rgb).resize((esc.mw, esc.mh), Image.NEAREST)
    img = Image.blend(esc.mapa, calor, 0.55)
    dr = ImageDraw.Draw(img)
    k = esc.escala
    dr.line([((i * distancia.PASO + distancia.PASO / 2) * k, (j * distancia.PASO + distancia.DY) * k)
             for j, i in mapa["ruta"]], fill=(255, 255, 255), width=3)
    ix, iy = mapa["inicio"]
    dr.ellipse((ix * k - 8, iy * k - 8, ix * k + 8, iy * k + 8), outline=(255, 255, 255), width=3)
    ruta = carpeta / "graficos" / "mapa_distancia.png"
    img.save(ruta)
    return ruta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["generacion", "mejor", "evolucion", "graficos", "todo", "vivo"])
    p.add_argument("--corrida", default="rStage01")
    p.add_argument("--gen", type=int, default=-1)
    p.add_argument("--cada", type=int, default=10)
    p.add_argument("--velocidad", type=int, default=None,
                   help="frames del juego por frame de video (1 por defecto; 4 en vivo)")
    a = p.parse_args()
    if a.modo == "vivo":
        vivo(a.corrida, a.velocidad or 4)
        return
    a.velocidad = a.velocidad or 1
    if a.modo == "generacion":
        video_generacion(a.corrida, a.gen, a.velocidad)
    elif a.modo == "mejor":
        video_mejor(a.corrida, a.gen)
    elif a.modo == "evolucion":
        video_evolucion(a.corrida, a.cada, a.velocidad)
    elif a.modo == "graficos":
        graficos(a.corrida)
    else:
        graficos(a.corrida)
        gens = almacen.generaciones_guardadas(almacen.dir_corrida(a.corrida))
        for g in sorted(set(gens[::a.cada]) | {gens[-1]}):
            video_generacion(a.corrida, g, a.velocidad)
            video_mejor(a.corrida, g)
        video_evolucion(a.corrida, a.cada, a.velocidad)


if __name__ == "__main__":
    main()
