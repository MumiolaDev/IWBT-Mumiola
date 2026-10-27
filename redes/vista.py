"""Vista semántica: lo que ve la red en cada frame.

Una grilla de `ancho`×`alto` celdas de `celda` px centrada en el jugador, con un canal por tipo de cosa (sólido,
plataforma, mortal, save, salida, agua, fuera de la sala). Cada celda vale la fracción de su área que cubre la
máscara de colisión de algún objeto de ese tipo (o de sus hijos): así una púa es un triángulo y no un cuadrado.
Los tipos se definen por la jerarquía de objetos del engine de IWBTG (`block`, `playerKiller`, ...), que comparten
casi todos los fangames, así que la misma vista sirve para cualquiera de ellos. El cálculo lo hace el emulador
(`GRID` en `rl.rs`), sin dibujar.

    uv run python -m redes.vista probar                              # velocidad y canales en el comienzo de rStage01
    uv run python -m redes.vista video --corrida GA_beta_corregida   # el mejor del GA con su vista al lado
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from iwbtgb_rl import ACTIONS, IWannaBeTheGBEnv
from iwbtgb_rl.client import Emulator, Grid

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Canal:
    nombre: str
    objetos: tuple[str, ...]          # con sus hijos
    excluidos: tuple[str, ...] = ()   # hijos de estos quedan fuera
    solo_visibles: bool = False       # instancias invisibles (trampas ocultas) no cuentan
    color: tuple[int, int, int] = (255, 255, 255)


# Engine estándar de IWBTG (el del GB y la mayoría de los fangames). `platform` es hija de `block`, pero se
# atraviesa desde abajo: va en su propio canal. `roomChanger` es invisible pero es la salida de la sala.
CANALES_IWBTG = (
    Canal("sólido", ("block",), ("platform",), color=(150, 150, 160)),
    Canal("plataforma", ("platform",), color=(90, 170, 255)),
    Canal("mortal", ("playerKiller",), solo_visibles=True, color=(235, 60, 60)),
    Canal("save", ("savePoint",), color=(60, 220, 120)),
    Canal("salida", ("roomChanger",), color=(250, 200, 40)),
    Canal("agua", ("water",), color=(40, 200, 220)),
)
CANAL_FUERA = Canal("fuera de la sala", (), color=(110, 60, 140))  # lo calcula Python con el tamaño de la room


@dataclass
class ConfigVista:
    celda: int = 8          # px por celda
    ancho: int = 64         # celdas (64 × 8 = 512 px)
    alto: int = 48          # celdas (48 × 8 = 384 px)
    muestras: int = 4       # puntos por eje y celda (4 × 4 = cada 2 px)
    objeto: str = "player"
    campos: tuple[str, ...] = ("djump",)   # variables del jugador que también ve la red
    canales: tuple[Canal, ...] = field(default_factory=lambda: CANALES_IWBTG)

    @property
    def n_canales(self) -> int:
        return len(self.canales) + 1  # + fuera de la sala

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigVista":
        d = dict(d)
        d["campos"] = tuple(d.get("campos", ()))
        if "canales" in d:
            d["canales"] = tuple(Canal(**{**c, "objetos": tuple(c["objetos"]),
                                          "excluidos": tuple(c.get("excluidos", ())),
                                          "color": tuple(c.get("color", (255, 255, 255)))})
                                 for c in d["canales"])
        return cls(**d)


class Vista:
    """Configura la grilla en un emulador y arma la observación (canales + variables del jugador)."""

    def __init__(self, emu: Emulator, cfg: ConfigVista | None = None):
        self.emu = emu
        self.cfg = cfg = cfg or ConfigVista()
        emu.track(cfg.objeto)
        faltan = emu.grid_config(cfg.celda, cfg.ancho, cfg.alto, cfg.muestras,
                                 [(list(c.objetos), list(c.excluidos), c.solo_visibles) for c in cfg.canales],
                                 list(cfg.campos))
        # Un objeto que este juego no tiene deja su canal vacío (o sin la exclusión): se avisa, no es un error
        self.faltan = faltan
        self._salas: dict[int, tuple[int, int]] = {}

    def tamano_sala(self, room: int) -> tuple[int, int]:
        if room not in self._salas:
            info = self.emu.info()
            self._salas[info.id] = (info.width, info.height)
        return self._salas.get(room, (0, 0))

    def completar(self, g: Grid) -> np.ndarray:
        """Canales del emulador + el de fuera de la sala, uint8 (canales, alto, ancho)."""
        c = self.cfg
        w, h = self.tamano_sala(g.room)
        x0 = g.centro[0] - c.ancho * c.celda // 2
        y0 = g.centro[1] - c.alto * c.celda // 2
        xs = x0 + np.arange(c.ancho) * c.celda          # borde izquierdo de cada celda
        ys = y0 + np.arange(c.alto) * c.celda
        # fracción de la celda fuera de [0, w) × [0, h)
        dentro_x = np.clip(np.minimum(xs + c.celda, w) - np.maximum(xs, 0), 0, c.celda) / c.celda
        dentro_y = np.clip(np.minimum(ys + c.celda, h) - np.maximum(ys, 0), 0, c.celda) / c.celda
        fuera = (1.0 - dentro_y[:, None] * dentro_x[None, :]) * 255
        return np.concatenate([g.canales, fuera.astype(np.uint8)[None]], axis=0)

    def leer(self) -> tuple[Grid, np.ndarray]:
        g = self.emu.grid()
        return g, self.completar(g)


def extras(g: Grid) -> np.ndarray:
    """Variables del jugador para la red, en escala ~[-1, 1]: hspeed / 3, vspeed / 9 y los campos (djump)."""
    campos = [0.0 if np.isnan(v) else float(v) for v in g.campos]
    return np.array([g.hspeed / 3.0, g.vspeed / 9.0, *campos], np.float32)


# -- dibujo -----------------------------------------------------------------------------------------------

def pintar(canales: np.ndarray, cfg: ConfigVista, px: int = 8) -> Image.Image:
    """Imagen RGB de la grilla (`px` píxeles por celda): cada canal con su color, mezclados por cobertura, y el
    jugador en el centro."""
    todos = (*cfg.canales, CANAL_FUERA)
    img = np.full((cfg.alto, cfg.ancho, 3), 18.0)
    # de atrás hacia adelante: fuera, sólido, ..., mortal encima de todo
    orden = [len(todos) - 1] + [i for i, c in enumerate(todos[:-1]) if c.nombre != "mortal"] + \
            [i for i, c in enumerate(todos) if c.nombre == "mortal"]
    for i in orden:
        a = canales[i].astype(np.float32)[..., None] / 255.0
        img = img * (1 - a) + np.array(todos[i].color, np.float32) * a
    im = Image.fromarray(img.astype(np.uint8)).resize((cfg.ancho * px, cfg.alto * px), Image.NEAREST)
    d = ImageDraw.Draw(im)
    # jugador: caja de 10×20 px del engine (de x−5 a x+5, de y−12 a y+8), en el centro de la grilla
    cx, cy = cfg.ancho * px / 2, cfg.alto * px / 2
    k = px / cfg.celda
    d.rectangle((cx - 5 * k, cy - 12 * k, cx + 5 * k, cy + 8 * k), outline=(255, 255, 255), width=2)
    return im


def leyenda(d: ImageDraw.ImageDraw, x: int, y: int, cfg: ConfigVista, fuente) -> None:
    for i, c in enumerate((*cfg.canales, CANAL_FUERA)):
        d.rectangle((x, y + 34 * i, x + 22, y + 34 * i + 22), fill=c.color)
        d.text((x + 34, y + 34 * i + 11), c.nombre, fill=(230, 230, 225), font=fuente, anchor="lm")


# -- pruebas ----------------------------------------------------------------------------------------------

def probar(n: int = 2000) -> None:
    env = IWannaBeTheGBEnv(difficulty="medium", start_room="rStage01", obs_type="state")
    try:
        vista = Vista(env.emu)
        if vista.faltan:
            print("objetos que este juego no tiene:", ", ".join(vista.faltan))
        env.emu.save(90)
        g, canales = vista.leer()
        print(f"room {g.room}, centro {g.centro}, jugador {'vivo' if g.vivo else 'no está'}, "
              f"hspeed {g.hspeed}, vspeed {g.vspeed}, campos {g.campos}")
        for c, m in zip((*vista.cfg.canales, CANAL_FUERA), canales):
            print(f"  {c.nombre:17s} celdas con algo: {int((m > 0).sum()):5d}   cubiertas del todo: {int((m == 255).sum()):5d}")
        for con_grilla in (False, True, False, True):
            env.emu.load(90)
            rng = np.random.default_rng(0)  # mismas teclas en cada pasada
            t = time.perf_counter()
            for _ in range(n):
                env.emu.step(ACTIONS[rng.integers(len(ACTIONS))], frames=1, obs=False, no_draw=True,
                             grid=con_grilla)
            dt = time.perf_counter() - t
            print(f"{'con' if con_grilla else 'sin'} grilla: {n / dt:7.0f} frames/s")
    finally:
        env.close()


def video(corrida: str, gen: int = -1, salida: Path | None = None, segundos: float | None = None) -> Path:
    """El mejor agente de una corrida del GA (partida continua, sin save heredado) con su vista al lado."""
    from ga import almacen
    from ga.nucleo import TECLA_R, Resultado, preparar
    from ga.visualizar import FPS, H, W, Video, fuente

    carpeta = almacen.dir_corrida(corrida)
    cfg = almacen.cargar_config(carpeta)
    gens = almacen.generaciones_guardadas(carpeta)
    gen = gens[-1] if gen < 0 else gen
    r = max(almacen.cargar_generacion(carpeta, gen), key=Resultado.clave)
    if r.inicio is not None:
        raise SystemExit("el mejor parte de un save heredado: usa una corrida sin herederos")
    salida = salida or ROOT / "corridas_redes" / "_vista" / f"{corrida}_gen{gen:04d}.mp4"
    env = IWannaBeTheGBEnv(difficulty=cfg.difficulty, start_room=cfg.start_room, obs_type="state",
                           seed=cfg.seed_juego)
    emu = env.emu
    vista = Vista(emu)
    vc = vista.cfg
    preparar(emu, cfg, None)
    video = Video(salida)
    f_med, f_chica = fuente(26, True), fuente(20)
    px = 12                               # 64 × 12 = 768, 48 × 12 = 576
    juego_w = W - vc.ancho * px - 60      # el juego a la izquierda, escalado
    juego_h = juego_w * 600 // 800
    n = len(r.traza) if segundos is None else min(len(r.traza), int(segundos * FPS))
    try:
        for f in range(n):
            a = int(r.traza[f]["accion"])
            teclas = ACTIONS[a] if a < len(ACTIONS) else (("r",) if a == TECLA_R else ())
            res = emu.step(teclas, frames=1, obs=True, grid=True)
            canales = vista.completar(res.grid)
            img = Image.new("RGB", (W, H), (26, 26, 25))
            img.paste(Image.fromarray(res.obs).resize((juego_w, juego_h), Image.NEAREST), (20, 60))
            img.paste(pintar(canales, vc, px), (W - vc.ancho * px - 20, 60))
            d = ImageDraw.Draw(img)
            d.text((20, 20), f"Juego · {corrida}, mejor de la gen. {gen}", fill=(240, 239, 236), font=f_med)
            d.text((W - vc.ancho * px - 20, 20), f"Lo que verá la red · {vc.ancho}×{vc.alto} celdas de {vc.celda} px",
                   fill=(240, 239, 236), font=f_med)
            leyenda(d, W - vc.ancho * px - 20, 60 + vc.alto * px + 30, vc, f_chica)
            g = res.grid
            d.text((W - vc.ancho * px + 300, 60 + vc.alto * px + 30),
                   f"hspeed {g.hspeed:+.1f}   vspeed {g.vspeed:+.1f}   djump {g.campos[0]:.0f}"
                   + ("" if g.vivo else "   (sin jugador)"), fill=(195, 194, 183), font=f_chica)
            video.escribir(img)
    finally:
        video.cerrar()
        env.close()
    return salida


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["probar", "video"])
    p.add_argument("--corrida", default="GA_beta_corregida")
    p.add_argument("--gen", type=int, default=-1)
    p.add_argument("--segundos", type=float, default=None)
    a = p.parse_args()
    if a.modo == "probar":
        probar()
    else:
        print("video:", video(a.corrida, a.gen, segundos=a.segundos))


if __name__ == "__main__":
    main()
