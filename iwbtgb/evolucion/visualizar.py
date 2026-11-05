"""Videos, en vivo y gráficos de una corrida de redes (las dos líneas: `python -m iwbtgb.redes|rayos visualizar`).

    visualizar entreno       --corrida rayos01 --quien mejor|todas|alternar [--velocidad 1]   # ventana
    visualizar entreno-video --corrida rayos01 --quien todas [--gen -1]
    visualizar prueba        --corrida rayos01|azar|solucion [--dificultad 0.5] [--semilla 1000000] [--verificar]
    visualizar prueba-video  --corrida rayos01 [--dificultad -1] [--salas 10]
    visualizar muestras      [--dificultad -1] [--semilla 0] [--salas 12] [--verificar]
    visualizar mejor         --corrida red01 [--sala rStage02] [--gen -1] [--espera 0]
    visualizar vivo-red      --corrida red01 [--sala rStage02]                                # ventana
    visualizar vivo          --corrida red05 [--velocidad 4]                                  # ventana
    visualizar generacion    --corrida red05 [--gen -1]
    visualizar graficos      --corrida red01

`entreno`: las salas de una generación de una corrida del gimnasio tal como las jugó la población (`mejor`: la mejor
red emulada, con lo que ve y cómo le fue al entrenar; `todas`: la mejor emulada y las demás como fantasmas desde sus
trazas; `alternar`: una sala de cada). En vivo repite la última generación y pasa a la nueva cuando aparece.

`prueba`: salas nuevas, que la red nunca vio, una tras otra (¿generaliza?); `azar` y `solucion` (la búsqueda resuelve
cada sala) sirven para ver salas del gimnasio. `muestras`: una imagen con salas generadas (y su solución).

`mejor` y `vivo-red`: la mejor red (la que llega más lejos) en un nivel, el suyo u otro que no vio. `vivo` y
`generacion`: todas las redes de una generación de un nivel sobre su mapa, con el dibujo del GA.

Con `--red`, la red por dentro en vez del gráfico de fitness. Los en vivo corren con tope de memoria. Videos en
<corridas>/<nombre>/videos/; con `azar`, `solucion` y `muestras`, en <corridas>/_gimnasio/.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..ga.nucleo import Config as ConfigGA
from ..ga.visualizar import cuadros_generacion
from ..medios import (AQUA, DORADO, FONDO, ROJO, TEXTO, TEXTO_2, W, Cerrada, EnVivo, Video, con_tope_de_memoria,
                      en_ventana, estilo_graficos, fuente, guardar_grafico, hexa)
from ..rutas import ROOT
from . import almacen
from .episodio import Escenario
from .gimnasio import Gimnasio, PoliticaAzar, PoliticaGrabada, PoliticaRed, generar, resolver
from .pantallas import FPS_VIDEO, PantallaGimnasio, PantallaNivel, PantallaPoblacion, Ritmo
from .salas import config_en_sala

FPS = FPS_VIDEO
MAX_FRAMES = 1000            # 20 s por sala con `azar` y `solucion` (una red usa la duración de su corrida)
PRESUPUESTO = 600            # iteraciones de la búsqueda por sala


def dificultad_de(dificultad: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(0, 1)) if dificultad < 0 else dificultad


def reglas_de_salas(cfg) -> tuple[bool, bool, int]:
    """(salida en la pared, trampas, trampas mínimas) de las salas nuevas para una red: las de su entrenamiento si
    entrenó en el gimnasio; si no, salida en la pared y con trampas."""
    if cfg.entorno == "gimnasio":
        return bool(cfg.gim_salida_borde), bool(cfg.gim_trampas), cfg.gim_trampas_min
    return True, True, 0


# -- el entrenamiento en el gimnasio: las salas de una generación, como las jugó la población ------------------

def generacion(carpeta: Path, gen: int) -> tuple[int, dict]:
    gens = almacen.generaciones_guardadas(carpeta)
    if not gens:
        raise SystemExit(f"{carpeta.name} todavía no tiene generaciones")
    gen = gens[-1] if gen < 0 else gen
    d = almacen.cargar_generacion(carpeta, gen)
    if "gimnasio" not in d:
        raise SystemExit(f"{carpeta.name} no entrena en el gimnasio")
    return gen, d


def salas_de(d: dict) -> list:
    """Las salas de una generación, con las reglas con que se generaron (sin las claves: las primeras corridas)."""
    borde = bool(d["salida_borde"]) if "salida_borde" in d else False
    trampas = bool(d["trampas"]) if "trampas" in d else False
    minimo = int(d["trampas_min"]) if "trampas_min" in d else 0
    return [generar(int(s), float(f), borde, trampas, minimo) for s, f in zip(d["semillas"], d["dificultades"])]


def trazas_de(d: dict) -> list[list[np.ndarray]] | None:
    """[red][sala] → traza por frame de esa partida; None si la generación no guardó las de todas sus salas."""
    largos = d["traza_largo"]
    if largos.ndim != 2 or not largos.sum():
        return None
    fin = np.cumsum(largos.ravel())
    trozos = np.split(d["trazas"], fin[:-1])
    n, s = largos.shape
    return [[trozos[i * s + k] for k in range(s)] for i in range(n)]


def sin_trazas(corrida: str, g: int) -> SystemExit:
    return SystemExit(f"la generación {g} de {corrida} no guardó las trazas de sus salas: «todas las redes» necesita "
                      "una corrida más nueva (detenla y continúala para que las generaciones nuevas las guarden). «La "
                      "mejor red» sí funciona")


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


def entreno(linea, corrida: str, quien: str, mostrar, velocidad: int = 1, gen: int = -1, en_vivo: bool = True,
            parar=None, con_red: bool = False) -> None:
    """Las salas de una generación tal como las jugó la población (`quien`: "mejor", "todas" o "alternar"). En
    vivo (`gen` = -1) repite las salas de la última generación guardada y pasa a la nueva cuando aparece; termina
    cuando `parar()` (si se da) devuelve True, entre generaciones."""
    carpeta = linea.dir_corrida(corrida)
    g, d = generacion(carpeta, gen)
    if quien != "mejor" and trazas_de(d) is None:
        raise sin_trazas(corrida, g)
    cfg, gim, politica = linea.preparar_gimnasio(corrida, g)
    politica.red.registrar = True
    tope, paciencia, cada = cfg.max_frames, int(cfg.paciencia * 50), 2 * velocidad
    p_mejor = PantallaGimnasio(politica.nombre, politica, "Entrenamiento en vivo", linea, con_red)
    p_todas = PantallaPoblacion(linea, corrida)    # todas juntas, sin lo que ve (se ve en la de la mejor)
    n_sala = 0
    try:
        while True:
            if politica.cargar(g):
                p_mejor.agente = politica.nombre
            mejor = almacen.indice_mejor(d)
            salas = salas_de(d)
            trazas = trazas_de(d)
            metricas = almacen.leer_metricas(carpeta)
            anteriores: list[tuple[int, int]] = []
            for k, sala in enumerate(salas):
                gim.construir(sala)
                fines = [str(f) for f in d["fines"][:, k]]
                ver_todas = trazas is not None and (quien == "todas" or (quien == "alternar" and n_sala % 2 == 0))
                if ver_todas:
                    p_todas.nueva_sala(g, k, sala, [t[k] for t in trazas], fines, mejor, anteriores, metricas,
                                       len(salas))
                    _sala_todas(gim, politica, p_todas, tope, paciencia, cada, mostrar)
                else:
                    _sala_mejor(gim, politica, p_mejor, sala, g, k, len(salas), fines[mejor],
                                float(d["avances"][mejor, k]), tope, paciencia, cada, mostrar)
                anteriores.append((fines.count("llegó"), len(fines)))
                n_sala += 1
                print(f"gen {g} sala {k + 1}/{len(salas)} ({sala.nombre}): llegaron {fines.count('llegó')} de "
                      f"{len(fines)} · la mejor: {fines[mejor]}", flush=True)
            if not en_vivo or (parar is not None and parar()):
                break
            g, d = generacion(carpeta, gen)
    finally:
        gim.close()


def entreno_vivo(linea, corrida: str, quien: str, velocidad: int = 1, con_red: bool = False) -> None:
    if quien != "mejor":   # antes de abrir la ventana
        g, d = generacion(linea.dir_corrida(corrida), -1)
        if trazas_de(d) is None:
            raise sin_trazas(corrida, g)
    nombres = {"mejor": "la mejor red", "todas": "todas las redes", "alternar": "todas y la mejor"}
    en_ventana(f"Entrenamiento en vivo · {corrida} · {nombres[quien]}",
               lambda mostrar: entreno(linea, corrida, quien, mostrar, velocidad, con_red=con_red), FPS)


def entreno_video(linea, corrida: str, quien: str, gen: int = -1, velocidad: int = 1, con_red: bool = False) -> Path:
    g, d = generacion(linea.dir_corrida(corrida), gen)
    if quien != "mejor" and trazas_de(d) is None:
        raise sin_trazas(corrida, g)
    ruta = linea.dir_corrida(corrida) / "videos" / f"entrenamiento_gen{g:04d}_{quien}.mp4"
    v = Video(ruta, fps=FPS)
    try:
        entreno(linea, corrida, quien, v.escribir, velocidad, g, en_vivo=False, con_red=con_red)
    finally:
        v.cerrar()
    return ruta


# -- la prueba con salas nuevas ------------------------------------------------------------------------------

def jugar_salas(linea, agente: str, dificultad: float, semilla: int, salas: int | None, verificar: bool, mostrar,
                recargar: bool = False, velocidad: int = 1, gen: int = -1, con_red: bool = False) -> None:
    """Salas nuevas una tras otra (`salas` = None: sin fin), cada cuadro a `mostrar`. `agente`: una corrida (su mejor
    red, con la duración y la paciencia de su corrida; con `recargar` toma la nueva cuando aparece), "azar" o
    "solucion" (la búsqueda resuelve cada sala y se ve su solución)."""
    if agente in ("azar", "solucion"):
        gim, politica = Gimnasio(), PoliticaAzar() if agente == "azar" else None
        nombre = "acciones al azar" if agente == "azar" else "solución de la búsqueda"
        tope, paciencia, (borde, trampas, minimo) = MAX_FRAMES, 0, (True, True, 0)
    else:
        cfg, gim, politica = linea.preparar_gimnasio(agente, gen)
        politica.red.registrar = True
        nombre = politica.nombre
        tope, paciencia = cfg.max_frames, int(cfg.paciencia * 50)
        borde, trampas, minimo = reglas_de_salas(cfg)
    pantalla = PantallaGimnasio(nombre, politica, "Gimnasio, salas nuevas", linea, con_red)
    rng = np.random.default_rng(semilla)
    try:
        n, s = 0, semilla
        while salas is None or n < salas:
            sala = generar(s, dificultad_de(dificultad, rng), borde, trampas, minimo)
            s += 1
            gim.construir(sala)
            solucion = None
            if politica is None or verificar:
                solucion = resolver(gim, PRESUPUESTO, sala.semilla)
                if solucion is None:
                    print(f"sala {sala.nombre}: sin solución encontrada, se salta", flush=True)
                    continue
            n += 1
            pol = PoliticaGrabada(solucion) if politica is None else politica
            if recargar and isinstance(pol, PoliticaRed) and pol.cargar(-1):
                pantalla.agente = pol.nombre
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


def prueba_vivo(linea, agente: str, dificultad: float, semilla: int, verificar: bool, velocidad: int = 1,
                con_red: bool = False) -> None:
    titulo = f"Gimnasio, salas nuevas · {agente}" + (f" (x{velocidad})" if velocidad > 1 else "")
    en_ventana(titulo, lambda mostrar: jugar_salas(linea, agente, dificultad, semilla, None, verificar, mostrar,
                                                   recargar=True, velocidad=velocidad, con_red=con_red), FPS)


def video_prueba(linea, agente: str, dificultad: float, semilla: int, salas: int, gen: int = -1,
                 verificar: bool = False, velocidad: int = 1, con_red: bool = False) -> Path:
    dif = "azar" if dificultad < 0 else f"{dificultad:.2f}"
    x = f"_x{velocidad}" if velocidad > 1 else ""
    if agente in ("azar", "solucion"):
        ruta = linea.corridas / "_gimnasio" / "videos" / f"{agente}_dif{dif}_s{semilla}{x}.mp4"
    else:
        gen = almacen.red_de(linea.dir_corrida(agente), gen)[0]
        ruta = linea.dir_corrida(agente) / "videos" / f"prueba_gen{gen:04d}_dif{dif}_s{semilla}{x}.mp4"
    v = Video(ruta, fps=FPS)
    try:
        jugar_salas(linea, agente, dificultad, semilla, salas, verificar, v.escribir, velocidad=velocidad, gen=gen,
                    con_red=con_red)
    finally:
        v.cerrar()
    return ruta


def muestras(linea, dificultad: float, semilla: int, salas: int = 12, verificar: bool = False) -> Path:
    """Mosaico de 4 columnas con salas del gimnasio (y la solución encontrada, con `verificar`)."""
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
    ruta = linea.corridas / "_gimnasio" / "muestras" / f"dif{dif}_s{semilla}{'_verificadas' if verificar else ''}.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(ruta)
    if verificar:
        print(f"con solución: {resueltas} de {salas}", flush=True)
    print(f"imagen: {ruta}", flush=True)
    return ruta


# -- la mejor red en un nivel --------------------------------------------------------------------------------

def en_sala(linea, corrida: str, sala: str | None, objetivo: str | None = None, max_segundos: float | None = None):
    """(carpeta, config, jugador para la sala pedida (la de la corrida si es None), texto para el título)."""
    carpeta = linea.dir_corrida(corrida)
    cfg, mapa, entrenada = config_en_sala(linea, carpeta, linea.cargar_config(carpeta), sala, objetivo)
    if max_segundos:
        cfg.max_segundos = max_segundos
    donde = "" if entrenada else f" · en {cfg.start_room} (no la vio al entrenar)"
    return carpeta, cfg, linea.jugador_nivel(cfg, mapa), donde


def video_mejor(linea, corrida: str, gen: int = -1, espera: int = 0, salida: Path | None = None,
                sala: str | None = None, objetivo: str | None = None, max_segundos: float | None = None,
                con_red: bool = False) -> Path:
    carpeta, cfg, jugador, donde = en_sala(linea, corrida, sala, objetivo, max_segundos)
    gen, genoma, fitness = almacen.red_de(carpeta, gen)
    nombre = f"gen_{gen:04d}_mejor_{cfg.start_room}" + (f"_espera{espera}" if espera else "")
    salida = salida or carpeta / "videos" / f"{nombre}.mp4"
    video = Video(salida, fps=FPS_VIDEO)
    pantalla = PantallaNivel(linea, cfg, jugador, carpeta, con_red)
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


def vivo_red(linea, corrida: str, sala: str | None = None, velocidad: int = 1, mostrar=None, parar=None,
             con_red: bool = False) -> None:
    """La mejor red jugando un nivel sin parar; toma la nueva entre partidas. Sin `mostrar`, en una ventana.
    Termina cuando `parar()` (si se da) devuelve True, entre partidas."""
    carpeta, cfg, jugador, donde = en_sala(linea, corrida, sala)
    pantalla = PantallaNivel(linea, cfg, jugador, carpeta, con_red)
    gen, genoma, n = -1, None, 0
    ventana = None
    if mostrar is None:
        ventana = EnVivo(f"{corrida} · red en vivo" + (f" (x{velocidad})" if velocidad > 1 else ""), FPS_VIDEO,
                         lambda: f"gen {gen} · {n} partidas")
        mostrar = ventana.mostrar
    historial: list[str] = []
    try:
        while parar is None or not parar():
            try:
                g, genoma_nuevo, fitness = almacen.red_de(carpeta, -1)
                if g != gen:
                    pantalla.red(g, fitness)
                    gen, genoma = g, genoma_nuevo
            except (OSError, ValueError, EOFError, KeyError, IndexError):  # se está guardando justo ahora
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
            for _ in range(FPS_VIDEO):
                mostrar(img)
    except (KeyboardInterrupt, Cerrada):
        pass
    finally:
        if ventana is not None:
            ventana.cerrar()
        jugador.close()


# -- todas las redes de una generación de un nivel, con el dibujo del GA ------------------------------------------

@dataclass
class RedEnMapa:
    """Una red con lo que lee `ga.visualizar.cuadros_generacion` (su primer escenario)."""
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


def config_ga(carpeta: Path, cfg) -> ConfigGA:
    """La config del GA con la que `ga.visualizar` dibuja el mapa, la salida y las zonas de la sala de la corrida."""
    return ConfigGA(
        nombre=carpeta.name, start_room=cfg.start_room, difficulty=cfg.difficulty, seed_juego=cfg.seed_juego,
        max_segundos=cfg.max_segundos, sala_objetivo=cfg.sala_objetivo, save_points="", resets=0,
        zonas_prohibidas=";".join(",".join(f"{v:g}" for v in z) for z in cfg.zonas()),
        room_entrenamiento=cfg.room_entrenamiento, room_objetivo=cfg.room_objetivo, room_ancho=cfg.room_ancho,
        room_alto=cfg.room_alto, mapa_distancia=str((carpeta / "distancia.npz").relative_to(ROOT)))


def poblacion(cfg, d: dict) -> list[RedEnMapa]:
    if "trazas" not in d:
        raise ValueError("esta generación no tiene trazas (son de una corrida más nueva)")
    fin = np.cumsum(d["traza_largo"])
    redes = []
    for i in range(len(d["fitness"])):
        f = str(d["fines"][i][0])
        redes.append(RedEnMapa(
            traza=d["trazas"][fin[i] - d["traza_largo"][i]:fin[i]], fitness=float(d["fitness"][i]),
            progreso=float(d["avances"][i][0]), saves=int(d["saves"][i][0]) if "saves" in d else 0,
            muertes=int(d["muertes"][i][0]) if "muertes" in d else 0, frames=int(d["traza_largo"][i]),
            llego=f == "llegó", murio=f in ("murió", "salió", "estancado") and not cfg.reintentos))
    return redes


def cuadros_todas(linea, corrida: str, gen: int, velocidad: int = 1, en_vivo: bool = False):
    carpeta = linea.dir_corrida(corrida)
    cfg = linea.cargar_config(carpeta)
    gens = almacen.generaciones_guardadas(carpeta)
    gen = gens[gen] if gen < 0 else gen
    pob = poblacion(cfg, almacen.cargar_generacion(carpeta, gen))
    return cuadros_generacion(carpeta, config_ga(carpeta, cfg), gen, pob, velocidad, en_vivo=en_vivo)


def video_generacion(linea, corrida: str, gen: int = -1, velocidad: int = 1) -> Path:
    carpeta = linea.dir_corrida(corrida)
    gen = almacen.generaciones_guardadas(carpeta)[gen] if gen < 0 else gen
    video = Video(carpeta / "videos" / f"gen_{gen:04d}_todas.mp4")
    try:
        for img in cuadros_todas(linea, corrida, gen, velocidad):
            video.escribir(img)
    finally:
        video.cerrar()
    return video.ruta


def vivo(linea, corrida: str, velocidad: int = 4) -> None:
    """Como el en vivo del GA: todas las redes de la última generación guardada, y salta a la más nueva."""
    carpeta = linea.dir_corrida(corrida)
    mostrada = -1
    ventana = EnVivo(f"{corrida} · todas las redes en vivo (x{velocidad})", detalle=lambda: f"gen {mostrada}")
    try:
        while ventana.abierta():
            ventana.revisar()
            gens = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
            if not gens or gens[-1] == mostrada:
                time.sleep(1)
                continue
            mostrada = gens[-1]
            try:  # si se guarda justo mientras se lee, se reintenta en la próxima vuelta
                cuadros = cuadros_todas(linea, corrida, mostrada, velocidad, en_vivo=True)
            except (OSError, ValueError, EOFError, KeyError) as e:
                if isinstance(e, ValueError) and "trazas" in str(e):
                    raise SystemExit(f"{corrida}: {e}. Usa «vivo-red» (la mejor red jugando).")
                mostrada = -1
                time.sleep(1)
                continue
            for img in cuadros:
                ventana.mostrar(img)
            del cuadros
    except (KeyboardInterrupt, Cerrada):
        pass
    finally:
        ventana.cerrar()


# -- gráficos ---------------------------------------------------------------------------------------------------

def graficos(linea, corrida: str) -> list[Path]:
    """Avance, fitness, cómo terminan las partidas, muertes y saves y, en el gimnasio, salas superadas y dificultad."""
    from ..medios import AZUL, GRIS, NARANJA
    plt = estilo_graficos()
    carpeta = linea.dir_corrida(corrida)
    m = almacen.leer_metricas(carpeta)
    if not m:
        raise SystemExit(f"{corrida} todavía no tiene generaciones")
    destino = carpeta / "graficos"
    destino.mkdir(exist_ok=True)
    g = [r["gen"] for r in m]
    salidas: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["avance_max"] for r in m], color=hexa(AZUL), label="lo más lejos en algún escenario")
    ax.plot(g, [r["avance_mejor"] for r in m], color=hexa(AQUA), label="la mejor red (promedio de escenarios)")
    ax.plot(g, [r["avance_media"] for r in m], color=hexa(NARANJA), label="media de la población")
    ax.set_title("Avance hacia la salida", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("% del camino")
    ax.set_ylim(0, max(5, max(r["avance_max"] for r in m) * 1.1))
    ax.legend(frameon=False, loc="upper left")
    guardar_grafico(plt, fig, destino / "avance.png", salidas)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(g, [r["fitness_max"] for r in m], color=hexa(AZUL), label="máximo")
    ax.plot(g, [r["fitness_media"] for r in m], color=hexa(NARANJA), label="media")
    ax.set_title("Fitness por generación", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("fitness")
    ax.legend(frameon=False, loc="upper left")
    guardar_grafico(plt, fig, destino / "fitness.png", salidas)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    claves = [("murio_pct", "murió", ROJO), ("estancado_pct", "se estancó", GRIS), ("tiempo_pct", "tiempo", AZUL),
              ("salio_pct", "salió de la sala", DORADO)]
    ax.stackplot(g, *[[r[k] for r in m] for k, _, _ in claves], labels=[e for _, e, _ in claves],
                 colors=[hexa(c) for _, _, c in claves], alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(g, [r["nichos"] for r in m], color=hexa(TEXTO), linewidth=1.5, label="nichos (eje derecho)")
    ax2.set_ylabel("nichos")
    ax2.grid(False)
    ax.set_title("Cómo terminan las partidas", loc="left", fontsize=18)
    ax.set_xlabel("generación")
    ax.set_ylabel("% de las partidas")
    ax.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", ncols=3)
    guardar_grafico(plt, fig, destino / "fines.png", salidas)

    con = [r for r in m if "muertes_media" in r]
    if con:
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot([r["gen"] for r in con], [r["muertes_media"] for r in con], color=hexa(ROJO),
                label="muertes por partida (media)")
        ax.set_xlabel("generación")
        ax.set_ylabel("muertes")
        ax2 = ax.twinx()
        ax.plot([r["gen"] for r in con], [r.get("quieto_pct", np.nan) / 10 for r in con], color=hexa(GRIS),
                label="tiempo quieto (% / 10)")
        ax2.plot([r["gen"] for r in con], [r["saves_media"] for r in con], color=hexa(AQUA), label="saves (media)")
        ax2.plot([r["gen"] for r in con], [r["saves_max"] for r in con], color=hexa(DORADO), linewidth=1.5,
                 label="saves (máximo)")
        ax2.plot([r["gen"] for r in con], [r.get("celdas_media", np.nan) / 10 for r in con], color=hexa(NARANJA),
                 label="celdas exploradas (media / 10)")
        ax2.set_ylabel("saves")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
        ax.set_title("Muertes y saves", loc="left", fontsize=18)
        guardar_grafico(plt, fig, destino / "muertes_saves.png", salidas)

    gim = [r for r in m if "dificultad_max" in r]
    if gim:
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot([r["gen"] for r in gim], [r["salas_pct"] for r in gim], color=hexa(AQUA),
                label="salas superadas (población, %)")
        ax.set_ylabel("% de salas")
        ax.set_ylim(0, 100)
        ax2 = ax.twinx()
        ax2.plot([r["gen"] for r in gim], [r["dificultad_max"] for r in gim], color=hexa(DORADO),
                 label="dificultad máxima (eje derecho)")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("dificultad")
        ax2.grid(False)
        for r0, r1 in zip(gim, gim[1:]):
            if r1.get("trampas") and not r0.get("trampas"):
                ax.axvline(r1["gen"], color=hexa(TEXTO), linestyle=":", linewidth=1)
                ax.text(r1["gen"], 96, " trampas", color=hexa(TEXTO), fontsize=11, va="top")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
        ax.set_title("Gimnasio: salas superadas y dificultad", loc="left", fontsize=18)
        ax.set_xlabel("generación")
        guardar_grafico(plt, fig, destino / "gimnasio.png", salidas)
    return salidas


def main(linea, argv=None) -> None:
    p = argparse.ArgumentParser(prog=f"python -m {linea.paquete} visualizar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["entreno", "entreno-video", "prueba", "prueba-video", "muestras", "mejor",
                                    "vivo-red", "vivo", "generacion", "graficos"])
    p.add_argument("--corrida", default=None, help="una corrida; en prueba también «azar» o «solucion»")
    p.add_argument("--quien", choices=["mejor", "todas", "alternar"], default="mejor")
    p.add_argument("--gen", type=int, default=-1)
    p.add_argument("--dificultad", type=float, default=0.5, help="gimnasio: 0 a 1; -1 = al azar en cada sala")
    p.add_argument("--semilla", type=int, default=None, help="gimnasio: semilla de la primera sala")
    p.add_argument("--salas", type=int, default=None, help="cuántas salas (prueba-video: 10; muestras: 12)")
    p.add_argument("--verificar", action="store_true", help="gimnasio: saltar las salas sin solución encontrada")
    p.add_argument("--sala", default=None, help="mejor/vivo-red: otro nivel (rStage02 …)")
    p.add_argument("--espera", type=int, default=0)
    p.add_argument("--velocidad", type=int, default=None, help="por defecto 4 en «vivo» y 1 en lo demás")
    p.add_argument("--red", action="store_true", help="la red por dentro en vez del gráfico de fitness")
    a = p.parse_args(argv)
    if a.corrida is None and a.modo != "muestras":
        p.error("falta --corrida")
    velocidad = a.velocidad or (4 if a.modo == "vivo" else 1)
    semilla = a.semilla if a.semilla is not None else (0 if a.modo == "muestras" else 1_000_000)
    if a.modo in ("entreno", "prueba", "vivo-red", "vivo"):
        con_tope_de_memoria()
    if a.modo == "entreno":
        entreno_vivo(linea, a.corrida, a.quien, velocidad, a.red)
    elif a.modo == "entreno-video":
        entreno_video(linea, a.corrida, a.quien, a.gen, velocidad, a.red)
    elif a.modo == "prueba":
        prueba_vivo(linea, a.corrida, a.dificultad, semilla, a.verificar, velocidad, a.red)
    elif a.modo == "prueba-video":
        video_prueba(linea, a.corrida, a.dificultad, semilla, a.salas or 10, a.gen, a.verificar, velocidad, a.red)
    elif a.modo == "muestras":
        muestras(linea, a.dificultad, semilla, a.salas or 12, a.verificar)
    elif a.modo == "mejor":
        video_mejor(linea, a.corrida, a.gen, a.espera, sala=a.sala, con_red=a.red)
    elif a.modo == "vivo-red":
        vivo_red(linea, a.corrida, a.sala, velocidad, con_red=a.red)
    elif a.modo == "vivo":
        vivo(linea, a.corrida, velocidad)
    elif a.modo == "generacion":
        video_generacion(linea, a.corrida, a.gen, velocidad)
    else:
        graficos(linea, a.corrida)
