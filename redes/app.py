"""App de las redes: crear corridas, entrenarlas, probarlas en otras salas, videos, gráficos y en vivo.

    uv run python -m redes.app

Es aparte de la del GA (`ga.app`), aunque se usa igual. Por debajo ejecuta `redes.entrenar`, `redes.probar`,
`redes.visualizar` y `redes.gimnasio_ver` como procesos aparte, así que lo que se hace acá se puede seguir desde la terminal y al revés:
un entrenamiento lanzado desde la terminal también aparece (y se puede detener) desde la app.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from ga.app import Proceso, spin

from . import almacen
from .config import Config
from .entrenar import FIJOS
from .episodio import red_para
from .salas import SALAS

LOG = Path.home() / ".cache" / "iwbtgb-redes.log"  # salida de la app cuando se abre desde el menú (sin terminal)
DIFICULTADES = ["medium", "hard", "veryhard", "impossible"]
FINES_CORTOS = {"llegó": "llegó", "murió": "murió", "estancado": "estancada", "tiempo": "tiempo", "salió": "salió"}


def entrenamientos_en_curso() -> dict[str, int]:
    """{corrida: pid} de los `redes.entrenar` que están corriendo (lanzados desde la app o desde la terminal)."""
    vivos = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        # el proceso de Python (no el `uv run` que lo envuelve ni sus workers)
        if len(args) > 3 and args[1] == b"-m" and args[2] == b"redes.entrenar" and b"--nombre" in args:
            i = args.index(b"--nombre")
            if i + 1 < len(args):
                vivos[args[i + 1].decode(errors="replace")] = int(d.name)
    return vivos


def doble(minimo: float, maximo: float, valor: float, decimales: int, sufijo: str = "",
          paso: float | None = None) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimo, maximo)
    s.setDecimals(decimales)
    s.setValue(valor)
    if paso:
        s.setSingleStep(paso)
    if sufijo:
        s.setSuffix(sufijo)
    return s


def combo_salas(primera: str | None = None) -> QComboBox:
    c = QComboBox()
    if primera:
        c.addItem(primera)
    c.addItems(SALAS)
    return c


class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWBTGB · redes neuronales")
        self.resize(1250, 860)
        self.procesos: list[Proceso] = []
        self.externos: dict[str, int] = {}

        # -- izquierda: corridas y resumen ------------------------------------------------------------
        self.lista = QListWidget()
        self.lista.currentItemChanged.connect(self.al_elegir)
        b_nueva = QPushButton("Nueva red")
        b_nueva.clicked.connect(self.nueva)
        b_act = QPushButton("Actualizar")
        b_act.clicked.connect(self.actualizar_lista)
        b_carpeta = QPushButton("Abrir carpeta")
        b_carpeta.clicked.connect(lambda: self.abrir(""))
        b_borrar = QPushButton("Borrar corrida…")
        b_borrar.clicked.connect(self.borrar)
        self.resumen = QLabel()
        self.resumen.setWordWrap(True)
        self.resumen.setAlignment(Qt.AlignTop)
        self.resumen.setTextInteractionFlags(Qt.TextSelectableByMouse)
        izq = QVBoxLayout()
        izq.addWidget(QLabel("<b>Corridas de redes</b>"))
        izq.addWidget(self.lista, 1)
        fila = QHBoxLayout()
        for b in (b_nueva, b_act, b_carpeta):
            fila.addWidget(b)
        izq.addLayout(fila)
        izq.addWidget(b_borrar)
        caja = QGroupBox("Resumen")
        QVBoxLayout(caja).addWidget(self.resumen)
        izq.addWidget(caja, 2)
        w_izq = QWidget()
        w_izq.setLayout(izq)

        # -- pestañas -------------------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.addTab(self.pestana_entrenar(), "Crear y entrenar")
        self.tabs.addTab(self.pestana_probar(), "Probar")
        self.tabs.addTab(self.pestana_visualizar(), "Videos y en vivo")

        # -- abajo: procesos y registro ---------------------------------------------------------------
        self.lista_proc = QListWidget()
        self.lista_proc.setMaximumHeight(80)
        b_det = QPushButton("Detener seleccionado")
        b_det.clicked.connect(self.detener_seleccionado)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("monospace", 9))
        abajo = QVBoxLayout()
        fila = QHBoxLayout()
        fila.addWidget(QLabel("<b>Procesos de la app</b>"))
        fila.addStretch()
        fila.addWidget(b_det)
        abajo.addLayout(fila)
        abajo.addWidget(self.lista_proc)
        abajo.addWidget(self.log, 1)
        w_abajo = QWidget()
        w_abajo.setLayout(abajo)

        der = QSplitter(Qt.Vertical)
        der.addWidget(self.tabs)
        der.addWidget(w_abajo)
        der.setSizes([760, 220])
        raiz = QSplitter()
        raiz.addWidget(w_izq)
        raiz.addWidget(der)
        raiz.setSizes([320, 930])
        self.setCentralWidget(raiz)

        self.actualizar_lista()
        if self.lista.count() == 0:
            self.nueva()
        # Entrenamientos lanzados desde la terminal y progreso: se revisan cada pocos segundos
        self.reloj = QTimer(self)
        self.reloj.timeout.connect(self.revisar)
        self.reloj.start(5000)
        self.revisar()

    # -- pestaña Crear y entrenar -------------------------------------------------------------------
    def pestana_entrenar(self) -> QWidget:
        d = Config()
        self._llenando = False

        # 1 · Corrida
        self.f_nombre = QLineEdit()
        self.f_desde = QComboBox()
        self.f_desde.setToolTip("En vez de empezar con pesos al azar, parte de la mejor red de otra corrida (p. ej. "
                                "una entrenada en el gimnasio, para seguir en un nivel). La forma de la red se copia "
                                "de ella")
        self.f_desde.currentIndexChanged.connect(self.al_cambiar_desde)

        # 2 · Dónde practica
        self.f_entorno = QComboBox()
        self.f_entorno.addItem("Un nivel del juego", "nivel")
        self.f_entorno.addItem("Gimnasio: salas generadas al azar", "gimnasio")
        self.f_entorno.currentIndexChanged.connect(lambda _: self.al_cambiar_entorno(sugerir=True))
        self.l_entorno = QLabel()
        self.l_entorno.setWordWrap(True)
        self.l_entorno.setStyleSheet("color: gray")
        self.f_stage = combo_salas()
        self.f_stage.currentIndexChanged.connect(
            lambda i: self.f_objetivo.setCurrentIndex(min(i + 1, len(SALAS) - 1)))
        self.f_objetivo = combo_salas()
        self.f_dificultad = QComboBox()
        self.f_dificultad.addItems(DIFICULTADES)
        self.f_dificultad.setToolTip("Dificultad del juego (el portal que se elige al empezar)")
        self.f_gim_min = doble(0, 1, d.gim_dificultad_min, 2, paso=0.1)
        self.f_gim_max = doble(0, 1, d.gim_dificultad_max, 2, paso=0.1)
        self.w_gim_dif = QWidget()
        fila = QHBoxLayout(self.w_gim_dif)
        fila.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("de"), self.f_gim_min, QLabel("a"), self.f_gim_max):
            fila.addWidget(w)
        fila.addStretch()
        self.w_gim_dif.setToolTip("Cada sala sale con una dificultad al azar en este rango. 0: escalones, púas y pozos "
                                  "chicos; desde 0,3 muros (doble salto); 0,4 techos de púas; 0,5 islas; 0,6 "
                                  "plataformas móviles. Los obstáculos crecen con ella. Se puede subir al continuar")

        # 3 · Partidas
        self.f_escenarios = spin(1, 100, d.escenarios)
        self.f_segundos = doble(1, 600, d.max_segundos, 0, " s")
        self.f_segundos.setToolTip("Tope de cada partida (escenario o sala)")
        self.f_paciencia = doble(0.5, 600, d.paciencia, 1, " s")
        self.f_paciencia.setToolTip("Si pasa este tiempo sin acercarse a la salida, la partida termina (en el nivel "
                                    "con reintentos: aprieta R)")
        self.f_espera = spin(0, 500, d.espera_max)
        self.f_espera.setSuffix(" frames")
        self.f_espera.setToolTip("Los escenarios después del primero empiezan con una espera al azar sin control "
                                 "(desfasa plataformas y trampas: que no memorice un solo recorrido)")
        self.f_reintentos = QCheckBox("Al morir, apretar R y seguir hasta que se acabe el tiempo")
        self.f_reintentos.toggled.connect(lambda on: self.f_espera_r.setEnabled(on))
        self.f_espera_r = spin(0, 500, d.espera_reintento)
        self.f_espera_r.setSuffix(" frames")
        self.f_espera_r.setToolTip("Espera al azar sin control tras reaparecer. Con 0, la red (sin memoria) repite "
                                   "exactamente la vida anterior")

        # 4 · La red
        self.l_red = QLabel()
        self.l_red.setWordWrap(True)
        self.l_red.setStyleSheet("color: gray")
        self.f_cuadros = spin(1, 8, d.cuadros)
        self.f_cuadros.setToolTip("Cuántas grillas ve a la vez: la actual y las de las decisiones anteriores. Con 1 "
                                  "no puede saber hacia dónde se mueve nada")
        self.f_filtros1 = spin(1, 128, d.filtros1)
        self.f_filtros1.setToolTip("Capa 1 (convolución 4×4): cada detector aprende a reconocer un patrón chico de la "
                                   "grilla (4×4 celdas = 32×32 px), como un borde de suelo o una púa. Más detectores "
                                   "= más patrones distintos, pero más pesos que ajustar")
        self.f_filtros2 = spin(1, 256, d.filtros2)
        self.f_filtros2.setToolTip("Capa 2 (convolución 3×3 sobre la capa 1): combina los patrones chicos en "
                                   "figuras más grandes (un pozo con púas, un muro alto)")
        self.f_oculta = spin(4, 2048, d.oculta)
        self.f_oculta.setToolTip("Capa densa: neuronas que juntan todo lo detectado en la grilla con la velocidad y "
                                 "el doble salto, y de las que sale la puntuación de cada una de las 12 acciones")
        self.f_fpd = spin(1, 25, d.frames_por_decision)
        self.f_fpd.setSuffix(" frames")
        self.f_fpd.setToolTip("Cada cuántos frames decide (mantiene las teclas entre medio); 50 frames = 1 s. "
                              "Menos = más preciso, pero más lento de entrenar")
        self.f_soltar = QCheckBox("Soltar el salto solo para poder volver a saltar")
        self.f_soltar.setToolTip("En el juego el salto sale al apretar la tecla, no al mantenerla. Si la red sigue "
                                 "pidiendo salto y el personaje ya no sube, se suelta la tecla un frame: mantenerla "
                                 "da salto completo, doble salto en la cima y vuelve a saltar al caer. Sin esto, la red "
                                 "tiene que aprender a soltarla ella")
        self.n_pesos = QLabel()
        for s in (self.f_cuadros, self.f_filtros1, self.f_filtros2, self.f_oculta, self.f_fpd):
            s.valueChanged.connect(self.mostrar_pesos)

        # 5 · Recompensas
        self.f_r_avance = doble(0, 100_000, d.r_avance, 1)
        self.f_r_avance.setToolTip("Puntos por recorrer todo el camino hasta la salida (proporcional al avance)")
        self.f_r_objetivo = doble(0, 100_000, d.r_objetivo, 1)
        self.f_r_rapidez = doble(0, 100_000, d.r_rapidez, 2)
        self.f_r_rapidez.setToolTip("Castigo máximo por tardar en llegar a su mejor punto (desempata entre iguales)")
        self.f_castigo = doble(0, 100_000, d.castigo_muerte, 2)
        self.f_castigo.setToolTip("Se resta por cada muerte, salida de la sala o atasco (con reintentos, por cada R)")
        self.f_quieto = doble(0, 100_000, d.castigo_quieto, 2)
        self.f_quieto.setToolTip("Se resta por cada segundo sin desplazarse (en una plataforma móvil no cuenta)")
        self.f_r_save = doble(0, 100_000, d.r_save, 1)
        self.f_r_save.setToolTip("Por cada save point nuevo activado (los del comienzo no cuentan)")
        self.f_explorar = doble(0, 100_000, d.r_explorar, 2)
        self.f_explorar.setToolTip("Por cada celda distinta de la sala por la que pasó vivo en la partida: probar otro "
                                   "camino vale aunque todavía no acerque a la salida")
        self.f_celda = spin(4, 512, d.celda_explorar)
        self.f_celda.setSuffix(" px")
        self.f_celda.setToolTip("Lado de las celdas que cuentan para la exploración")

        # 6 · Evolución
        self.f_poblacion = spin(2, 5000, d.poblacion)
        self.f_padres = spin(1, 1000, d.padres)
        self.f_padres.setToolTip("Las mejores (una por nicho primero) tienen hijos: copias con los pesos mutados")
        self.f_elite = spin(0, 1000, d.elite)
        self.f_elite.setToolTip("Las mejores pasan sin cambios a la generación siguiente")
        self.f_lejos = QCheckBox("La que llega más lejos pasa siempre y tiene hijos")
        self.f_lejos.setToolTip("Además de la élite por fitness: así no se pierde el mayor avance aunque haya muerto "
                                "mucho o tenga castigos")
        self.f_sigma = doble(0.0001, 1, d.sigma, 4, paso=0.005)
        self.f_sigma.setToolTip("Tamaño de la mutación de los pesos (ruido normal × sigma)")
        self.f_generaciones = spin(1, 1_000_000, d.generaciones)
        self.f_workers = spin(1, os.cpu_count() or 1, d.workers)
        self.f_workers.setToolTip("Emuladores en paralelo (uno por núcleo, más o menos)")
        self.f_seed = spin(0, 1_000_000, d.seed)
        self.f_extra = QLineEdit()
        self.f_extra.setPlaceholderText("opcional, p. ej. --nicho-px 32")

        def grupo(titulo: str, filas: list[tuple[str, QWidget]], arriba: QWidget | None = None):
            g = QGroupBox(titulo)
            v = QVBoxLayout(g)
            if arriba is not None:
                v.addWidget(arriba)
            f = QFormLayout()
            for etiqueta, w in filas:
                f.addRow(etiqueta, w)
            v.addLayout(f)
            return g, f

        g_corrida, _ = grupo("1 · Corrida", [("Nombre", self.f_nombre), ("Partir de la red de", self.f_desde)])
        g_donde, self.form_donde = grupo("2 · Dónde practica", [
            ("Entorno", self.f_entorno), ("Sala", self.f_stage), ("Sala de llegada", self.f_objetivo),
            ("Dificultad del juego", self.f_dificultad), ("Dificultad de las salas", self.w_gim_dif)],
            arriba=self.l_entorno)
        g_partidas, self.form_partidas = grupo("3 · Partidas de cada red", [
            ("Escenarios por red", self.f_escenarios), ("Duración máxima", self.f_segundos),
            ("Paciencia sin avanzar", self.f_paciencia), ("Espera al azar al comenzar", self.f_espera),
            ("", self.f_reintentos), ("Espera al reaparecer", self.f_espera_r)])
        g_red, _ = grupo("4 · La red", [
            ("Grillas que ve a la vez", self.f_cuadros), ("Detectores de patrones chicos", self.f_filtros1),
            ("Detectores de patrones grandes", self.f_filtros2), ("Neuronas de decisión", self.f_oculta),
            ("Decide cada", self.f_fpd), ("", self.f_soltar), ("Tamaño", self.n_pesos)], arriba=self.l_red)
        g_premios, self.form_premios = grupo("5 · Recompensas (por partida)", [
            ("Avance (todo el camino)", self.f_r_avance), ("Llegar a la salida", self.f_r_objetivo),
            ("Rapidez (castigo máx.)", self.f_r_rapidez), ("Castigo por muerte", self.f_castigo),
            ("Castigo por segundo quieto", self.f_quieto), ("Premio por save nuevo", self.f_r_save),
            ("Premio por celda explorada", self.f_explorar), ("Tamaño de la celda", self.f_celda)])
        g_evo, _ = grupo("6 · Evolución", [
            ("Redes por generación", self.f_poblacion), ("Padres", self.f_padres),
            ("Élite (pasan sin cambios)", self.f_elite), ("", self.f_lejos), ("Sigma de la mutación", self.f_sigma),
            ("Hasta la generación", self.f_generaciones), ("Procesos en paralelo", self.f_workers),
            ("Semilla", self.f_seed)])
        izq, der = QVBoxLayout(), QVBoxLayout()
        for g in (g_corrida, g_donde, g_partidas):
            izq.addWidget(g)
        for g in (g_red, g_premios, g_evo):
            der.addWidget(g)
        izq.addStretch()
        der.addStretch()
        columnas = QHBoxLayout()
        columnas.addLayout(izq, 1)
        columnas.addLayout(der, 1)

        self.b_entrenar = QPushButton("Crear y entrenar desde cero")
        self.b_entrenar.clicked.connect(lambda: self.entrenar(continuar=False))
        self.b_continuar = QPushButton("Continuar")
        self.b_continuar.setToolTip("Sigue desde la última generación guardada. Se pueden cambiar recompensas, "
                                    "evolución, partidas y la dificultad de las salas; no el entorno ni la red")
        self.b_continuar.clicked.connect(lambda: self.entrenar(continuar=True))
        self.b_detener = QPushButton("Detener")
        self.b_detener.clicked.connect(self.detener_entrenamiento)
        self.barra = QProgressBar()
        self.estado = QLabel("Sin entrenamiento en curso")
        self.estado.setWordWrap(True)

        fila = QHBoxLayout()
        for b in (self.b_entrenar, self.b_continuar, self.b_detener):
            fila.addWidget(b)
        v = QVBoxLayout()
        v.addLayout(columnas)
        f = QFormLayout()
        f.addRow("Otras opciones", self.f_extra)
        v.addLayout(f)
        v.addLayout(fila)
        v.addWidget(self.barra)
        v.addWidget(self.estado)
        v.addStretch()
        w = QWidget()
        w.setLayout(v)
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        return scroll

    def mostrar_pesos(self) -> None:
        cfg = Config(cuadros=self.f_cuadros.value(), filtros1=self.f_filtros1.value(),
                     filtros2=self.f_filtros2.value(), oculta=self.f_oculta.value())
        vista = cfg.config_vista
        n = red_para(cfg).n_pesos
        self.n_pesos.setText(f"{n:,} pesos".replace(",", "."))
        self.l_red.setText(
            f"La red ve una grilla de {vista.ancho}×{vista.alto} celdas de {vista.celda} px alrededor del jugador, con "
            f"{vista.n_canales} capas de información (sólido, plataforma, púas, save, salida, agua, fuera de la "
            f"sala), y las {self.f_cuadros.value()} últimas a la vez. {self.f_filtros1.value()} detectores buscan "
            f"patrones chicos; {self.f_filtros2.value()} los combinan en figuras más grandes; "
            f"{self.f_oculta.value()} neuronas de decisión juntan eso con la velocidad y el doble salto y puntúan "
            f"las 12 acciones (decide cada {self.f_fpd.value()} frame{'s' if self.f_fpd.value() > 1 else ''}).")

    def llenar_formulario(self, cfg: Config) -> None:
        self._llenando = True
        self.f_nombre.setText(cfg.nombre)
        self.llenar_desde(cfg.desde)
        self.f_entorno.setCurrentIndex(max(0, self.f_entorno.findData(cfg.entorno)))
        self.f_stage.setCurrentText(cfg.start_room)
        self.f_objetivo.setCurrentText(cfg.sala_objetivo)
        self.f_dificultad.setCurrentText(cfg.difficulty)
        self.f_gim_min.setValue(cfg.gim_dificultad_min)
        self.f_gim_max.setValue(cfg.gim_dificultad_max)
        self.f_cuadros.setValue(cfg.cuadros)
        self.f_filtros1.setValue(cfg.filtros1)
        self.f_filtros2.setValue(cfg.filtros2)
        self.f_oculta.setValue(cfg.oculta)
        self.f_fpd.setValue(cfg.frames_por_decision)
        self.f_soltar.setChecked(bool(cfg.soltar_salto))
        self.f_escenarios.setValue(cfg.escenarios)
        self.f_espera.setValue(cfg.espera_max)
        self.f_segundos.setValue(cfg.max_segundos)
        self.f_paciencia.setValue(cfg.paciencia)
        self.f_reintentos.setChecked(bool(cfg.reintentos))
        self.f_espera_r.setValue(cfg.espera_reintento)
        self.f_espera_r.setEnabled(bool(cfg.reintentos))
        self.f_r_avance.setValue(cfg.r_avance)
        self.f_r_objetivo.setValue(cfg.r_objetivo)
        self.f_r_rapidez.setValue(cfg.r_rapidez)
        self.f_castigo.setValue(cfg.castigo_muerte)
        self.f_quieto.setValue(cfg.castigo_quieto)
        self.f_r_save.setValue(cfg.r_save)
        self.f_explorar.setValue(cfg.r_explorar)
        self.f_celda.setValue(cfg.celda_explorar)
        self.f_poblacion.setValue(cfg.poblacion)
        self.f_padres.setValue(cfg.padres)
        self.f_elite.setValue(cfg.elite)
        self.f_sigma.setValue(cfg.sigma)
        self.f_lejos.setChecked(bool(cfg.elite_mas_lejos))
        self.f_generaciones.setValue(cfg.generaciones)
        self.f_workers.setValue(min(cfg.workers, self.f_workers.maximum()))
        self.f_seed.setValue(cfg.seed)
        self.f_extra.clear()
        self.al_cambiar_entorno()
        self.al_cambiar_desde()
        self.mostrar_pesos()
        self._llenando = False

    def llenar_desde(self, actual: str = "") -> None:
        self.f_desde.blockSignals(True)
        self.f_desde.clear()
        self.f_desde.addItem("nada: pesos al azar", "")
        for p in sorted(almacen.CORRIDAS.glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not p.parent.name.startswith("_") and almacen.generaciones_guardadas(p.parent):
                self.f_desde.addItem(p.parent.name, p.parent.name)
        if actual and self.f_desde.findData(actual) < 0:
            self.f_desde.addItem(actual, actual)
        self.f_desde.setCurrentIndex(max(0, self.f_desde.findData(actual)))
        self.f_desde.blockSignals(False)

    def al_cambiar_entorno(self, sugerir: bool = False) -> None:
        """Muestra solo los campos que usa el entorno elegido."""
        gim = self.f_entorno.currentData() == "gimnasio"
        for w in (self.f_stage, self.f_objetivo, self.f_dificultad):
            self.form_donde.setRowVisible(w, not gim)
        self.form_donde.setRowVisible(self.w_gim_dif, gim)
        for w in (self.f_espera, self.f_reintentos, self.f_espera_r):
            self.form_partidas.setRowVisible(w, not gim)
        for w in (self.f_quieto, self.f_r_save, self.f_explorar, self.f_celda):
            self.form_premios.setRowVisible(w, not gim)
        self.form_partidas.labelForField(self.f_escenarios).setText("Salas por generación" if gim else
                                                                    "Escenarios por red")
        self.f_escenarios.setToolTip(
            "Salas nuevas en cada generación, las mismas para todas las redes (no las puede memorizar); su fitness "
            "es el promedio. Más salas = más justo, pero más lento" if gim else
            "Partidas por red y generación: la primera desde el comienzo y las demás con una espera al azar; su "
            "fitness es el promedio (para que no memorice un solo recorrido)")
        self.l_entorno.setText(
            "Cada partida es una sala de una pantalla generada al azar dentro del juego (saltos, púas, pozos, muros, "
            "techos, islas, plataformas móviles), con la física real. Termina al llegar, morir, a la duración máxima "
            "o sin avanzar. Para practicar antes de un nivel: después crea otra corrida que parta de esta red."
            if gim else
            "Entrena en una sala del GB hasta la sala de llegada; el avance se mide por el camino más corto a la "
            "salida.")
        if sugerir and not self._llenando:     # valores razonables para cada entorno
            self.f_escenarios.setValue(8 if gim else Config.escenarios)
            self.f_segundos.setValue(15 if gim else Config.max_segundos)
            self.f_paciencia.setValue(4 if gim else Config.paciencia)

    def al_cambiar_desde(self) -> None:
        """Con una red de partida, la forma de la red y la vista son las suyas (no se pueden elegir)."""
        origen = self.f_desde.currentData()
        for w in (self.f_cuadros, self.f_filtros1, self.f_filtros2, self.f_oculta, self.f_fpd):
            w.setEnabled(not origen)
        if origen and almacen.dir_corrida(origen).exists():
            fuente = almacen.cargar_config(almacen.dir_corrida(origen))
            self.f_cuadros.setValue(fuente.cuadros)
            self.f_filtros1.setValue(fuente.filtros1)
            self.f_filtros2.setValue(fuente.filtros2)
            self.f_oculta.setValue(fuente.oculta)
            self.f_fpd.setValue(fuente.frames_por_decision)

    def valores(self) -> dict:
        gim = self.f_entorno.currentData() == "gimnasio"
        return {
            "entorno": self.f_entorno.currentData(), "desde": self.f_desde.currentData(),
            "start_room": self.f_stage.currentText(), "sala_objetivo": self.f_objetivo.currentText(),
            "difficulty": self.f_dificultad.currentText(),
            "gim_dificultad_min": float(self.f_gim_min.value()),
            "gim_dificultad_max": float(max(self.f_gim_max.value(), self.f_gim_min.value())),
            "cuadros": self.f_cuadros.value(), "filtros1": self.f_filtros1.value(),
            "filtros2": self.f_filtros2.value(), "oculta": self.f_oculta.value(),
            "frames_por_decision": self.f_fpd.value(), "soltar_salto": int(self.f_soltar.isChecked()),
            "escenarios": self.f_escenarios.value(), "espera_max": self.f_espera.value(),
            "max_segundos": float(self.f_segundos.value()), "paciencia": float(self.f_paciencia.value()),
            "reintentos": int(self.f_reintentos.isChecked() and not gim), "espera_reintento": self.f_espera_r.value(),
            "r_avance": float(self.f_r_avance.value()), "r_objetivo": float(self.f_r_objetivo.value()),
            "r_rapidez": float(self.f_r_rapidez.value()), "castigo_muerte": float(self.f_castigo.value()),
            "castigo_quieto": float(self.f_quieto.value()), "r_save": float(self.f_r_save.value()),
            "r_explorar": float(self.f_explorar.value()), "celda_explorar": self.f_celda.value(),
            "poblacion": self.f_poblacion.value(), "padres": self.f_padres.value(), "elite": self.f_elite.value(),
            "elite_mas_lejos": int(self.f_lejos.isChecked()),
            "sigma": float(self.f_sigma.value()), "generaciones": self.f_generaciones.value(),
            "workers": self.f_workers.value(), "seed": self.f_seed.value(),
        }

    def entrenar(self, continuar: bool) -> None:
        nombre = self.f_nombre.text().strip()
        if not re.fullmatch(r"[\w.-]+", nombre) or nombre.startswith("_"):
            QMessageBox.warning(self, "Nombre", "Usa un nombre sin espacios (letras, números, _ . -) que no empiece "
                                "con _.")
            return
        if nombre in self.entrenando():
            QMessageBox.information(self, "Entrenar", f"«{nombre}» ya se está entrenando.")
            return
        vals = self.valores()
        if vals["padres"] > vals["poblacion"] or vals["elite"] > vals["poblacion"]:
            QMessageBox.warning(self, "Evolución", "Los padres y la élite no pueden ser más que las redes por "
                                "generación.")
            return
        carpeta = almacen.dir_corrida(nombre)
        hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
        args = ["redes.entrenar", "--nombre", nombre]
        if continuar:
            if not hechas:
                QMessageBox.information(self, "Continuar", f"«{nombre}» no tiene generaciones guardadas.")
                return
            guardada = almacen.cargar_config(carpeta)
            cambiados = [k for k in FIJOS if k in vals and getattr(guardada, k) != vals[k]]
            if cambiados:
                QMessageBox.information(self, "Continuar", "Al continuar no se pueden cambiar el entorno, la sala, la "
                                        "dificultad del juego, la red de partida ni la forma de la red ("
                                        + ", ".join(cambiados) + "): se usan los de la corrida. Para probar otros, "
                                        "crea una red nueva.")
            vals = {k: v for k, v in vals.items() if k not in FIJOS and (getattr(guardada, k) != v
                                                                          or k == "generaciones")}
            if vals["generaciones"] <= hechas[-1] + 1:
                QMessageBox.information(self, "Continuar", f"Ya tiene {hechas[-1] + 1} generaciones: "
                                        "sube «Hasta la generación».")
                return
            args.append("--continuar")
        elif hechas:
            r = QMessageBox.question(self, "Sobrescribir", f"«{nombre}» ya tiene {len(hechas)} generaciones. "
                                     "¿Borrarlas y empezar de cero?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
            if r != QMessageBox.Yes:
                return
            args.append("--sobrescribir")
        for k, v in vals.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        try:
            args += shlex.split(self.f_extra.text())
        except ValueError as e:
            QMessageBox.warning(self, "Otras opciones", str(e))
            return
        self.lanzar(f"entrenar {nombre}", args)
        self.estado.setText(f"Entrenando «{nombre}»… (si es nueva, primero arma el mapa de la sala)")
        QTimer.singleShot(1500, self.revisar)

    def entrenando(self) -> dict[str, int]:
        """{corrida: pid} en curso, de la app o de la terminal."""
        propios = {p.args[2]: p.popen.pid for p in self.procesos if p.args[0] == "redes.entrenar" and p.vivo()}
        return {**self.externos, **propios}

    def detener_entrenamiento(self) -> None:
        nombre = self.corrida_actual() or self.f_nombre.text().strip()
        for p in self.procesos:
            if p.args[0] == "redes.entrenar" and p.args[2] == nombre:
                p.detener()
                self.estado.setText(f"Deteniendo «{nombre}»: termina la generación en curso…")
                return
        pid = self.externos.get(nombre)
        if pid:
            os.kill(pid, signal.SIGINT)  # lanzado desde la terminal: Ctrl+C, guarda lo hecho
            self.estado.setText(f"Deteniendo «{nombre}» (lanzado desde la terminal)…")

    def revisar(self) -> None:
        """Entrenamientos en curso (también los de la terminal), botones, barra y resumen."""
        self.externos = entrenamientos_en_curso()
        nombre = self.corrida_actual() or self.f_nombre.text().strip()
        corriendo = nombre in self.entrenando()
        self.b_entrenar.setEnabled(not corriendo)
        self.b_continuar.setEnabled(not corriendo)
        self.b_detener.setEnabled(corriendo)
        if not nombre:
            return
        carpeta = almacen.dir_corrida(nombre)
        try:
            cfg = almacen.cargar_config(carpeta)
            filas = almacen.leer_metricas(carpeta)
        except (OSError, ValueError):
            return
        if corriendo or filas:
            self.barra.setRange(0, cfg.generaciones)
            self.barra.setValue(min(len(filas), cfg.generaciones))
        if corriendo:
            if filas:
                u = filas[-1]
                salas = (f" · salas superadas {u['salas_pct']:.0f} %"
                         if cfg.entorno == "gimnasio" and "salas_pct" in u else "")
                self.estado.setText(f"Entrenando «{nombre}»"
                                    f"{' (lanzado desde la terminal)' if nombre in self.externos else ''} · "
                                    f"generación {int(u['gen']) + 1} de {cfg.generaciones} · avance del mejor "
                                    f"{u['avance_mejor']:.1f} %{salas} · {u['segundos_reloj']:.0f} s por generación")
            self.mostrar_resumen()
        elif not any(p.args[0] == "redes.entrenar" for p in self.procesos):
            self.estado.setText(f"«{nombre}»: sin entrenamiento en curso")

    # -- pestaña Probar -------------------------------------------------------------------------------
    def pestana_probar(self) -> QWidget:
        self.p_gen = spin(-1, 1_000_000, -1, "última")
        self.p_sala = QComboBox()
        self.p_sala.addItem("Donde entrenó", "suya")
        self.p_sala.addItem("Gimnasio: salas generadas", "gimnasio")
        self.p_sala.addItem("Todos los niveles (rStage01 a rStage17)", "todas")
        for s in SALAS:
            self.p_sala.addItem(s, s)
        self.p_sala.currentIndexChanged.connect(self.al_cambiar_prueba)
        self.p_escenarios = spin(1, 200, 10)
        self.p_dif = doble(0, 1, 0.5, 2, paso=0.1)
        self.p_dif_azar = QCheckBox("al azar en cada sala")
        self.p_dif_azar.toggled.connect(lambda on: self.p_dif.setEnabled(not on))
        self.w_p_dif = QWidget()
        fila = QHBoxLayout(self.w_p_dif)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.addWidget(self.p_dif)
        fila.addWidget(self.p_dif_azar)
        fila.addStretch()
        self.p_verificar = QCheckBox("Solo salas con solución (la búsqueda tarda unos segundos por sala)")
        self.p_segundos = doble(0, 600, 0, 0, " s")
        self.p_segundos.setSpecialValueText("el de la corrida")
        self.p_video = QCheckBox("También un video")
        b = QPushButton("Probar la mejor red")
        b.clicked.connect(self.probar)
        self.form_probar = QFormLayout()
        self.form_probar.addRow("Red de la generación", self.p_gen)
        self.form_probar.addRow("Dónde", self.p_sala)
        self.form_probar.addRow("Partidas por sala", self.p_escenarios)
        self.form_probar.addRow("Dificultad de las salas", self.w_p_dif)
        self.form_probar.addRow("", self.p_verificar)
        self.form_probar.addRow("Tope por partida", self.p_segundos)
        self.form_probar.addRow("", self.p_video)
        self.form_probar.addRow("", b)

        self.tabla = QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(["Dónde", "Gen", "Avance medio", "Máximo", "Llegó", "Cómo terminan",
                                              "Fecha"])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tabla.horizontalHeader().setStretchLastSection(True)

        v = QVBoxLayout()
        explicacion = QLabel("Juega la mejor red sin entrenarla más. Donde entrenó mide si es robusta (esperas o salas "
                             "que no vio); en otros niveles o en el gimnasio, si <b>generaliza</b>. Las salas del "
                             "gimnasio de la prueba no son las de su entrenamiento.")
        explicacion.setWordWrap(True)
        v.addWidget(explicacion)
        v.addLayout(self.form_probar)
        v.addWidget(QLabel("<b>Resultados</b> (donde entrenó, en cursiva)"))
        v.addWidget(self.tabla, 1)
        w = QWidget()
        w.setLayout(v)
        self.al_cambiar_prueba()
        return w

    def donde_prueba(self) -> str:
        """"gimnasio", "todas", un nivel o "" (el nivel donde entrenó)."""
        donde = self.p_sala.currentData()
        if donde == "suya":
            cfg = self.config_actual()
            return "gimnasio" if cfg and cfg.entorno == "gimnasio" else ""
        return donde

    def al_cambiar_prueba(self) -> None:
        gim = self.donde_prueba() == "gimnasio"
        self.form_probar.setRowVisible(self.w_p_dif, gim)
        self.form_probar.setRowVisible(self.p_verificar, gim)
        self.form_probar.setRowVisible(self.p_segundos, not gim)
        self.form_probar.labelForField(self.p_escenarios).setText("Salas" if gim else "Partidas por sala")
        self.p_escenarios.setToolTip("Cuántas salas generadas juega (de una partida cada una)" if gim else
                                     "Partidas por sala, con esperas repartidas entre 0 y la espera máxima")

    def probar(self) -> None:
        nombre = self.corrida_con_generaciones("Probar")
        if not nombre:
            return
        args = ["redes.probar", "--corrida", nombre, "--gen", str(self.p_gen.value()),
                "--escenarios", str(self.p_escenarios.value())]
        donde = self.donde_prueba()
        if donde == "gimnasio":
            dif = -1.0 if self.p_dif_azar.isChecked() else self.p_dif.value()
            args += ["--gimnasio", "--dificultad", f"{dif:g}"]
            if self.p_verificar.isChecked():
                args.append("--verificar")
        else:
            if donde:
                args += ["--sala", donde]
            if self.p_segundos.value() > 0:
                args += ["--max-segundos", str(self.p_segundos.value())]
        if self.p_video.isChecked():
            args.append("--video")
        self.lanzar(f"probar {nombre}", args)

    def mostrar_pruebas(self) -> None:
        self.tabla.setRowCount(0)
        nombre = self.corrida_actual()
        if not nombre:
            return
        filas = []
        for ruta in (almacen.dir_corrida(nombre) / "pruebas").glob("*.json"):
            try:
                filas.append(json.loads(ruta.read_text()))
            except (OSError, ValueError):
                pass
        filas.sort(key=lambda r: (not r["entrenada"], r["sala"], -r["gen"]))
        cursiva = QFont()
        cursiva.setItalic(True)
        for r in filas:
            fila = self.tabla.rowCount()
            self.tabla.insertRow(fila)
            fines = ", ".join(f"{n} {FINES_CORTOS.get(f, f)}" for f, n in
                              sorted(r["fines"].items(), key=lambda x: -x[1]))
            textos = [r["sala"], str(r["gen"]), f"{100 * r['avance_medio']:.1f} %", f"{100 * r['avance_max']:.1f} %",
                      f"{r['llegaron']} de {len(r['partidas'])}", fines, r["fecha"]]
            for c, t in enumerate(textos):
                it = QTableWidgetItem(t)
                if r["entrenada"]:
                    it.setFont(cursiva)
                if c in (1, 2, 3):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tabla.setItem(fila, c, it)

    # -- pestaña Videos y en vivo -------------------------------------------------------------------
    def pestana_visualizar(self) -> QWidget:
        self.v_quien = QComboBox()
        self.v_quien.addItem("La mejor red de la corrida", "mejor")
        self.v_quien.addItem("Todas las redes de la generación", "todas")
        self.v_quien.addItem("Acciones al azar (gimnasio)", "azar")
        self.v_quien.addItem("Solución de la búsqueda (gimnasio)", "solucion")
        self.v_quien.setToolTip("«La mejor» = la que llega más lejos. «Todas» las dibuja juntas, como el GA (donde "
                                "entrenó). Al azar y la solución sirven para ver las salas del gimnasio: la solución "
                                "comprueba que cada sala se pueda pasar")
        self.v_donde = QComboBox()
        self.v_donde.addItem("Donde entrenó (en el gimnasio: el entrenamiento en vivo)", "suya")
        self.v_donde.addItem("Prueba en el gimnasio: salas nuevas que nunca vio", "gimnasio")
        for s in SALAS:
            self.v_donde.addItem(s, s)
        self.v_donde.setToolTip("Donde entrenó: su nivel o, si entrena en el gimnasio, las mismas salas que jugó la "
                                "población en la última generación (en vivo pasa a la nueva: se ve cómo aprende). La "
                                "prueba en el gimnasio y los otros niveles son salas que la red nunca vio: sirven "
                                "para ver si generaliza")
        for c in (self.v_quien, self.v_donde):
            c.currentIndexChanged.connect(self.al_cambiar_visual)
        self.v_gen = spin(-1, 1_000_000, -1, "última")
        self.v_gen.setToolTip("Para los videos; en vivo juega siempre la última y salta a la nueva mientras entrena")
        self.v_espera = spin(0, 500, 0)
        self.v_espera.setSuffix(" frames")
        self.v_espera.setToolTip("Frames sin control al comienzo (0 = el escenario de siempre)")
        self.v_dif = doble(0, 1, 0.5, 2, paso=0.1)
        self.v_dif_azar = QCheckBox("al azar en cada sala")
        self.v_dif_azar.toggled.connect(lambda on: self.v_dif.setEnabled(not on))
        self.w_v_dif = QWidget()
        fila = QHBoxLayout(self.w_v_dif)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.addWidget(self.v_dif)
        fila.addWidget(self.v_dif_azar)
        fila.addStretch()
        self.v_semilla = spin(0, 1_000_000, 0)
        self.v_semilla.setToolTip("Las salas van de esta semilla en adelante: la misma semilla da la misma sala")
        self.v_salas = spin(1, 200, 10)
        self.v_salas.setToolTip("Salas del video o de la imagen de muestras")
        self.v_verificar = QCheckBox("Solo salas con solución (la búsqueda tarda unos segundos por sala)")
        self.v_vel = spin(1, 16, 1)
        self.v_vel.setSuffix("×")
        self.v_vel.setToolTip("Veces el tiempo real (en vivo y el video de todas las redes); se saltan cuadros")
        self.form_visual = QFormLayout()
        for etiqueta, w in (("Quién juega", self.v_quien), ("Dónde", self.v_donde), ("Generación (videos)", self.v_gen),
                            ("Espera al comenzar (video)", self.v_espera), ("Dificultad de las salas", self.w_v_dif),
                            ("Semilla de la primera sala", self.v_semilla), ("Salas (video y muestras)", self.v_salas),
                            ("", self.v_verificar), ("Velocidad", self.v_vel)):
            self.form_visual.addRow(etiqueta, w)

        grilla = QGridLayout()
        botones = [
            ("En vivo (ventana)", "vivo", "Una ventana que juega sin parar; con una corrida entrenando, toma la red "
                                          "nueva. Corre con tope de memoria: sirve para streams"),
            ("Grabar video", "video", "Un video en la carpeta videos/ de la corrida (o de _gimnasio)"),
            ("Muestras de salas (imagen)", "muestras", "Una imagen con salas del gimnasio; con «solo salas con "
                                                       "solución», cada una con el camino que encontró la búsqueda"),
            ("Gráficos de la corrida", "graficos", "Avance, fitness, cómo terminan las partidas, muertes y saves"),
            ("Abrir carpeta de videos", "carpeta-videos", ""),
            ("Abrir carpeta del gimnasio", "carpeta-gimnasio", ""),
        ]
        for i, (texto, modo, ayuda) in enumerate(botones):
            b = QPushButton(texto)
            if ayuda:
                b.setToolTip(ayuda)
            b.clicked.connect(lambda _=False, m=modo: self.visualizar(m))
            grilla.addWidget(b, i // 3, i % 3)
        self.v_nota = QLabel()
        self.v_nota.setWordWrap(True)
        self.v_nota.setStyleSheet("color: gray")

        self.grafico = QLabel("Pulsa «Gráficos de la corrida» o «Muestras de salas».")
        self.grafico.setAlignment(Qt.AlignCenter)
        self.grafico.setMinimumHeight(200)
        v = QVBoxLayout()
        v.addLayout(self.form_visual)
        v.addWidget(self.v_nota)
        v.addLayout(grilla)
        v.addWidget(self.grafico, 1)
        w = QWidget()
        w.setLayout(v)
        self.al_cambiar_visual()
        return w

    def config_actual(self) -> Config | None:
        nombre = self.corrida_actual()
        try:
            return almacen.cargar_config(almacen.dir_corrida(nombre)) if nombre else None
        except (OSError, ValueError):
            return None

    def donde_visual(self) -> str:
        """Dónde se juega en «Videos y en vivo»: "entreno" (las salas de la generación de una corrida del gimnasio),
        "gimnasio" (salas nuevas), un nivel, o "" (el nivel donde entrenó)."""
        quien, donde = self.v_quien.currentData(), self.v_donde.currentData()
        cfg = self.config_actual()
        suya = "entreno" if cfg and cfg.entorno == "gimnasio" else ""
        if quien in ("azar", "solucion"):
            return "gimnasio"
        if quien == "todas":
            return suya
        return suya if donde == "suya" else donde

    def al_cambiar_visual(self) -> None:
        quien, donde = self.v_quien.currentData(), self.donde_visual()
        gim = donde == "gimnasio"
        self.v_donde.setEnabled(quien == "mejor")
        f = self.form_visual
        f.setRowVisible(self.v_gen, quien in ("mejor", "todas"))
        f.setRowVisible(self.v_espera, quien == "mejor" and donde not in ("gimnasio", "entreno"))
        for w in (self.w_v_dif, self.v_semilla, self.v_salas, self.v_verificar):
            f.setRowVisible(w, gim)
        if donde == "entreno":
            nota = ("<b>Entrenamiento en vivo</b>: " +
                    ("la mejor red de la última generación en las mismas salas que jugó la población, con lo que ve "
                     "por dentro y cómo le fue al entrenar." if quien == "mejor" else
                     "toda la población en las salas de su generación: la mejor emulada en vivo y las demás como "
                     "fantasmas, desde lo que jugaron al entrenar.") +
                    " En vivo, cuando aparece una generación nueva pasa a sus salas: se ve cómo aprende.")
        elif quien == "todas":
            nota = "Todas las redes de la generación sobre el mapa del nivel donde entrenó, como el GA."
        elif gim and quien == "mejor":
            nota = ("<b>Prueba</b>: salas nuevas que la red nunca vio, una tras otra, con el historial de cuántas "
                    "supera: sirve para ver si generaliza.")
        else:
            nota = {"azar": "Para ver salas del gimnasio con alguien moviéndose.",
                    "solucion": "Antes de cada sala, una búsqueda encuentra cómo pasarla y se ve la solución."}.get(
                quien, "El video muestra el juego, el puntaje, la red por dentro (nodos y conexiones) y qué acción "
                       "prefiere en cada momento.")
        self.v_nota.setText(nota)

    def visualizar(self, modo: str) -> None:
        if modo == "carpeta-videos":
            self.abrir("videos")
            return
        if modo == "carpeta-gimnasio":
            self.abrir_ruta(almacen.CORRIDAS / "_gimnasio")
            return
        quien, donde = self.v_quien.currentData(), self.donde_visual()
        dif = -1.0 if self.v_dif_azar.isChecked() else self.v_dif.value()
        gimnasio = ["--dificultad", f"{dif:g}", "--semilla", str(self.v_semilla.value())]
        if self.v_verificar.isChecked():
            gimnasio.append("--verificar")
        if modo == "muestras":
            self.lanzar("muestras de salas", ["redes.gimnasio_ver", "muestras", *gimnasio,
                                              "--salas", str(self.v_salas.value())])
            return
        nombre = ""
        if modo == "graficos" or quien in ("mejor", "todas"):
            nombre = self.corrida_con_generaciones("Videos y en vivo")
            if not nombre:
                return
        if modo == "graficos":
            self.lanzar(f"gráficos {nombre}", ["redes.visualizar", "graficos", "--corrida", nombre])
            return
        vel = ["--velocidad", str(self.v_vel.value())]
        gen = ["--gen", str(self.v_gen.value())]
        if donde == "entreno":
            args = (["redes.gimnasio_ver", "entreno", "--agente", nombre, "--quien", quien, *vel] if modo == "vivo"
                    else ["redes.gimnasio_ver", "entreno-video", "--agente", nombre, "--quien", quien, *gen, *vel])
        elif quien == "todas":
            args = (["redes.visualizar", "vivo", "--corrida", nombre, *vel] if modo == "vivo" else
                    ["redes.visualizar", "generacion", "--corrida", nombre, *gen, *vel])
        elif donde == "gimnasio":
            agente = nombre or quien
            args = ["redes.gimnasio_ver", modo, "--agente", agente, *gimnasio]
            args += vel if modo == "vivo" else ["--salas", str(self.v_salas.value()), *(gen if nombre else [])]
        else:
            sala = ["--sala", donde] if donde else []
            args = (["redes.visualizar", "vivo-red", "--corrida", nombre, *sala, *vel] if modo == "vivo" else
                    ["redes.visualizar", "mejor", "--corrida", nombre, *sala, *gen,
                     "--espera", str(self.v_espera.value())])
        self.lanzar(f"{'en vivo' if modo == 'vivo' else 'video'} {nombre or quien}", args)

    def mostrar_imagen(self, ruta: str) -> None:
        pix = QPixmap(ruta)
        if not pix.isNull():
            self.grafico.setPixmap(pix.scaled(max(self.grafico.width(), 400), max(self.grafico.height(), 200),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def abrir_ruta(self, ruta: Path) -> None:
        ruta.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    def mostrar_grafico(self) -> None:
        nombre = self.corrida_actual()
        ruta = almacen.dir_corrida(nombre) / "graficos" / "avance.png" if nombre else None
        if ruta and ruta.exists():
            self.mostrar_imagen(str(ruta))
        else:
            self.grafico.clear()
            self.grafico.setText("Pulsa «Gráficos de la corrida» o «Muestras de salas».")

    # -- corridas -----------------------------------------------------------------------------------
    def actualizar_lista(self) -> None:
        actual = self.corrida_actual()
        self.lista.blockSignals(True)
        self.lista.clear()
        carpetas = sorted((p for p in almacen.CORRIDAS.glob("*/config.json") if not p.parent.name.startswith("_")),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for p in carpetas:
            n = len(almacen.generaciones_guardadas(p.parent))
            item = QListWidgetItem(f"{p.parent.name}   ({n} gen)")
            item.setData(Qt.UserRole, p.parent.name)
            self.lista.addItem(item)
            if p.parent.name == actual:
                self.lista.setCurrentItem(item)
        self.lista.blockSignals(False)
        if self.lista.currentItem() is None and self.lista.count():
            self.lista.setCurrentRow(0)
        else:
            self.mostrar_todo()

    def corrida_actual(self) -> str:
        item = self.lista.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def corrida_con_generaciones(self, titulo: str) -> str:
        nombre = self.corrida_actual()
        if not nombre or not almacen.generaciones_guardadas(almacen.dir_corrida(nombre)):
            QMessageBox.information(self, titulo, "Elige a la izquierda una corrida con generaciones.")
            return ""
        return nombre

    def al_elegir(self, item, _anterior) -> None:
        if item is None:
            return
        cfg = almacen.cargar_config(almacen.dir_corrida(item.data(Qt.UserRole)))
        self.llenar_formulario(cfg)
        if cfg.entorno == "gimnasio":   # videos y pruebas, por defecto con la dificultad con que entrena
            self.v_dif.setValue(cfg.gim_dificultad_max)
            self.p_dif.setValue(cfg.gim_dificultad_max)
        self.mostrar_todo()
        self.revisar()

    def mostrar_todo(self) -> None:
        self.mostrar_resumen()
        self.mostrar_pruebas()
        self.mostrar_grafico()
        self.al_cambiar_prueba()
        self.al_cambiar_visual()

    def nueva(self) -> None:
        self.lista.blockSignals(True)
        self.lista.setCurrentItem(None)
        self.lista.blockSignals(False)
        cfg = Config()
        n = 1
        while almacen.dir_corrida(f"red{n:02d}").exists():
            n += 1
        cfg.nombre = f"red{n:02d}"
        self.llenar_formulario(cfg)
        self.resumen.setText("Red nueva: ajusta los parámetros y pulsa «Crear y entrenar desde cero».")
        self.tabla.setRowCount(0)
        self.mostrar_grafico()
        self.tabs.setCurrentIndex(0)
        self.revisar()

    def mostrar_resumen(self) -> None:
        nombre = self.corrida_actual()
        if not nombre:
            return
        carpeta = almacen.dir_corrida(nombre)
        try:
            cfg = almacen.cargar_config(carpeta)
        except (OSError, ValueError):
            return
        gim = cfg.entorno == "gimnasio"
        pesos = f"{red_para(cfg).n_pesos:,}".replace(",", ".")
        premios = [(cfg.r_avance, f"avance {cfg.r_avance:g}"), (cfg.r_objetivo, f"llegar {cfg.r_objetivo:g}"),
                   (cfg.r_rapidez, f"rapidez −{cfg.r_rapidez:g}"), (cfg.castigo_muerte, f"muerte −{cfg.castigo_muerte:g}")]
        if not gim:
            premios += [(cfg.castigo_quieto, f"quieto −{cfg.castigo_quieto:g}/s"), (cfg.r_save, f"save +{cfg.r_save:g}"),
                        (cfg.r_explorar, f"celda de {cfg.celda_explorar} px +{cfg.r_explorar:g}")]
        premios = [t for v, t in premios if v]   # las que valen 0 no se aplican
        texto = [
            f"<b>{nombre}</b>",
            (f"<b>gimnasio</b>: {cfg.escenarios} salas nuevas por generación, dificultad "
             f"{cfg.gim_dificultad_min:g}–{cfg.gim_dificultad_max:g}" if gim else
             f"<b>nivel</b>: {cfg.start_room} → {cfg.sala_objetivo} · {cfg.difficulty}"),
            (f"partidas de hasta {cfg.max_segundos:g} s (o {cfg.paciencia:g} s sin avanzar); terminan al morir"
             if gim else
             f"{cfg.escenarios} escenarios de hasta {cfg.max_segundos:g} s (espera al azar hasta {cfg.espera_max} "
             f"frames, paciencia {cfg.paciencia:g} s) · " +
             (f"al morir aprieta R (espera al reaparecer hasta {cfg.espera_reintento} frames)" if cfg.reintentos
              else "terminan al morir")),
            f"red: {cfg.cuadros} grilla{'s' if cfg.cuadros > 1 else ''} → {cfg.filtros1} detectores chicos → "
            f"{cfg.filtros2} grandes → {cfg.oculta} neuronas → 12 acciones ({pesos} pesos), decide cada "
            f"{cfg.frames_por_decision} frame{'s' if cfg.frames_por_decision > 1 else ''}"
            f"{', suelta el salto sola' if cfg.soltar_salto else ''}",
            f"{cfg.poblacion} redes · {cfg.padres} padres · élite {cfg.elite}"
            f"{' + la que llega más lejos' if cfg.elite_mas_lejos else ''} · sigma {cfg.sigma:g}",
            "recompensas: " + " · ".join(premios),
        ]
        if cfg.desde:
            texto.append(f"parte de la mejor red de <b>{cfg.desde}</b> (gen {cfg.desde_gen})")
        if nombre in self.entrenando():
            texto.append(f"<span style='color:#199e70'><b>entrenando ahora</b></span>"
                         f"{' (desde la terminal)' if nombre in self.externos else ''}")
        try:
            filas = almacen.leer_metricas(carpeta)
        except (OSError, ValueError):
            filas = []
        if filas:
            u = filas[-1]
            horas = sum(f.get("segundos_reloj", 0) for f in filas) / 3600
            mejor = max(filas, key=lambda f: f["avance_mejor"])
            texto += [f"<br>{len(filas)} de {cfg.generaciones} generaciones ({horas:.1f} h de cálculo, "
                      f"{u['segundos_reloj']:.0f} s la última)"]
            if gim and "salas_pct" in u:
                texto.append(f"<b>salas superadas: {u['salas_pct']:.1f} %</b> de todas las partidas · la mejor red "
                             f"{u['salas_mejor']:.0f} de {cfg.escenarios}")
            texto += [f"mejor red (promedio de {'salas' if gim else 'escenarios'}): <b>{u['avance_mejor']:.1f} %</b> · "
                      f"récord {mejor['avance_mejor']:.1f} % (gen {mejor['gen']:.0f})",
                      f"media de la población: {u['avance_media']:.1f} % · nichos: {u['nichos']:.0f}"]
            if not gim:
                texto.append(f"lo más lejos en un escenario: {max(f['avance_max'] for f in filas):.1f} % · "
                             f"llegaron a la salida: {u['llegaron']:.0f}")
            texto.append(f"muertes por partida: {u['muertes_media']:.1f} · quieto {u.get('quieto_pct', 0):.0f} % del "
                         f"tiempo · saves: media {u['saves_media']:.2f}, máx {u['saves_max']:.0f}"
                         if cfg.reintentos and not gim and "muertes_media" in u else
                         f"murió {u['murio_pct']:.0f} % · se estancó {u['estancado_pct']:.0f} % · tiempo "
                         f"{u['tiempo_pct']:.0f} %" + ("" if gim else f" · salió {u['salio_pct']:.0f} %"))
            if "celdas_media" in u and cfg.r_explorar and not gim:
                texto.append(f"celdas exploradas por partida: media {u['celdas_media']:.1f}, máx {u['celdas_max']:.0f}")
        self.resumen.setText("<br>".join(texto))

    def borrar(self) -> None:
        nombre = self.corrida_actual()
        if not nombre:
            return
        if nombre in self.entrenando() or any(nombre in p.args for p in self.procesos):
            QMessageBox.information(self, "Borrar", f"Hay un proceso usando «{nombre}»: detenlo primero.")
            return
        carpeta = almacen.dir_corrida(nombre)
        n = len(almacen.generaciones_guardadas(carpeta))
        mb = sum(f.stat().st_size for f in carpeta.rglob("*") if f.is_file()) / 1e6
        r = QMessageBox.question(self, "Borrar corrida", f"¿Borrar «{nombre}» ({n} generaciones, {mb:.0f} MB) "
                                 "con sus redes, pruebas, videos y gráficos?\n\nNo se puede deshacer.",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        shutil.rmtree(carpeta)
        self.log.appendPlainText(f"borrada la corrida {nombre} ({mb:.0f} MB)")
        self.lista.setCurrentItem(None)
        self.actualizar_lista()
        if self.lista.count() == 0:
            self.nueva()

    def abrir(self, sub: str) -> None:
        nombre = self.corrida_actual()
        ruta = almacen.dir_corrida(nombre) / sub if nombre else almacen.CORRIDAS
        ruta.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    # -- procesos -----------------------------------------------------------------------------------
    def lanzar(self, etiqueta: str, args: list[str]) -> Proceso:
        p = Proceso(etiqueta, args)
        p.linea.connect(self.al_recibir)
        p.termino.connect(self.al_terminar)
        self.log.appendPlainText(f"$ python -m {shlex.join(args)}")
        p.iniciar()
        self.procesos.append(p)
        item = QListWidgetItem(etiqueta)
        item.setData(Qt.UserRole, id(p))
        self.lista_proc.addItem(item)
        return p

    @Slot(object, str)
    def al_recibir(self, p: Proceso, linea: str) -> None:
        self.log.appendPlainText(f"[{p.etiqueta}] {linea}")
        if p.args[0] == "redes.gimnasio_ver" and linea.startswith("imagen: "):
            self.mostrar_imagen(linea.removeprefix("imagen: ").strip())
        if p.args[0] == "redes.entrenar" and linea.startswith("gen "):
            self.revisar()
            if self.lista.findItems(f"{p.args[2]} ", Qt.MatchStartsWith) == []:
                self.actualizar_lista()  # corrida nueva: aparece en la lista con su primera generación

    @Slot(object, int)
    def al_terminar(self, p: Proceso, codigo: int) -> None:
        estado = "terminó" if codigo == 0 else ("detenido" if codigo in (130, -2, -9) else f"falló (código {codigo})")
        self.log.appendPlainText(f"[{p.etiqueta}] {estado}")
        self.procesos.remove(p)
        for i in range(self.lista_proc.count()):
            if self.lista_proc.item(i).data(Qt.UserRole) == id(p):
                self.lista_proc.takeItem(i)
                break
        if p.args[0] == "redes.entrenar":
            self.estado.setText(f"Entrenamiento de «{p.args[2]}» {estado}.")
        self.actualizar_lista()
        self.revisar()

    def detener_seleccionado(self) -> None:
        item = self.lista_proc.currentItem()
        if item:
            for p in self.procesos:
                if id(p) == item.data(Qt.UserRole):
                    p.detener()

    def closeEvent(self, ev) -> None:
        if self.procesos:
            r = QMessageBox.question(self, "Salir", f"Hay {len(self.procesos)} proceso(s) de la app en curso. "
                                     "¿Detenerlos y salir? (Las generaciones completas quedan guardadas; lo lanzado "
                                     "desde la terminal sigue corriendo.)")
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            for p in self.procesos:
                if p.vivo():
                    os.kill(p.popen.pid, signal.SIGINT)
            for p in list(self.procesos):
                try:
                    p.popen.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                p._matar_grupo()
        ev.accept()


def main():
    if not sys.stdout or not sys.stdout.isatty():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(LOG, "a", buffering=1)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")  # avisos inofensivos de Wayland
    app = QApplication(sys.argv)
    app.setDesktopFileName("iwbtgb-redes")  # app_id en Wayland: lo asocia al acceso directo del menú
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icono.png"))))

    def error(tipo, valor, tb):  # sin terminal, un error inesperado se muestra en una ventana
        texto = "".join(traceback.format_exception(tipo, valor, tb))
        print(texto, flush=True)
        QMessageBox.critical(None, "Error", f"{valor}\n\nDetalle en {LOG}")

    sys.excepthook = error
    v = Ventana()
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
