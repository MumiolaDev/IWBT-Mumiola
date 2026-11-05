"""App de una línea de redes (`python -m iwbtgb.redes app` o `iwbtgb.rayos app`): crear corridas, entrenarlas,
probarlas, videos, en vivo y transmisión.

Por debajo ejecuta los comandos de la línea (`entrenar`, `probar`, `visualizar`, `transmitir`) como procesos aparte,
así que lo que se hace acá se puede seguir desde la terminal y al revés: un entrenamiento lanzado desde la terminal
también aparece (y se puede detener) desde la app. Las secciones de lo que ve la red y de la red son de cada línea
(`FormularioRed`). La app abierta usa el código con el que se abrió: tras cambiar la config o la app, reabrirla.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..emulador import SALAS
from ..qt import DIFICULTADES, Proceso, combo_salas, doble, en_curso, spin
from . import almacen
from .transmitir import CONFIG as CONFIG_TRANSMISION, destino_configurado, oculto

FINES_CORTOS = {"llegó": "llegó", "murió": "murió", "estancado": "estancada", "tiempo": "tiempo", "salió": "salió"}


class FormularioRed:
    """Las secciones de la pestaña «Crear y entrenar» propias de una línea (lo que ve la red y la red). `cambio` se
    llama cuando cambia algo que mueve el número de pesos."""

    def __init__(self, cambio, d):
        self.cambio = cambio

    def grupos(self) -> list[tuple[str, list[tuple[str, QWidget]], QWidget | None]]:
        """[(título sin número, [(etiqueta, control)], texto de arriba o None)]."""
        raise NotImplementedError

    def config(self):
        """Una config con la forma de red del formulario (para contar sus pesos)."""
        raise NotImplementedError

    def explicar(self, cfg) -> None:
        """Actualiza los textos que explican lo que ve y el tamaño de la red."""

    def llenar_forma(self, cfg) -> None:
        """Pone en el formulario lo que ve y la forma de la red de `cfg` (también al partir de otra red)."""
        raise NotImplementedError

    def llenar(self, cfg) -> None:
        self.llenar_forma(cfg)

    def valores(self) -> dict:
        raise NotImplementedError

    def fijos(self) -> list[QWidget]:
        """Los controles de lo que ve y la forma de la red: no se eligen al partir de otra red."""
        raise NotImplementedError

    def resumen(self, cfg) -> list[str]:
        """Líneas del resumen de una corrida: lo que ve y la red."""
        raise NotImplementedError


class Ventana(QMainWindow):
    def __init__(self, linea, Formulario):
        super().__init__()
        self.linea, self.Formulario = linea, Formulario
        self.setWindowTitle(f"IWBTGB · {linea.titulo}")
        self.resize(1250, 880)
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
        izq.addWidget(QLabel(f"<b>Corridas de {linea.titulo}</b>"))
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
        self.tabs.addTab(self.pestana_visualizar(), "Videos, en vivo y transmisión")

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
        d = self.linea.Config()
        self._llenando = False

        # Corrida
        self.f_nombre = QLineEdit()
        self.f_desde = QComboBox()
        self.f_desde.setToolTip("En vez de empezar con pesos al azar, parte de la mejor red de otra corrida (p. ej. "
                                "una entrenada en el gimnasio, para seguir en un nivel). Lo que ve y la forma de la "
                                "red se copian de ella")
        self.f_desde.currentIndexChanged.connect(self.al_cambiar_desde)

        # Dónde practica
        self.f_entorno = QComboBox()
        self.f_entorno.addItem("Gimnasio: salas generadas al azar", "gimnasio")
        self.f_entorno.addItem("Un nivel del juego", "nivel")
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
                                  "plataformas móviles. Los obstáculos crecen con ella")
        self.f_gim_borde = QCheckBox("Salida en la pared derecha, como en los niveles")
        self.f_gim_borde.setToolTip("La salida es una abertura en la pared derecha. Sin marcar: un portal a 2 celdas "
                                    "de la pared")
        self.f_gim_trampas = QCheckBox("Trampas desde el comienzo: púas que saltan, caen o se deslizan y cerezas")
        self.f_gim_trampas.setToolTip("Un disparador invisible suelta una púa o una cereza cuando el jugador pasa: púa "
                                      "que salta (dificultad 0,2), que cae o cereza (0,3), lateral (0,4). Con el "
                                      "currículo se activan solas al llegar a su dificultad")
        self.f_gim_trampas_min = spin(0, 2, d.gim_trampas_min, "las del azar")
        self.f_gim_trampas_min.setSuffix(" por sala")
        self.f_gim_trampas_min.setToolTip("Con trampas, cada sala trae al menos tantas, aunque su dificultad sea baja "
                                          "(0: las que salgan al azar según la dificultad)")
        self.f_curriculo = QCheckBox("Currículo: subir la dificultad sola cuando la supera")
        self.f_curriculo.setToolTip("Cuando la población supera en promedio el umbral de salas durante esas "
                                    "generaciones, la dificultad máxima sube un paso (hasta 1). Desde la dificultad "
                                    "indicada aparecen las trampas")
        self.f_cur_umbral = doble(1, 100, d.curriculo_umbral, 0, " %")
        self.f_cur_gens = spin(1, 1000, d.curriculo_gens)
        self.f_cur_paso = doble(0.01, 1, d.curriculo_paso, 2, paso=0.05)
        self.f_cur_trampas = doble(0, 2, d.curriculo_trampas, 2, paso=0.1)
        self.f_cur_trampas.setToolTip("Dificultad máxima desde la que se activan las trampas (2 = nunca)")
        self.w_curriculo = QWidget()
        fila = QHBoxLayout(self.w_curriculo)
        fila.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("con"), self.f_cur_umbral, QLabel("en"), self.f_cur_gens, QLabel("gen. sube"),
                  self.f_cur_paso, QLabel("· trampas desde"), self.f_cur_trampas):
            fila.addWidget(w)
        fila.addStretch()
        self.f_curriculo.toggled.connect(lambda on: self.w_curriculo.setEnabled(on))

        # Partidas
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

        # Lo que ve y la red: de la línea
        self.red = self.Formulario(self.mostrar_pesos, d)

        # Recompensas
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

        # Evolución
        self.f_poblacion = spin(2, 5000, d.poblacion)
        self.f_padres = spin(1, 1000, d.padres)
        self.f_padres.setToolTip("Las mejores (una por nicho primero) tienen hijos: copias con los pesos mutados")
        self.f_elite = spin(0, 1000, d.elite)
        self.f_elite.setToolTip("Las mejores pasan sin cambios a la generación siguiente")
        self.f_lejos = QCheckBox("La que llega más lejos pasa siempre y tiene hijos")
        self.f_sigma = doble(0.0001, 1, d.sigma, 4, paso=0.005)
        self.f_sigma.setToolTip("Tamaño de la mutación de los pesos (ruido normal × sigma)")
        self.f_generaciones = spin(1, 1_000_000, d.generaciones)
        self.f_workers = spin(1, os.cpu_count() or 1, min(d.workers, os.cpu_count() or 1))
        self.f_workers.setToolTip("Emuladores en paralelo (uno por núcleo, más o menos; deja 2 libres si vas a "
                                  "transmitir)")
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
            ("Dificultad del juego", self.f_dificultad), ("Dificultad de las salas", self.w_gim_dif),
            ("", self.f_gim_borde), ("", self.f_gim_trampas), ("Trampas mínimas", self.f_gim_trampas_min),
            ("", self.f_curriculo), ("", self.w_curriculo)],
            arriba=self.l_entorno)
        g_partidas, self.form_partidas = grupo("3 · Partidas de cada red", [
            ("Escenarios por red", self.f_escenarios), ("Duración máxima", self.f_segundos),
            ("Paciencia sin avanzar", self.f_paciencia), ("Espera al azar al comenzar", self.f_espera),
            ("", self.f_reintentos), ("Espera al reaparecer", self.f_espera_r)])
        n = 4
        g_red = []
        for titulo, filas, arriba in self.red.grupos():
            g_red.append(grupo(f"{n} · {titulo}", filas, arriba)[0])
            n += 1
        g_premios, self.form_premios = grupo(f"{n} · Recompensas (por partida)", [
            ("Avance (todo el camino)", self.f_r_avance), ("Llegar a la salida", self.f_r_objetivo),
            ("Rapidez (castigo máx.)", self.f_r_rapidez), ("Castigo por muerte", self.f_castigo),
            ("Castigo por segundo quieto", self.f_quieto), ("Premio por save nuevo", self.f_r_save),
            ("Premio por celda explorada", self.f_explorar), ("Tamaño de la celda", self.f_celda)])
        g_evo, _ = grupo(f"{n + 1} · Evolución", [
            ("Redes por generación", self.f_poblacion), ("Padres", self.f_padres),
            ("Élite (pasan sin cambios)", self.f_elite), ("", self.f_lejos), ("Sigma de la mutación", self.f_sigma),
            ("Hasta la generación", self.f_generaciones), ("Procesos en paralelo", self.f_workers),
            ("Semilla", self.f_seed)])
        izq, der = QVBoxLayout(), QVBoxLayout()
        for g in (g_corrida, g_donde, g_partidas, g_premios):
            izq.addWidget(g)
        for g in (*g_red, g_evo):
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
                                    "evolución, partidas, la dificultad de las salas y el currículo; no el entorno, lo "
                                    "que ve ni la red")
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
        self.red.explicar(self.red.config())

    def llenar_formulario(self, cfg) -> None:
        self._llenando = True
        self.f_nombre.setText(cfg.nombre)
        self.llenar_desde(cfg.desde)
        self.f_entorno.setCurrentIndex(max(0, self.f_entorno.findData(cfg.entorno)))
        self.f_stage.setCurrentText(cfg.start_room)
        self.f_objetivo.setCurrentText(cfg.sala_objetivo)
        self.f_dificultad.setCurrentText(cfg.difficulty)
        self.f_gim_min.setValue(cfg.gim_dificultad_min)
        self.f_gim_max.setValue(cfg.gim_dificultad_max)
        self.f_gim_borde.setChecked(bool(cfg.gim_salida_borde))
        self.f_gim_trampas.setChecked(bool(cfg.gim_trampas))
        self.f_gim_trampas_min.setValue(cfg.gim_trampas_min)
        self.f_curriculo.setChecked(bool(cfg.gim_curriculo))
        self.w_curriculo.setEnabled(bool(cfg.gim_curriculo))
        self.f_cur_umbral.setValue(cfg.curriculo_umbral)
        self.f_cur_gens.setValue(cfg.curriculo_gens)
        self.f_cur_paso.setValue(cfg.curriculo_paso)
        self.f_cur_trampas.setValue(cfg.curriculo_trampas)
        self.red.llenar(cfg)
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
        for p in sorted(self.linea.corridas.glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
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
        for w in (self.w_gim_dif, self.f_gim_borde, self.f_gim_trampas, self.f_gim_trampas_min, self.f_curriculo,
                  self.w_curriculo):
            self.form_donde.setRowVisible(w, gim)
        for w in (self.f_espera, self.f_reintentos, self.f_espera_r):
            self.form_partidas.setRowVisible(w, not gim)
        for w in (self.f_quieto, self.f_r_save, self.f_explorar, self.f_celda):
            self.form_premios.setRowVisible(w, not gim)
        self.form_partidas.labelForField(self.f_escenarios).setText("Salas por generación" if gim else
                                                                    "Escenarios por red")
        self.f_escenarios.setToolTip(
            "Salas nuevas en cada generación, las mismas para todas las redes (no las puede memorizar); su fitness "
            "es el promedio" if gim else
            "Partidas por red y generación: la primera desde el comienzo y las demás con una espera al azar")
        self.l_entorno.setText(
            "Cada partida es una sala de una pantalla generada al azar dentro del juego, con la física real. Termina "
            "al llegar, morir, a la duración máxima o sin avanzar. Después, para seguir en un nivel, crea otra corrida "
            "que parta de esta red." if gim else
            "Entrena en una sala del GB hasta la sala de llegada; el avance se mide por el camino más corto a la "
            "salida. Antes conviene probar la red ahí («Probar»): es su primer intento sin haberla visto.")
        if sugerir and not self._llenando:     # valores razonables para cada entorno
            self.f_escenarios.setValue(8 if gim else 3)
            self.f_segundos.setValue(15 if gim else 30)
            self.f_paciencia.setValue(8)

    def al_cambiar_desde(self) -> None:
        """Con una red de partida, lo que ve y la forma de la red son los suyos (no se pueden elegir)."""
        origen = self.f_desde.currentData()
        for w in self.red.fijos():
            w.setEnabled(not origen)
        if origen and self.linea.dir_corrida(origen).exists():
            self.red.llenar_forma(self.linea.cargar_config(self.linea.dir_corrida(origen)))

    def valores(self) -> dict:
        gim = self.f_entorno.currentData() == "gimnasio"
        return {
            "entorno": self.f_entorno.currentData(), "desde": self.f_desde.currentData(),
            "start_room": self.f_stage.currentText(), "sala_objetivo": self.f_objetivo.currentText(),
            "difficulty": self.f_dificultad.currentText(),
            "gim_dificultad_min": float(self.f_gim_min.value()),
            "gim_dificultad_max": float(max(self.f_gim_max.value(), self.f_gim_min.value())),
            "gim_salida_borde": int(self.f_gim_borde.isChecked()),
            "gim_trampas": int(self.f_gim_trampas.isChecked()), "gim_trampas_min": self.f_gim_trampas_min.value(),
            "gim_curriculo": int(self.f_curriculo.isChecked()),
            "curriculo_umbral": float(self.f_cur_umbral.value()), "curriculo_gens": self.f_cur_gens.value(),
            "curriculo_paso": float(self.f_cur_paso.value()), "curriculo_trampas": float(self.f_cur_trampas.value()),
            **self.red.valores(),
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
        carpeta = self.linea.dir_corrida(nombre)
        hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
        args = ["entrenar", "--nombre", nombre]
        if continuar:
            if not hechas:
                QMessageBox.information(self, "Continuar", f"«{nombre}» no tiene generaciones guardadas.")
                return
            guardada = self.linea.cargar_config(carpeta)
            fijos = self.linea.fijos
            cambiados = [k for k in fijos if k in vals and getattr(guardada, k) != vals[k]]
            if cambiados:
                QMessageBox.information(self, "Continuar", "Al continuar no se pueden cambiar el entorno, la sala, la "
                                        "dificultad del juego, la red de partida, lo que ve ni la forma de la red ("
                                        + ", ".join(cambiados) + "): se usan los de la corrida. Para probar otros, "
                                        "crea una red nueva.")
            vals = {k: v for k, v in vals.items() if k not in fijos and (getattr(guardada, k) != v
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
        propios = {p.args[2]: p.popen.pid for p in self.procesos if p.args[0] == "entrenar" and p.vivo()}
        return {**self.externos, **propios}

    def detener_entrenamiento(self) -> None:
        nombre = self.corrida_actual() or self.f_nombre.text().strip()
        for p in self.procesos:
            if p.args[0] == "entrenar" and p.args[2] == nombre:
                p.detener()
                self.estado.setText(f"Deteniendo «{nombre}»: termina la generación en curso…")
                return
        pid = self.externos.get(nombre)
        if pid:
            os.kill(pid, signal.SIGINT)  # lanzado desde la terminal: Ctrl+C, guarda lo hecho
            self.estado.setText(f"Deteniendo «{nombre}» (lanzado desde la terminal)…")

    def revisar(self) -> None:
        """Entrenamientos en curso (también los de la terminal), botones, barra y resumen."""
        self.externos = en_curso(self.linea.paquete)
        nombre = self.corrida_actual() or self.f_nombre.text().strip()
        corriendo = nombre in self.entrenando()
        self.b_entrenar.setEnabled(not corriendo)
        self.b_continuar.setEnabled(not corriendo)
        self.b_detener.setEnabled(corriendo)
        self.mostrar_transmision()
        if not nombre:
            return
        carpeta = self.linea.dir_corrida(nombre)
        try:
            cfg = self.linea.cargar_config(carpeta)
            filas = almacen.leer_metricas(carpeta)
        except (OSError, ValueError):
            return
        if corriendo or filas:
            self.barra.setRange(0, cfg.generaciones)
            self.barra.setValue(min(len(filas), cfg.generaciones))
        if corriendo:
            if filas:
                u = filas[-1]
                salas = (f" · salas superadas {u['salas_pct']:.0f} % · dificultad hasta {cfg.gim_dificultad_max:g}"
                         if cfg.entorno == "gimnasio" and "salas_pct" in u else "")
                self.estado.setText(f"Entrenando «{nombre}»"
                                    f"{' (lanzado desde la terminal)' if nombre in self.externos else ''} · "
                                    f"generación {int(u['gen']) + 1} de {cfg.generaciones} · avance del mejor "
                                    f"{u['avance_mejor']:.1f} %{salas} · {u['segundos_reloj']:.0f} s por generación")
            self.mostrar_resumen()
        elif not any(p.args[0] == "entrenar" for p in self.procesos):
            self.estado.setText(f"«{nombre}»: sin entrenamiento en curso")

    # -- pestaña Probar -------------------------------------------------------------------------------
    def pestana_probar(self) -> QWidget:
        self.p_gen = spin(-1, 1_000_000, -1, "última")
        self.p_sala = QComboBox()
        self.p_sala.addItem("Gimnasio: salas generadas que nunca vio", "gimnasio")
        self.p_sala.addItem("Su nivel (start_room)", "suya")
        self.p_sala.addItem("Todos los niveles (rStage01 a rStage17)", "todas")
        for s in SALAS:
            self.p_sala.addItem(s, s)
        self.p_sala.currentIndexChanged.connect(self.al_cambiar_prueba)
        self.p_escenarios = spin(1, 200, 40)
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
        explicacion = QLabel("Juega la mejor red sin entrenarla más. En el gimnasio, salas que nunca vio (el examen: "
                             "mismas semillas cada vez, para comparar generaciones); en un nivel, cuánto avanza. "
                             "<b>Antes de entrenar en un nivel, pruébala ahí</b>: es su primer intento sin haberlo "
                             "visto. Todos los niveles miden si <b>generaliza</b>.")
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
        """"gimnasio", "todas", un nivel o "" (su nivel)."""
        donde = self.p_sala.currentData()
        return "" if donde == "suya" else donde

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
        args = ["probar", "--corrida", nombre, "--gen", str(self.p_gen.value()),
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
        for ruta in (self.linea.dir_corrida(nombre) / "pruebas").glob("*.json"):
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

    # -- pestaña Videos, en vivo y transmisión ------------------------------------------------------
    def pestana_visualizar(self) -> QWidget:
        self.v_quien = QComboBox()
        self.v_quien.addItem("Todas las redes de la generación juntas", "todas")
        self.v_quien.addItem("La mejor red de la corrida", "mejor")
        self.v_quien.addItem("Alternar: una sala con todas y otra con la mejor", "alternar")
        self.v_quien.addItem("Acciones al azar (salas del gimnasio)", "azar")
        self.v_quien.addItem("Solución de la búsqueda (salas del gimnasio)", "solucion")
        self.v_quien.setToolTip("«Todas»: toda la población a la vez, como en los videos del GA (la mejor emulada y "
                                "las demás desde lo que jugaron al entrenar). «La mejor» = la que llega más lejos, con "
                                "lo que ve y qué acción prefiere. Al azar y la solución sirven para ver salas del "
                                "gimnasio: la solución comprueba que cada sala se pueda pasar")
        self.v_donde = QComboBox()
        self.v_donde.addItem("Donde entrenó (en el gimnasio: el entrenamiento en vivo)", "suya")
        self.v_donde.addItem("Prueba en el gimnasio: salas nuevas que nunca vio", "gimnasio")
        for s in SALAS:
            self.v_donde.addItem(s, s)
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
        self.v_semilla = spin(0, 10_000_000, 1_000_000)
        self.v_semilla.setToolTip("Las salas van de esta semilla en adelante: la misma semilla da la misma sala")
        self.v_salas = spin(1, 200, 10)
        self.v_salas.setToolTip("Salas del video o de la imagen de muestras")
        self.v_verificar = QCheckBox("Solo salas con solución (la búsqueda tarda unos segundos por sala)")
        self.v_red = QCheckBox("La red por dentro en vez del gráfico de fitness")
        self.v_red.setToolTip("Con la mejor red: dibuja sus capas (nodos y conexiones) a la derecha del juego")
        self.v_vel = spin(1, 16, 1)
        self.v_vel.setSuffix("×")
        self.v_vel.setToolTip("Veces el tiempo real; se saltan cuadros")
        self.form_visual = QFormLayout()
        for etiqueta, w in (("Quién juega", self.v_quien), ("Dónde", self.v_donde), ("Generación (videos)", self.v_gen),
                            ("Espera al comenzar (video)", self.v_espera), ("Dificultad de las salas", self.w_v_dif),
                            ("Semilla de la primera sala", self.v_semilla), ("Salas (video y muestras)", self.v_salas),
                            ("", self.v_verificar), ("", self.v_red), ("Velocidad", self.v_vel)):
            self.form_visual.addRow(etiqueta, w)

        grilla = QGridLayout()
        botones = [
            ("En vivo (ventana)", "vivo", "Una ventana que juega sin parar; con una corrida entrenando, toma la red "
                                          "nueva. Corre con tope de memoria: sirve para dejarla horas"),
            ("Grabar video", "video", "Un video en la carpeta videos/ de la corrida (o de _gimnasio)"),
            ("Gráficos de la corrida", "graficos", "Avance, fitness, cómo terminan las partidas, muertes y saves y, en "
                                                   "el gimnasio, salas superadas y dificultad"),
            ("Muestras de salas (imagen)", "muestras", "Una imagen con salas del gimnasio; con «solo salas con "
                                                       "solución», cada una con el camino que encontró la búsqueda"),
            ("Abrir carpeta de videos", "carpeta-videos", ""),
            ("Abrir carpeta del gimnasio", "carpeta-gimnasio", ""),
        ]
        for i, (texto, modo, ayuda) in enumerate(botones):
            b = QPushButton(texto)
            if ayuda:
                b.setToolTip(ayuda)
            b.clicked.connect(lambda _=False, m=modo: self.visualizar(m))
            grilla.addWidget(b, i // 2, i % 2)
        self.v_nota = QLabel()
        self.v_nota.setWordWrap(True)
        self.v_nota.setStyleSheet("color: gray")

        # Transmisión
        self.t_corrida = QComboBox()
        self.t_corrida.addItem("La corrida elegida a la izquierda", "elegida")
        self.t_corrida.addItem("La que se esté entrenando (cambia sola)", "auto")
        self.t_alto = QComboBox()
        self.t_alto.addItem("720p (3000 kb/s)", (720, 3000))
        self.t_alto.addItem("1080p (6000 kb/s)", (1080, 6000))
        self.t_destino = QLabel()
        self.t_destino.setWordWrap(True)
        self.t_destino.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.b_transmitir = QPushButton("Transmitir en vivo")
        self.b_transmitir.clicked.connect(self.transmitir)
        self.b_probar_t = QPushButton("Probar 2 min a un archivo")
        self.b_probar_t.setToolTip("Lo mismo que se transmitiría, en videos/transmision_prueba.flv de la corrida")
        self.b_probar_t.clicked.connect(lambda: self.transmitir(prueba=True))
        self.b_detener_t = QPushButton("Detener transmisión")
        self.b_detener_t.clicked.connect(self.detener_transmision)
        g_t = QGroupBox("Transmisión en vivo (YouTube, Twitch…)")
        ft = QFormLayout(g_t)
        ft.addRow("Qué se transmite", self.t_corrida)
        ft.addRow("Calidad", self.t_alto)
        ft.addRow("Destino", self.t_destino)
        fila = QHBoxLayout()
        for b in (self.b_transmitir, self.b_probar_t, self.b_detener_t):
            fila.addWidget(b)
        ft.addRow(fila)

        self.grafico = QLabel("Pulsa «Gráficos de la corrida» o «Muestras de salas».")
        self.grafico.setAlignment(Qt.AlignCenter)
        self.grafico.setMinimumHeight(160)
        v = QVBoxLayout()
        v.addLayout(self.form_visual)
        v.addWidget(self.v_nota)
        v.addLayout(grilla)
        v.addWidget(g_t)
        v.addWidget(self.grafico, 1)
        w = QWidget()
        w.setLayout(v)
        self.al_cambiar_visual()
        self.mostrar_transmision()
        return w

    def config_actual(self):
        nombre = self.corrida_actual()
        try:
            return self.linea.cargar_config(self.linea.dir_corrida(nombre)) if nombre else None
        except (OSError, ValueError):
            return None

    def donde_visual(self) -> str:
        """"entreno" (las salas de la generación de una corrida del gimnasio), "gimnasio" (salas nuevas), un nivel o
        "" (el nivel donde entrenó)."""
        quien, donde = self.v_quien.currentData(), self.v_donde.currentData()
        if quien in ("azar", "solucion"):
            return "gimnasio"
        cfg = self.config_actual()
        suya = "entreno" if cfg is None or cfg.entorno == "gimnasio" else ""
        if quien != "mejor":
            return suya
        return suya if donde == "suya" else donde

    def al_cambiar_visual(self) -> None:
        quien, donde = self.v_quien.currentData(), self.donde_visual()
        gim = donde == "gimnasio"
        self.v_donde.setEnabled(quien == "mejor")
        f = self.form_visual
        f.setRowVisible(self.v_gen, quien in ("mejor", "todas", "alternar"))
        f.setRowVisible(self.v_espera, quien == "mejor" and donde not in ("gimnasio", "entreno"))
        f.setRowVisible(self.v_red, quien in ("mejor", "alternar"))
        for w in (self.w_v_dif, self.v_semilla, self.v_salas, self.v_verificar):
            f.setRowVisible(w, gim)
        if quien in ("azar", "solucion"):
            nota = {"azar": "Para ver salas del gimnasio con alguien moviéndose.",
                    "solucion": "Antes de cada sala, una búsqueda encuentra cómo pasarla y se ve la solución."}[quien]
        elif quien != "mejor" and donde == "":
            nota = ("<b>Todas las redes</b> de la última generación sobre el mapa del nivel, como en los videos del GA "
                    "(desde lo que jugaron al entrenar). En vivo salta a la generación nueva.")
        elif donde == "entreno":
            nota = ("<b>Entrenamiento en vivo</b>: las salas de la última generación como las jugó la población. " +
                    {"mejor": "La mejor red, con lo que ve y cómo le fue al entrenar.",
                     "todas": "La mejor emulada en vivo y las demás como fantasmas.",
                     "alternar": "Una sala con toda la población y la siguiente con la mejor."}[quien] +
                    " Cuando aparece una generación nueva pasa a sus salas: se ve cómo aprende.")
        elif gim:
            nota = ("<b>Prueba</b>: salas nuevas que la red nunca vio, una tras otra, con el historial de cuántas "
                    "supera: sirve para ver si generaliza.")
        else:
            nota = ("La mejor red en el nivel: el juego con lo que ve, el puntaje, el gráfico de fitness (o la red por "
                    "dentro) y qué acción prefiere en cada momento.")
        self.v_nota.setText(nota)

    def visualizar(self, modo: str) -> None:
        if modo == "carpeta-videos":
            self.abrir("videos")
            return
        if modo == "carpeta-gimnasio":
            ruta = self.linea.corridas / "_gimnasio"
            ruta.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))
            return
        quien, donde = self.v_quien.currentData(), self.donde_visual()
        dif = -1.0 if self.v_dif_azar.isChecked() else self.v_dif.value()
        gimnasio = ["--dificultad", f"{dif:g}", "--semilla", str(self.v_semilla.value())]
        if self.v_verificar.isChecked():
            gimnasio.append("--verificar")
        if modo == "muestras":
            self.lanzar("muestras de salas", ["visualizar", "muestras", *gimnasio, "--salas", str(self.v_salas.value())])
            return
        nombre = quien if quien in ("azar", "solucion") and modo != "graficos" else \
            self.corrida_con_generaciones("Videos y en vivo")
        if not nombre:
            return
        if modo == "graficos":
            self.lanzar(f"gráficos {nombre}", ["visualizar", "graficos", "--corrida", nombre])
            return
        vel = ["--velocidad", str(self.v_vel.value())]
        gen = ["--gen", str(self.v_gen.value())] if quien not in ("azar", "solucion") else []
        red = ["--red"] if self.v_red.isChecked() and quien in ("mejor", "alternar") else []
        if quien in ("todas", "alternar") and donde == "":     # todas en un nivel: el dibujo del GA
            args = (["visualizar", "vivo", "--corrida", nombre, *vel] if modo == "vivo" else
                    ["visualizar", "generacion", "--corrida", nombre, *gen, *vel])
        elif donde == "entreno":
            args = (["visualizar", "entreno", "--corrida", nombre, "--quien", quien, *vel] if modo == "vivo"
                    else ["visualizar", "entreno-video", "--corrida", nombre, "--quien", quien, *gen, *vel])
        elif donde == "gimnasio":
            args = (["visualizar", "prueba", "--corrida", nombre, *gimnasio, *vel] if modo == "vivo" else
                    ["visualizar", "prueba-video", "--corrida", nombre, *gimnasio, "--salas",
                     str(self.v_salas.value()), *gen, *vel])
        else:
            sala = ["--sala", donde] if donde else []
            args = (["visualizar", "vivo-red", "--corrida", nombre, *sala, *vel] if modo == "vivo" else
                    ["visualizar", "mejor", "--corrida", nombre, *sala, *gen, "--espera", str(self.v_espera.value())])
        self.lanzar(f"{'en vivo' if modo == 'vivo' else 'video'} {nombre}", args + red)

    # -- transmisión ----------------------------------------------------------------------------------
    def transmision_en_curso(self) -> Proceso | None:
        return next((p for p in self.procesos if p.args[0] == "transmitir" and p.vivo()), None)

    def mostrar_transmision(self) -> None:
        if not hasattr(self, "t_destino"):
            return
        destino = destino_configurado()
        self.t_destino.setText(oculto(destino) if destino else
                               f"<span style='color:#d95926'>sin configurar</span>: crea {CONFIG_TRANSMISION} con "
                               '{"url": "rtmp://a.rtmp.youtube.com/live2", "clave": "…"} (la clave del stream, que da '
                               "YouTube o Twitch; no va al repositorio)")
        en_curso_ = self.transmision_en_curso() is not None
        self.b_transmitir.setEnabled(bool(destino) and not en_curso_)
        self.b_probar_t.setEnabled(not en_curso_)
        self.b_detener_t.setEnabled(en_curso_)

    def transmitir(self, prueba: bool = False) -> None:
        if self.t_corrida.currentData() == "auto" and not prueba:
            nombre = "auto"
        else:
            nombre = self.corrida_con_generaciones("Transmitir")
            if not nombre:
                return
        alto, bitrate = self.t_alto.currentData()
        quien = self.v_quien.currentData() if self.v_quien.currentData() in ("mejor", "todas", "alternar") else \
            "alternar"
        args = ["transmitir", "--corrida", nombre, "--quien", quien, "--alto", str(alto), "--bitrate", str(bitrate)]
        if prueba:
            args += ["--destino", str(self.linea.dir_corrida(nombre) / "videos" / "transmision_prueba.flv")]
            p = self.lanzar(f"prueba de transmisión {nombre}", args)
            QTimer.singleShot(120_000, p.detener)
        else:
            que = "la corrida que se esté entrenando" if nombre == "auto" else f"«{nombre}»"
            r = QMessageBox.question(self, "Transmitir", f"¿Empezar a transmitir {que} a "
                                     f"{oculto(destino_configurado())}? Queda en vivo hasta que la detengas.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
            self.lanzar(f"transmisión {nombre}", args)
        self.mostrar_transmision()

    def detener_transmision(self) -> None:
        p = self.transmision_en_curso()
        if p is not None:
            p.detener()

    def mostrar_imagen(self, ruta: str) -> None:
        pix = QPixmap(ruta)
        if not pix.isNull():
            self.grafico.setPixmap(pix.scaled(max(self.grafico.width(), 400), max(self.grafico.height(), 160),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mostrar_grafico(self) -> None:
        nombre = self.corrida_actual()
        carpeta = self.linea.dir_corrida(nombre) / "graficos" if nombre else None
        ruta = next((carpeta / n for n in ("gimnasio.png", "avance.png") if carpeta and (carpeta / n).exists()), None)
        if ruta:
            self.mostrar_imagen(str(ruta))
        else:
            self.grafico.clear()
            self.grafico.setText("Pulsa «Gráficos de la corrida» o «Muestras de salas».")

    # -- corridas -----------------------------------------------------------------------------------
    def actualizar_lista(self) -> None:
        actual = self.corrida_actual()
        self.lista.blockSignals(True)
        self.lista.clear()
        carpetas = sorted((p for p in self.linea.corridas.glob("*/config.json") if not p.parent.name.startswith("_")),
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
        if not nombre or not almacen.generaciones_guardadas(self.linea.dir_corrida(nombre)):
            QMessageBox.information(self, titulo, "Elige a la izquierda una corrida con generaciones.")
            return ""
        return nombre

    def al_elegir(self, item, _anterior) -> None:
        if item is None:
            return
        cfg = self.linea.cargar_config(self.linea.dir_corrida(item.data(Qt.UserRole)))
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
        cfg = self.linea.Config()
        n = 1
        while self.linea.dir_corrida(f"{self.linea.prefijo}{n:02d}").exists():
            n += 1
        cfg.nombre = f"{self.linea.prefijo}{n:02d}"
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
        carpeta = self.linea.dir_corrida(nombre)
        try:
            cfg = self.linea.cargar_config(carpeta)
        except (OSError, ValueError):
            return
        gim = cfg.entorno == "gimnasio"
        premios = [(cfg.r_avance, f"avance {cfg.r_avance:g}"), (cfg.r_objetivo, f"llegar {cfg.r_objetivo:g}"),
                   (cfg.r_rapidez, f"rapidez −{cfg.r_rapidez:g}"), (cfg.castigo_muerte, f"muerte −{cfg.castigo_muerte:g}")]
        if not gim:
            premios += [(cfg.castigo_quieto, f"quieto −{cfg.castigo_quieto:g}/s"), (cfg.r_save, f"save +{cfg.r_save:g}"),
                        (cfg.r_explorar, f"celda de {cfg.celda_explorar} px +{cfg.r_explorar:g}")]
        premios = [t for v, t in premios if v]   # las que valen 0 no se aplican
        texto = [
            f"<b>{nombre}</b>",
            (f"<b>gimnasio</b>: {cfg.escenarios} salas nuevas por generación, dificultad "
             f"{cfg.gim_dificultad_min:g}–{cfg.gim_dificultad_max:g}, salida "
             f"{'en la pared' if cfg.gim_salida_borde else 'en un portal antes de la pared'}"
             f"{', con trampas' if cfg.gim_trampas else ', sin trampas'}"
             f"{f' (al menos {cfg.gim_trampas_min} por sala)' if cfg.gim_trampas and cfg.gim_trampas_min else ''}"
             + (f" · currículo: +{cfg.curriculo_paso:g} con {cfg.curriculo_umbral:g} % en {cfg.curriculo_gens} gen."
                if cfg.gim_curriculo else "") if gim else
             f"<b>nivel</b>: {cfg.start_room} → {cfg.sala_objetivo} · {cfg.difficulty}"),
            (f"partidas de hasta {cfg.max_segundos:g} s (o {cfg.paciencia:g} s sin avanzar); terminan al morir"
             if gim else
             f"{cfg.escenarios} escenarios de hasta {cfg.max_segundos:g} s (espera al azar hasta {cfg.espera_max} "
             f"frames, paciencia {cfg.paciencia:g} s) · " +
             (f"al morir aprieta R (espera al reaparecer hasta {cfg.espera_reintento} frames)" if cfg.reintentos
              else "terminan al morir")),
            *self.red.resumen(cfg),
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
        carpeta = self.linea.dir_corrida(nombre)
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
        ruta = self.linea.dir_corrida(nombre) / sub if nombre else self.linea.corridas
        ruta.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    # -- procesos -----------------------------------------------------------------------------------
    def lanzar(self, etiqueta: str, args: list[str]) -> Proceso:
        p = Proceso(etiqueta, self.linea.paquete, args)
        p.linea.connect(self.al_recibir)
        p.termino.connect(self.al_terminar)
        self.log.appendPlainText(f"$ python -m {self.linea.paquete} {shlex.join(args)}")
        p.iniciar()
        self.procesos.append(p)
        item = QListWidgetItem(etiqueta)
        item.setData(Qt.UserRole, id(p))
        self.lista_proc.addItem(item)
        return p

    @Slot(object, str)
    def al_recibir(self, p: Proceso, linea: str) -> None:
        self.log.appendPlainText(f"[{p.etiqueta}] {linea}")
        if p.args[:2] == ["visualizar", "muestras"] and linea.startswith("imagen: "):
            self.mostrar_imagen(linea.removeprefix("imagen: ").strip())
        if p.args[0] == "visualizar" and linea.startswith("gráfico: ") and linea.endswith(
                ("gimnasio.png", "avance.png")):
            self.mostrar_grafico()
        if p.args[0] == "entrenar" and (linea.startswith("gen ") or linea.startswith("currículo")):
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
        if p.args[0] == "entrenar":
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
