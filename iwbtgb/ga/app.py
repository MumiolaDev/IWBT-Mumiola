"""App del algoritmo genético: lanzar entrenamientos y visualizaciones sin escribir comandos.

    uv run python -m iwbtgb.ga app

Por debajo ejecuta `python -m iwbtgb.ga entrenar|visualizar` como procesos aparte (lo mismo que la línea de comandos),
así que lo que se entrena acá se puede seguir desde la terminal y al revés.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..emulador import SALAS as STAGES
from ..qt import DIFICULTADES, Proceso, ejecutar, spin
from . import almacen
from .nucleo import Config

PAQUETE = "iwbtgb.ga"
LINEA_GEN = re.compile(r"^gen\s+(\d+) \|.*camino máx\s+([\d.]+) % \| nichos\s+(\d+).*muertes\s+([\d.]+) %")


class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWBTGB · algoritmo genético")
        self.resize(1100, 760)
        self.procesos: list[Proceso] = []
        self.entrenamiento: Proceso | None = None
        self.gen_objetivo = 0

        # -- izquierda: corridas y resumen ------------------------------------------------------------
        self.lista = QListWidget()
        self.lista.currentItemChanged.connect(self.al_elegir)
        b_nueva = QPushButton("Nueva corrida")
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
        izq.addWidget(QLabel("<b>Corridas</b>"))
        izq.addWidget(self.lista, 1)
        fila = QHBoxLayout()
        for b in (b_nueva, b_act, b_carpeta):
            fila.addWidget(b)
        izq.addLayout(fila)
        izq.addWidget(b_borrar)
        caja = QGroupBox("Resumen")
        QVBoxLayout(caja).addWidget(self.resumen)
        izq.addWidget(caja, 1)
        w_izq = QWidget()
        w_izq.setLayout(izq)

        # -- pestañas -------------------------------------------------------------------------------
        tabs = QTabWidget()
        tabs.addTab(self.pestana_entrenar(), "Entrenar")
        tabs.addTab(self.pestana_visualizar(), "Visualizar")

        # -- abajo: procesos y registro ---------------------------------------------------------------
        self.lista_proc = QListWidget()
        self.lista_proc.setMaximumHeight(90)
        b_det = QPushButton("Detener seleccionado")
        b_det.clicked.connect(self.detener_seleccionado)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("monospace", 9))
        abajo = QVBoxLayout()
        fila = QHBoxLayout()
        fila.addWidget(QLabel("<b>Procesos activos</b>"))
        fila.addStretch()
        fila.addWidget(b_det)
        abajo.addLayout(fila)
        abajo.addWidget(self.lista_proc)
        abajo.addWidget(self.log, 1)
        w_abajo = QWidget()
        w_abajo.setLayout(abajo)

        der = QSplitter(Qt.Vertical)
        der.addWidget(tabs)
        der.addWidget(w_abajo)
        der.setSizes([380, 380])
        raiz = QSplitter()
        raiz.addWidget(w_izq)
        raiz.addWidget(der)
        raiz.setSizes([300, 800])
        self.setCentralWidget(raiz)

        self.actualizar_lista()
        if self.lista.count() == 0:
            self.nueva()

    # -- pestaña Entrenar ---------------------------------------------------------------------------
    def pestana_entrenar(self) -> QWidget:
        d = Config()
        self.f_nombre = QLineEdit()
        self.f_stage = QComboBox()
        self.f_stage.addItems(STAGES)
        self.f_stage.currentIndexChanged.connect(
            lambda i: self.f_objetivo.setCurrentIndex(min(i + 1, len(STAGES) - 1)))
        self.f_objetivo = QComboBox()
        self.f_objetivo.addItems(STAGES)
        self.f_dificultad = QComboBox()
        self.f_dificultad.addItems(DIFICULTADES)
        self.f_poblacion = spin(2, 5000, d.poblacion)
        self.f_padres = spin(1, 1000, d.padres)
        self.f_generaciones = spin(1, 1_000_000, d.generaciones)
        self.f_segundos = QDoubleSpinBox()
        self.f_segundos.setRange(1, 3600)
        self.f_segundos.setDecimals(0)
        self.f_segundos.setSuffix(" s")
        self.f_segundos.setValue(d.max_segundos)
        self.f_resets = spin(0, 100, d.resets)
        self.f_resets.setToolTip("Muertes que puede reintentar (R → último save) antes de quedar descalificado")
        self.f_desde_save = spin(0, 100, round(100 * d.prob_desde_save))
        self.f_desde_save.setSuffix(" %")
        self.f_desde_save.setToolTip("Hijos de un padre que activó un save point nuevo que parten desde ese save, "
                                     "con el tiempo y los reintentos completos")
        self.f_workers = spin(1, os.cpu_count() or 1, d.workers)
        self.f_seed = spin(0, 1_000_000, d.seed)
        self.f_extra = QLineEdit()
        self.f_extra.setPlaceholderText("opcional, p. ej. --r-avance 500 --castigo-muerte 30")

        form = QFormLayout()
        form.addRow("Nombre", self.f_nombre)
        form.addRow("Stage", self.f_stage)
        form.addRow("Sala objetivo", self.f_objetivo)
        form.addRow("Dificultad", self.f_dificultad)
        form.addRow("Agentes por generación", self.f_poblacion)
        form.addRow("Padres (pasan como élite)", self.f_padres)
        form.addRow("Hasta la generación", self.f_generaciones)
        form.addRow("Duración del intento", self.f_segundos)
        form.addRow("Reintentos (R al último save)", self.f_resets)
        form.addRow("Hijos que parten del save heredado", self.f_desde_save)
        form.addRow("Procesos en paralelo", self.f_workers)
        form.addRow("Semilla", self.f_seed)
        form.addRow("Otras opciones", self.f_extra)

        self.b_entrenar = QPushButton("Entrenar desde cero")
        self.b_entrenar.clicked.connect(lambda: self.entrenar(continuar=False))
        self.b_continuar = QPushButton("Continuar")
        self.b_continuar.clicked.connect(lambda: self.entrenar(continuar=True))
        self.b_detener = QPushButton("Detener")
        self.b_detener.clicked.connect(lambda: self.entrenamiento and self.entrenamiento.detener())
        self.b_detener.setEnabled(False)
        self.barra = QProgressBar()
        self.estado = QLabel("Sin entrenamiento en curso")

        fila = QHBoxLayout()
        for b in (self.b_entrenar, self.b_continuar, self.b_detener):
            fila.addWidget(b)
        v = QVBoxLayout()
        v.addLayout(form)
        v.addLayout(fila)
        v.addWidget(self.barra)
        v.addWidget(self.estado)
        v.addStretch()
        w = QWidget()
        w.setLayout(v)
        return w

    def llenar_formulario(self, cfg: Config) -> None:
        self.f_nombre.setText(cfg.nombre)
        self.f_stage.setCurrentText(cfg.start_room or "rStage01")
        self.f_objetivo.setCurrentText(cfg.sala_objetivo)
        self.f_dificultad.setCurrentText(cfg.difficulty)
        self.f_poblacion.setValue(cfg.poblacion)
        self.f_padres.setValue(cfg.padres)
        self.f_generaciones.setValue(cfg.generaciones)
        self.f_segundos.setValue(cfg.max_segundos)
        self.f_resets.setValue(cfg.resets)
        self.f_desde_save.setValue(round(100 * cfg.prob_desde_save))
        self.f_seed.setValue(cfg.seed)
        self.f_extra.clear()

    def valores(self) -> dict:
        # rStage01 = inicio normal del juego (start_room vacío)
        return {
            "start_room": None if self.f_stage.currentText() == "rStage01" else self.f_stage.currentText(),
            "sala_objetivo": self.f_objetivo.currentText(),
            "difficulty": self.f_dificultad.currentText(),
            "poblacion": self.f_poblacion.value(),
            "padres": self.f_padres.value(),
            "elite": self.f_padres.value(),
            "generaciones": self.f_generaciones.value(),
            "max_segundos": float(self.f_segundos.value()),
            "resets": self.f_resets.value(),
            "prob_desde_save": self.f_desde_save.value() / 100,
            "workers": self.f_workers.value(),
            "seed": self.f_seed.value(),
        }

    def entrenar(self, continuar: bool) -> None:
        nombre = self.f_nombre.text().strip()
        if not re.fullmatch(r"[\w.-]+", nombre):
            QMessageBox.warning(self, "Nombre", "Usa un nombre sin espacios (letras, números, _ . -).")
            return
        carpeta = almacen.dir_corrida(nombre)
        hechas = almacen.generaciones_guardadas(carpeta) if carpeta.exists() else []
        vals = self.valores()
        args = ["entrenar", "--nombre", nombre]
        if continuar:
            if not hechas:
                QMessageBox.information(self, "Continuar", f"«{nombre}» no tiene generaciones guardadas.")
                return
            # Al continuar solo se pasa lo que cambió; la sala y la dificultad quedan como estaban
            guardada = almacen.cargar_config(carpeta)
            for k in ("start_room", "sala_objetivo", "difficulty"):
                vals.pop(k)
            if vals["padres"] == guardada.padres:  # una corrida con élite distinta de padres la conserva
                vals.pop("elite")
            vals = {k: v for k, v in vals.items() if getattr(guardada, k) != v or k == "generaciones"}
            if "resets" in vals:
                r = QMessageBox.question(self, "Reintentos", "Cambiar los reintentos a mitad de una corrida mezcla "
                                         "puntajes con reglas distintas (la élite no se vuelve a evaluar). Para "
                                         "comparar bien conviene una corrida nueva. ¿Continuar igual?")
                if r != QMessageBox.Yes:
                    return
                if not guardada.save_points:  # corrida anterior a los save points: también se activan
                    vals["save_points"] = "auto"
            if "prob_desde_save" in vals and not guardada.save_points:
                vals["save_points"] = "auto"
            if vals["generaciones"] <= hechas[-1] + 1:
                QMessageBox.information(self, "Continuar", f"Ya tiene {hechas[-1] + 1} generaciones: "
                                        "sube «Hasta la generación».")
                return
            args.append("--continuar")
        elif hechas:
            r = QMessageBox.question(self, "Sobrescribir", f"«{nombre}» ya tiene {len(hechas)} generaciones. "
                                     "¿Borrarlas y empezar de cero?")
            if r != QMessageBox.Yes:
                return
            args.append("--sobrescribir")
        for k, v in vals.items():
            if v is not None:
                args += [f"--{k.replace('_', '-')}", str(v)]
        try:
            args += shlex.split(self.f_extra.text())
        except ValueError as e:
            QMessageBox.warning(self, "Otras opciones", str(e))
            return

        self.gen_objetivo = vals["generaciones"]
        self.barra.setRange(0, self.gen_objetivo)
        self.barra.setValue(hechas[-1] + 1 if continuar else 0)
        self.estado.setText(f"Entrenando «{nombre}»… (la primera generación tarda más: arma el mapa de la sala)")
        self.entrenamiento = self.lanzar(f"entrenar {nombre}", args)
        self.botones_entrenar(True)

    def botones_entrenar(self, corriendo: bool) -> None:
        self.b_entrenar.setEnabled(not corriendo)
        self.b_continuar.setEnabled(not corriendo)
        self.b_detener.setEnabled(corriendo)

    def linea_entrenamiento(self, linea: str) -> None:
        m = LINEA_GEN.match(linea) if self.entrenamiento else None
        if m:
            gen = int(m[1])
            self.barra.setValue(gen + 1)
            self.estado.setText(f"Generación {gen + 1} de {self.gen_objetivo} · camino máx {m[2]} % · "
                                f"nichos {m[3]} · muertes {m[4]} %")
            if self.corrida_actual() == self.entrenamiento.args[2]:
                self.mostrar_resumen()

    # -- pestaña Visualizar -------------------------------------------------------------------------
    def pestana_visualizar(self) -> QWidget:
        self.v_gen = spin(-1, 1_000_000, -1, "última")
        self.v_cada = spin(1, 10_000, 10)
        self.v_vel = spin(0, 64, 0, "automática")
        form = QFormLayout()
        form.addRow("Generación", self.v_gen)
        form.addRow("Cada cuántas generaciones (evolución y todo)", self.v_cada)
        form.addRow("Velocidad (frames por cuadro de video)", self.v_vel)

        botones = [
            ("En vivo (ventana)", "vivo", "Repite la última generación guardada; se actualiza sola mientras entrena"),
            ("Video: todos juntos", "generacion", "Todos los agentes de la generación elegida"),
            ("Video: el mejor", "mejor", "El mejor agente de la generación elegida, con cámara"),
            ("Video: evolución", "evolucion", "El mejor de cada N generaciones, uno tras otro"),
            ("Gráficos", "graficos", "Progreso, diversidad, supervivencia, mapa de muertes y de distancia"),
            ("Todo", "todo", "Gráficos + videos cada N generaciones + evolución (puede tardar)"),
        ]
        grilla = QGridLayout()
        for i, (texto, modo, ayuda) in enumerate(botones):
            b = QPushButton(texto)
            b.setToolTip(ayuda)
            b.clicked.connect(lambda _=False, m=modo: self.visualizar(m))
            grilla.addWidget(b, i // 3, i % 3)
        b_vid = QPushButton("Abrir carpeta de videos")
        b_vid.clicked.connect(lambda: self.abrir("videos"))
        b_gra = QPushButton("Abrir carpeta de gráficos")
        b_gra.clicked.connect(lambda: self.abrir("graficos"))
        grilla.addWidget(b_vid, 2, 0)
        grilla.addWidget(b_gra, 2, 1)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addLayout(grilla)
        v.addWidget(QLabel("Los videos van a <i>corridas/&lt;nombre&gt;/videos/</i> y los gráficos a "
                           "<i>corridas/&lt;nombre&gt;/graficos/</i>."))
        v.addStretch()
        w = QWidget()
        w.setLayout(v)
        return w

    def visualizar(self, modo: str) -> None:
        nombre = self.corrida_actual()
        if not nombre or not almacen.generaciones_guardadas(almacen.dir_corrida(nombre)):
            QMessageBox.information(self, "Visualizar", "Elige a la izquierda una corrida con generaciones.")
            return
        args = ["visualizar", modo, "--corrida", nombre]
        if modo in ("generacion", "mejor"):
            args += ["--gen", str(self.v_gen.value())]
        if modo in ("evolucion", "todo"):
            args += ["--cada", str(self.v_cada.value())]
        if self.v_vel.value() > 0 and modo != "mejor":
            args += ["--velocidad", str(self.v_vel.value())]
        self.lanzar(f"{modo} {nombre}", args)

    # -- corridas -----------------------------------------------------------------------------------
    def actualizar_lista(self) -> None:
        actual = self.corrida_actual()
        self.lista.blockSignals(True)
        self.lista.clear()
        carpetas = sorted((p for p in almacen.CORRIDAS.glob("*/config.json")),
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
            self.mostrar_resumen()

    def corrida_actual(self) -> str:
        item = self.lista.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def al_elegir(self, item, _anterior) -> None:
        if item is None:
            return
        self.llenar_formulario(almacen.cargar_config(almacen.dir_corrida(item.data(Qt.UserRole))))
        self.mostrar_resumen()

    def nueva(self) -> None:
        self.lista.blockSignals(True)
        self.lista.setCurrentItem(None)
        self.lista.blockSignals(False)
        cfg = Config()
        n = 1
        while almacen.dir_corrida(f"corrida{n:02d}").exists():
            n += 1
        cfg.nombre = f"corrida{n:02d}"
        self.llenar_formulario(cfg)
        self.resumen.setText("Corrida nueva: ajusta los parámetros y pulsa «Entrenar desde cero».")

    def mostrar_resumen(self) -> None:
        nombre = self.corrida_actual()
        if not nombre:
            return
        carpeta = almacen.dir_corrida(nombre)
        cfg = almacen.cargar_config(carpeta)
        texto = [f"<b>{nombre}</b>",
                 f"{cfg.start_room or 'rStage01'} → {cfg.sala_objetivo} · {cfg.difficulty}",
                 f"{cfg.poblacion} agentes · {cfg.padres} padres · {cfg.max_segundos:.0f} s por intento",
                 f"selección: {cfg.seleccion}",
                 f"zonas prohibidas: {cfg.zonas_prohibidas or 'ninguna'}",
                 f"reintentos: {cfg.resets} · save points: "
                 f"{len(cfg.save_points.split(';')) if cfg.save_points and cfg.save_points != 'auto' else 0} · "
                 f"parten del save heredado: {100 * cfg.prob_desde_save:.0f} %"]
        try:
            filas = almacen.leer_metricas(carpeta)
        except (FileNotFoundError, ValueError):
            filas = []
        if filas:
            u = filas[-1]
            horas = sum(f.get("segundos_reloj", 0) for f in filas) / 3600
            texto.append(f"<br>{len(filas)} de {cfg.generaciones} generaciones ({horas:.1f} h de cálculo)")
            if "progreso_max" in u:
                texto.append(f"camino máx: <b>{max(f['progreso_max'] for f in filas):.1f} %</b>")
                texto.append(f"nichos: {u['nichos']:.0f}")
            if "saves_max" in u:
                texto.append(f"saves máx: {max(f['saves_max'] for f in filas):.0f} · "
                             f"reintentos medios: {u['reintentos_media']:.1f}"
                             + (f" · desde un save: {u['desde_save_pct']:.0f} %" if "desde_save_pct" in u else ""))
            texto.append(f"llegaron a la salida: {u['llegaron']:.0f}")
            texto.append(f"muertes: {u['muertes_pct']:.0f} % · vida media {u['segundos_medio']:.1f} s")
        self.resumen.setText("<br>".join(texto))

    def borrar(self) -> None:
        nombre = self.corrida_actual()
        if not nombre:
            return
        if any(nombre in p.args for p in self.procesos):
            QMessageBox.information(self, "Borrar", f"Hay un proceso usando «{nombre}»: detenlo primero.")
            return
        carpeta = almacen.dir_corrida(nombre)
        n = len(almacen.generaciones_guardadas(carpeta))
        mb = sum(f.stat().st_size for f in carpeta.rglob("*") if f.is_file()) / 1e6
        r = QMessageBox.question(self, "Borrar corrida", f"¿Borrar «{nombre}» ({n} generaciones, {mb:.0f} MB) "
                                 "con sus videos y gráficos?\n\nNo se puede deshacer.",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        shutil.rmtree(carpeta)
        (almacen.CORRIDAS / f"{nombre}.log").unlink(missing_ok=True)
        self.log.appendPlainText(f"borrada la corrida {nombre} ({mb:.0f} MB)")
        self.lista.setCurrentItem(None)
        self.actualizar_lista()

    def abrir(self, sub: str) -> None:
        nombre = self.corrida_actual()
        ruta = almacen.dir_corrida(nombre) / sub if nombre else almacen.CORRIDAS
        ruta.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    # -- procesos -----------------------------------------------------------------------------------
    def lanzar(self, etiqueta: str, args: list[str]) -> Proceso:
        p = Proceso(etiqueta, PAQUETE, args)
        p.linea.connect(self.al_recibir)
        p.termino.connect(self.al_terminar)
        self.log.appendPlainText(f"$ python -m {PAQUETE} {shlex.join(args)}")
        p.iniciar()
        self.procesos.append(p)
        item = QListWidgetItem(etiqueta)
        item.setData(Qt.UserRole, id(p))
        self.lista_proc.addItem(item)
        return p

    @Slot(object, str)
    def al_recibir(self, p: Proceso, linea: str) -> None:
        self.log.appendPlainText(f"[{p.etiqueta}] {linea}")
        if p is self.entrenamiento:
            self.linea_entrenamiento(linea)

    @Slot(object, int)
    def al_terminar(self, p: Proceso, codigo: int) -> None:
        estado = "terminó" if codigo == 0 else ("detenido" if codigo in (130, -2, -9) else f"falló (código {codigo})")
        self.log.appendPlainText(f"[{p.etiqueta}] {estado}")
        self.procesos.remove(p)
        for i in range(self.lista_proc.count()):
            if self.lista_proc.item(i).data(Qt.UserRole) == id(p):
                self.lista_proc.takeItem(i)
                break
        if p is self.entrenamiento:
            self.entrenamiento = None
            self.botones_entrenar(False)
            self.estado.setText(f"Entrenamiento {estado}. " + self.estado.text())
        self.actualizar_lista()

    def detener_seleccionado(self) -> None:
        item = self.lista_proc.currentItem()
        if item:
            for p in self.procesos:
                if id(p) == item.data(Qt.UserRole):
                    p.detener()

    def closeEvent(self, ev) -> None:
        if self.procesos:
            r = QMessageBox.question(self, "Salir", f"Hay {len(self.procesos)} proceso(s) en curso. ¿Detenerlos y "
                                     "salir? (Las generaciones completas quedan guardadas.)")
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


def main(argv=None):
    ejecutar(Ventana, "iwbtgb-ga", Path(__file__).with_name("icono.png"))
