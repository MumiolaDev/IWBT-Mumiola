"""App de `rayos`: `uv run python -m iwbtgb.rayos app` (en el menú: iwbtgb-rayos.desktop). La ventana es la común
(`evolucion.app`); aquí, las secciones de lo que ve (los rayos) y de la red (sus capas)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from ..evolucion.app import FormularioRed, Ventana
from ..qt import ejecutar, spin
from . import LINEA, observacion
from .config import Config
from .jugadores import red_para

MAX_CAPAS = 6


class EditorCapas(QWidget):
    """Cuántas capas ocultas y cuántas neuronas tiene cada una."""

    def __init__(self, al_cambiar):
        super().__init__()
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 0, 0, 0)
        self.n = spin(1, MAX_CAPAS, 2)
        self.n.setToolTip("Capas ocultas entre lo que ve y las acciones")
        self.n.valueChanged.connect(self._mostrar)
        self.n.valueChanged.connect(al_cambiar)
        fila.addWidget(self.n)
        fila.addWidget(QLabel("capas:"))
        self.nodos: list[QSpinBox] = []
        for k in range(MAX_CAPAS):
            s = spin(1, 4096, 32)
            s.setToolTip(f"Neuronas de la capa {k + 1}")
            s.valueChanged.connect(al_cambiar)
            self.nodos.append(s)
            fila.addWidget(s)
        fila.addStretch()
        self._mostrar()

    def _mostrar(self) -> None:
        for k, s in enumerate(self.nodos):
            s.setVisible(k < self.n.value())

    def texto(self) -> str:
        return ",".join(str(s.value()) for s in self.nodos[:self.n.value()])

    def poner(self, texto: str) -> None:
        capas = [int(v) for v in str(texto).replace(" ", "").split(",") if v][:MAX_CAPAS] or [64]
        self.n.setValue(len(capas))
        for s, v in zip(self.nodos, capas):
            s.setValue(v)
        self._mostrar()


class FormularioRayos(FormularioRed):
    def __init__(self, cambio, d: Config):
        super().__init__(cambio, d)
        self.l_ve = QLabel()
        self.l_ve.setWordWrap(True)
        self.l_ve.setStyleSheet("color: gray")
        self.f_rayos = spin(4, 512, d.n_rayos)
        self.f_rayos.setToolTip("Rayos repartidos en todas las direcciones desde el jugador hasta el borde de la "
                                "pantalla (los corta el primer sólido)")
        self.f_paso = spin(1, 16, d.paso_px)
        self.f_paso.setSuffix(" px")
        self.f_paso.setToolTip("Cada cuántos píxeles mira el rayo dentro de un objeto (precisión del borde de una púa)")
        self.f_cuadros = spin(1, 8, d.cuadros)
        self.f_cuadros.setToolTip("Cuántas lecturas de los rayos ve a la vez: la actual y las de las decisiones "
                                  "anteriores. Con 1 no puede saber hacia dónde se mueve algo")
        self.f_ver_vel = QCheckBox("su velocidad")
        self.f_ver_doble = QCheckBox("si le queda el doble salto")
        self.f_ver_dir = QCheckBox("hacia dónde avanza el nivel")
        self.f_ver_dir.setToolTip("La dirección del camino más corto a la salida desde donde está (como la ve un "
                                  "jugador que mira el mapa)")
        self.f_ver_mira = QCheckBox("hacia qué lado mira")
        self.w_ver = QWidget()
        fila = QHBoxLayout(self.w_ver)
        fila.setContentsMargins(0, 0, 0, 0)
        for w in (self.f_ver_vel, self.f_ver_doble, self.f_ver_dir, self.f_ver_mira):
            fila.addWidget(w)
            w.toggled.connect(cambio)
        fila.addStretch()

        self.f_capas = EditorCapas(cambio)
        self.f_activacion = QComboBox()
        self.f_activacion.addItem("ReLU (se apaga en negativo)", "relu")
        self.f_activacion.addItem("tanh (entre −1 y 1)", "tanh")
        self.f_fpd = spin(1, 25, d.frames_por_decision)
        self.f_fpd.setSuffix(" frames")
        self.f_fpd.setToolTip("Cada cuántos frames decide (mantiene las teclas entre medio); 50 frames = 1 s")
        self.f_soltar = QCheckBox("Soltar el salto solo para poder volver a saltar")
        self.f_soltar.setToolTip("En el juego el salto sale al apretar la tecla, no al mantenerla. Si la red sigue "
                                 "pidiendo salto y el personaje ya no sube, se suelta la tecla un frame")
        self.n_pesos = QLabel()
        self.n_pesos.setWordWrap(True)
        for s in (self.f_rayos, self.f_cuadros, self.f_fpd):
            s.valueChanged.connect(cambio)

    def grupos(self):
        return [("Lo que ve", [("Rayos", self.f_rayos), ("Precisión", self.f_paso), ("Lecturas a la vez", self.f_cuadros),
                               ("Además ve", self.w_ver)], self.l_ve),
                ("La red", [("Capas ocultas", self.f_capas), ("Activación", self.f_activacion),
                            ("Decide cada", self.f_fpd), ("", self.f_soltar), ("Tamaño", self.n_pesos)], None)]

    def config(self) -> Config:
        return Config(n_rayos=self.f_rayos.value(), cuadros=self.f_cuadros.value(), capas=self.f_capas.texto(),
                      ver_velocidad=int(self.f_ver_vel.isChecked()), ver_doble_salto=int(self.f_ver_doble.isChecked()),
                      ver_direccion=int(self.f_ver_dir.isChecked()), ver_mirando=int(self.f_ver_mira.isChecked()))

    def explicar(self, cfg: Config) -> None:
        red = red_para(cfg)
        extras = observacion.nombres_extras(cfg)
        n_rayos = cfg.n_rayos * observacion.POR_RAYO * cfg.cuadros
        capas = " → ".join(str(n) for n in cfg.lista_capas)
        self.n_pesos.setText(f"{red.n_entrada} entradas → {capas} → 12 acciones · "
                             f"<b>{red.n_pesos:,} pesos</b>".replace(",", "."))
        self.l_ve.setText(
            f"{cfg.n_rayos} rayos salen del jugador en todas las direcciones hasta el borde de la pantalla; los corta "
            f"el primer sólido. Cada uno dice qué tan cerca tiene lo primero de cada tipo: sólido, plataforma, mortal, "
            f"save, salida, agua y el borde ({observacion.POR_RAYO} valores; así ve una púa delante de un muro y también "
            f"el muro). Con {cfg.cuadros} lectura{'s' if cfg.cuadros > 1 else ''} a la vez son {n_rayos} valores"
            + (f", más {len(extras)} del jugador ({', '.join(extras)})." if extras else "."))

    def llenar_forma(self, cfg: Config) -> None:
        self.f_rayos.setValue(cfg.n_rayos)
        self.f_paso.setValue(cfg.paso_px)
        self.f_cuadros.setValue(cfg.cuadros)
        self.f_ver_vel.setChecked(bool(cfg.ver_velocidad))
        self.f_ver_doble.setChecked(bool(cfg.ver_doble_salto))
        self.f_ver_dir.setChecked(bool(cfg.ver_direccion))
        self.f_ver_mira.setChecked(bool(cfg.ver_mirando))
        self.f_capas.poner(cfg.capas)
        self.f_activacion.setCurrentIndex(max(0, self.f_activacion.findData(cfg.activacion)))
        self.f_fpd.setValue(cfg.frames_por_decision)

    def llenar(self, cfg: Config) -> None:
        self.llenar_forma(cfg)
        self.f_soltar.setChecked(bool(cfg.soltar_salto))

    def valores(self) -> dict:
        return {"n_rayos": self.f_rayos.value(), "paso_px": self.f_paso.value(), "cuadros": self.f_cuadros.value(),
                "ver_velocidad": int(self.f_ver_vel.isChecked()), "ver_doble_salto": int(self.f_ver_doble.isChecked()),
                "ver_direccion": int(self.f_ver_dir.isChecked()), "ver_mirando": int(self.f_ver_mira.isChecked()),
                "capas": self.f_capas.texto(), "activacion": self.f_activacion.currentData(),
                "frames_por_decision": self.f_fpd.value(), "soltar_salto": int(self.f_soltar.isChecked())}

    def fijos(self):
        return [self.f_rayos, self.f_paso, self.f_cuadros, self.w_ver, self.f_capas, self.f_activacion, self.f_fpd]

    def resumen(self, cfg: Config) -> list[str]:
        red = red_para(cfg)
        extras = observacion.nombres_extras(cfg)
        pesos = f"{red.n_pesos:,}".replace(",", ".")
        return [f"ve: {cfg.n_rayos} rayos × {cfg.cuadros} lectura{'s' if cfg.cuadros > 1 else ''}"
                + (f" + {', '.join(extras)}" if extras else ""),
                f"red: {red.n_entrada} → {' → '.join(map(str, cfg.lista_capas))} → 12 ({pesos} pesos, "
                f"{cfg.activacion}), decide cada {cfg.frames_por_decision} "
                f"frame{'s' if cfg.frames_por_decision > 1 else ''}"]


def main(argv=None):
    ejecutar(lambda: Ventana(LINEA, FormularioRayos), "iwbtgb-rayos", Path(__file__).with_name("icono.png"))
