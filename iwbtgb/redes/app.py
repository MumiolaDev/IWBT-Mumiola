"""App de `redes`: `uv run python -m iwbtgb.redes app` (en el menú: iwbtgb-redes.desktop). La ventana es la común
(`evolucion.app`); aquí, la sección de la red (la CNN y la grilla que ve)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QLabel

from ..evolucion.app import FormularioRed, Ventana
from ..qt import ejecutar, spin
from . import LINEA
from .config import Config
from .jugadores import red_para


class FormularioGrilla(FormularioRed):
    def __init__(self, cambio, d: Config):
        super().__init__(cambio, d)
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
            s.valueChanged.connect(cambio)

    def grupos(self):
        return [("La red", [("Grillas que ve a la vez", self.f_cuadros),
                            ("Detectores de patrones chicos", self.f_filtros1),
                            ("Detectores de patrones grandes", self.f_filtros2), ("Neuronas de decisión", self.f_oculta),
                            ("Decide cada", self.f_fpd), ("", self.f_soltar), ("Tamaño", self.n_pesos)], self.l_red)]

    def config(self) -> Config:
        return Config(cuadros=self.f_cuadros.value(), filtros1=self.f_filtros1.value(),
                      filtros2=self.f_filtros2.value(), oculta=self.f_oculta.value(),
                      frames_por_decision=self.f_fpd.value())

    def explicar(self, cfg: Config) -> None:
        vista = cfg.config_vista
        self.n_pesos.setText(f"{red_para(cfg).n_pesos:,} pesos".replace(",", "."))
        self.l_red.setText(
            f"La red ve una grilla de {vista.ancho}×{vista.alto} celdas de {vista.celda} px alrededor del jugador, con "
            f"{vista.n_canales} capas de información (sólido, plataforma, púas, save, salida, agua, fuera de la "
            f"sala), y las {cfg.cuadros} últimas a la vez. {cfg.filtros1} detectores buscan patrones chicos; "
            f"{cfg.filtros2} los combinan en figuras más grandes; {cfg.oculta} neuronas de decisión juntan eso con la "
            f"velocidad y el doble salto y puntúan las 12 acciones (decide cada {cfg.frames_por_decision} "
            f"frame{'s' if cfg.frames_por_decision > 1 else ''}).")

    def llenar_forma(self, cfg: Config) -> None:
        self.f_cuadros.setValue(cfg.cuadros)
        self.f_filtros1.setValue(cfg.filtros1)
        self.f_filtros2.setValue(cfg.filtros2)
        self.f_oculta.setValue(cfg.oculta)
        self.f_fpd.setValue(cfg.frames_por_decision)

    def llenar(self, cfg: Config) -> None:
        self.llenar_forma(cfg)
        self.f_soltar.setChecked(bool(cfg.soltar_salto))

    def valores(self) -> dict:
        return {"cuadros": self.f_cuadros.value(), "filtros1": self.f_filtros1.value(),
                "filtros2": self.f_filtros2.value(), "oculta": self.f_oculta.value(),
                "frames_por_decision": self.f_fpd.value(), "soltar_salto": int(self.f_soltar.isChecked())}

    def fijos(self):
        return [self.f_cuadros, self.f_filtros1, self.f_filtros2, self.f_oculta, self.f_fpd]

    def resumen(self, cfg: Config) -> list[str]:
        pesos = f"{red_para(cfg).n_pesos:,}".replace(",", ".")
        return [f"red: {cfg.cuadros} grilla{'s' if cfg.cuadros > 1 else ''} → {cfg.filtros1} detectores chicos → "
                f"{cfg.filtros2} grandes → {cfg.oculta} neuronas → 12 acciones ({pesos} pesos), decide cada "
                f"{cfg.frames_por_decision} frame{'s' if cfg.frames_por_decision > 1 else ''}"
                f"{', suelta el salto sola' if cfg.soltar_salto else ''}"]


def main(argv=None):
    ejecutar(lambda: Ventana(LINEA, FormularioGrilla), "iwbtgb-redes", Path(__file__).with_name("icono.png"))
