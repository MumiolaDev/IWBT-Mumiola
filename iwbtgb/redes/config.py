"""Configuración de una corrida de `redes`: la común (`evolucion.config`) más la vista y la forma de la CNN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..evolucion.config import ConfigBase
from .vista import ConfigVista


@dataclass
class Config(ConfigBase):
    nombre: str = "red01"
    entorno: str = "nivel"
    gim_dificultad_max: float = 0.5
    gim_trampas: int = 1
    gim_curriculo: int = 0
    escenarios: int = 3
    max_segundos: float = 30.0
    r_avance: float = 100.0
    castigo_muerte: float = 3.0
    padres: int = 20
    elite: int = 5
    sigma: float = 0.02
    workers: int = 10
    # -- la red: CNN chica sobre la grilla + variables del jugador
    filtros1: int = 8                   # conv 4×4, paso 4
    filtros2: int = 16                  # conv 3×3, paso 2
    oculta: int = 64                    # capa densa
    vista: dict = field(default_factory=lambda: ConfigVista().to_dict())

    ANTERIORES: ClassVar[dict] = {
        "castigo_muerte": 0.0, "r_save": 0.0, "reintentos": 0, "castigo_quieto": 0.0, "elite_mas_lejos": 0,
        "r_explorar": 0.0, "entorno": "nivel", "desde": "", "soltar_salto": 0, "gim_salida_borde": 0,
        "gim_trampas": 0}

    @property
    def config_vista(self) -> ConfigVista:
        return ConfigVista.from_dict(self.vista)
