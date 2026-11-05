"""Configuración de una corrida de `rayos`: la común (`evolucion.config`) más los rayos y las capas de la red."""

from __future__ import annotations

from dataclasses import dataclass

from ..evolucion.config import ConfigBase


@dataclass
class Config(ConfigBase):
    nombre: str = "rayos01"
    # -- lo que ve: rayos desde el jugador hasta el borde de la pantalla, cortados por el primer sólido. Cada rayo
    #    dice qué tan cerca está lo primero de cada tipo (sólido, plataforma, mortal, save, salida, agua, borde).
    n_rayos: int = 64
    paso_px: int = 2                    # precisión de los rayos (px)
    ver_velocidad: int = 1              # velocidad horizontal y vertical del jugador
    ver_doble_salto: int = 1            # si le queda el doble salto
    ver_direccion: int = 1              # hacia dónde avanza el nivel desde donde está (el camino más corto a la salida)
    ver_mirando: int = 1                # hacia qué lado mira (dispara hacia ese lado)
    # -- la red: densa, con estas capas ocultas (neuronas de cada una, separadas por coma)
    capas: str = "64,32"
    activacion: str = "relu"            # "relu" o "tanh"

    @property
    def lista_capas(self) -> list[int]:
        return [int(n) for n in str(self.capas).replace(" ", "").split(",") if n]

    def problema(self) -> str | None:
        if not self.lista_capas:
            return "--capas: al menos una capa oculta, p. ej. 64,32"
        if self.activacion not in ("relu", "tanh"):
            return f"--activacion: relu o tanh, no {self.activacion}"
        return None
