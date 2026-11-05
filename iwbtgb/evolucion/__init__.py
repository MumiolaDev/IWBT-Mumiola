"""Neuroevolución (Deep GA) común a las dos redes del proyecto: la que ve una grilla (`iwbtgb.redes`) y la que ve con
rayos (`iwbtgb.rayos`). Ver docs/redes.md.

Cada línea se describe con una `Linea` (lo que ve su red, la red y cómo se dibuja) y el resto es de aquí: el gimnasio
de salas generadas, las partidas en un nivel, el entrenamiento, las pruebas, los videos, la transmisión y la app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CAMPOS_JUEGO = ("start_room", "sala_objetivo", "difficulty", "seed_juego", "entorno", "desde", "desde_gen")


@dataclass(frozen=True)
class Linea:
    nombre: str                       # "redes" o "rayos"
    paquete: str                      # el de `python -m`
    titulo: str                       # "redes neuronales", "redes de rayos"
    corridas: Path
    Config: type
    prefijo: str                      # nombre de una corrida nueva: <prefijo>01, <prefijo>02…
    forma_red: tuple[str, ...]        # lo que ve y la forma de la red: se copia con --desde y no cambia al continuar
    red_para: Callable                # cfg → red
    jugador_nivel: Callable           # (cfg, mapa) → episodio.Jugador
    gimnasio: Callable                # cfg → gimnasio.Gimnasio con lo que ve su red
    politica: Callable                # (cfg, gimnasio) → gimnasio.PoliticaRed
    dibujo: Callable                  # (red, cfg, caja, etiquetas) → la red dibujada por dentro (`--red`)
    sobre_juego: Callable | None = None   # (img, res, esquina de la vista) → algo encima del juego (los rayos)
    leyenda: Callable | None = None       # (draw, x, y) → la leyenda de lo que dibuja `sobre_juego`

    @property
    def fijos(self) -> tuple[str, ...]:
        """Lo que no se puede cambiar al continuar una corrida: el juego que ve y la forma de la red."""
        return CAMPOS_JUEGO + self.forma_red

    def dir_corrida(self, nombre: str) -> Path:
        return self.corridas / nombre

    def cargar_config(self, carpeta: Path):
        return self.Config.from_dict(json.loads((carpeta / "config.json").read_text()))

    def corridas_guardadas(self) -> list[str]:
        if not self.corridas.exists():
            return []
        return sorted(p.name for p in self.corridas.iterdir() if (p / "config.json").exists())

    def jugador_gimnasio(self, cfg):
        from .gimnasio import JugadorGimnasio
        gim = self.gimnasio(cfg)
        return JugadorGimnasio(cfg, gim, self.politica(cfg, gim))

    def preparar_gimnasio(self, corrida: str, gen: int = -1):
        """(config, gimnasio con su vista, la mejor red de la generación) de una corrida."""
        carpeta = self.dir_corrida(corrida)
        cfg = self.cargar_config(carpeta)
        gim = self.gimnasio(cfg)
        politica = self.politica(cfg, gim)
        politica.carpeta = carpeta
        politica.cargar(gen)
        return cfg, gim, politica
