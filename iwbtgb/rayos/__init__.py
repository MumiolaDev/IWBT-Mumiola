"""Red neuronal que ve con rayos (distancia y tipo de lo que tiene alrededor), entrenada con el Deep GA de
`iwbtgb.evolucion`. Ver docs/redes.md."""

from ..evolucion import Linea
from ..rutas import CORRIDAS_RAYOS
from .config import Config
from .dibujo import DibujoRedDensa, leyenda, rayos_sobre_juego
from .jugadores import JugadorRayos, PoliticaRayos, gimnasio, red_para

LINEA = Linea(
    nombre="rayos", paquete="iwbtgb.rayos", titulo="redes de rayos", corridas=CORRIDAS_RAYOS, Config=Config,
    prefijo="rayos", forma_red=("n_rayos", "paso_px", "cuadros", "capas", "activacion", "ver_velocidad",
                                "ver_doble_salto", "ver_direccion", "ver_mirando", "frames_por_decision"),
    red_para=red_para, jugador_nivel=JugadorRayos, gimnasio=gimnasio, politica=PoliticaRayos, dibujo=DibujoRedDensa,
    sobre_juego=rayos_sobre_juego, leyenda=leyenda,
)
