"""Red neuronal que ve una grilla semántica alrededor del jugador (una CNN chica), entrenada con el Deep GA de
`iwbtgb.evolucion`. Ver docs/redes.md."""

from ..evolucion import Linea
from ..rutas import CORRIDAS_REDES
from .config import Config
from .dibujo import DibujoRed
from .jugadores import JugadorGrilla, PoliticaGrilla, gimnasio, red_para

LINEA = Linea(
    nombre="redes", paquete="iwbtgb.redes", titulo="redes neuronales", corridas=CORRIDAS_REDES, Config=Config,
    prefijo="red", forma_red=("cuadros", "filtros1", "filtros2", "oculta", "frames_por_decision", "vista"),
    red_para=red_para, jugador_nivel=JugadorGrilla, gimnasio=gimnasio, politica=PoliticaGrilla, dibujo=DibujoRed,
)
