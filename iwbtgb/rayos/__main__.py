from ..comandos import EVOLUCION, despachar
from . import LINEA

despachar("iwbtgb.rayos", {
    **EVOLUCION,
    "app": (".app:main", "interfaz para crear, entrenar, probar, ver y transmitir corridas"),
}, LINEA)
