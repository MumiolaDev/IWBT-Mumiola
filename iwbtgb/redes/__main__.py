from ..comandos import EVOLUCION, despachar
from . import LINEA

despachar("iwbtgb.redes", {
    **EVOLUCION,
    "vista": (".vista:main", "la vista semántica: probar su velocidad, o el mejor del GA con su vista al lado"),
    "app": (".app:main", "interfaz para crear, entrenar, probar y ver corridas"),
}, LINEA)
