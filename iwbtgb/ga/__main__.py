from ..comandos import despachar

despachar("iwbtgb.ga", {
    "entrenar": (".entrenar:main", "algoritmo genético (opciones = campos de Config en nucleo.py)"),
    "explorar": (".explorar:main", "Go-Explore (archivo de celdas)"),
    "visualizar": (".visualizar:main", "videos, gráficos y en vivo de una corrida"),
    "app": (".app:main", "interfaz para entrenar y visualizar"),
})
