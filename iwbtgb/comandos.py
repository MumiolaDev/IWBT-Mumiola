"""`python -m iwbtgb.<línea> <comando> [opciones]`: cada línea declara sus comandos y este módulo los despacha."""

from __future__ import annotations

import importlib
import sys


# Los comandos de la neuroevolución, comunes a `redes` y `rayos` (reciben la línea)
EVOLUCION = {
    "entrenar": ("iwbtgb.evolucion.entrenar:main", "Deep GA en el gimnasio o en un nivel (opciones = la config)"),
    "probar": ("iwbtgb.evolucion.probar:main", "la mejor red en salas que no vio o en otros niveles"),
    "visualizar": ("iwbtgb.evolucion.visualizar:main", "videos, en vivo y gráficos"),
    "transmitir": ("iwbtgb.evolucion.transmitir:main", "stream RTMP del entrenamiento"),
}


def despachar(paquete: str, comandos: dict[str, tuple[str, str]], linea=None, argv=None) -> None:
    """`comandos` = {nombre: ("módulo:función", descripción)}; la función recibe el resto de los argumentos (y antes
    `linea`, si es de `iwbtgb.evolucion`). El módulo se importa solo al usarlo (la app no carga lo de los videos, ni
    el entrenamiento la de Qt)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in comandos:
        print(f"uso: python -m {paquete} <comando> [opciones]   (--help en cada comando)\n")
        for nombre, (_, descripcion) in comandos.items():
            print(f"  {nombre:12s} {descripcion}")
        sys.exit(0 if argv[:1] in (["-h"], ["--help"]) else 2)
    comando, resto = argv[0], argv[1:]
    modulo, funcion = comandos[comando][0].split(":")
    principal = getattr(importlib.import_module(modulo, paquete), funcion)
    try:
        if modulo.startswith("iwbtgb.evolucion"):
            principal(linea, resto)
        else:
            principal(resto)
    except KeyboardInterrupt:  # Ctrl+C o «Detener» en una app: lo completo ya quedó guardado
        if comando == "entrenar":
            print("detenido; para seguir: --continuar", flush=True)
        sys.exit(130)
