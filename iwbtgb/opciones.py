"""Opciones de línea de comandos a partir de los campos de una config (dataclass): `max_segundos` → `--max-segundos`."""

from __future__ import annotations

import argparse
from dataclasses import fields

SI, NO = ("1", "true", "si", "sí", "yes"), ("0", "false", "no")


def booleano(texto: str) -> bool:
    t = texto.strip().lower()
    if t in SI:
        return True
    if t in NO:
        return False
    raise argparse.ArgumentTypeError(f"se esperaba sí o no (1 o 0), vino {texto!r}")


def tipo_de(campo) -> type | callable:
    """int, float, `booleano` o str, según la anotación (con `from __future__ import annotations` es un texto)."""
    nombre = campo.type if isinstance(campo.type, str) else getattr(campo.type, "__name__", str(campo.type))
    return {"int": int, "float": float, "bool": booleano}.get(nombre.split("|")[0].strip(), str)


def agregar(parser: argparse.ArgumentParser, cls, omitir=()) -> None:
    """Una opción por campo de `cls` (salvo `omitir`), con default None: así se sabe cuáles se dieron."""
    for f in fields(cls):
        if f.name not in omitir:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=tipo_de(f), default=None)


def dadas(args: argparse.Namespace, cls, omitir=()) -> dict:
    """{campo: valor} de las opciones que se dieron."""
    return {f.name: getattr(args, f.name) for f in fields(cls)
            if f.name not in omitir and getattr(args, f.name, None) is not None}
