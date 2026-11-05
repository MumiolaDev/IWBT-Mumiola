"""Disco de una corrida: <corridas de la línea>/<nombre>/ con config.json, distancia.npz, metricas.csv, log.txt,
gen_NNNN.npz, redes/, pruebas/ y videos/.

Cada `gen_NNNN.npz` guarda la población como linaje respecto de la generación anterior (`padre`, `semilla`, `sigma`):
un hijo es su padre más una mutación; la élite, su padre sin cambios (semilla −1); en la generación 0, `semilla` es la
de los pesos iniciales (padre −1). Con eso se reconstruye el genoma de cualquiera. Además van el fitness, los avances
y fines por escenario, las acciones y las trazas por frame (`traza.TRAZA_DTYPE`; en el gimnasio, de todas las salas)
y, en el gimnasio, las semillas y dificultades de sus salas y las reglas con que se generaron. La mejor red va también
a `redes/` (pesos y genoma), para usarla sin reconstruirla.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .genoma import Genoma, genoma_de

CAMPOS_METRICAS = [
    "gen", "fitness_max", "fitness_media", "avance_max", "avance_media", "avance_mejor", "llegaron", "nichos",
    "murio_pct", "estancado_pct", "tiempo_pct", "salio_pct", "frames_medio", "segundos_reloj",
    "muertes_media", "saves_max", "saves_media", "quieto_pct", "celdas_media", "celdas_max",
    "salas_pct", "salas_mejor", "dificultad_min", "dificultad_max", "trampas",
]


def guardar_config(carpeta: Path, cfg) -> None:
    tmp = carpeta / "config.json.tmp"
    tmp.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(carpeta / "config.json")


def generaciones_guardadas(carpeta: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in carpeta.glob("gen_*.npz"))


def guardar_generacion(carpeta: Path, gen: int, datos: dict) -> None:
    # Se escribe aparte y se renombra: si se detiene el entrenamiento a mitad, no queda un .npz roto
    tmp = carpeta / f"gen_{gen:04d}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez_compressed(f, **datos)
    tmp.replace(carpeta / f"gen_{gen:04d}.npz")


def cargar_generacion(carpeta: Path, gen: int, claves=None) -> dict:
    """Los arreglos de la generación (solo `claves`, si se dan: cada uno se descomprime entero al leerlo)."""
    with np.load(carpeta / f"gen_{gen:04d}.npz") as z:
        return {k: z[k] for k in (z.files if claves is None else claves) if k in z.files}


def indice_mejor(d: dict) -> int:
    """La mejor red de una generación = la que llega más lejos en su primer escenario (el que muestran los videos);
    en el gimnasio, la de más avance promedio en sus salas. A igual avance, la de mayor fitness. La selección para
    reproducirse sigue siendo por fitness."""
    avance = d["avance"] if "gimnasio" in d else d["avances"][:, 0]
    return int(np.lexsort((d["fitness"], avance))[-1])


def genomas(carpeta: Path, hasta: int | None = None) -> list[Genoma]:
    """Genomas de la población de la generación `hasta` (la última si es None), siguiendo el linaje."""
    hasta = generaciones_guardadas(carpeta)[-1] if hasta is None else hasta
    actual: list[Genoma] = []
    base = genoma_de(json.loads((carpeta / "config.json").read_text()).get("genoma_inicial", []))  # `desde`
    for g in range(hasta + 1):
        d = cargar_generacion(carpeta, g, ("padre", "semilla", "sigma"))
        nuevos = []
        for padre, semilla, sigma in zip(d["padre"], d["semilla"], d["sigma"]):
            if padre < 0 and base:
                nuevos.append(base if sigma == 0 else base + ((int(semilla), float(sigma)),))
            elif padre < 0:
                nuevos.append((int(semilla),))
            elif semilla < 0:
                nuevos.append(actual[padre])
            else:
                nuevos.append(actual[padre] + ((int(semilla), float(sigma)),))
        actual = nuevos
    return actual


def red_de(carpeta: Path, gen: int = -1) -> tuple[int, Genoma, float]:
    """(generación, genoma, fitness) de la mejor red de `gen` (la última con -1): la que llega más lejos. El genoma
    sale de redes/ si está ahí (la última y cada `guardar_red_cada`) o, si no, del linaje."""
    gen = generaciones_guardadas(carpeta)[-1] if gen < 0 else gen
    d = cargar_generacion(carpeta, gen, ("fitness", "avance", "avances", "gimnasio"))
    i = indice_mejor(d)
    for nombre in ("mejor", f"gen_{gen:04d}"):
        ruta = carpeta / "redes" / f"{nombre}.json"
        try:
            info = json.loads(ruta.read_text())
        except (OSError, ValueError):  # no está, o se está escribiendo justo ahora
            continue
        if info.get("gen") == gen and info.get("agente") == i:
            return gen, genoma_de(info["genoma"]), float(d["fitness"][i])
    return gen, genomas(carpeta, gen)[i], float(d["fitness"][i])


def guardar_red(carpeta: Path, nombre: str, pesos: np.ndarray, info: dict) -> None:
    (carpeta / "redes").mkdir(exist_ok=True)
    np.save(carpeta / "redes" / f"{nombre}.npy", pesos)
    (carpeta / "redes" / f"{nombre}.json").write_text(json.dumps(info, ensure_ascii=False))


def agregar_metricas(carpeta: Path, fila: dict) -> None:
    ruta = carpeta / "metricas.csv"
    if ruta.exists():
        with ruta.open() as f:
            campos = next(csv.reader(f), [])
        if campos != CAMPOS_METRICAS:  # corrida de antes de alguna columna nueva: se reescribe con todas
            with ruta.open() as f:
                filas = list(csv.DictReader(f))
            with ruta.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CAMPOS_METRICAS)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in CAMPOS_METRICAS} for r in filas)
    nuevo = not ruta.exists()
    with ruta.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_METRICAS)
        if nuevo:
            w.writeheader()
        w.writerow({k: fila.get(k, "") for k in CAMPOS_METRICAS})


def leer_metricas(carpeta: Path) -> list[dict]:
    ruta = carpeta / "metricas.csv"
    if not ruta.exists():
        return []
    with ruta.open() as f:
        return [{k: float(v) for k, v in fila.items() if v != ""} for fila in csv.DictReader(f)]
