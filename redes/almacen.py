"""Disco: corridas_redes/<nombre>/ con config.json, distancia.npz, metricas.csv, gen_NNNN.npz y redes/.

Cada `gen_NNNN.npz` guarda la población como linaje respecto de la generación anterior (`padre`, `semilla`,
`sigma`): un hijo es su padre más una mutación; la élite, su padre sin cambios (semilla −1); en la generación 0,
`semilla` es la de los pesos iniciales (padre −1). Con eso se reconstruye el genoma de cualquiera. Los pesos
completos del mejor van a `redes/` (para usarlos sin reconstruir).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import Config
from .modelo import Genoma

ROOT = Path(__file__).resolve().parent.parent
CORRIDAS = ROOT / "corridas_redes"

CAMPOS_METRICAS = [
    "gen", "fitness_max", "fitness_media", "avance_max", "avance_media", "avance_mejor", "llegaron", "nichos",
    "murio_pct", "estancado_pct", "tiempo_pct", "salio_pct", "frames_medio", "segundos_reloj",
    "muertes_media", "saves_max", "saves_media", "quieto_pct", "celdas_media", "celdas_max",
    "salas_pct", "salas_mejor",
]


def dir_corrida(nombre: str) -> Path:
    return CORRIDAS / nombre


def guardar_config(carpeta: Path, cfg: Config) -> None:
    (carpeta / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def cargar_config(carpeta: Path) -> Config:
    return Config.from_dict(json.loads((carpeta / "config.json").read_text()))


def generaciones_guardadas(carpeta: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in carpeta.glob("gen_*.npz"))


def guardar_generacion(carpeta: Path, gen: int, datos: dict) -> None:
    tmp = carpeta / f"gen_{gen:04d}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez_compressed(f, **datos)
    tmp.replace(carpeta / f"gen_{gen:04d}.npz")


def cargar_generacion(carpeta: Path, gen: int) -> dict:
    with np.load(carpeta / f"gen_{gen:04d}.npz") as z:
        return {k: z[k] for k in z.files}


def indice_mejor(d: dict) -> int:
    """La mejor red de una generación = la que llega más lejos en su primer escenario (el que muestran los videos);
    en el gimnasio, la de más avance promedio en sus salas. A igual avance, la de mayor fitness. La selección para
    reproducirse sigue siendo por fitness."""
    avance = d["avance"] if "gimnasio" in d else d["avances"][:, 0]
    return int(np.lexsort((d["fitness"], avance))[-1])


def genoma_de(lista) -> Genoma:
    """Genoma guardado en JSON ([semilla, [s, sigma], ...]) → tupla."""
    return tuple([int(lista[0])] + [(int(s), float(g)) for s, g in lista[1:]]) if len(lista) else ()


def genomas(carpeta: Path, hasta: int | None = None) -> list[Genoma]:
    """Genomas de la población de la generación `hasta` (la última si es None), siguiendo el linaje."""
    gens = generaciones_guardadas(carpeta)
    hasta = gens[-1] if hasta is None else hasta
    actual: list[Genoma] = []
    base = genoma_de(cargar_config(carpeta).genoma_inicial)   # corrida que parte de otra red (`desde`)
    for g in range(hasta + 1):
        d = cargar_generacion(carpeta, g)
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
