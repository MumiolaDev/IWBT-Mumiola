"""Guardado y carga de generaciones: `corridas/<nombre>/gen_NNNN.npz` + `metricas.csv` + `config.json`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .nucleo import TRAZA_DTYPE, Config, Genoma, Resultado, compactar

ROOT = Path(__file__).resolve().parent.parent
CORRIDAS = ROOT / "corridas"

CAMPOS_METRICAS = [
    "gen", "fitness_max", "fitness_media", "fitness_mediana", "fitness_min",
    "pantallas_max", "pantallas_media", "llegaron", "celdas_max", "celdas_media",
    "muertes_pct", "segundos_medio", "segundos_mejor", "evaluados", "segundos_reloj",
    "progreso_max", "progreso_final_max", "progreso_media", "nichos",
]


def dir_corrida(nombre: str) -> Path:
    return CORRIDAS / nombre


def guardar_generacion(carpeta: Path, gen: int, pob: list[Resultado]) -> Path:
    largos = np.array([len(r.genoma) for r in pob], np.int64)
    trazas_largo = np.array([len(r.traza) for r in pob], np.int64)
    trazas = [r.traza if r.traza.dtype == TRAZA_DTYPE else compactar(r.traza) for r in pob]
    ruta = carpeta / f"gen_{gen:04d}.npz"
    # Se escribe aparte y se renombra: si se detiene el entrenamiento a mitad, no queda un .npz roto
    tmp = ruta.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            fitness=np.array([r.fitness for r in pob]),
            murio=np.array([r.murio for r in pob]),
            frames=np.array([r.frames for r in pob]),
            llego=np.array([r.llego for r in pob]),
            pantallas=np.array([r.pantallas for r in pob]),
            celdas=np.array([r.celdas for r in pob]),
            frame_progreso=np.array([r.frame_progreso for r in pob]),
            progreso=np.array([r.progreso for r in pob]),
            progreso_max=np.array([r.progreso_max for r in pob]),
            origen=np.array([r.origen for r in pob]),
            padres=np.array([",".join(map(str, r.padres)) for r in pob]),
            genoma_largo=largos,
            genoma_acciones=np.concatenate([r.genoma.acciones for r in pob]),
            genoma_frames=np.concatenate([r.genoma.frames for r in pob]),
            traza_largo=trazas_largo,
            trazas=np.concatenate(trazas) if trazas else np.zeros(0, TRAZA_DTYPE),
        )
    tmp.replace(ruta)
    return ruta


def cargar_generacion(carpeta: Path, gen: int) -> list[Resultado]:
    # Cada acceso a un npz descomprime el arreglo entero: se lee cada uno una sola vez
    with np.load(carpeta / f"gen_{gen:04d}.npz") as z:
        d = {k: z[k] for k in z.files}
    pob = []
    g_fin = np.cumsum(d["genoma_largo"])
    t_fin = np.cumsum(d["traza_largo"])
    for i in range(len(d["fitness"])):
        g0 = g_fin[i] - d["genoma_largo"][i]
        t0 = t_fin[i] - d["traza_largo"][i]
        padres = tuple(int(p) for p in str(d["padres"][i]).split(",") if p)
        pob.append(Resultado(
            Genoma(d["genoma_acciones"][g0:g_fin[i]].copy(), d["genoma_frames"][g0:g_fin[i]].copy()),
            float(d["fitness"][i]), bool(d["murio"][i]), int(d["frames"][i]), int(d["llego"][i]),
            int(d["pantallas"][i]), int(d["celdas"][i]), int(d["frame_progreso"][i]),
            d["trazas"][t0:t_fin[i]].copy(), padres, str(d["origen"][i]),
            float(d["progreso"][i]) if "progreso" in d else 0.0,
            float(d["progreso_max"][i]) if "progreso_max" in d else 0.0,
        ))
    return pob


def generaciones_guardadas(carpeta: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in carpeta.glob("gen_*.npz"))


def guardar_config(carpeta: Path, cfg: Config) -> None:
    (carpeta / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def cargar_config(carpeta: Path) -> Config:
    d = json.loads((carpeta / "config.json").read_text())
    d.setdefault("zonas_prohibidas", "")  # corridas anteriores a las zonas prohibidas: sin zonas
    return Config(**d)


def agregar_metricas(carpeta: Path, fila: dict) -> None:
    ruta = carpeta / "metricas.csv"
    nuevo = not ruta.exists()
    # Una corrida vieja conserva sus columnas
    campos = CAMPOS_METRICAS if nuevo else ruta.open().readline().strip().split(",")
    with ruta.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        w.writerow({k: fila.get(k, "") for k in campos})


def leer_metricas(carpeta: Path) -> list[dict]:
    with (carpeta / "metricas.csv").open() as f:
        return [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]
