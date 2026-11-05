"""Guardado y carga de generaciones: `corridas/<nombre>/gen_NNNN.npz` + `metricas.csv` + `config.json`."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..rutas import CORRIDAS_GA as CORRIDAS
from .nucleo import TRAZA_DTYPE, Config, Genoma, Inicio, Resultado, compactar

CAMPOS_METRICAS = [
    "gen", "fitness_max", "fitness_media", "fitness_mediana", "fitness_min",
    "pantallas_max", "pantallas_media", "llegaron", "celdas_max", "celdas_media",
    "muertes_pct", "segundos_medio", "segundos_mejor", "evaluados", "segundos_reloj",
    "progreso_max", "progreso_final_max", "progreso_media", "nichos",
    "saves_max", "saves_media", "reintentos_media", "desde_save_pct",
]


# Save heredado y saves activados: JSON por agente (los archivos de partida en hexadecimal)
def _archivos_json(archivos: tuple) -> list:
    return [[nombre, datos.hex()] for nombre, datos in archivos]


def _archivos_de_json(lista: list) -> tuple:
    return tuple((nombre, bytes.fromhex(h)) for nombre, h in lista)


def inicio_a_json(inicio: Inicio | None) -> str:
    if inicio is None:
        return ""
    return json.dumps([inicio.save, _archivos_json(inicio.archivos), list(inicio.saves),
                       [list(p) for p in inicio.pantallas], inicio.gen, inicio.agente, inicio.frame])


def inicio_de_json(texto: str) -> Inicio | None:
    if not texto:
        return None
    save, archivos, saves, pantallas, gen, agente, frame = json.loads(texto)
    return Inicio(save, _archivos_de_json(archivos), tuple(saves), tuple(tuple(p) for p in pantallas), gen, agente,
                  frame)


def nuevos_a_json(nuevos: tuple) -> str:
    return json.dumps([[k, frame, g, _archivos_json(a)] for k, frame, g, a in nuevos]) if nuevos else ""


def nuevos_de_json(texto: str) -> tuple:
    return tuple((k, frame, g, _archivos_de_json(a)) for k, frame, g, a in json.loads(texto)) if texto else ()


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
            muertes=np.array([r.muertes for r in pob]),
            saves=np.array([r.saves for r in pob]),
            inicio=np.array([inicio_a_json(r.inicio) for r in pob]),
            nuevos=np.array([nuevos_a_json(r.nuevos) for r in pob]),
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
        genoma = Genoma(d["genoma_acciones"][g0:g_fin[i]].copy(), d["genoma_frames"][g0:g_fin[i]].copy())
        traza = d["trazas"][t0:t_fin[i]].copy()
        pob.append(Resultado(
            genoma, float(d["fitness"][i]), bool(d["murio"][i]), int(d["frames"][i]), int(d["llego"][i]),
            int(d["pantallas"][i]), int(d["celdas"][i]), int(d["frame_progreso"][i]),
            traza, padres, str(d["origen"][i]),
            float(d["progreso"][i]) if "progreso" in d else 0.0,
            float(d["progreso_max"][i]) if "progreso_max" in d else 0.0,
            int(d["muertes"][i]) if "muertes" in d else 0,
            int(d["saves"][i]) if "saves" in d else 0,
            inicio_de_json(str(d["inicio"][i])) if "inicio" in d else None,
            nuevos_de_json(str(d["nuevos"][i])) if "nuevos" in d else (),
        ))
    return pob


def cadena(carpeta: Path, r: Resultado) -> list[tuple[int | None, Resultado, int | None]]:
    """Tramos del recorrido completo de `r`: [(generación, agente, frame donde termina su tramo)], desde el
    ancestro que partió del comienzo hasta `r` (último tramo, sin corte). Cada tramo termina al activar el save
    desde el que parte el siguiente."""
    tramos = [(None, r, None)]
    while r.inicio is not None:
        ini = r.inicio
        r = cargar_generacion(carpeta, ini.gen)[ini.agente]
        tramos.insert(0, (ini.gen, r, ini.frame))
    return tramos


def generaciones_guardadas(carpeta: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in carpeta.glob("gen_*.npz"))


def guardar_config(carpeta: Path, cfg: Config) -> None:
    (carpeta / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def cargar_config(carpeta: Path) -> Config:
    d = json.loads((carpeta / "config.json").read_text())
    # Corridas anteriores a cada regla nueva: siguen con sus condiciones
    d.setdefault("zonas_prohibidas", "")
    d.setdefault("resets", 0)
    d.setdefault("save_points", "")
    d.setdefault("prob_desde_save", 0.0)
    nombres = {f.name for f in fields(Config)}   # sin los campos de reglas que ya no existen
    return Config(**{k: v for k, v in d.items() if k in nombres})


def agregar_metricas(carpeta: Path, fila: dict, campos_nuevos: list[str] = CAMPOS_METRICAS) -> None:
    ruta = carpeta / "metricas.csv"
    nuevo = not ruta.exists()
    # Una corrida vieja conserva sus columnas
    campos = campos_nuevos if nuevo else ruta.open().readline().strip().split(",")
    with ruta.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        w.writerow({k: fila.get(k, "") for k in campos})


def leer_metricas(carpeta: Path) -> list[dict]:
    with (carpeta / "metricas.csv").open() as f:
        return [{k: float(v) for k, v in fila.items() if v != ""} for fila in csv.DictReader(f)]
