"""Workers de multiprocessing con un emulador cada uno."""

from __future__ import annotations

import multiprocessing.util
import os
import signal

_recurso = None


def cerrar_al_salir(recurso) -> None:
    """Cierra `recurso.close()` (su emulador) cuando el worker sale normal (el centinela del pool) o con SIGTERM
    (`Pool.terminate`). Sin esto la carpeta del emulador queda en /tmp, que está en RAM."""
    global _recurso
    _recurso = recurso
    multiprocessing.util.Finalize(None, _cerrar, exitpriority=10)
    signal.signal(signal.SIGTERM, _sigterm)


def _cerrar() -> None:
    global _recurso
    if _recurso is not None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # el pool suele mandarlo mientras se cierra: que termine
        r, _recurso = _recurso, None
        r.close()


def _sigterm(*_) -> None:
    _cerrar()
    os._exit(0)


def un_hilo_por_worker() -> None:
    """Un hilo de BLAS por worker (los workers heredan estas variables al crearse): si no, numpy abre uno por núcleo
    en cada worker, la máquina queda sobrecargada y algunos workers se traban al cerrar."""
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
