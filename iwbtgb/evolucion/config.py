"""Configuración común de una corrida de neuroevolución (se guarda en `config.json`; cada campo es una opción de
`entrenar`). Cada línea (`redes`, `rayos`) agrega lo que ve su red y la forma de la red, y puede cambiar los valores
por defecto."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar

from ..distancia import ZONAS_AUTO, rectangulos


@dataclass
class ConfigBase:
    nombre: str = ""
    # -- juego
    difficulty: str = "medium"
    start_room: str = "rStage01"
    sala_objetivo: str = "rStage02"
    zonas_prohibidas: str = "auto"      # "x0,y0,x1,y1;..." o "auto" (`distancia.ZONAS_AUTO`)
    seed_juego: int = 0
    # -- dónde practica: "gimnasio" (salas generadas, nuevas cada generación) o "nivel" (start_room). En el
    #    gimnasio, `escenarios` = salas por generación y `max_segundos` = tope por sala.
    entorno: str = "gimnasio"
    gim_dificultad_min: float = 0.0     # cada sala con una dificultad al azar entre estas dos
    gim_dificultad_max: float = 0.3
    gim_salida_borde: int = 1           # la salida es una abertura en la pared derecha; 0 = un portal a 2 celdas
    gim_trampas: int = 0                # púas que saltan (desde dificultad 0,2), caen o cerezas (0,3), laterales (0,4)
    gim_trampas_min: int = 0            # con trampas: al menos tantas en cada sala, a cualquier dificultad
    # -- currículo: cuando la población supera bastantes salas, el gimnasio se pone más difícil solo
    gim_curriculo: int = 1
    curriculo_umbral: float = 50.0      # % de salas superadas (promedio de la población) en las últimas…
    curriculo_gens: int = 10            # …generaciones para subir la dificultad máxima
    curriculo_paso: float = 0.1         # cuánto sube (hasta 1)
    curriculo_trampas: float = 0.5      # con la dificultad máxima desde esta, se activan las trampas (2 = nunca)
    # -- partir de la mejor red de otra corrida (misma forma de red y de lo que ve, que se copian de ella)
    desde: str = ""
    desde_gen: int = -1                 # su generación (-1 = la última)
    # -- decisión
    cuadros: int = 2                    # observaciones apiladas: la actual y las de decisiones anteriores
    frames_por_decision: int = 2        # la red decide cada tantos frames y mantiene las teclas entre medio
    soltar_salto: int = 1               # el salto sale al apretar: si la red sigue pidiendo salto y el jugador ya no
                                        # sube, se suelta la tecla un frame (`gimnasio.plan_teclas`)
    # -- evaluación: cada red juega varios escenarios y su fitness es el promedio (para que no memorice)
    escenarios: int = 8                 # en el nivel, la primera partida desde el comienzo y las demás con una
                                        # espera al azar; en el gimnasio, salas (nuevas cada generación)
    espera_max: int = 60                # frames sin control al inicio (desfasa plataformas y trampas)
    max_segundos: float = 15.0          # tope por partida (escenario o sala)
    paciencia: float = 8.0              # s sin mejorar el avance: termina el escenario
    # -- recompensa (`episodio.desglose`)
    r_avance: float = 300.0             # × mejor fracción del camino a la salida alcanzada
    r_objetivo: float = 200.0           # por llegar a la salida
    r_rapidez: float = 1.0              # − r_rapidez × (frame del mejor avance / tope): a igual avance, el más rápido
    castigo_muerte: float = 1.0         # − por cada muerte (o salida, o atasco con reintentos)
    r_save: float = 10.0                # en el nivel: por cada save point nuevo activado (salvo los del comienzo)
    castigo_quieto: float = 2.0         # en el nivel: − por cada segundo sin desplazarse
    r_explorar: float = 0.5             # en el nivel: por cada celda distinta de la sala por la que pasó vivo
    celda_explorar: int = 32            # lado de esas celdas (px)
    # -- reintentos (en el nivel): al morir aprieta R (vuelve al último save) y sigue hasta que se acaba el tiempo
    reintentos: int = 1                 # 0 = la partida termina al morir, salir o estancarse
    frames_reset: int = 25              # lo que tarda en apretar R después de morir (0,5 s)
    espera_reintento: int = 30          # tras reaparecer, espera al azar sin control (0 a esto, frames): sin ella una
                                        # red sin memoria repite exacta la vida anterior
    # -- evolución (Deep GA: sin cruce, mutación gaussiana de todos los pesos)
    poblacion: int = 200
    padres: int = 50                    # los mejores, priorizando uno por nicho
    elite: int = 10                     # pasan sin cambios (y se vuelven a evaluar con los escenarios nuevos)
    elite_mas_lejos: int = 1            # la que llega más lejos pasa siempre como élite (y es padre)
    sigma: float = 0.025                # desviación de la mutación
    nicho_px: int = 64                  # nicho = cuadro donde terminó el primer escenario
    generaciones: int = 1000
    workers: int = 12
    seed: int = 1
    guardar_red_cada: int = 10          # pesos completos de la mejor cada tantas generaciones (además de la última)
    # -- resuelto al crear la corrida
    room_entrenamiento: int = -1
    room_objetivo: int = -1
    room_ancho: int = 0
    room_alto: int = 0
    genoma_inicial: list = field(default_factory=list)   # el genoma de `desde`

    # Valores para las corridas guardadas antes de que existiera cada campo: así siguen con las reglas de entonces
    ANTERIORES: ClassVar[dict] = {}

    @property
    def max_frames(self) -> int:
        return int(self.max_segundos * 50)

    def zonas(self) -> tuple:
        texto = ZONAS_AUTO.get(self.start_room, "") if self.zonas_prohibidas == "auto" else self.zonas_prohibidas
        return rectangulos(texto)

    def problema(self) -> str | None:
        """Qué tiene de inválido esta config (None si nada)."""
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        nombres = {f.name for f in fields(cls)}
        d = {**cls.ANTERIORES, **d}
        if d.get("entorno") == "gimnasio":  # las primeras del gimnasio tenían sus propios campos
            d["escenarios"] = d.pop("gim_salas", d.get("escenarios"))
            d["max_segundos"] = d.pop("gim_segundos", d.get("max_segundos"))
        return cls(**{k: v for k, v in d.items() if k in nombres})
