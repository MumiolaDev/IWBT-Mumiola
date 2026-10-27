"""Configuración de una corrida de redes (se guarda en `config.json`; cada campo es una opción de `entrenar`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .vista import ConfigVista

# Rectángulos de la sala donde entrar cuenta como muerte (como en el GA): la sala de portales de rStage01
ZONAS_AUTO = {"rStage01": "800,0,1600,608"}

# Valores para las corridas guardadas antes de que existiera cada campo (así siguen con las reglas de entonces)
ANTERIORES = {"castigo_muerte": 0.0, "r_save": 0.0, "reintentos": 0, "castigo_quieto": 0.0, "elite_mas_lejos": 0,
              "r_explorar": 0.0, "entorno": "nivel", "desde": "", "soltar_salto": 0}


@dataclass
class Config:
    nombre: str = "red01"
    # -- juego
    difficulty: str = "medium"
    start_room: str = "rStage01"
    sala_objetivo: str = "rStage02"
    zonas_prohibidas: str = "auto"      # "x0,y0,x1,y1;..." o "auto"
    seed_juego: int = 0
    # -- dónde entrena: "nivel" (start_room) o "gimnasio" (salas generadas, nuevas cada generación; ver gimnasio.py).
    #    En el gimnasio, `escenarios` = salas por generación y `max_segundos` = tope por sala.
    entorno: str = "nivel"
    gim_dificultad_min: float = 0.0     # cada sala con una dificultad al azar entre estas dos
    gim_dificultad_max: float = 0.5
    # -- partir de una red ya entrenada (p. ej. en el gimnasio) en vez de pesos al azar: la población inicial son
    #    mutaciones de la mejor red de esa corrida (misma forma de red y vista, que se copian de ella)
    desde: str = ""
    desde_gen: int = -1                 # su generación (-1 = la última)
    # -- red (CNN chica sobre la grilla + variables del jugador)
    cuadros: int = 2                    # grillas apiladas: la actual y la de la decisión anterior (para ver movimiento)
    filtros1: int = 8                   # conv 4×4, paso 4
    filtros2: int = 16                  # conv 3×3, paso 2
    oculta: int = 64                    # capa densa
    frames_por_decision: int = 2        # la red decide cada tantos frames y mantiene las teclas entre medio
    soltar_salto: int = 1               # el salto sale al apretar, no al mantener: si la red sigue pidiendo salto y el
                                        # jugador ya no sube, se suelta la tecla un frame (para que pueda volver a
                                        # saltar o hacer el doble salto); 0 = la red tiene que soltarla ella
    # -- evaluación: cada red juega varios escenarios y su fitness es el promedio (para que no memorice)
    escenarios: int = 3                 # partidas por red: en el nivel, la primera desde el comienzo y las demás
                                        # con una espera al azar; en el gimnasio, salas (nuevas cada generación)
    espera_max: int = 60                # frames sin control al inicio (desfasa plataformas y trampas)
    max_segundos: float = 30.0          # tope por partida (escenario o sala)
    paciencia: float = 8.0              # s sin mejorar el avance: termina el escenario
    # -- recompensa
    r_avance: float = 100.0             # × mejor fracción del camino a la salida (Dijkstra) alcanzada
    r_objetivo: float = 200.0           # por llegar a la sala objetivo
    r_rapidez: float = 1.0              # − r_rapidez × (frame del mejor avance / tope): a igual avance, el más rápido
    castigo_muerte: float = 3.0         # − por cada muerte (o salida, o atasco con reintentos)
    r_save: float = 10.0                # por cada save point nuevo activado (salvo los del comienzo)
    castigo_quieto: float = 2.0         # − por cada segundo quieto (sin desplazarse; en una plataforma móvil no lo
                                        # está): más que morir, para que esperar el fin del tiempo no convenga
    r_explorar: float = 0.5             # por cada celda distinta de la sala por la que pasó vivo en la partida (para
                                        # que probar otro camino valga aunque todavía no acerque a la salida)
    celda_explorar: int = 32            # lado de esas celdas (px)
    # -- reintentos: al morir aprieta R (vuelve al último save) y sigue hasta que se acaba el tiempo
    reintentos: int = 1                 # 1 = sí; 0 = la partida termina al morir, salir o estancarse
    frames_reset: int = 25              # lo que tarda en apretar R después de morir (0,5 s)
    espera_reintento: int = 30          # tras reaparecer, espera al azar sin control (0 a esto, frames): sin
                                        # ella la red, que no tiene memoria, repite la vida anterior igual
    # -- evolución (Deep GA: sin cruce, mutación gaussiana de todos los pesos)
    poblacion: int = 200
    padres: int = 20                    # los mejores, priorizando uno por nicho
    elite: int = 5                      # pasan sin cambios (y se vuelven a evaluar con los escenarios nuevos)
    elite_mas_lejos: int = 1            # 1 = la que llega más lejos pasa siempre como élite (y es padre), aunque su
                                        # fitness no esté entre los mejores
    sigma: float = 0.02                 # desviación de la mutación
    nicho_px: int = 64                  # nicho = cuadro donde terminó el primer escenario
    generaciones: int = 1000
    workers: int = 10
    seed: int = 1
    guardar_red_cada: int = 10          # pesos completos del mejor cada tantas generaciones (además de la última)
    # -- vista (dict de ConfigVista)
    vista: dict = field(default_factory=lambda: ConfigVista().to_dict())
    # -- resuelto al crear la corrida
    room_entrenamiento: int = -1
    room_objetivo: int = -1
    room_ancho: int = 0
    room_alto: int = 0
    genoma_inicial: list = field(default_factory=list)   # el genoma de `desde` (linaje de semillas)

    @property
    def config_vista(self) -> ConfigVista:
        return ConfigVista.from_dict(self.vista)

    @property
    def max_frames(self) -> int:
        return int(self.max_segundos * 50)

    def zonas(self) -> list[tuple[float, float, float, float]]:
        texto = ZONAS_AUTO.get(self.start_room, "") if self.zonas_prohibidas == "auto" else self.zonas_prohibidas
        return [tuple(float(v) for v in z.split(",")) for z in texto.split(";") if z.strip()]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        nombres = {f.name for f in fields(cls)}
        d = {**ANTERIORES, **d}  # corridas creadas antes de cada regla: la regla apagada
        if d.get("entorno") == "gimnasio":  # las primeras del gimnasio tenían sus propios campos (2026-10-27)
            d["escenarios"] = d.pop("gim_salas", d.get("escenarios"))
            d["max_segundos"] = d.pop("gim_segundos", d.get("max_segundos"))
        return cls(**{k: v for k, v in d.items() if k in nombres})
