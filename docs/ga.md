# Algoritmo genético con TAS y Go-Explore (`iwbtgb/ga/`)

Para el video blog de YouTube. Un agente es una secuencia fija de teclas (un TAS) para un nivel: el juego es
determinista desde el savestate inicial, así que el genoma reproduce siempre la misma partida. No generaliza a otros
niveles, pero aprovecha los savestates para buscar rápido. Corridas en `corridas/` (enlace a
`/mnt/datos/iwbtgb/corridas`). En pausa desde el 2026-10-26 (`GA_beta_corregida` detenida en la gen 1637, en ~47 % del
camino).

```bash
uv run python -m iwbtgb.ga entrenar --nombre salida01 [--continuar] [--generaciones 500]   # opciones = campos de Config
uv run python -m iwbtgb.ga visualizar todo --corrida salida01 --cada 10    # videos y gráficos de la corrida
uv run python -m iwbtgb.ga visualizar vivo --corrida salida01              # ventana con la última generación
uv run python -m iwbtgb.ga explorar --nombre ge01 --workers 4 --iteraciones 2000   # Go-Explore; --continuar
uv run python -m iwbtgb.ga app                                             # interfaz (en el menú: iwbtgb-ga.desktop)
```

Cada campo de `Config` (`nucleo.py`) es una opción (`max_segundos` → `--max-segundos`; los booleanos aceptan `1/0`,
`sí/no`, `true/false`). La config usada queda en `corridas/<nombre>/config.json` y `--continuar` la respeta.

| Archivo | Rol |
|---|---|
| `nucleo.py` | `Config`, genoma, evaluación por familias, fitness, cruce y mutación, save heredado |
| `entrenar.py` | Bucle de generaciones; guarda `gen_NNNN.npz` (genomas + trazas por frame) y `metricas.csv` |
| `explorar.py` | Go-Explore |
| `almacen.py` | Formato en disco |
| `recursos.py` | Mapa completo de la room (sin jugador ni `camera`) y sprites del jugador extraídos del .exe |
| `visualizar.py` | Videos 1920×1080: `generacion` (todos juntos), `mejor` (partida real, recorrido completo por tramos), `evolucion`, `graficos`, `vivo` |
| `app.py` | Interfaz Qt: lista de corridas, entrenar/continuar/detener, visualizaciones, borrar corridas |

## El algoritmo

- **Genoma**: genes `(acción, frames)`: mantener una de las 12 acciones durante 1..`max_frames_gen` frames. Si el
  agente sobrevive más de lo que cubre su genoma, se le agregan genes al azar (con su semilla) y quedan guardados.
- **Evaluación por familias**: los hijos de un padre parten del savestate del padre en el gen donde se separan
  (`MAX_SLOTS` = 6 por familia). Da lo mismo que jugar desde cero (verificado).
- **Fitness** (`nucleo.puntuar`): `r_cercania` (1000) × fracción del camino recorrida al final + `r_avance` (300) ×
  mejor fracción + `r_pantalla` (300) por pantalla del camino a la salida + `r_save` (150) por save point nuevo −
  `castigo_reset` (10) por reintento − `castigo_muerte` (50) al quedar descalificado; + `r_objetivo` y `r_tiempo` por
  segundo sobrante si llega. Solo cuenta lo que pasa en la room de entrenamiento.
- **Distancia a la salida** (`iwbtgb/distancia.py`): al crear la corrida se mide la room en una grilla de 16 px con la
  máscara del jugador (`place_meeting` contra `block`, sin las plataformas, que se mueven; `playerKiller` cuesta 8
  veces más) y se calcula con Dijkstra la distancia por el laberinto hasta el `roomChanger` cuyo `roomTo` es la sala
  objetivo. Queda en `distancia.npz` (y `graficos/mapa_distancia.png`). Las "pantallas del camino" son las que cruza
  el camino más corto.
- **Selección por nichos**: dos agentes que terminan en el mismo cuadro de 64 px son del mismo nicho; los `padres`
  mejores de nichos distintos pasan como élite y se reparten los hijos. Un hijo que repetiría exactamente una partida
  ya vista se vuelve a mutar.
- **Mutación**: 50 % ensayo y error en una ventana antes de una de sus muertes o del mejor momento, 30 % en un
  momento al azar, 20 % rama nueva con teclas al azar; más un cruce en un punto del tiempo con otro padre que parta del
  mismo lugar (`prob_cruce`).
- **Zonas prohibidas** (`zonas_prohibidas`, "auto" = `distancia.ZONAS_AUTO`): en rStage01, la pantalla de arriba a la
  derecha (800,0)–(1600,608), la sala de portales que abren los jefes; entrar cuenta como morir, igual que irse a otra
  room (`salir_es_muerte`; el hueco del techo en x≈183 lleva a rStage07). Al crear la corrida "auto" se reemplaza por
  los rectángulos.
- **Reintentos** (`resets`): al morir, entrar a una zona prohibida o irse de la sala, el agente espera `frames_reset`
  (25 frames), aprieta R y el juego lo devuelve a su último save, como a un jugador. La muerte siguiente a los
  reintentos lo descalifica.
- **Save points** (`save_points` "auto" = los `savePoint` de la sala en la dificultad, salvo los de zonas
  prohibidas; en rStage01 medium, 6). Un save se detecta porque cambian los primeros 9 bytes (sala, x, y) de
  `save/saveN`; cuál fue, por el `savePoint` con `image_index >= 1`.
- **Save heredado** (`prob_desde_save`): cada hijo de un padre que activó un save point nuevo parte, con esa
  probabilidad, desde ese save (el más cercano a la salida), con el tiempo y los reintentos completos. `Inicio` guarda
  el save, los archivos de `save/`, los saves y pantallas ya logrados por los ancestros y quién lo activó. Los hijos de
  un agente que parte de un save parten del mismo save, así que bajar `prob_desde_save` no devuelve a nadie al
  comienzo. `visualizar mejor` muestra el recorrido completo tramo por tramo (`almacen.cadena`); en cada unión hay un
  salto de 20–40 px (reaparece en el save, sin velocidad).
- **Compatibilidad**: `almacen.cargar_config` da a las corridas viejas las reglas que no tenían (`setdefault`: sin
  zonas, sin reintentos, sin save points ni save heredado) e ignora campos que ya no existen (las reglas del
  experimento 1: recompensa por exploración y selección por torneo, que se quitaron el 2026-11-05).
- **Borrar corridas**: botón "Borrar corrida…" en la app (pide confirmación y no deja borrar una en uso), o
  `rm -r corridas/<nombre>`.

`visualizar vivo` aguanta horas (stream): corre en un scope de systemd con `MemoryMax=1500M` sin swap, ffplay muere
con él (`-autoexit` + `PR_SET_PDEATHSIG`) y se reinicia si pasa de 800 MB, y cada 10 min escribe una línea con la
memoria. La app lanza cada comando en su propio grupo de procesos: "Detener" manda Ctrl+C (deja guardadas las
generaciones completas) y a los 10 s mata el grupo, emuladores y ffplay incluidos.

## Go-Explore (`explorar.py`)

En vez de una población, un archivo de celdas. Celda = cuadro de `celda_px` (16) px + signo de la velocidad vertical +
si queda doble salto (`celda_salto`) + posición relativa de la plataforma móvil más cercana en cuadros de 48 px dentro
de `radio_plataforma` (`celda_plataforma`; el recorrido de las plataformas se precalcula con `frames_plataformas`).
Para cada celda se guarda el camino más corto (en frames) con que se llegó, como cadena de `Nodo`s inmutables.

- Cada iteración elige `workers × tareas_por_worker` celdas con peso
  `(1/√(1+elegida) + peso_vista/√(1+vista)) × e^(beta·(progreso − mejor progreso))` y desde cada una prueba
  `exploraciones` ráfagas de teclas al azar de hasta `frames_exploracion` (100) frames.
- Volver a una celda: cada worker guarda `cache` (16) savestates (LRU); si no tiene la celda, carga el ancestro más
  cercano que tenga y repite los genes. El principal lleva una copia exacta de cada caché y manda cada celda al worker
  que menos frames tiene que repetir; cada tarea guarda además una "escala" (un ancestro ≥150 frames antes).
- Sin reintentos ni save points. Verificado: los genes de una celda jugados desde el comienzo reproducen su traza.
- Disco: `config.json` (una `Config` con `seleccion = "archivo"`, para que la app y `visualizar` la lean),
  `explorar.json`, `archivo.pkl`, `metricas.csv` y, cada `guardar_cada` iteraciones, `gen_NNNN.npz` con las `muestra`
  celdas más avanzadas y `graficos/mapa_archivo.png`. `entrenar` se niega a continuar una corrida con `explorar.json`.
- Con el emulador corregido llegó al 48,7 % en 4000 iteraciones: no logra montarse en la plataforma móvil (288–608,
  y≈1120) para subir al save de (672,1056).

## Estado de las corridas (2026-10-26)

Todas en rStage01 → rStage02, medium, selección por nichos. "Camino" = mejor fracción del camino a la salida.

**El muro del ~20,5 % era un error del emulador**: la máscara de las púas con fila de relleno era el rectángulo entero
(ver `docs/emulador.md`, «Color clave»). Corregido el 2026-10-25. Las corridas hasta `200A_30P` usan el emulador
viejo: para repetirlas (videos) hay que usar el binario respaldado
(`OPENGMK_EMULATOR=/mnt/datos/iwbtgb/emulador_antiguo/gm8emulator`) o `OPENGMK_GM7_CLAVE_ESQUINA=1`.

| Corrida | Agentes / padres | Intento | Reintentos | Save heredado | Gens | Camino máx | Notas |
|---|---|---|---|---|---|---|---|
| `150AStage01_00` | 150 / 25 | 120 s | 2 | 30 % | 690 | 19,9 % | Estancada en 18,6 % desde la gen 318; con reintentos y save heredado subió a 19,9 % |
| `300A_Stage01_01` | 300 / 10 | 120 s | 0 | 0 % (al crearla, >0) | 5000 | 20,5 % | 100 % parte de un save desde la gen 5; sin avance desde la ~750; desde la ~2500 los nichos bajaron a 10 |
| `100A_Stage01_01` | 100 / 30 | 120 s | 0 | 0 % | 716 | 9,9 % | Estancada desde la gen 200 |
| `50A_Stage01_01` | 50 / 10 | 120 s | 0 | 100 % | 501 | 9,8 % | |
| `200A_30P` | 200 / 30 | 90 s | 1 | 30 % | 1736 | 20,0 % | Detenida al encontrar el error del emulador |
| `_prueba_ga_alfa` | 200 / 30 | 90 s | 0 | 30 % | 2000 | 75,7 % | Emulador corregido: 49,8 % en la gen 1800, 75,7 % en la 1999, 5 saves. El mejor final juega 12 frames desde el save 5 y muere: casi todo su avance es heredado |
| `GA_beta_corregida` | 200 / 50 | 180 s | 0 | 0 % | 1637 | ~47 % | Emulador corregido, partida continua (hipótesis del usuario: sin herederos también llega). Detenida el 2026-10-26 |
| `test_sala` | 10 / 3 | 60 s | 0 | 100 % | 300 | 3,8 % | Prueba del usuario |

Los experimentos 1 y 2 (corridas `rStage01` y `salida01`, ya borradas) están en el informe
(https://claude.ai/artifact/XBoYLppCkJP2CkYwzmkP8E; no incluye reintentos ni save heredado).

Lo que se sabe:

- Con pocos padres (10) la diversidad se pierde: los nichos quedan iguales al número de padres.
- Con `prob_desde_save = 0` y sin reintentos, el video del mejor es una sola partida continua.
- Disco: ~0,6 GB por cada ~700 generaciones de 150 agentes con intentos de 120 s.

Ideas sin probar: que no todos los hijos de un agente que parte de un save hereden ese save (una fracción vuelve al
save del abuelo o al comienzo); nichos que incluyan el save de inicio; re-evaluar la élite al cambiar las reglas a
mitad de una corrida (hoy solo se re-evalúan los agentes cuyos saves no se conocen); para Go-Explore, la fase de la
plataforma móvil en la celda.
