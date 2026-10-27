# Entorno de aprendizaje reforzado

> **Estado (2026-10-23):** hay un algoritmo genético sobre secuencias de entradas en `ga/` (ver "Algoritmo
> genético" al final). Con PPO u otro RL por gradiente todavía no se ha entrenado nada.

## Piezas

| Capa | Archivo | Rol |
|---|---|---|
| Servidor en el emulador | `vendor/OpenGMK/gm8emulator/src/game/rl.rs` | `--rl-socket RUTA [--rl-seed N]`: corre sin ventana ni audio y responde a un cliente por socket Unix |
| Cliente Python | `iwbtgb_rl/client.py` (`Emulator`) | Lanza el emulador sobre una copia del `.exe` en un directorio temporal y habla el protocolo |
| Entorno Gymnasium | `iwbtgb_rl/env.py` (`IWannaBeTheGBEnv`, id `IWannaBeTheGB-v0`) | Arranque, savestate inicial, acciones, observaciones, recompensa |
| Ejemplos | `examples/random_agent.py`, `parallel_envs.py`, `ver_agente.py` | Prueba, paralelismo y visualización (ventana en vivo o video `.mp4`) |

## Uso

```python
import gymnasium as gym
import iwbtgb_rl  # registra IWannaBeTheGB-v0

env = gym.make("IWannaBeTheGB-v0", obs_type="gray", frameskip=4)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()  # cierra el proceso del emulador
```

```bash
uv run python examples/random_agent.py
N_ENVS=8 uv run python examples/parallel_envs.py
uv run python examples/ver_agente.py                             # ventana, velocidad real
uv run python examples/ver_agente.py --video videos/x.mp4 --episodios 5
```

Para ver a un agente propio en `ver_agente.py`, reemplaza `politica()`. Si entrenaste con `VecFrameStack`, hay
que agregar al script el apilamiento de los últimos frames.

### Arranque y episodios

Al crearse, el entorno avanza por el título y el menú pulsando Shift, llega a `rSelectStage` (la sala de
dificultades) y teletransporta al jugador sobre el portal `warpStart` de la dificultad elegida. Las
posiciones de los portales: medium (96, 384), hard (672, 384), veryhard (64, 224) e impossible (704, 224).
Cuando llega a `rStage01`, guarda el slot 0, y cada `reset()` lo carga.

Con `start_room`, después de entrar al nivel hace `room_goto` y mueve al jugador a
`playerStart + (17, 23)`. Eso es necesario porque `player` es persistente y conserva las coordenadas de la
room anterior, que pueden quedar dentro de un muro.

El episodio termina cuando muere el jugador: su instancia desaparece y aparecen `GAMEOVER` y
`PlayerDeathParts`.

### Acciones (`Discrete(12)`)

`0` nada, `1` ←, `2` →, `3` Shift, `4` ←+Shift, `5` →+Shift, `6` Z, `7` ←+Z, `8` →+Z, `9` Shift+Z,
`10` ←+Shift+Z y `11` →+Shift+Z. La altura del salto depende de cuántos pasos se mantenga Shift, y volver a
presionarlo en el aire hace el doble salto.

### Parámetros

| Parámetro | Defecto | Nota |
|---|---|---|
| `difficulty` | `"medium"` | medium, hard, veryhard o impossible |
| `start_room` | `None` | p. ej. `"rStage05"` o `"rBoss01"` |
| `frameskip` | 4 | el juego corre a 50 FPS |
| `skip_draw` | False | dibuja solo el último frame del paso (~25 % más rápido; los eventos Draw intermedios no corren) |
| `obs_type` | `"rgb"` | `rgb` (150×200×3), `gray` (150×200×1) o `state` (`[x, y, hspeed, vspeed, room]`) |
| `downscale` | 4 | sobre los 800×600 originales |
| `max_episode_steps` | 5000 | truncado |
| `noop_reset_max` | 0 | frames aleatorios sin acción al hacer reset (para que no memorice) |
| `cell_size` / `cell_reward` | 32 / 0.1 | recompensa por celda nueva visitada en el episodio |
| `room_reward` | 10.0 | por room nueva |
| `death_penalty` | 1.0 | |
| `seed` | 0 | semilla del RNG del juego |
| `show_window` | False | variable de entorno `OPENGMK_RL_SHOW=1` |

Para otra recompensa, hereda y sobreescribe `compute_reward(state, dead)`. `state` es `None` si el jugador
murió y, si no, un `dict` con `x`, `y`, `hspeed`, `vspeed` y `room`.

## Protocolo del socket

Está documentado en la cabecera de `rl.rs`. Todo es little-endian y cada petición empieza con 1 byte de
opcode.

| Op | Petición | Respuesta |
|---|---|---|
| 1 STEP | `u8 n, u8 vk[n], u32 frames, u8 flags` (bit 0 = obs, bit 1 = skip_draw) | `u8 status [, str error], i32 room, u64 frame [, obs]` |
| 2 SAVE / 3 LOAD | `u32 slot` | `u8 status` |
| 4 OBS | — | `obs` = `u32 w, u32 h, RGB w*h*3` (fila superior primero) |
| 5 QUERY | `u16 len, nombre del objeto` | `u32 n, (i32 id, f64 x, y, hspeed, vspeed)×n` (incluye objetos hijos) |
| 6 INFO | — | `i32 room, u32 w, h, speed, u16 len, nombre, u64 frame` |
| 7 EXEC | `u32 len, GML` | `u8 status`, y luego `u8 0 + f64` o `u8 1 + str` (valor de `return`) o el error |
| 8 TRACK | `u16 len, nombre del objeto` | `u8 status`: elige el objeto que describen las trazas |
| 255 QUIT | — | — |

Flags de STEP: bit 0 = obs, bit 1 = `skip_draw`, bit 2 = traza, bit 3 = `no_draw` (no dibuja ningún frame) y
bit 4 = `stop_on_gone` (corta el paso cuando desaparece el objeto seguido, p. ej. al morir). Con traza, tras la
obs viene `u32 n` y un registro por frame: `i32 room, u8 alive, f32 x, y, i32 sprite_index, f32 image_index,
f32 image_xscale` (`TRACE_DTYPE` en `client.py`).

`status`: 0 = ok, 1 = el juego terminó (se reinicia) y 2 = error GML. Las teclas son por nivel: las que no
vienen en un STEP se sueltan. Cada savestate guarda también las teclas que estaban presionadas.

Desde Python, con `Emulator(show_window=True)`, los métodos son `step(keys, frames, obs, skip_draw)`,
`save(slot)`, `load(slot)`, `observe()`, `query("player")`, `info()` y `exec("...")`. Los nombres de
objetos, rooms y variables se pueden buscar en `decompiled/gml/`.

## Verificado

- **Determinismo**: la misma secuencia de acciones desde un savestate produce frames idénticos (comparados por
  hash).
- **Muerte**: se detecta. Un agente aleatorio completa episodios; el video y la ventana en vivo funcionan.
- **Dibujar no cambia la lógica**: las trazas con `no_draw`, `skip_draw` y dibujando cada frame son idénticas
  (12 genomas al azar en `rStage01`). Sin dibujar, un emulador va a ~670 FPS.
- **Savestates**: ~14 ms y ~11 MB cada uno. Jugar desde un savestate da la misma trayectoria que jugar desde
  el inicio (32/32 hijos en la prueba de `ga/`).
- **Rendimiento** (16 hilos, `rStage01`): ~2800 FPS en el título y ~290 FPS en el nivel. Un entorno da
  ~60 pasos/s y 8 en paralelo, ~330 pasos/s. Cada frame cuesta ~3,5 ms: ~1,8 ms de dibujado y el resto del
  intérprete GML. `observe()` cuesta ~3 ms.

## Ideas y pendientes para cuando se retome

- No se ha entrenado ningún agente con RL. El candidato es PPO de Stable-Baselines3 (`CnnPolicy`,
  `SubprocVecEnv`, `VecFrameStack(4)`), que hay que agregar con `uv add "stable-baselines3[extra]"`. No se ha
  probado. Las redes con neuroevolución (`redes/`, `docs/redes.md`) usan la grilla semántica en vez de la imagen y
  un gimnasio de salas generadas; si el GA se estanca, PPO sobre ese gimnasio es el siguiente paso.
- El usuario **no quiere** un currículo con puntos de partida a mitad de nivel para las redes (la red debe
  aprender desde el comienzo); el currículo es la dificultad del gimnasio.
- Posibles optimizaciones:
  - reducir la imagen dentro de Rust para no transferir 1,4 MB por paso;
  - devolver el estado del jugador dentro de STEP para ahorrar viajes de ida y vuelta.
- En `rStage02`, moverse a la izquierda desplazó al jugador 1467 px en 40 frames. No se verificó si es un
  teletransporte propio del nivel.
- Las rutas de socket Unix tienen un máximo de 108 caracteres: el cliente las crea en `/tmp/iwbtgb_*`.

## Algoritmo genético (`ga/`)

Población de secuencias de entradas `(acción, frames)` evaluadas en paralelo, un emulador por proceso. El
diseño, la recompensa y los parámetros están en el informe del experimento
(https://claude.ai/artifact/XBoYLppCkJP2CkYwzmkP8E) y en `ga/nucleo.py` (`Config`).

```bash
uv run python -m ga.entrenar --nombre salida01                     # experimento 2 (valores por defecto)
uv run python -m ga.entrenar --nombre salida01 --continuar --generaciones 500
uv run python -m ga.visualizar todo --corrida salida01 --cada 10   # videos y gráficos
uv run python -m ga.visualizar vivo --corrida salida01 --velocidad 8   # en otra terminal, mientras entrena
```

Cada campo de `Config` es una opción de línea de comandos (`max_segundos` → `--max-segundos`). La config
usada queda en `corridas/<nombre>/config.json` y `--continuar` la respeta.

Interfaz gráfica (PySide6): `uv run python -m ga.app`. Lanza los mismos comandos como procesos aparte, cada uno
en su propio grupo de procesos. "Detener" manda Ctrl+C al proceso principal, que deja guardadas las
generaciones completas, y a los 10 s mata el grupo entero, emuladores y ffplay incluidos. El formulario cubre
agentes, padres (= élite), generaciones, duración, reintentos, % de hijos que parten del save heredado,
procesos y semilla; el resto va en "Otras opciones". Al continuar solo pasa los valores que cambiaron. Las generaciones se
escriben a un `.tmp` y se renombran, así que un corte nunca deja un `.npz` roto. `corridas/` es un enlace a
`/mnt/datos/iwbtgb/corridas`.

Hay dos experimentos:

| | Experimento 1 (`corridas/rStage01`) | Experimento 2 (`corridas/salida01`, valores por defecto) |
|---|---|---|
| Recompensa | `exploracion`: +100 por pantalla, +1 por celda de 32 px, −10 por morir | `salida`: 1000 × fracción del camino al final + 300 × mejor fracción + 300 por pantalla del camino − 50 por morir |
| Episodio | 3 min | 1 min; irse a otra room (`rStage07`, `rHideArea01`) es morir |
| Selección | torneo de 3, élite 5, 100 agentes | 10 padres de nichos distintos (pasan como élite), 4 hijos cada uno, 50 agentes |
| Mutación | ventana antes de morir/estancarse | 50 % esa ventana, 30 % un momento al azar, 20 % rama nueva con teclas al azar; sin clones |

**Distancia a la salida (`ga/distancia.py`).** Al crear la corrida se mide la room en una grilla de 16 px con la
máscara del jugador (`place_meeting` contra `block`, sin plataformas, que se mueven; `playerKiller` cuesta
8 veces más) y se calcula con Dijkstra la distancia por el laberinto hasta el `roomChanger` cuyo `roomTo` es
la sala objetivo. Queda en `corridas/<nombre>/distancia.npz` (más `graficos/mapa_distancia.png`). Las
"pantallas del camino" son las que cruza el camino más corto: en `rStage01` la de arriba a la derecha (hub
de warps cerrados) no cuenta.

**Zonas prohibidas.** `zonas_prohibidas` ("x0,y0,x1,y1;..." en px de la room de entrenamiento) marca zonas sin
salida donde entrar cuenta como morir, igual que irse a otra room: la traza se corta justo antes de entrar. Con
"auto" (por defecto) se usan las de `ZONAS_AUTO` en `ga/nucleo.py`; hoy solo `rStage01`: la pantalla de
arriba a la derecha (800,0)–(1600,608), la sala de portales que abren los jefes, donde en la corrida
`150AStage01_00` se quedaban vivos 47 de 150 agentes (ocupaban muchos nichos con fitness ~55). Al crear la
corrida "auto" se reemplaza por los rectángulos, así que `config.json` dice exactamente qué se usó. Las corridas
anteriores no tienen zonas (al continuarlas se puede agregar `--zonas-prohibidas auto`; `""` = ninguna). Los
videos dibujan la zona en rojo.

**Reintentos y save points** (corridas creadas desde el 2026-10-23). Al morir, entrar a una zona prohibida o
irse de la sala, el agente espera `frames_reset` (25 frames, 0,5 s), aprieta R y el juego lo devuelve a su
último save, como haría un jugador (R → `loadGame` → `game_restart`, 1 frame en `rInit`). Tiene `resets` (3)
reintentos; la muerte siguiente lo descalifica. Los genes siguen desde donde iban. Todo el tiempo de espera
cuenta dentro de `max_segundos`.

- Puntaje: + `r_save` (150) por cada save point distinto activado (disparándole), − `castigo_reset` (10) por
  reintento, − `castigo_muerte` (50) al quedar descalificado. La cercanía "al final" es la del mejor final de
  vida (donde murió o donde lo pilló el tiempo), no la del save al que volvió.
- `save_points` = "auto" se resuelve al crear la corrida con los `savePoint` (y sus hijos) que existen en la
  dificultad, salvo los de zonas prohibidas. En `rStage01`, medium: 6.
- Un save se detecta porque cambian los primeros 9 bytes (sala, x, y) de `save/saveN` en la carpeta del
  emulador; cuál fue, por el `savePoint` con `image_index >= 1`.
- **Los savestates del cliente incluyen los archivos de `save/`** (salvo los `.bmp`): el emulador no los guarda
  y, sin eso, cargar un savestate y apretar R llevaría al save de otra partida. Con esto la evaluación por
  familias sigue siendo idéntica a jugar desde cero (verificado 8 de 8 con reintentos).
- La traza guarda por frame el gen (`gen`) y las teclas (`accion`: 0–11, 254 = espera, 255 = R). Los videos
  repiten la partida con esas teclas. Las corridas anteriores se completan al cargarlas (sin reintentos).
- Las mutaciones locales caen antes de una de sus muertes (cualquiera) o del mejor momento; el cruce y la rama
  cortan por gen usando `gen`.
- Corridas anteriores: `resets` = 0 y sin save points. Cambiar `--resets` al continuar mezcla puntajes.

**Save heredado** (`prob_desde_save`, 0,3; corridas anteriores: 0). Cada hijo de un padre que activó un save
point nuevo parte, con esa probabilidad, desde ese save (el más cercano a la salida de los que activó), con
`max_segundos` y los reintentos completos. Es la ventaja real del save: no repetir el camino hasta él.

- Partir del save (`nucleo.preparar`): savestate inicial, se escriben los archivos de `save/` que tenía el padre
  justo después de guardar y se aprieta R. El juego lo deja en el save, igual que a un jugador.
- `Inicio` (en `Resultado.inicio`, JSON en el `.npz`) guarda: el save, esos archivos, los saves y las pantallas
  del camino ya logrados por los ancestros (cuentan en el fitness), y la generación, el agente y el frame que lo
  activaron. `Resultado.nuevos` = saves activados en la propia partida, con su gen, frame y archivos.
- Genoma del hijo: lo que el padre hizo después de activar el save, con cambios en los primeros ~2 s. Los hijos
  de un agente que parte de un save parten del mismo save. El cruce solo se hace entre padres que parten del
  mismo lugar. La evaluación por familias sigue siendo idéntica a jugar desde cero (verificado 8 de 8).
- Al continuar una corrida de antes, los agentes cuyos saves no se conocen (los que tienen más saves que los
  registrados en `nuevos`, como la élite) se vuelven a jugar con las reglas actuales y la generación se guarda de
  nuevo.
- `video_mejor` muestra el recorrido completo: tramo por tramo, cada ancestro juega hasta activar el save
  desde el que parte el siguiente (`almacen.cadena`).
- Métrica nueva: `desde_save_pct`. En una prueba sobre `150AStage01_00` pasó de 0 a 70 % en 3 generaciones y
  los nichos bajaron de 25 a 13: muchos agentes mueren en el mismo lugar después del mismo save.
- **Bajar `prob_desde_save` (incluso a 0) no devuelve a nadie al comienzo**: los hijos de un agente que parte
  de un save siempre parten de ese save. Solo deja de crear linajes nuevos. Una vez que toda la población parte
  de un save (pasa en ~5 generaciones), el comienzo de la sala no se vuelve a jugar.

**Borrar corridas.** Botón "Borrar corrida…" en la app (pide confirmación y no deja borrar una corrida que esté
usando un proceso), o `rm -r corridas/<nombre>` (borra en el disco de datos, por el enlace).

**Nichos.** Dos agentes que terminan (mueren o se les acaba el tiempo) en el mismo cuadro de 64 px son del
mismo nicho; solo el mejor de cada nicho puede ser padre.

| Archivo | Rol |
|---|---|
| `ga/nucleo.py` | `Config`, genoma, evaluación por familias (savestate del padre donde se separa cada hijo), fitness, cruce y mutación |
| `ga/distancia.py` | Grilla de colisiones, salidas y mapa de distancias (Dijkstra) para la recompensa `salida` |
| `ga/entrenar.py` | Bucle de generaciones; guarda `corridas/<nombre>/gen_NNNN.npz` (genomas + trazas por frame) y `metricas.csv` |
| `ga/almacen.py` | Formato en disco |
| `ga/recursos.py` | Mapa completo de la room (sin jugador ni `camera`) y sprites del jugador extraídos del .exe |
| `ga/visualizar.py` | Videos 1920x1080: `generacion` (todos juntos), `mejor` (partida real, recorrido completo por tramos), `evolucion`, `graficos`, `vivo`. `vivo` aguanta horas (stream): corre en un scope de systemd con `MemoryMax=1500M` sin swap, ffplay muere con él (`-autoexit` + `PR_SET_PDEATHSIG`) y se reinicia si pasa de 800 MB, y cada 10 min escribe una línea con la memoria |
| `ga/app.py` | Interfaz Qt: lista de corridas, entrenar/continuar/detener, visualizaciones, borrar corridas |
| `iwbtgb_rl/client.py` | Cliente del emulador; `save`/`load` copian también `save/` (`archivos()`, `restaurar_archivos()`) |

- `camera` mueve la vista en pasos de 800x608, pero la vista mide 800x600: hay 8 px de cada pantalla que nunca
  se ven.
- `rStage01` tiene un hueco en el techo (x≈183) que entra a `rStage07` y devuelve al jugador al caer; por eso
  el fitness solo cuenta lo que pasa dentro de la room de entrenamiento (y en el experimento 2 es muerte).
- La salida real de `rStage01` es el borde derecho abajo (`roomChanger` en (1600, 1152)): el jugador cambia de
  room al salir de la room sobre un `roomChanger` (si sale por otro lado, muere). Los warps de la pantalla de
  arriba a la derecha solo se abren tras vencer a los jefes.
- Las plataformas móviles (`movingPlatform`, hijas de `platform` → `block`) tapan pasillos en el frame inicial.
- La posición del jugador puede salir levemente de la room (x = -3e-15): se recorta antes de calcular pantallas.

## Go-Explore (`ga/explorar.py`)

Alternativa al GA para el muro de precisión (ver "Estado de las corridas"): en vez de una población, un archivo
de celdas. Celda = cuadro de `celda_px` (16) px de la sala + signo de la velocidad vertical (sube, baja, quieto)
+ si queda doble salto (`celda_salto`) + posición relativa de la plataforma móvil más cercana, en cuadros de
48 px dentro de `radio_plataforma` (160) px (`celda_plataforma`; las plataformas se mueven solas, su recorrido se
precalcula con `frames_plataformas`). Las corridas viejas siguen sin esas dos partes (`setdefault` en False).
Para cada celda se guarda el camino más corto (en frames) con que se llegó, como cadena de `Nodo`s (genes y traza
desde el nodo padre; inmutables, así una celda que mejora no rompe los caminos de sus descendientes).

```bash
uv run python -m ga.explorar --nombre ge01 --workers 4 --iteraciones 2000
uv run python -m ga.explorar --nombre ge01 --continuar --iteraciones 5000
```

- Cada iteración elige `workers × tareas_por_worker` celdas con peso
  `(1/√(1+elegida) + peso_vista/√(1+vista)) × e^(beta·(progreso − mejor progreso))` y desde cada una prueba
  `exploraciones` ráfagas de teclas al azar de hasta `frames_exploracion` (100) frames. Morir, irse de la sala o
  entrar a una zona prohibida corta la ráfaga, pero lo ganado antes queda.
- Volver a una celda: cada worker guarda `cache` (16) savestates (LRU); si no tiene la celda, carga el ancestro
  más cercano que tenga (o el comienzo de la sala) y repite los genes. El principal lleva una copia exacta de
  cada caché y manda cada celda al worker que menos frames tiene que repetir (`frames_repeticion` en las métricas).
  Cada tarea guarda además una "escala" (un ancestro ≥150 frames antes del destino) para que las celdas vecinas
  no repitan todo el camino; con `beta` alto (50) sin escalas la caché se saturaba.
- Sin reintentos ni save points: el savestate ya es "el save".
- Verificado: los genes de una celda jugados desde el comienzo de la sala reproducen su traza frame a frame
  (10 de 10).
- Disco: `config.json` (una `Config` normal, `seleccion = "archivo"`, para que la app y `ga.visualizar` la lean),
  `explorar.json` (`ConfigExplorar`), `archivo.pkl` (para `--continuar`), `metricas.csv` (una fila por
  iteración; columnas extra `archivo_celdas`, `nuevas`, `mejoradas`, …) y, cada `guardar_cada` iteraciones,
  `gen_NNNN.npz` (NNNN = iteración) con las `muestra` (30) celdas más avanzadas y su partida completa, y
  `graficos/mapa_archivo.png` (celdas coloreadas por iteración de descubrimiento + camino de la mejor).
  `ga.visualizar mejor/generacion/graficos` funcionan igual.
- `ga.entrenar` se niega a continuar una corrida que tenga `explorar.json`. La app la lista, pero "Continuar"
  desde la app no sirve: se lanza por línea de comandos.

## Estado de las corridas (2026-10-26)

Todas en `rStage01` → `rStage02`, medium, recompensa `salida`, selección por nichos. "Camino" = mejor fracción
del camino a la salida alcanzada. Las lanza el usuario desde la app; los valores de config son los actuales
(pueden haber cambiado al continuar).

**El muro del ~20,5 % era un error del emulador**, no del algoritmo: el lector GM7 tomaba mal el color
transparente de los sprites con una fila extra abajo (`sprSpikeUp` y `sprSpikeRight` de 32×33, entre otros; ver
`docs/formato-gm7.md`) y la máscara de esas púas era el rectángulo entero. Pasar el muro de púas de (384–416,
512–576) era imposible. Corregido el 2026-10-25. **Todas las corridas hasta `200A_30P` usan el emulador viejo**:
para repetirlas (videos) hay que usar el binario respaldado,
`OPENGMK_EMULATOR=/mnt/datos/iwbtgb/emulador_antiguo/gm8emulator`, o el actual con `OPENGMK_GM7_CLAVE_ESQUINA=1`.
Con el corregido, sus partidas no se reproducen.

| Corrida | Agentes / padres | Intento | Reintentos | Save heredado | Gens | Camino máx | Notas |
|---|---|---|---|---|---|---|---|
| `150AStage01_00` | 150 / 25 | 120 s | 2 | 30 % | 690 | 19,9 % | Estancada en 18,6 % desde la gen 318 (save de (672,512)); con reintentos y save heredado subió a 19,9 % |
| `300A_Stage01_01` | 300 / 10 | 120 s | 0 | 0 % (al crearla, >0) | 5000 (terminó) | 20,5 % | 100 % parte de un save desde la gen 5. Sin avance desde la gen ~750; desde la ~2500 los nichos bajaron a 10 (= padres) |
| `100A_Stage01_01` | 100 / 30 | 120 s | 0 | 0 % | 716 | 9,9 % | Recreada sin reintentos ni save heredado; estancada desde la gen 200 |
| `50A_Stage01_01` | 50 / 10 | 120 s | 0 | 100 % | 501 (terminó) | 9,8 % | |
| `200A_30P` | 200 / 30 | 90 s | 1 | 30 % | 1736 (detenida) | 20,0 % | Detenida al encontrar el error del emulador |
| `_prueba_ga_alfa` | 200 / 30 | 90 s | 0 | 30 % | 2000 (terminó) | 75,7 % | Prueba con el emulador corregido: 49,8 % en la gen 1800, 75,7 % en la 1999, 5 saves. El mejor final juega 12 frames desde el save 5 y muere: casi todo su avance es heredado |
| `GA_beta_corregida` | 200 / 50 | 180 s | 0 | 0 % | 1637 (detenida el 2026-10-26) | ~47 % | Emulador corregido. Hipótesis del usuario: sin herederos también llega, en más generaciones, y el video es una partida continua. Se estancó en ~47 % y se detuvo para empezar con las redes (`docs/redes.md`) |
| `test_sala` | 10 / 3 | 60 s | 0 | 100 % | 300 | 3,8 % | Prueba del usuario |

Las corridas de los experimentos 1 y 2 (`rStage01`, `salida01`) ya no existen: el usuario las borró. Sus
resultados están en el informe.

Lo que se sabe:

- El cuello de botella del 18–20 % (pasar el save de (672,512)) era el error de las púas del emulador.
- Con el save heredado, el video del mejor encadena tramos: el heredero parte del save tras una R (reaparece en
  el punto del save, con la sala reiniciada y sin velocidad), así que en cada unión hay un salto de 20–40 px.
  Con `prob_desde_save = 0` y sin reintentos, el video es una sola partida continua.
- Go-Explore con el emulador corregido (prueba de 4000 iteraciones, ya borrada) llegó al 48,7 %: no logra
  montarse en la plataforma móvil (288–608, y≈1120) para subir al save de (672,1056).
- Con pocos padres (10), la diversidad se pierde: los nichos quedan iguales al número de padres.
- Disco: ~0,6 GB por cada ~700 generaciones de 150 agentes con intentos de 120 s.

Ideas sin probar:

- Que no todos los hijos de un agente que parte de un save hereden ese save. Por ejemplo, una fracción vuelve al
  save del abuelo o al comienzo, para no perder rutas alternativas.
- Nichos que incluyan el save de inicio, o más padres.
- Re-evaluar la élite al cambiar las reglas a mitad de una corrida. Hoy solo se re-evalúan los agentes cuyos
  saves no se conocen.

