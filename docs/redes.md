# Redes neuronales (`redes/`)

Proyecto aparte del GA con TAS (`ga/`). El objetivo es una red que **generalice**: que juegue niveles que no vio,
idealmente de otros fangames de IWBTG. El GA produce una secuencia fija de teclas para una partida determinista; la
red decide cada pocos frames mirando el juego. Comparte con el GA el emulador y `iwbtgb_rl`, pero tiene sus propias
corridas (`corridas_redes/` → `/mnt/datos/iwbtgb/corridas_redes`), app y videos. No mezclarlo con `ga/`.

## Estado (2026-10-28)

- **Entrenar en el nivel no funciona con este método**: 7 corridas de Deep GA en rStage01 (`red01`–`red07`,
  configuraciones muy distintas) chocan en el 7–8,7 % del camino, en la trampa `spikeTrigger01` del pozo derecho
  (736, 288): una púa que sube a 10 px/frame cuando el jugador cae. Caer derecho da todo el avance hasta ahí y
  desviarse (lo que hace el GA con TAS) no suma al principio: óptimo local engañoso, la población termina en clones.
  Más tiempo no ayuda (`red05`: 790 generaciones, sin cambios desde la ~350).
- **Enfoque actual: el gimnasio** (`redes/gimnasio.py`): salas de una pantalla generadas al azar dentro del juego
  real, con dificultad creciente y trampas, nuevas en cada generación (no se pueden memorizar). Se entrena ahí
  desde cero y después se pasa a un nivel partiendo de esa red (`--desde`). El usuario rechazó explícitamente un
  currículo con puntos de partida a mitad de nivel: la red debe aprender desde cero empezando por el comienzo.
- **`gym03`** (en curso): red chica, dificultad 0,3–0,8, salida en la pared, sin trampas (se pueden activar al
  continuar). Ver "Corridas".
- Plan acordado: subir la dificultad por etapas (0–0,3 → 0–0,6 → 0–1) hasta que la mejor red supere ~70 % de un
  examen fijo de 40 salas o se estanque; luego `--desde` en rStage01 y comparar contra `red05`–`red07` (¿llega al
  ~8,7 % en pocas generaciones y lo pasa?). Si el GA se estanca igual, el siguiente paso es cambiar de método (rayos
  en vez de la grilla y/o PPO), no más ajustes de parámetros.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | `Config` (cada campo es una opción de `entrenar`) y `ANTERIORES` (valores para corridas viejas) |
| `vista.py` | Vista semántica (grilla por tipo de objeto, la calcula el emulador) |
| `modelo.py` | La CNN en numpy, genoma de semillas y caché de pesos |
| `episodio.py` | Una red jugando un nivel (`Jugador`), `Partida`, `Evaluacion`, `desglose` (fitness), workers |
| `gimnasio.py` | Generador de salas, `Gimnasio` (emulador en rStageBace), políticas, `resolver`, `JugadorGimnasio` |
| `entrenar.py` | Deep GA: población, evaluación en paralelo, selección, guardado; `partir_de` (`--desde`) |
| `almacen.py` | Disco de una corrida: config, generaciones, genomas, métricas, `indice_mejor` |
| `probar.py`, `salas.py` | La mejor red en otras salas o en el gimnasio, sin entrenar |
| `visualizar.py` | Videos, en vivo y gráficos en un nivel (`Pantalla`, `Ritmo`) |
| `gimnasio_ver.py` | Muestras, prueba con salas nuevas y entrenamiento en vivo del gimnasio |
| `dibujo_red.py` | La red dibujada por dentro (nodos y pesos) para los videos |
| `app.py` | Interfaz Qt |

## Vista semántica (`vista.py`)

Lo que ve la red: una grilla de 64×48 celdas de 8 px (512×384 px) centrada en el jugador, con un canal por tipo de
cosa. Cada celda vale la fracción (0–255) de sus 4×4 puntos de muestra que cae dentro de la **máscara de colisión**
de algún objeto de ese tipo (las púas son triángulos). Más hspeed, vspeed y `djump` del jugador.

| Canal | Objetos (con sus hijos) | Notas |
|---|---|---|
| sólido | `block` salvo `platform` | `block` es invisible: es la colisión de los tiles. Incluye `blockNotVis` y `blockDisappear` |
| plataforma | `platform` (incluye `movingPlatform`) | se atraviesa desde abajo |
| mortal | `playerKiller` | solo instancias visibles: las trampas ocultas no se ven, como para un humano |
| save | `savePoint` | |
| salida | `roomChanger` (incluye `warp`) | invisible en los niveles, pero es la salida |
| agua | `water` | |
| fuera de la sala | — | lo calcula Python con el tamaño de la room |

Es la jerarquía del engine estándar de IWBTG: otros fangames deberían servir igual (`Vista.faltan` avisa si falta
un objeto). El cálculo lo hace el emulador: `OP_GRID_CONFIG` (9) y `OP_GRID` (10) o `step(grid=True)` (flag 32);
en el cliente, `Emulator.grid_config`, `Emulator.grid`, `Grid`. Costo ~0,45 ms por decisión.

```bash
uv run python -m redes.vista probar                              # canales al comienzo de rStage01
uv run python -m redes.vista video --corrida GA_beta_corregida   # el mejor del GA con su vista al lado
```

## La red (`modelo.py`)

Grilla (7 canales × `cuadros` grillas apiladas: la actual y las de decisiones anteriores) → conv 4×4 paso 4
(`filtros1`) → conv 3×3 paso 2 (`filtros2`) → + hspeed, vspeed, djump → densa (`oculta`) → 12 acciones (la de
mayor puntuación). Decide cada `frames_por_decision` frames y mantiene las teclas entre medio. En la app: «grillas
que ve a la vez», «detectores de patrones chicos/grandes», «neuronas de decisión».

- Por defecto 2/8/16/64, decide cada 2 frames (53.156 pesos). El barrido del 2026-10-27 mostró que una red chica
  (2 grillas, 4/8 filtros, 16 ocultas, 7.608 pesos) rinde igual o mejor con el GA y cuesta menos de la mitad.
- **Genoma = linaje** (como *Deep Neuroevolution*): `(semilla inicial, (semilla, sigma), ...)`. Pesos = He inicial
  + suma de mutaciones gaussianas. Cada worker tiene una caché LRU de pesos y los hermanos se mandan juntos (el
  padre se reconstruye una vez).
- `Red.parches` recorta cada grilla en parches 4×4 una sola vez (la primera conv es un producto de matrices sobre
  los parches concatenados de las grillas apiladas). `Red.registrar` guarda las activaciones de cada capa en `act`
  (para dibujarlas; apagado al entrenar).

## Entrenamiento (`entrenar.py`, `episodio.py`)

```bash
uv run python -m redes.entrenar --nombre red01 --workers 10        # opciones = campos de redes/config.py
uv run python -m redes.entrenar --nombre red01 --continuar --generaciones 2000
uv run python -m redes.entrenar --nombre gym04 --entorno gimnasio --escenarios 8 --max-segundos 15
uv run python -m redes.entrenar --nombre gym04_stage01 --desde gym04          # sigue en un nivel desde esa red
```

**Deep GA**: sin cruce; cada generación, `elite` pasan sin cambios (y se re-evalúan) y el resto son hijos (padre al
azar entre los `padres` mejores, primero el mejor de cada nicho de `nicho_px`, + mutación de `sigma`). Con
`elite_mas_lejos` la que llega más lejos pasa siempre y tiene hijos. La selección es por **ranking**: solo importan
las proporciones entre los términos del fitness, no la escala.

**Evaluación**: cada red juega `escenarios` partidas y su fitness es el promedio de `episodio.desglose`:

| Término | Campo (por defecto) | Notas |
|---|---|---|
| avance | `r_avance` (100) × mejor fracción del camino | nivel: Dijkstra de `ga/distancia.py`; gimnasio: BFS |
| llegar | `r_objetivo` (200) | |
| rapidez | − `r_rapidez` (1) × frame del mejor avance / tope | con 1 solo desempata; no hay otro castigo por tiempo |
| muerte | − `castigo_muerte` (3) por muerte, salida o atasco | |
| saves | `r_save` (10) por save nuevo | solo nivel: los de la sala salvo los del comienzo (avance < 5 %) y de zonas prohibidas |
| quieto | − `castigo_quieto` (2) por segundo sin desplazarse ≥ 0,5 px | solo nivel; no usa `hspeed` (en una `movingPlatform` el juego mueve al jugador sin velocidad) |
| exploración | `r_explorar` (0,5) por celda de `celda_explorar` px visitada | solo nivel |

- **En el nivel**: el primer escenario empieza en el comienzo y los demás con una espera al azar sin control
  (`espera_max`, nueva cada generación). Termina a los `max_segundos`, o con `reintentos` = 0 al morir, salir de
  la sala (o entrar a una zona prohibida: la sala de portales de rStage01) o tras `paciencia` sin avanzar. Con
  `reintentos` = 1 esas causas no terminan: espera `frames_reset`, aprieta R (vuelve al último save) y espera al azar
  0–`espera_reintento` frames (sin esa espera, una red sin memoria repite exacta la vida anterior).
- **En el gimnasio** (`entorno = "gimnasio"`): `escenarios` = salas nuevas por generación (iguales para todas las
  redes), dificultad uniforme entre `gim_dificultad_min` y `_max`; `max_segundos` = tope por sala. Sin reintentos.
- **Soltar el salto** (`soltar_salto`): el salto sale al *apretar* la tecla. Una red que pide «→↑» en decisiones
  seguidas la mantiene apretada y no vuelve a saltar. Con la regla (`gimnasio.plan_teclas`), si sigue pidiendo
  salto y el jugador no sube (`vspeed >= 0`), el primer frame de esa decisión va sin salto: mantenerla da salto
  completo, doble salto en la cima y otro salto al aterrizar.
- **La mejor red** (videos, en vivo, pruebas, `redes/mejor.npy`) = `almacen.indice_mejor`: la que llega más lejos en
  su primer escenario (en el gimnasio, la de más avance promedio); a igual avance, la de más fitness.
- **Partir de otra red** (`desde`, `desde_gen`): copia la forma de red y la vista de esa corrida
  (`entrenar.FORMA_RED`) y guarda el genoma de su mejor red en `genoma_inicial`; la gen 0 es esa red sin cambios
  más mutaciones suyas, y `almacen.genomas` la reconstruye con esa base.
- **Al continuar** no se pueden cambiar `entrenar.FIJOS` (sala, dificultad del juego, forma de la red, entorno,
  red de partida). Sí recompensas, evolución, partidas y la dificultad, salida y trampas del gimnasio.
- **Compatibilidad**: cada regla nueva tiene en `config.ANTERIORES` el valor que la apaga para corridas creadas
  antes (`Config.from_dict`). Las primeras corridas del gimnasio tenían `gim_salas`/`gim_segundos`: se convierten a
  `escenarios`/`max_segundos`. `metricas.csv` viejos se reescriben con las columnas nuevas al continuar.
- **Disco** (`corridas_redes/<nombre>/`): `config.json`, `distancia.npz` (rStage01, también en las del gimnasio),
  `metricas.csv`, `log.txt` (no redirigir la salida ahí), `gen_NNNN.npz` (linaje `padre`/`semilla`/`sigma`,
  fitness, avances y fines por escenario, posición final, acciones y trazas por frame (`ga.nucleo.TRAZA_DTYPE`):
  en el nivel del primer escenario, en el gimnasio de todas las salas (`traza_largo` (redes, salas), ~0,7 MB por
  generación con 200 redes × 4 salas); en el gimnasio además `gimnasio`, `semillas`, `dificultades` (float64),
  `salida_borde` y `trampas`), `redes/` (`mejor.npy` + `.json` con el genoma, y cada `guardar_red_cada`
  generaciones), `pruebas/` y `videos/`.

### Rendimiento

- Cada worker es un par Python + emulador que se turnan (~1,1 núcleos). Un hilo de BLAS por worker
  (`OPENBLAS_NUM_THREADS=1` antes de crear el pool; sin eso la carga llegaba a 33 y algún worker se trababa al
  cerrar). Al terminar: `pool.close()` y `join()`. Con 16 workers en el Ryzen 7 2700 (16 hilos) la CPU se satura.
- En rStage01 el emulador tarda ~1–1,4 ms por frame (~970 instancias, 722 `block`; OpenGMK busca colisiones
  recorriéndolas todas). En el gimnasio, ~5.200 frames/s (~120 instancias).
- Muerto, ~9–11 ms por frame (la sangre): los `frames_reset` antes de R no se emulan (`Jugador.muerto`; R reinicia
  la sala, así que no cambian nada). Verificado: mismos resultados emulando o no.
- Al arrancar muchos emuladores a la vez, alguno puede abortar (código −6): el pool lo reemplaza solo.

## Gimnasio (`gimnasio.py`)

Salas de una pantalla (25 × 19 celdas de 32 px; la cámara avanza de a 800 × 608) armadas en `rStageBace` (una sala
del juego casi vacía: jugador, cámara, `world`, `playMusic`) creando con GML las instancias de los niveles y los
tiles de ladrillo de rStage01 (`tile01`: (0, 32) con borde donde hay aire encima, (32, 32) liso). Fondo
`backgroundIWBTG`. Física y colisiones exactas. Construir una sala ~1 ms; cada intento parte de un savestate
(cargar ~17 ms, guardar ~11 ms); al morir no hay R. `Gimnasio` guarda hasta 64 salas armadas (slots 10–73).

**Generador** (`generar(semilla, dificultad, salida_borde, trampas)`): el jugador a la izquierda, la salida a la
derecha y el suelo avanzando por obstáculos según la dificultad (0 a 1). Medidas según la física del jugador:
salto 8,5 (~90 px), doble 7 (~150 px en total), 3 px por frame, gravedad 0,4.

| Obstáculo | Desde | |
|---|---|---|
| plano, escalón (±1–4), púas (1–3), pozo de púas (1–5) | 0 | crecen con la dificultad |
| muro (3–4 de alto) | 0,3 | doble salto |
| techo de púas a 4 celdas | 0,4 | pasa el salto simple, no el doble; desde 0,7 con una púa en el suelo |
| islas (pozo ancho con bloques sueltos) | 0,5 | |
| plataforma móvil | 0,6 | |
| **púa que salta** (trampa) | 0,2 | una del fondo de un pozo sube al llegar al borde: esperar y saltar |
| **púa o cereza que cae** (trampa) | 0,3 | cuelga sobre el camino; el disparador va tantas columnas antes como camina el jugador mientras cae: quien sigue derecho llega justo debajo (mató a 26 de 27) |
| **púa lateral** (trampa) | 0,4 | al final de un pasillo, se desliza por el suelo hacia el jugador: saltarla |

- **Trampas** (`gim_trampas`): las piezas de los niveles, un disparador invisible `triggerNN` y una púa o cereza
  `spikeTriggerNN`/`cherryTriggerNN` que sale con `direction`/`trapSpeed` al tocarlo. Todos los disparadores de un
  número se activan juntos: cada trampa de una sala usa otro (`NUMEROS_TRAMPA`). Sin trampas, el catálogo es el de
  antes y las salas salen idénticas. Los bloques falsos (`blockDisappear`) quedaron fuera: la vista los ve como
  suelo, igual que los bloques reales de las islas (haría falta otro canal, otra forma de red).
- **Salida** (`gim_salida_borde`): una abertura de 3 celdas en la pared derecha a la altura del suelo final, con un
  `warp` en cada celda (`onCollide = false`: solo se ve; llegar = la caja del jugador, x−5…x+5, y−12,2…y+7,8, toca
  alguna celda de `Sala.celdas_salida`). Antes era un portal a 2 celdas de la pared que las redes pasaban de largo.
- **Versiones**: cada generación guarda `salida_borde` y `trampas`, y `gimnasio_ver.salas_de` reconstruye sus salas
  igual (sin las claves: portal y sin trampas). Así se pueden activar al continuar sin romper el en vivo de las
  generaciones viejas. **Al agregar obstáculos nuevos, hacerlo detrás de otra regla así**: cambiar el catálogo
  cambia el azar de todas las salas.
- **Avance**: BFS en celdas de 16 px desde la salida (sin física), como fracción del camino desde el inicio.
- **¿Tiene solución?** `resolver`: búsqueda tipo Go-Explore (celdas de 32 px con savestates, elegidas con peso
  exp(6·avance)/√(1+visitas), 2–12 acciones al azar de 2–16 frames). Con 600 iteraciones resuelve ~11 de 12 salas
  (también con trampas) en 2–15 s cada una. Al entrenar no se verifica: las salas sin solución le tocan igual a todas.
- **Políticas**: `PoliticaRed` (la mejor red de una corrida), `PoliticaCNN` (pesos dados, la usan los workers),
  `PoliticaAzar`, `PoliticaGrabada` (p. ej. la solución de `resolver`).

## App (`app.py`)

```bash
uv run python -m redes.app      # también en el menú: ~/.local/share/applications/iwbtgb-redes.desktop
```

Reutiliza `Proceso` y `spin` de `ga.app`. A la izquierda, las corridas y su resumen. Pestañas:

- **Crear y entrenar**: 1 · Corrida (nombre, partir de la red de), 2 · Dónde practica (nivel o gimnasio, con
  dificultad de las salas, salida en la pared y trampas), 3 · Partidas de cada red, 4 · La red (con descripción y
  número de pesos), 5 · Recompensas, 6 · Evolución. Solo se muestran los campos del entorno elegido
  (`QFormLayout.setRowVisible`); al cambiar de entorno se sugieren valores. Detecta y detiene (SIGINT) los
  entrenamientos lanzados desde la terminal (busca en `/proc`).
- **Probar**: «Dónde»: donde entrenó, el gimnasio (N salas, dificultad, solo con solución), todos los niveles o
  uno; tabla con los resultados de `pruebas/`.
- **Videos y en vivo**: «Quién juega» (la mejor red, todas las de la generación, al azar, la solución) × «Dónde»
  (donde entrenó —en una corrida del gimnasio, el entrenamiento en vivo—, prueba en el gimnasio con salas nuevas u
  otro nivel); botones en vivo, grabar video, muestras de salas, gráficos y carpetas. La app elige
  `redes.visualizar` o `redes.gimnasio_ver` según la combinación.

La app abierta usa el código con el que se abrió (los procesos que lanza usan el del disco): tras cambiar `Config`
o la app, hay que cerrarla y reabrirla.

## Probar (`probar.py`, `salas.py`)

```bash
uv run python -m redes.probar --corrida red01                                  # su sala, 10 esperas
uv run python -m redes.probar --corrida red01 --sala rStage02 rStage03 [--max-segundos 60] [--video]
uv run python -m redes.probar --corrida gym03 --gimnasio --escenarios 40 --dificultad -1 [--verificar]
```

En un nivel, la mejor red juega `--escenarios` partidas por sala con esperas fijas repartidas entre 0 y
`espera_max`; en otra sala el objetivo es la siguiente y su mapa de distancias se arma la primera vez en
`corridas_redes/_salas/`. rStage02 y rStage03 matan casi al empezar con trampas ocultas. En el gimnasio, salas desde
la semilla 1.000.000 (no son las del entrenamiento), con la duración, la paciencia, la salida y las trampas de la
corrida. Resultados en `pruebas/*.json`. **Para medir el progreso usar un examen fijo así**: el % de salas
superadas por generación del log sube y baja mucho (depende de qué tan difíciles salieron esas salas).

## Videos y en vivo (`visualizar.py`, `gimnasio_ver.py`, `dibujo_red.py`)

```bash
uv run python -m redes.visualizar mejor      --corrida red01 [--gen -1] [--espera 0] [--sala rStage02]
uv run python -m redes.visualizar vivo-red   --corrida red01 [--sala rStage02]    # la mejor red, ventana
uv run python -m redes.visualizar generacion --corrida red05 [--gen -1]           # todas juntas en el nivel
uv run python -m redes.visualizar vivo       --corrida red05 [--velocidad 4]      # todas juntas, ventana
uv run python -m redes.visualizar graficos   --corrida red01
uv run python -m redes.gimnasio_ver entreno  --agente gym03 --quien mejor|todas   # entrenamiento en vivo
uv run python -m redes.gimnasio_ver entreno-video --agente gym03 --quien todas [--gen -1]
uv run python -m redes.gimnasio_ver vivo     --agente gym03 [--dificultad 0.5]   # prueba: salas nuevas
uv run python -m redes.gimnasio_ver video    --agente solucion --salas 10
uv run python -m redes.gimnasio_ver muestras --dificultad -1 [--verificar]      # mosaico de 12 salas
```

- **La mejor red** (nivel o gimnasio): el juego a 1280×960 y un panel con tiempo, avance, puntaje (nivel: en curso
  con su desglose y el fitness de la generación; gimnasio: sala, obstáculos e historial de salas superadas), **la
  red dibujada por dentro** y la preferencia por acción.
- **Todas juntas en el nivel**: las trazas del primer escenario dibujadas con `ga.visualizar.cuadros_generacion`
  (adaptadores `_Red`, `_cfg_ga`). Corridas sin trazas (`red01`–`red04`): solo la mejor red.
- **Entrenamiento en vivo del gimnasio** (`entreno`): las salas de una generación tal como las jugó la población.
  «La mejor»: emulada, con la nota de cómo le fue al entrenar (avisa si al repetirla termina distinto; verificado
  igual con `gym02`). «Todas»: la mejor emulada (así se mueven las plataformas) y las demás como fantasmas desde sus
  trazas, con cruces donde murieron, conteo y el gráfico de salas superadas; si la mejor termina antes, se recarga
  la sala, se esconde al jugador y se adelanta al mismo frame. En vivo repite la última generación y pasa a la
  nueva. Necesita generaciones con trazas de todas las salas (desde el 2026-10-27; avisa si no las hay).
- **Prueba con salas nuevas** (`vivo`/`video`): la mejor red en salas que nunca vio, una tras otra: mide si
  generaliza (por eso se ven muchas más salas que las del entrenamiento).
- **Dibujo de la red** (`DibujoRed`): columnas vista por tipo → filtros capa 1 → filtros capa 2 + velocidad y doble
  salto → ocultas → acciones; cada filtro es un nodo y el peso entre nodos es el promedio de sus pesos. Líneas fijas
  por red (naranja +, azul −, las 160 más fuertes por par de capas), círculos encendidos con la activación; con más
  de 32 ocultas, las 32 de más peso hacia la salida. ~35 ms al cambiar de red, ~1,5 ms por cuadro.
- **Ritmo**: todo a 25 cuadros/s, uno cada `2·velocidad` frames de juego según el contador del emulador
  (`visualizar.Ritmo`, `Gimnasio.jugar(cada=...)`): a ×N se emula más juego por cuadro, no más cuadros. El emulador
  solo dibuja los pasos que se muestran y la parte fija del panel se arma una vez por sala. A ×1 llega a tiempo real
  aun con la CPU ocupada por un entrenamiento (~40 ms por cuadro); a ×2–×4 con una red que decide cada frame el
  límite son sus decisiones.
- Los en vivo corren con tope de memoria (`ga.visualizar._con_tope_de_memoria`), como el del GA.

## Corridas

### En el nivel (rStage01 → rStage02)

| Corrida | Red (grillas/f1/f2/ocultas, decide cada) | Población, sigma | Reglas | Gens | Récord |
|---|---|---|---|---|---|
| `red01` | 2/8/16/64, 2 | 200, 0,02 | defecto de entonces, 3 escenarios de 30 s | 169 | 8,5 % (gen 76) |
| `red02` | 2/8/16/64, 2 | 200 (50 padres), 0,05 | igual, para probar diversidad | 378 | 8,7 % |
| `red03` | 5/8/16/33, 1 | 100, 0,01 | 2 escenarios de 60 s | 65 | 7,2 % |
| `red04` | 3/8/16/32, 1 | 200, 0,02 | reintentos, muerte 1: aprendió a quedarse quieta (12,9 de 20 s) → castigo por quieto | 11 | 3,9 % |
| `red05` | 2/8/16/32, 2 | 100, 0,02 | reintentos, muerte 3, quieto 1, 20 s | 790 | 8,8 % (sin cambios desde la ~350; clones) → premio por explorar |
| `red06` | 5/8/16/32, 2 | 300, 0,05 | reintentos, muerte 5, explorar 5 → 1, 120 s | 44 | 8,7 % |
| `red07` | 5/8/8/32, 1 | 200, 0,05 | sin reintentos, 30 s, explorar 1 | 145 | 7,3 % |

Todas mueren en la trampa del pozo derecho (o antes). La primera prueba de la mejor red de `red07` en el gimnasio:
muere en ~1 s o no avanza (no generaliza).

### En el gimnasio

| Corrida | Red | Partidas | Dificultad, salida, trampas | Gens | Resultado |
|---|---|---|---|---|---|
| `gym01` | 5/8/16/64, 1 (55.844) | 8 salas × 15 s | 0–0,3, portal, no | 13 | ~2 % de salas (sin castigo por muerte ni soltar el salto) |
| `gym02` | 5/8/16/32, 1 (30.756) | 4 × 15 s, sigma 0,025 | 0–0,3, portal, no | 76 | 12 → 19 → 13 % de la población; la mejor 2,8 de 4 |
| `gym03` | 2/4/8/16, 2 (7.608) | 8 × 15 s, sigma 0,025, 200 redes | 0,3–0,8, pared, no | 63+ (en curso) | ~30 % de la población; la mejor 6 de 8 |

### Barrido de configuraciones del gimnasio (2026-10-27)

Cuatro corridas cortas iguales salvo una cosa (población 60, 25 generaciones, dificultad 0–0,3, salida en la pared,
soltar el salto, recompensas de `gym02`), y la mejor red de cada una en las mismas 40 salas fijas (semillas
5.000.000+):

| | Cambio | Superó (de 40) | Avance medio | s/gen (7 workers) |
|---|---|---|---|---|
| A | `gym02` escalada: 5/8/16/32, decide cada frame, sigma 0,025, 6 salas | 12 | 58,8 % | 48 |
| B | sigma 0,05 | 11 | 58,7 % | 49 |
| C | red chica 2/4/8/16, decide cada 2 frames | **15** | **61,1 %** | **21** |
| D | 12 salas por generación | 9 | 55,5 % | 73 |

Las diferencias son del orden del ruido (±3), salvo que la red chica rinde igual o mejor y cuesta menos de la mitad.
El % de salas por generación oscila mucho en todas (también con 12 salas); la mejor red sí se estabiliza con más
salas. Todas se aplanan en ~20 % de la población y ~30–40 % de la mejor tras ~10 generaciones.

## Pendiente

- Etapas de dificultad de `gym03` y examen fijo; activar trampas; transferir a rStage01 con `--desde`.
- Si el GA se estanca: rayos en vez de la grilla (prototipo posible sobre la grilla sin tocar Rust) y/o PPO.
- Premio por llegar rápido solo a las que llegan (propuesto, no implementado; hoy `r_rapidez` solo desempata).
- Gimnasio: salas de varias pantallas y verticales, salidas en otros bordes, saves a los que hay que disparar,
  bloques falsos (requiere otro canal de vista), biblioteca de salas verificadas.
- Memoria entre muertes (las trampas ocultas de rStage02/03 no se pueden anticipar sin ella).
