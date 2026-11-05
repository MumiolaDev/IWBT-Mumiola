# Redes neuronales: la grilla (`iwbtgb/redes`) y los rayos (`iwbtgb/rayos`)

El objetivo es una red que **generalice**: que juegue niveles que no vio, idealmente de otros fangames de IWBTG, y que
vaya superando el juego por etapas mientras se transmite en vivo. A diferencia del GA con TAS (`docs/ga.md`), la red
decide cada pocos frames mirando el juego. Hay dos redes, que solo difieren en lo que ven y en la red:

- **`redes`**: una CNN chica que ve una grilla semántica alrededor del jugador. Corridas en `corridas_redes/`.
- **`rayos`**: una red densa que ve 64 rayos que salen del jugador. Corridas en `corridas_rayos/`.

Todo lo demás es común (`iwbtgb/evolucion`): el Deep GA, el gimnasio de salas generadas, las partidas en un nivel, el
fitness, las pruebas, los videos, la transmisión y la app. Un cambio ahí vale para las dos. Las carpetas de corridas
son enlaces a `/mnt/datos/iwbtgb/`; cada una tiene además `_salas/` (mapas de otros niveles) y `_gimnasio/` (muestras,
videos de `azar`/`solucion` y la caché de sprites).

## Estado (2026-11-05)

- **Entrenar en el nivel se estanca**: 7 corridas de `redes` (`red01`–`red07`) y 5 de `rayos` (`rayos02`, `03`, `05`,
  `06` y `07`) chocan en el 7–8,9 % de rStage01, en la trampa `spikeTrigger01` del pozo derecho (736, 288): una púa que sube a
  10 px/frame cuando el jugador cae. Caer derecho da todo el avance hasta ahí y desviarse (lo que hace el GA con TAS) no
  suma al principio: óptimo local engañoso, la población termina en clones. Más tiempo no ayuda.
- **El gimnasio**: salas de una pantalla generadas dentro del juego real, nuevas en cada generación (no se pueden
  memorizar), con dificultad creciente y trampas. Se entrena ahí desde cero y después se sigue en un nivel desde esa
  red (`--desde`). El usuario rechazó explícitamente un currículo con puntos de partida a mitad de nivel: la red debe
  aprender desde el comienzo; el currículo es la dificultad del gimnasio.
- **Los rayos generalizan mejor**: en las mismas 40 salas fijas, la mejor red de rayos (64,32) superó 24 contra 15 de
  la mejor CNN (ver «Corridas»). `rayos04` (gimnasio con currículo hasta dificultad 1 y trampas) supera el 57 % de sus
  salas, pero al seguir en rStage01 (`rayos05`) se estanca igual en el pozo derecho, como `rayos06` y `rayos07`.
- La transmisión está probada a un archivo; falta probarla contra un servidor real con la clave del usuario.
- Si se estanca: memoria entre muertes (una red recurrente chica) o PPO (`iwbtgb/entorno.py`), no más ajustes de
  parámetros.

## Comandos

Iguales para las dos (`iwbtgb.redes` o `iwbtgb.rayos`); sin comando, cada una los lista.

```bash
uv run python -m iwbtgb.rayos app                                         # interfaz (en el menú: iwbtgb-rayos.desktop)
uv run python -m iwbtgb.rayos entrenar --nombre rayos08 --entorno gimnasio    # opciones = campos de su Config
uv run python -m iwbtgb.rayos entrenar --nombre rayos08 --continuar --generaciones 2000
uv run python -m iwbtgb.rayos entrenar --nombre rayos08_s01 --entorno nivel --desde rayos08   # seguir en rStage01
uv run python -m iwbtgb.rayos probar --corrida rayos08 --gimnasio --escenarios 40 [--dificultad -1] [--verificar]
uv run python -m iwbtgb.rayos probar --corrida rayos08 --sala rStage01 | todas [--video]
uv run python -m iwbtgb.rayos visualizar entreno --corrida rayos08 --quien mejor|todas|alternar   # en vivo
uv run python -m iwbtgb.rayos visualizar entreno-video --corrida rayos08 --quien todas [--gen -1]
uv run python -m iwbtgb.rayos visualizar prueba --corrida rayos08|azar|solucion [--dificultad 0.5]   # salas nuevas
uv run python -m iwbtgb.rayos visualizar prueba-video --corrida rayos08 --salas 10
uv run python -m iwbtgb.rayos visualizar mejor --corrida rayos08_s01 [--sala rStage02] [--gen -1] [--espera 0]
uv run python -m iwbtgb.rayos visualizar vivo-red --corrida rayos08_s01 [--sala rStage02]
uv run python -m iwbtgb.rayos visualizar vivo|generacion --corrida rayos08_s01    # todas las redes sobre el mapa
uv run python -m iwbtgb.rayos visualizar muestras --dificultad -1 --verificar      # mosaico de salas generadas
uv run python -m iwbtgb.rayos visualizar graficos --corrida rayos08
uv run python -m iwbtgb.rayos transmitir --corrida auto [--destino prueba.flv]
uv run python -m iwbtgb.redes vista probar | video --corrida GA_beta_corregida   # la grilla (solo redes)
```

`--red` en `visualizar` dibuja la red por dentro en vez del gráfico de fitness (el usuario prefirió el gráfico y dejó
el dibujo para más adelante).

## Archivos

| Archivo | Qué hace |
|---|---|
| `evolucion/__init__.py` | `Linea`: lo que el núcleo necesita de una red (su config, cómo arma su red, su jugador, su gimnasio, su política y su dibujo) |
| `evolucion/config.py` | `ConfigBase`: los campos comunes (juego, gimnasio, currículo, partidas, recompensa, reintentos, evolución) |
| `evolucion/genoma.py` | Genoma de semillas, `Pesos` (iniciales, ruido, reconstrucción con caché) y `Cache` |
| `evolucion/episodio.py` | Una red en un nivel: `Escenario`, `Partida`, `Evaluacion`, `desglose` (fitness), `Jugador`, `plan_teclas` |
| `evolucion/gimnasio.py` | Generador de salas, `Gimnasio`, políticas, `JugadorGimnasio`, `resolver` |
| `evolucion/entrenar.py`, `trabajador.py` | Deep GA: población, evaluación en paralelo, selección, currículo, guardado; los workers |
| `evolucion/almacen.py`, `salas.py` | Disco de una corrida; niveles donde jugar y sus mapas de distancias |
| `evolucion/probar.py` | Examen en salas que no vio y en niveles |
| `evolucion/pantallas.py`, `visualizar.py`, `transmitir.py` | Cuadros de video, videos y en vivo, stream RTMP |
| `evolucion/app.py` | La app (Qt); cada línea pone su `FormularioRed` (lo que ve y la red) |
| `redes/` | `config.py`, `vista.py` (la grilla), `modelo.py` (la CNN), `jugadores.py`, `dibujo.py`, `app.py` |
| `rayos/` | `config.py`, `observacion.py` (los rayos y la dirección del nivel), `modelo.py` (la red densa), `jugadores.py`, `dibujo.py` (la red por dentro y los rayos sobre el juego), `app.py` |

## Lo que ve cada red

### La grilla (`redes/vista.py`)

Una grilla de 64×48 celdas de 8 px (512×384 px) centrada en el jugador, con un canal por tipo de cosa. Cada celda vale
la fracción (0–255) de sus 4×4 puntos de muestra dentro de la **máscara de colisión** de algún objeto de ese tipo. Más
hspeed, vspeed y `djump` del jugador. La calcula el emulador (`docs/emulador.md`, «La grilla semántica»).

| Canal | Objetos (con sus hijos) | Notas |
|---|---|---|
| sólido | `block` salvo `platform` | `block` es invisible: es la colisión de los tiles. Incluye `blockNotVis` y `blockDisappear` |
| plataforma | `platform` (incluye `movingPlatform`) | se atraviesa desde abajo |
| mortal | `playerKiller` | solo instancias visibles: las trampas ocultas no se ven, como para un humano |
| save | `savePoint` | |
| salida | `roomChanger` (incluye `warp`) | invisible en los niveles, pero es la salida |
| agua | `water` | |
| fuera de la sala | — | lo calcula Python con el tamaño de la room |

Es la jerarquía del engine estándar de IWBTG: otros fangames deberían servir igual (`Vista.faltan` avisa si falta un
objeto).

### Los rayos (`rayos/observacion.py`)

`n_rayos` (64) rayos desde el centro de la caja del jugador hasta el borde de la pantalla, cortados por el primer
sólido (`block` sin `platform`: así se ve la púa y también el muro de atrás). Por rayo, 7 valores de **cercanía**
`1 − d/1000` (0 = no hay, 1 = pegado): sólido, plataforma, mortal (solo visibles), save, salida, agua y el borde. Se
apilan los rayos de las últimas `cuadros` decisiones y se suman, cada una con su opción `ver_*`: la velocidad
(`hspeed/3`, `vspeed/9`), el doble salto, la **dirección del nivel** (vector unitario hacia la celda más cercana a la
salida dentro de 3 celdas del mapa de distancias: la noción de «hacia dónde queda la salida» de un jugador que mira el
mapa) y hacia qué lado mira. Con 64 rayos y 2 cuadros son 902 entradas.

## Las redes

- **CNN** (`redes/modelo.py`): 7 canales × `cuadros` grillas → conv 4×4 paso 4 (`filtros1`) → conv 3×3 paso 2
  (`filtros2`) → + hspeed, vspeed, djump → densa (`oculta`) → 12 acciones. El barrido del 2026-10-27 mostró que una red
  chica (2 grillas, 4/8 filtros, 16 ocultas, 7.608 pesos) rinde igual o mejor y cuesta menos de la mitad.
- **Densa** (`rayos/modelo.py`): `entradas → capas ocultas (capas, p. ej. 64,32) → 12 acciones`, ReLU (He) o tanh
  (Xavier). 902→64→32→12 = 60.268 pesos.
- **Genoma = linaje** (*Deep Neuroevolution*, Such et al. 2017): `(semilla inicial, (semilla, sigma), ...)`. Pesos =
  iniciales + suma de mutaciones gaussianas. Cada worker tiene una caché LRU de pesos y los hermanos se mandan juntos
  (el padre se reconstruye una vez). Reconstruir sin caché crece con las generaciones (~0,8 s con 600 mutaciones).
- La red decide cada `frames_por_decision` frames y mantiene las teclas entre medio. **Soltar el salto**
  (`soltar_salto`, `episodio.plan_teclas`): el salto sale al apretar; si la red sigue pidiéndolo y el jugador no sube,
  el primer frame de esa decisión va sin salto.

## Entrenamiento

**Deep GA**: sin cruce; cada generación `elite` pasan sin cambios (y se re-evalúan) y el resto son hijos (padre al azar
entre los `padres` mejores, primero el mejor de cada nicho de `nicho_px`, + mutación de `sigma`). Con
`elite_mas_lejos`, la que llega más lejos pasa siempre y tiene hijos. La selección es por **ranking**: solo importan
las proporciones entre los términos del fitness.

**Evaluación**: cada red juega `escenarios` partidas y su fitness es el promedio de `episodio.desglose`:

| Término | Campo | Notas |
|---|---|---|
| avance | `r_avance` × mejor fracción del camino | nivel: Dijkstra de `iwbtgb/distancia.py`; gimnasio: BFS |
| llegar | `r_objetivo` | |
| rapidez | − `r_rapidez` × frame del mejor avance / tope | solo desempata |
| muerte | − `castigo_muerte` por muerte, salida o atasco | |
| saves | `r_save` por save nuevo | solo nivel: los de la sala salvo los del comienzo y de zonas prohibidas |
| quieto | − `castigo_quieto` por segundo sin desplazarse ≥ 0,5 px | solo nivel; no usa `hspeed` (una plataforma mueve al jugador sin velocidad) |
| exploración | `r_explorar` por celda de `celda_explorar` px visitada | solo nivel |

- **En el nivel**: el primer escenario empieza en el comienzo y los demás con una espera al azar sin control
  (`espera_max`). Termina a los `max_segundos`, o con `reintentos` = 0 al morir, salir de la sala (o entrar a una zona
  prohibida) o tras `paciencia` sin avanzar. Con `reintentos` = 1 esas causas no terminan: espera `frames_reset`,
  aprieta R (vuelve al último save) y espera al azar 0–`espera_reintento` frames (sin esa espera, una red sin memoria
  repite exacta la vida anterior).
- **Frames de muerto**: los `frames_reset` antes de R no se emulan al entrenar (la sangre cuesta ~10 ms por frame y R
  reinicia la sala). En los videos sí se ven, pero desde un savestate al que después se vuelve: la sangre usa el azar
  del juego y, si no, la partida del video se apartaría de la del entrenamiento en niveles con jefes o enemigos que lo
  usan (verificado: mismo estado del azar al terminar).
- **En el gimnasio** (`entorno = "gimnasio"`): `escenarios` = salas nuevas por generación (iguales para todas las
  redes), con dificultad uniforme entre `gim_dificultad_min` y `_max`; `max_segundos` = tope por sala. Sin reintentos.
- **Currículo** (`gim_curriculo`; por defecto encendido en `rayos`, apagado en `redes`): si en las últimas
  `curriculo_gens` generaciones (todas con la dificultad actual) la población superó en promedio `curriculo_umbral` %
  de las salas, `gim_dificultad_max` sube `curriculo_paso` (hasta 1); desde `curriculo_trampas` se activan las
  trampas. Se guarda en `config.json`, en el log y en las métricas. Los workers reciben la config con cada tarea.
- **La mejor red** (videos, en vivo, pruebas, `redes/mejor.npy` + `.json`) = `almacen.indice_mejor`: la que llega más
  lejos en su primer escenario (en el gimnasio, la de más avance promedio); a igual avance, la de más fitness.
- **Partir de otra red** (`desde`, `desde_gen`): copia lo que ve y la forma de la red de esa corrida (de la misma
  línea: `Linea.forma_red`) y guarda el genoma de su mejor red en `genoma_inicial`.
- **Al continuar** no se pueden cambiar el juego, el entorno, la red de partida, lo que ve ni la forma de la red
  (`Linea.fijos`, incluido `frames_por_decision`). Sí recompensas, evolución, partidas, el gimnasio y el currículo.
- **Compatibilidad**: cada regla nueva tiene un valor "apagado" para las corridas creadas antes
  (`Config.ANTERIORES`, aplicado en `from_dict`). Las primeras corridas del gimnasio tenían `gim_salas`/`gim_segundos`:
  se convierten a `escenarios`/`max_segundos`. Un `metricas.csv` viejo se reescribe con las columnas nuevas al
  continuar.
- **Disco** (`<corridas>/<nombre>/`): `config.json`, `distancia.npz` (de rStage01, también en las del gimnasio),
  `metricas.csv`, `log.txt` (no redirigir la salida ahí), `gen_NNNN.npz` (ver `evolucion/almacen.py`), `redes/`
  (`mejor` y cada `guardar_red_cada` generaciones: pesos y genoma), `pruebas/` y `videos/`.

### Rendimiento y memoria

- Cada worker es un par Python + emulador (~1,1 núcleos), con un hilo de BLAS (`workers.un_hilo_por_worker`: sin eso
  la carga llegaba a 33 y algún worker se trababa al cerrar). En el Ryzen 7 2700 (16 hilos), 16 workers saturan la CPU.
- Al arrancar muchos emuladores a la vez, alguno puede abortar (código −6): el pool lo reemplaza solo.
- **RAM** (16 GB, sin OOM killer: sin RAM la máquina se congela en vez de matar un proceso). Cada emulador del
  gimnasio usa ~150 MB más ~11 MB por sala armada; `Gimnasio` guarda a lo más `max_salas` (8, o las salas por
  generación) y rearmar una sala da lo mismo. `resolver` guarda a lo más 32 savestates (~560 MB por emulador en vez
  de ~1,6 GB): a las demás celdas vuelve repitiendo acciones desde su ancestro guardado más cercano, que da el mismo
  resultado. Antes de lanzar pruebas junto a un entrenamiento, mirar `MemAvailable`.

## Gimnasio (`evolucion/gimnasio.py`)

Salas de una pantalla (25 × 19 celdas de 32 px) armadas en `rStageBace` (una sala del juego casi vacía) creando con
GML las instancias de los niveles y los tiles de ladrillo de rStage01, con fondo `backgroundIWBTG`. Física y
colisiones exactas. Construir una sala ~1 ms; cada intento parte de un savestate (cargar ~17 ms, guardar ~11 ms).

**Generador** (`generar(semilla, dificultad, salida_borde, trampas, trampas_min)`): el jugador a la izquierda, la
salida a la derecha y el suelo avanzando por obstáculos según la dificultad (0 a 1), con medidas según la física del
jugador (salto 8,5 → ~90 px, doble 7 → ~150 px en total, 3 px por frame, gravedad 0,4).

| Obstáculo | Desde | |
|---|---|---|
| plano, escalón (±1–4), púas (1–3), pozo de púas (1–5) | 0 | crecen con la dificultad |
| muro (3–4 de alto) | 0,3 | doble salto |
| techo de púas a 4 celdas | 0,4 | pasa el salto simple, no el doble; desde 0,7 con una púa en el suelo |
| islas (pozo ancho con bloques sueltos) | 0,5 | |
| plataforma móvil | 0,6 | |
| **púa que salta** (trampa) | 0,2 | una del fondo de un pozo sube al llegar al borde: esperar y saltar |
| **púa o cereza que cae** (trampa) | 0,3 | cuelga sobre el camino; el disparador va tantas columnas antes como camina el jugador mientras cae: quien sigue derecho llega justo debajo |
| **púa lateral** (trampa) | 0,4 | al final de un pasillo, se desliza por el suelo hacia el jugador: saltarla |

- **Trampas** (`gim_trampas`): las piezas de los niveles, un disparador invisible `triggerNN` y una púa o cereza
  `spikeTriggerNN`/`cherryTriggerNN`. Cada trampa de una sala usa otro número. Los bloques falsos (`blockDisappear`)
  quedaron fuera: la grilla los ve como suelo (haría falta otro canal).
- **Trampas mínimas** (`gim_trampas_min`): con trampas, cada sala trae al menos tantas, a cualquier dificultad; si un
  obstáculo largo le quitó el lugar, se genera otra con un azar derivado de la semilla.
- **Salida** (`gim_salida_borde`): una abertura de 3 celdas en la pared derecha a la altura del suelo final, con un
  `warp` en cada celda (`onCollide = false`: llegar se detecta en Python, `Gimnasio.en_meta`). Sin ella, un portal a 2
  celdas de la pared (las primeras corridas: las redes lo pasaban de largo).
- **Versiones**: cada generación guarda `salida_borde`, `trampas` y `trampas_min`, y `visualizar.salas_de` reconstruye
  sus salas igual. **Al agregar obstáculos, hacerlo detrás de otra regla así**: cambiar el catálogo cambia el azar de
  todas las salas y rompería el en vivo de las generaciones guardadas.
- **Avance**: BFS en celdas de 16 px desde la salida (sin física), como fracción del camino desde el inicio.
- **¿Tiene solución?** `resolver`: búsqueda tipo Go-Explore (celdas de 32 px, elegidas con peso
  exp(6·avance)/√(1+visitas), 2–12 acciones al azar de 2–16 frames). Con 600 iteraciones resuelve ~11 de 12 salas en
  2–20 s cada una. Al entrenar no se verifica: las salas sin solución le tocan igual a todas.
- **Políticas**: `PoliticaRed` (una red de la línea; con su carpeta, la mejor de una generación), `PoliticaAzar` y
  `PoliticaGrabada` (p. ej. la solución de `resolver`).

## Probar

En un nivel, la mejor red juega `--escenarios` partidas por sala con esperas fijas repartidas entre 0 y `espera_max`;
en otra sala, el objetivo es la siguiente y su mapa se arma la primera vez en `<corridas>/_salas/`. rStage02 y rStage03
matan casi al empezar con trampas ocultas. En el gimnasio, salas desde la semilla 1.000.000 (no son las del
entrenamiento), con la duración, la paciencia y las reglas de salas de la corrida. Resultados en `pruebas/*.json` (la
app los muestra en una tabla). **Para medir el progreso usar un examen fijo así**: el % de salas por generación del
log sube y baja mucho según qué tan difíciles salieron.

Plan por etapas: (1) gimnasio desde cero con currículo; (2) antes de cada nivel, su primer intento
(`probar --sala rStageNN`); (3) seguir en ese nivel con `--desde`; (4) pendiente: un entorno «mixto» (gimnasio +
niveles ya superados en cada generación) para que no olvide (necesita varios emuladores por worker o armar varios
niveles en uno: cuidado con la RAM); (5) medir la generalización con `probar --sala todas`.

## Videos, en vivo y transmisión

- **La mejor red** (nivel o gimnasio): el juego a 1280×960 (con los rayos dibujados, en `rayos`) y un panel con
  tiempo, avance, puntaje (nivel: en curso con su desglose; gimnasio: sala, obstáculos e historial de salas superadas),
  el gráfico de fitness por generación y la preferencia por acción.
- **Todas juntas en el nivel** (`vivo`, `generacion`): las trazas del primer escenario dibujadas con
  `ga.visualizar.cuadros_generacion`.
- **Entrenamiento en vivo del gimnasio** (`entreno`): las salas de una generación tal como las jugó la población. «La
  mejor»: emulada, con la nota de cómo le fue al entrenar (avisa si al repetirla termina distinto). «Todas»: la mejor
  emulada (así se mueven las plataformas) y las demás como fantasmas desde sus trazas, con cruces donde murieron y los
  gráficos de fitness y de salas superadas. «Alternar»: una sala de cada. En vivo repite la última generación y pasa a
  la nueva.
- **Prueba con salas nuevas** (`prueba`): la mejor red en salas que nunca vio, una tras otra: mide si generaliza.
- **Ritmo**: todo a 25 cuadros/s, uno cada `2·velocidad` frames de juego según el contador del emulador
  (`pantallas.Ritmo`): a ×N se emula más juego por cuadro, no más cuadros. A ×1 llega a tiempo real aun con la CPU
  ocupada por un entrenamiento.
- Los en vivo y la transmisión corren con tope de memoria (`medios.con_tope_de_memoria`, 1.500 MB) y su ventana se
  vigila (`medios.EnVivo`).
- **Transmisión** (`transmitir`): el destino va en `~/.config/iwbtgb/transmision.json` (`{"url": "rtmp://…",
  "clave": "…"}`, nunca en el repo) o en `IWBTGB_RTMP`. Envía a ffmpeg 25 cuadros/s fijos desde un hilo (espera si el
  juego va adelantado, repite el último cuadro si se atrasa), con audio en silencio en tiempo real (YouTube lo exige),
  720p a 3000 kb/s o 1080p a 6000 kb/s y el rótulo «EN VIVO»; reconecta con espera creciente. Con `--corrida auto`
  sigue a la corrida de la línea que se esté entrenando. `--destino archivo.flv` sirve para probar (la app tiene
  «Probar 2 min a un archivo»).

## App (`python -m iwbtgb.redes|rayos app`)

Tres pestañas: **Crear y entrenar** (corrida, dónde practica con el gimnasio y el currículo, partidas, lo que ve y la
red de la línea, recompensas, evolución; solo se muestran los campos del entorno elegido); **Probar** (por defecto el
examen de 40 salas); **Videos, en vivo y transmisión** («Quién juega» × «Dónde», muestras de salas, gráficos y el grupo
de transmisión). Detecta y detiene (SIGINT) los entrenamientos lanzados desde la terminal (busca en `/proc`). La app
abierta usa el código con el que se abrió: tras cambiar la config o la app, cerrarla y reabrirla.

## Corridas

### `redes` en el nivel (rStage01 → rStage02)

| Corrida | Red (grillas/f1/f2/ocultas, decide cada) | Población, sigma | Reglas | Gens | Récord |
|---|---|---|---|---|---|
| `red01` | 2/8/16/64, 2 | 200, 0,02 | 3 escenarios de 30 s | 169 | 8,5 % (gen 76) |
| `red02` | 2/8/16/64, 2 | 200 (50 padres), 0,05 | igual, para probar diversidad | 378 | 8,7 % |
| `red03` | 5/8/16/33, 1 | 100, 0,01 | 2 escenarios de 60 s | 65 | 7,2 % |
| `red04` | 3/8/16/32, 1 | 200, 0,02 | reintentos, muerte 1: aprendió a quedarse quieta → castigo por quieto | 11 | 3,9 % |
| `red05` | 2/8/16/32, 2 | 100, 0,02 | reintentos, muerte 3, quieto 1, 20 s | 790 | 8,8 % (sin cambios desde la ~350; clones) → premio por explorar |
| `red06` | 5/8/16/32, 2 | 300, 0,05 | reintentos, muerte 5, explorar 5 → 1, 120 s | 44 | 8,7 % |
| `red07` | 5/8/8/32, 1 | 200, 0,05 | sin reintentos, 30 s, explorar 1 | 145 | 7,3 % |

### `redes` en el gimnasio

| Corrida | Red | Partidas | Dificultad, salida, trampas | Gens | Resultado |
|---|---|---|---|---|---|
| `gym01` | 5/8/16/64, 1 | 8 salas × 15 s | 0–0,3, portal, no | 13 | ~2 % de salas |
| `gym02` | 5/8/16/32, 1 | 4 × 15 s, sigma 0,025 | 0–0,3, portal, no | 76 | 12 → 19 → 13 % de la población; la mejor 2,8 de 4 |
| `gym03` | 2/4/8/16, 2 | 8 × 15 s, sigma 0,025, 200 redes | 0,3–0,8, pared, no | 331 | ~30 % de la población; la mejor 6 de 8; gen 330: 14 de 40 salas difíciles |

**Barrido del gimnasio (2026-10-27)**: población 60, 25 generaciones, dificultad 0–0,3, salida en la pared; la mejor
red de cada una en las mismas 40 salas fijas (semillas 5.000.000+). A (5/8/16/32, decide cada frame): 12 de 40; B
(sigma 0,05): 11; **C (red chica 2/4/8/16, decide cada 2 frames): 15, la mitad de tiempo**; D (12 salas por
generación): 9. Las diferencias son del orden del ruido (±3); todas se aplanan en ~20 % de la población tras ~10
generaciones.

### `rayos`

| Corrida | Entorno | Red | Resultado |
|---|---|---|---|
| `rayos01` | gimnasio, currículo; desde la gen. 13 en 0,5–1 con trampas | 64,32,32 relu | gen. 106: 17 de 40 salas difíciles |
| `rayos02` | rStage01 desde rayos01 (gen. 358) | 64,32,32 | 8,7 % en 468 gen. |
| `rayos03` | rStage01 desde cero; muerte 5, explorar en celdas de 8 px, sigma 0,015 | 16,16 tanh | 8,9 % desde la gen. 351; detenida en la 1067. El fitness siguió subiendo solo por explorar el comienzo |
| `rayos04` | gimnasio desde cero con currículo (llegó a 0–1), trampas, al menos una por sala | 64 | 644 gen.: 57,6 % de las salas de la población, la mejor 8 de 8 |
| `rayos05` | rStage01 desde rayos04 | 64 | 8,5 % en 168 gen. |
| `rayos06` | rStage01 desde cero | 100,100,100 | 8,7 % en 101 gen. |
| `rayos07` | rStage01 desde cero | 100,100 | 7,5 % en 23 gen. |

**Comparación con la CNN (2026-10-28)**, en las condiciones del barrido: rayos 902→64→32→12 (60.268 pesos) superó
**24** de 40 (avance medio 75,5 %); rayos con una capa (902→32→12) 5; la mejor CNN 15. La segunda capa importa. El %
de salas de la población por generación queda parecido (~15 %): la diferencia está en la mejor red, que generaliza
mucho mejor.

## Pendiente

- Entorno mixto (gimnasio + niveles superados); probar la transmisión contra YouTube o Twitch.
- Premio por llegar rápido solo a las que llegan (hoy `r_rapidez` solo desempata).
- Gimnasio: salas de varias pantallas y verticales, salidas en otros bordes, saves a los que hay que disparar,
  bloques falsos, biblioteca de salas verificadas.
- Memoria entre muertes (las trampas ocultas de rStage02/03 no se pueden anticipar sin ella), o PPO.
