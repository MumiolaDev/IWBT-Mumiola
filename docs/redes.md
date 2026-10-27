# Redes neuronales (`redes/`)

Proyecto aparte del GA con TAS (`ga/`). El objetivo es una red que **generalice**: que juegue niveles que no vio,
idealmente de otros fangames de IWBTG. El GA produce una secuencia fija de teclas para una partida determinista;
la red, en cambio, decide cada pocos frames mirando el juego. Comparte con el GA el emulador y `iwbtgb_rl`, pero
tiene sus propias corridas (`corridas_redes/` → `/mnt/datos/iwbtgb/corridas_redes`), app y videos.

Plan: vista semántica → genoma "red" con neuroevolución (Deep GA: mutación gaussiana de los pesos, élite y
nichos) → evaluar cada red en varias situaciones (saves, perturbaciones, varias salas) para que no memorice →
probar en salas que no vio. Después, PPO y memoria entre muertes para las trampas troll.

## Vista semántica (`redes/vista.py`)

Lo que ve la red: una grilla de 64×48 celdas de 8 px (512×384 px) centrada en el jugador, con un canal por tipo
de cosa. Cada celda vale la fracción (0–255) de sus 4×4 puntos de muestra que cae dentro de la **máscara de
colisión** de algún objeto de ese tipo, así que las púas son triángulos. Más hspeed, vspeed y `djump` del jugador.

| Canal | Objetos (con sus hijos) | Notas |
|---|---|---|
| sólido | `block` salvo `platform` | `block` es invisible: es la colisión de los tiles |
| plataforma | `platform` (incluye `movingPlatform`) | se atraviesa desde abajo |
| mortal | `playerKiller` | solo instancias visibles: las trampas ocultas no se ven, como para un humano |
| save | `savePoint` | |
| salida | `roomChanger` | invisible, pero es la salida |
| agua | `water` | |
| fuera de la sala | — | lo calcula Python con el tamaño de la room |

En el GB, `killPlayer` solo se llama al tocar un `playerKiller` (106 objetos heredan de él) o al salir de la
room. Los `trigger` (invisibles, activan trampas) no se muestran. Es la jerarquía del engine estándar de IWBTG:
otros fangames deberían servir con la misma configuración; si les falta un objeto, `Vista.faltan` lo dice.

El cálculo lo hace el emulador: `OP_GRID_CONFIG` (9) configura celda, tamaño, muestras, canales (objetos,
excluidos, solo visibles) y campos del jugador; `OP_GRID` (10) o `step(grid=True)` (flag 32) la devuelven. En el
cliente: `Emulator.grid_config`, `Emulator.grid`, `Grid`.

```bash
uv run python -m redes.vista probar                              # canales al comienzo de rStage01 y velocidad
uv run python -m redes.vista video --corrida GA_beta_corregida   # el mejor del GA con su vista al lado
```

Costo: ~20 % de velocidad por frame con grilla (medido con la máquina cargada, 161 vs 200 frames/s con teclas al
azar). Se puede bajar cacheando los canales estáticos.

## Núcleo: neuroevolución (`redes/entrenar.py`)

```bash
uv run python -m redes.entrenar --nombre red01 --workers 10                 # opciones = campos de redes/config.py
uv run python -m redes.entrenar --nombre red01 --continuar --generaciones 2000
uv run python -m redes.visualizar mejor --corrida red01 [--gen -1] [--espera 0]
```

- **Red** (`modelo.py`, numpy, sin GPU): grilla (7 canales × `cuadros` = 2, la actual y la de la decisión
  anterior) → conv 4×4 paso 4 (8 filtros) → conv 3×3 paso 2 (16) → + hspeed, vspeed, djump → densa 64 → 12
  acciones (la mayor). 53 156 pesos, 0,4 ms por decisión. Decide cada `frames_por_decision` (2) frames.
- **Genoma = linaje** (como *Deep Neuroevolution*): `(semilla inicial, (semilla, sigma), ...)`. Pesos = He
  inicial + suma de mutaciones gaussianas. Cada worker tiene una caché LRU de pesos y los hermanos se mandan
  juntos (el padre se reconstruye una vez). Sin caché, reconstruir 1500 mutaciones toma ~1,3 s.
- **Evaluación**: cada red juega `escenarios` (3): el comienzo de la sala y otros con `espera` al azar
  (0–`espera_max` frames sin control, nuevos cada generación). Termina al morir, salir de la sala o entrar a una
  zona prohibida, a los `max_segundos`, o tras `paciencia` s sin mejorar el avance.
- **Fitness** (promedio de escenarios): `r_avance` × mejor fracción del camino (Dijkstra de `ga/distancia.py`) +
  `r_objetivo` si llega − `r_rapidez` × frame del mejor avance / tope + `r_save` × saves nuevos (10) −
  `castigo_muerte` × muertes (3; también cuentan salir de la sala y atascarse). La selección es por ranking: solo
  importan las proporciones entre términos, no la escala.
- **Castigo por estar quieto** (`castigo_quieto` = 2 por segundo, desde 2026-10-26): `red04` (castigo por muerte
  1) aprendió a esperar el fin del tiempo (su mejor red pasaba 12,9 de 20 s quieta). Quieto = el jugador no se
  desplazó ≥ 0,5 px desde la decisión anterior (`QUIETO_PX`). No se usa `hspeed`/`vspeed`: sobre una
  `movingPlatform` el juego mueve al jugador sin cambiarle la velocidad (medido: hspeed ≈ 0, 2 px por decisión).
  Empujar una pared sí cuenta como quieto; saltar en el lugar no, pero la paciencia lo corta (R, castigo de muerte).
  Métrica `quieto_pct`. `metricas.csv` viejos se reescriben con las columnas nuevas al continuar.
- **Reintentos** (`reintentos` = 1, desde 2026-10-26): morir, salir o atascarse (`paciencia` sin avanzar) no
  terminan la partida: espera `frames_reset`, aprieta R y vuelve al último save hasta que se acaba el tiempo. Tras
  reaparecer espera al azar 0–`espera_reintento` frames sin control: **sin esa espera cada vida repite la anterior
  exacta** (la red no tiene memoria y R reinicia la sala igual; medido: 5 muertes seguidas en el mismo píxel). Con
  reintentos el nicho es el lugar del mejor avance. Cuesta ~8× por red (las partidas duran siempre el tope).
- **Saves con premio**: los `savePoint` de la sala salvo los del comienzo (avance < 5 %) y los de zonas
  prohibidas; en rStage01 quedan 5 (17, 49, 60, 76 y 91 % del camino). Se detecta el cambio del archivo de partida
  (`ga.nucleo.leer_save`) y cuál save tiene `image_index >= 1`.
- **Premio por explorar** (`r_explorar` = 0,5 por celda, `celda_explorar` = 32 px, desde 2026-10-26): suma por
  cada celda distinta de la sala por la que pasó vivo en la partida (todas las vidas; fuera de zonas prohibidas;
  se mira en cada decisión). Motivo: `red05` se estancó ~400 generaciones en 8,7 %, en la trampa
  `spikeTrigger01` (736, 288): una púa que sube a 10 px/frame por el pozo de x ≈ 760. Caer derecho da todo el
  avance hasta ahí y desviarse a la izquierda (lo que hace el GA) no suma nada al principio: óptimo local
  engañoso, con las mejores redes convertidas en clones. Con el premio, probar otro camino vale aunque no acerque
  a la salida. Métricas `celdas_media` y `celdas_max`; 0 en corridas anteriores.
- Corridas viejas: `config.ANTERIORES` deja apagadas las reglas que no existían (castigo 0, save 0, sin reintentos).
- **Un hilo de BLAS por worker** (`OPENBLAS_NUM_THREADS=1` antes de crear el pool): sin eso cada worker usaba
  ~120 % de CPU (carga 33 con 10 workers) y a veces uno se trababa al cerrar. Al terminar normal: `pool.close()`
  y `join()` (sin el SIGTERM de `terminate`). `red01`–`red03` se lanzaron antes del arreglo.
- **Rendimiento** (medido el 2026-10-26 con la config de `red06`): jugando vivo, el emulador tarda ~1–1,4 ms por
  frame (rStage01 tiene ~970 instancias, 722 `block`, y OpenGMK busca colisiones recorriéndolas todas, sin índice
  espacial); la grilla, ~0,45 ms por decisión; la red, ~0,7–1 ms por decisión. **Muerto, ~9–11 ms por frame**
  (la sangre): los `frames_reset` antes de R eran el 60 % del tiempo con redes que mueren mucho. Desde entonces
  no se emulan (`Jugador.muerto`): R reinicia la sala, así que no cambian nada; solo suman al reloj y a la traza
  (registros con `alive` = 0 en el último lugar). Verificado: mismas acciones, avance, muertes y celdas que lo
  grabado por `red06` con el código anterior (8/8), y 16/16 partidas iguales emulando o no. Los videos (con
  `al_frame`) sí los emulan, para que se vea la muerte. La red recorta cada grilla en parches una sola vez
  (`Red.parches`, concatenados en `salidas_parches`; mismos números que antes). Cada worker es un par Python +
  emulador que se turnan: ~1,1 núcleos.
- **"La mejor red"** (videos, en vivo, pruebas, `redes/mejor.npy`; desde 2026-10-26) = la que llega más lejos en
  su primer escenario, a igual avance la de más fitness (`almacen.indice_mejor`). La selección sigue por fitness.
  Con `elite_mas_lejos` (1; 0 en corridas anteriores) esa red pasa siempre como élite y entra a los padres,
  aunque su fitness no esté entre los mejores.
- **Selección**: los `padres` mejores, primero el mejor de cada nicho (cuadro de `nicho_px` donde terminó el
  primer escenario); `elite` pasan sin cambios pero se vuelven a evaluar; el resto son hijos (padre al azar +
  mutación de `sigma`). Sin cruce.
- **Disco** (`corridas_redes/<nombre>/`): `config.json`, `distancia.npz`, `metricas.csv`, `gen_NNNN.npz`
  (linaje `padre`/`semilla`/`sigma`, fitness, avances y fines por escenario, posición final, esperas) y
  `redes/` (`mejor.npy` + `.json` con el genoma; y cada `guardar_red_cada` generaciones). `almacen.genomas`
  reconstruye la población de cualquier generación (verificado: mismos pesos que `mejor.npy`).
- **Acciones grabadas**: cada `gen_NNNN.npz` guarda también `acciones` (índice de `ACTIONS` por decisión del
  primer escenario, relleno con 255): con `esperas[0]` se puede repetir la partida de cualquier red en el emulador
  sin reconstruir sus pesos (para una vista "todas las redes juntas" o el stream). Las generaciones anteriores al
  2026-10-26 no lo tienen. También `celdas` por escenario (con `r_explorar`).
- **Log**: `entrenar` escribe cada línea también en `corridas_redes/<nombre>/log.txt` (no redirigir la salida a
  ese mismo archivo).
- **Al continuar** no se pueden cambiar la sala, la dificultad ni la forma de la red (`entrenar.FIJOS`).

## App (`redes/app.py`)

```bash
uv run python -m redes.app      # también en el menú: ~/.local/share/applications/iwbtgb-redes.desktop
```

Aparte de la del GA, pero se usa igual (reutiliza `Proceso` y `spin` de `ga.app`). Pestañas:

- **Crear y entrenar** (rediseñada el 2026-10-27): grupos numerados 1 · Corrida (nombre, partir de la red de
  otra corrida), 2 · Dónde practica (nivel o gimnasio, con una explicación), 3 · Partidas de cada red, 4 · La red
  (con una descripción en palabras: «detectores de patrones chicos/grandes», «neuronas de decisión»), 5 ·
  Recompensas y 6 · Evolución. Solo se muestran los campos que usa el entorno elegido (`QFormLayout.setRowVisible`):
  en el gimnasio no aparecen sala, espera, reintentos, quieto, saves ni exploración, y «Escenarios por red» pasa a
  «Salas por generación». Al cambiar de entorno se sugieren valores (gimnasio: 8 salas, 15 s, paciencia 4 s).
  Entrenar desde cero, continuar y detener; detecta los `redes.entrenar` lanzados desde la terminal (busca en
  `/proc`) y también los puede detener (SIGINT). Barra y resumen se actualizan cada 5 s desde `metricas.csv`.
- **Probar**: «Dónde»: donde entrenó, el gimnasio (N salas, dificultad, solo con solución), todos los niveles o
  uno; corre `redes.probar` y muestra en una tabla los resultados guardados en `pruebas/`.
- **Videos y en vivo** (absorbió la pestaña «Gimnasio»): «Quién juega» (la mejor red, todas las de una
  generación, al azar o la solución de la búsqueda) × «Dónde» (donde entrenó, gimnasio u otro nivel); botones en
  vivo, grabar video, muestras de salas, gráficos y carpetas. La app elige `redes.visualizar` o
  `redes.gimnasio_ver` según la combinación.

## Probar (`redes/probar.py`, `redes/salas.py`)

```bash
uv run python -m redes.probar --corrida red01                                  # su sala, 10 esperas
uv run python -m redes.probar --corrida red01 --sala rStage02 rStage03 [--max-segundos 60] [--video]
uv run python -m redes.probar --corrida red01 --sala todas
```

La mejor red de la generación juega `--escenarios` partidas por sala con esperas fijas repartidas entre 0 y
`espera_max` (iguales en cada prueba, para comparar). En otra sala el objetivo es la siguiente (`rStageNN+1`) y su
mapa de distancias se arma la primera vez en `corridas_redes/_salas/<dificultad>_<sala>_<objetivo>/`. Resultado
en `corridas_redes/<nombre>/pruebas/<sala>_genNNNN.json`. Ojo: rStage02 y rStage03 matan casi al empezar con
trampas ocultas (en rStage02, una pared de púas aparece al saltar a la derecha): una red sin memoria no las puede
anticipar.

## Videos, gráficos y en vivo (`redes/visualizar.py`)

```bash
uv run python -m redes.visualizar mejor    --corrida red01 [--gen -1] [--espera 0] [--sala rStage02]
uv run python -m redes.visualizar generacion --corrida red05 [--gen -1] [--velocidad 1]   # todas juntas
uv run python -m redes.visualizar vivo     --corrida red05 [--velocidad 4]       # todas juntas, ventana
uv run python -m redes.visualizar vivo-red --corrida red01 [--sala rStage02]   # la mejor red, ventana
uv run python -m redes.visualizar graficos --corrida red01     # avance.png, fitness.png, fines.png
```

**Todas las redes juntas** (`generacion`, `vivo`, desde 2026-10-26): cada gen guarda la traza por frame del primer
escenario de cada red (`trazas` + `traza_largo`, `ga.nucleo.TRAZA_DTYPE`, ~15 KB por gen comprimido con 16 redes
de 8 s). Se dibuja con `ga.visualizar.cuadros_generacion` tal cual (mapa, cámara, minimapa, panel, gráfico): las
redes se adaptan a lo que lee (`_Red`, `_cfg_ga`). Corridas sin trazas (`red01`–`red04`): solo `vivo-red`.

`Pantalla` arma los cuadros del video y del en vivo de la mejor red (`vivo-red`): el juego a 1280×960 y un panel con
tiempo, avance, el puntaje de la partida en curso con su desglose (`episodio.desglose`, la misma fórmula del
fitness), el fitness de la red y el promedio y máximo de su generación, el gráfico de fitness y la preferencia por
acción. Sin la vista de la red (se quitó el 2026-10-26; `redes.vista video` la sigue mostrando). El en vivo
juega la mejor red en tiempo real (25 decisiones/s), partida tras partida con esperas al azar, y toma la nueva
mejor red entre partidas; corre con el mismo tope de memoria que el del GA
(`ga.visualizar._con_tope_de_memoria("redes.visualizar")`).

Pendiente: vista "todas las redes juntas" (con las acciones grabadas), escenarios desde saves, memoria entre
muertes (trampas), bajar el costo de la grilla.

## Gimnasio (`redes/gimnasio.py`, `redes/gimnasio_ver.py`; prototipo del 2026-10-27)

Salas generadas al azar para que una red practique situaciones variadas antes de los niveles del GB (que no ve
hasta la prueba). Motivo: las 7 corridas de Deep GA chocaron en la trampa del 8,5–8,7 % de rStage01, y entrenar
en el nivel mismo da una señal pobre (una partida larga = un número) y cara (~1 ms por frame).

- **Dentro del juego real**: `rStageBace` es una sala casi vacía (jugador, cámara, `world`, `playMusic`). Con GML
  se crean las instancias de los niveles (`block`, `spikeUp/Down/Left/Right`, `movingPlatform`) y los tiles de
  ladrillo de rStage01 (`tile01`: (0, 32) con borde donde hay aire encima, (32, 32) liso; `block` es invisible).
  Fondo `backgroundIWBTG`. La salida es un `warp` con `onCollide = false` (solo se ve): llegar se detecta en
  Python con la traza por frame y la caja del jugador (x−5…x+5, y−12,2…y+7,8). Física y colisiones exactas.
- **Velocidad**: ~5.200 frames/s por emulador (~120 instancias) contra ~900 en rStage01 (~970). Construir una sala
  tarda ~1 ms; cada intento parte de un savestate (cargar ~17 ms, guardar ~11 ms: es del emulador, no de copiar
  archivos). Al morir no hay R: se carga el savestate.
- **Generador** (`generar(semilla, dificultad)`): una pantalla de 25 × 19 celdas de 32 px (la cámara avanza de a
  800 × 608), jugador a la izquierda y salida a la derecha; el suelo avanza por obstáculos: plano, escalón (±1–4),
  púas (1–3), pozo de púas (1–5), muro (3–4 de alto, doble salto; desde 0,3), techo de púas a 4 celdas (pasa el
  salto simple, no el doble; desde 0,4; desde 0,7 con una púa en el suelo), islas (pozo ancho con bloques sueltos;
  desde 0,5) y plataforma móvil (desde 0,6). Medidas según la física del jugador: salto 8,5 (~90 px), doble 7
  (~150 px en total), 3 px por frame, gravedad 0,4.
- **Avance**: BFS en celdas de 16 px desde la salida (sin física), como fracción del camino desde el inicio.
- **¿Tiene solución?** `resolver`: búsqueda tipo Go-Explore (archivo de celdas de 32 px con savestates, se eligen
  con peso exp(6·avance)/√(1+visitas) y se prueban 2–12 acciones al azar de 2–16 frames). Con 600 iteraciones
  resuelve ~92 % de las salas (11 de 12 en una muestra con dificultad al azar) en 2–9 s cada una; las que no,
  se saltan con `--verificar`. Pesos de celda de 16 px y 1–6 acciones resolvían 1 de 9.
- **Políticas**: la mejor red de una corrida (`PoliticaRed`, con su vista), `azar` y la solución de la búsqueda
  (`PoliticaGrabada`). Al dibujar, cada decisión se emula de a 2 frames con las mismas teclas (el emulador las
  mantiene apretadas: da lo mismo), así los videos van a 25 cuadros/s fluidos.
- **Ver** (`redes.gimnasio_ver` y la pestaña «Videos y en vivo» de la app): `muestras` (mosaico de 4 columnas, con el
  camino de la solución si se verifica), `vivo` (ventana sin fin con el historial de salas superadas; la red se
  recarga si hay una nueva) y `video`. Salida en `corridas_redes/_gimnasio/` (o en `videos/` de la corrida).
- **En vivo a tiempo real** (2026-10-27): un cuadro cada `2·velocidad` frames de juego, siempre a 25 cuadros/s
  (antes una red que decide cada frame daba un cuadro por frame y la ventana iba a la mitad de la velocidad real, y
  ×N pedía N veces más cuadros por segundo); el emulador solo dibuja los pasos que se muestran; la parte fija del
  panel (sala, historial, rótulos) se arma una vez por sala. Medido con el entrenamiento de `gym01` ocupando la
  CPU: ~40 ms por cuadro a ×1 con una red (era ~65), ~25–30 ms al azar; a ×2–×4 con una red que decide cada frame
  el límite son sus decisiones (4–8 por cuadro).
- Primera prueba: la mejor red de `red07` (entrenada en rStage01) muere en las salas del gimnasio en ~1 s o se
  queda sin avanzar: no generaliza.
- **Entrenar en el gimnasio** (`entorno = "gimnasio"`, desde el 2026-10-27): el mismo Deep GA, pero cada
  generación sortea `escenarios` salas nuevas (semilla y dificultad uniforme entre `gim_dificultad_min` y `_max`)
  que juegan todas las redes; cada partida termina al llegar, morir, a los `max_segundos` o tras `paciencia` sin
  avanzar (sin reintentos). Las primeras corridas tenían `gim_salas`/`gim_segundos` aparte (la app mostraba dos
  duraciones y el en vivo usaba 30 s fijos): `Config.from_dict` los pasa a `escenarios`/`max_segundos`, y el en
  vivo y las pruebas usan ahora la duración y la paciencia de la corrida. Fitness = promedio de `desglose` por sala (avance BFS, llegar, rapidez, muerte; sin
  saves, quieto ni celdas). Los workers usan `gimnasio.JugadorGimnasio`, que devuelve las mismas
  `Partida`/`Evaluacion`, y guardan hasta 64 salas armadas (un savestate cada una). La mejor red = la de más
  avance promedio (`gen_NNNN.npz` lleva `gimnasio` y `dificultades`). Métricas `salas_pct` (salas superadas,
  % de todas las partidas) y `salas_mejor`. Sin trazas: «todas juntas» no aplica; «video de la mejor red» la
  muestra en rStage01 (prueba sin haberla visto). Prueba: 12 redes, 3 salas, ~3–5 s por generación.
- **Partir de otra red** (`desde`, `desde_gen`): la corrida nueva copia la forma de red y la vista de esa corrida
  (`entrenar.FORMA_RED`) y guarda el genoma de su mejor red en `genoma_inicial`; la gen 0 es esa red sin cambios
  más mutaciones suyas (`padre` = −1 con sigma 0 o con su semilla), y `almacen.genomas` lo reconstruye con esa
  base (verificado: mismos pesos que `mejor.npy`). Así se entrena en el gimnasio y después se sigue en rStage01.
  En la app: grupo «Dónde entrena» (entorno, salas, dificultades, segundos y «Partir de la red de»).
- Pendiente: PPO; rayos; trampas (triggers), salas de varias pantallas y verticales; biblioteca de salas
  verificadas (hoy se entrena sin verificar: ~8 % de las salas puede no tener solución, y le pasa igual a todas).

## Soltar el salto, dibujo de la red y ritmo de los videos (2026-10-27)

- **Soltar el salto** (`soltar_salto`, 1; 0 en corridas anteriores): en el juego el salto sale al *apretar* la
  tecla. Una red que elige «→↑» en decisiones seguidas la mantiene apretada y no vuelve a saltar nunca (se veía en
  las barras de preferencia). Con la regla (`gimnasio.plan_teclas`, usada por el nivel y el gimnasio), si la red
  sigue pidiendo salto con la tecla apretada y el jugador no está subiendo (`vspeed >= 0`), el primer frame de esa
  decisión va sin salto: mantenerla da salto completo, doble salto en la cima y otro salto al aterrizar. Medido en
  una sala plana con «↑» fijo: sin la regla un solo salto; con ella, 5 despegues en 200 frames.
- **Dibujo de la red** (`redes/dibujo_red.py`) en los videos y en vivo de la mejor red (nivel y gimnasio): capas
  en columnas (vista por tipo de objeto → filtros de la capa 1 → filtros de la capa 2 + velocidad y doble salto →
  ocultas → acciones); cada filtro es un nodo y el peso entre nodos es el promedio de sus pesos. Líneas fijas por
  red (naranja +, azul −, las 160 más fuertes por par de capas), círculos encendidos con la activación del paso
  (`Red.registrar` guarda `act`); con más de 32 ocultas se dibujan las 32 de más peso hacia la salida. Las líneas
  se arman una vez por red (~35 ms); cada cuadro cuesta ~1,5 ms. En el video del nivel reemplaza al mini gráfico
  de fitness (sigue en la tarjeta final y en «Gráficos»).
- **Ritmo**: los videos y el en vivo de la mejor red en el nivel van siempre a 25 cuadros/s, uno cada 2 frames de
  juego según el contador del emulador (`visualizar.Ritmo`): antes, una red que decide cada frame daba 50 cuadros
  por segundo de juego (video a la mitad de velocidad en vivo) y una que decide cada 5, 10 (video acelerado).

## Corridas

- `red01` (2026-10-26): configuración por defecto, 12 workers, ~33 s por generación. Gen 10: el mejor llega al
  6,4 % (el GA con TAS iba en ~5 % a esa altura). Se lanzó desde la terminal antes de la app (`setsid nohup`,
  log en `corridas_redes/red01/log.txt`); la app la ve y la puede detener; se sigue con `--continuar`.
  Detenida en la gen 168: estancada en 8,5 % desde la gen 76 (muere en las púas a la derecha del inicio), con
  solo 16–19 nichos.
- `red02` (2026-10-26): igual que `red01` pero con `--padres 50 --sigma 0.05`, para ver si el estancamiento es
  por falta de diversidad. Misma gen 0 (misma semilla). Primera con acciones grabadas. Lanzada desde la terminal
  (stdout a /dev/null porque `entrenar` ya escribe `log.txt`; stderr en `errores.txt`).
- `red05` (2026-10-26, lanzada por el usuario): castigo por muerte 3, quieto 1/s, 1 escenario, 20 s,
  población 100, 30 padres, sigma 0,02, `elite_mas_lejos`. Estancada en 8,73 % desde la gen ~350 (gen 751: fitness
  máx 19,96 sin cambios, 0 saves). Muere en la trampa `spikeTrigger01` del pozo derecho; las 6 mejores redes eran
  clones. Motivó el premio por explorar.
