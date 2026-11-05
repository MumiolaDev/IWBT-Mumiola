# I wanna be the GB en Linux, y algoritmos que aprenden a jugarlo

Port nativo a Linux de *I wanna be the GB*, un fangame japonés de 2009 de *I Wanna Be the Guy* hecho en **GameMaker
7.0**, y tres formas de entrenar a un agente para que lo pase. El objetivo final es transmitir en vivo al algoritmo
entrenando y pasándose el juego.

No se reescribió el juego: corre su código GML original sobre [OpenGMK](https://github.com/OpenGMK/OpenGMK), una
reimplementación en Rust del runtime de GameMaker 8, al que se le agregaron un lector de ejecutables GM7, la API de
GM7 que cambia y un modo de control externo para entrenar (`docs/emulador.md`).

## Uso rápido

```bash
./scripts/build.sh        # clona OpenGMK si falta, aplica el parche y compila (requiere cargo)
./scripts/jugar.sh        # jugar con ventana y audio (controles en game/ReadMe.txt: Shift salta, Z dispara)
uv sync                   # dependencias de Python (numpy, gymnasium, pillow, matplotlib, PySide6)

uv run python -m iwbtgb.ga app        # cada línea tiene su app (también en el menú de aplicaciones)
uv run python -m iwbtgb.redes app
uv run python -m iwbtgb.rayos app
```

Todo se usa como `python -m iwbtgb.<línea> <comando>`; sin comando, cada línea muestra los suyos y cada comando
tiene `--help`. Las apps lanzan esos mismos comandos como procesos aparte, así que lo que se hace en una se puede
seguir desde la terminal y al revés.

## Las tres líneas

| Línea | Qué aprende | Estado (2026-11-05) | Detalle |
|---|---|---|---|
| `iwbtgb.ga` | Una secuencia fija de teclas (un TAS) para un nivel, con algoritmo genético; o un archivo de celdas (Go-Explore) | La que más avanza: 75,7 % del camino de rStage01 (con save heredado), 47 % en una partida continua | `docs/ga.md` |
| `iwbtgb.redes` | Una red (CNN) que decide mirando una grilla semántica del juego, con Deep GA | 7–8,7 % de rStage01; ~30 % de las salas del gimnasio | `docs/redes.md` |
| `iwbtgb.rayos` | Una red densa que ve con 64 rayos, con el mismo Deep GA | La que mejor generaliza (24 de 40 salas nuevas); 8,7–8,9 % de rStage01. Tiene transmisión en vivo | `docs/redes.md` |

Las dos redes comparten todo salvo lo que ve la red y la red misma (`iwbtgb.evolucion`): el gimnasio de salas
generadas dentro del juego, las partidas en un nivel, el entrenamiento, las pruebas, los videos, la transmisión y la
app.

```bash
uv run python -m iwbtgb.ga entrenar --nombre salida01 --poblacion 50 --generaciones 300
uv run python -m iwbtgb.ga visualizar todo --corrida salida01 --cada 10
uv run python -m iwbtgb.rayos entrenar --nombre rayos08 --entorno gimnasio
uv run python -m iwbtgb.rayos probar --corrida rayos08 --gimnasio --escenarios 40       # examen en salas que no vio
uv run python -m iwbtgb.rayos visualizar entreno --corrida rayos08 --quien alternar     # el entrenamiento en vivo
uv run python -m iwbtgb.rayos transmitir --corrida auto                                 # stream RTMP
```

## Estructura

| Ruta | Contenido |
|---|---|
| `iwbtgb/emulador.py` | Cliente del emulador (`Emulator`) y arranque del juego hasta el comienzo de una sala (`arrancar`) |
| `iwbtgb/distancia.py` | Mapa de distancias a la salida por el laberinto (Dijkstra): la noción de avance de todas las líneas |
| `iwbtgb/traza.py`, `opciones.py`, `medios.py`, `workers.py`, `qt.py`, `comandos.py` | Lo común: trazas por frame, opciones de línea de comandos, videos y ventanas en vivo, workers, piezas de las apps y despacho de comandos |
| `iwbtgb/gm7.py` | Lector del `.exe` GM7: exporta el GML (`python -m iwbtgb.gm7 <exe> <carpeta>`) y lee sprites |
| `iwbtgb/entorno.py` | Entorno Gymnasium (`IWannaBeTheGB-v0`) para RL por gradiente; todavía no lo usa ninguna línea |
| `iwbtgb/ga/` | Algoritmo genético con TAS, Go-Explore, videos y su app |
| `iwbtgb/evolucion/` | Neuroevolución común a las dos redes |
| `iwbtgb/redes/`, `iwbtgb/rayos/` | Lo propio de cada red: lo que ve, la red, su dibujo y su sección de la app |
| `docs/` | `emulador.md` (formato GM7, cambios a OpenGMK, protocolo RL), `ga.md`, `redes.md` |
| `patches/opengmk-gm7-y-rl.patch` | Todos los cambios a OpenGMK (`vendor/OpenGMK`, commit `9e4282e`) |
| `scripts/` | `build.sh`, `jugar.sh`, `decompilar.sh` (genera `decompiled/i_wanna_be_the_GB.gmk`) |
| `game/` | Ejecutable original, ReadMe (Shift-JIS) y `save/` (partidas guardadas del juego) |
| `corridas/`, `corridas_redes/`, `corridas_rayos/` | Resultados de cada línea (enlaces al disco de datos, no versionados) |

## Estado del port

- El juego funciona en Linux: título, menú, partidas guardadas y niveles. No se ha jugado completo, así que podría
  haber otras diferencias GM7/GM8 en partes no recorridas.
- El modo de control es determinista (la misma secuencia de teclas desde un savestate da los mismos frames) y corre
  sin ventana a ~5.000 frames/s en salas chicas y ~700–1.000 en rStage01.
- El `.gmk` decompilado se puede abrir con [LateralGM](https://github.com/IsmAvatar/LateralGM) o con GameMaker 8.0
  vía Wine (no probado).

## Nota legal

*I wanna be the GB* es freeware de terceros. Este trabajo es para uso personal, investigación e interoperabilidad. No
redistribuyas el ejecutable ni sus assets o código decompilados sin permiso del autor.
