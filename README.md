# I wanna be the GB en Linux

Port nativo a Linux de *I wanna be the GB*, un fangame japonés de 2009 de *I Wanna Be the Guy* hecho en
**GameMaker 7.0**. Se logró con ingeniería inversa del ejecutable, e incluye un entorno
[Gymnasium](https://gymnasium.farama.org/) para aprendizaje reforzado.

No se reescribió el juego. Se usa [OpenGMK](https://github.com/OpenGMK/OpenGMK), una reimplementación en Rust
del runtime de GameMaker 8, al que se le agregaron:
- un lector de ejecutables GM7;
- compatibilidad con la API de GM7;
- un modo de control externo para RL.

El código GML original corre tal cual.

## Uso rápido

```bash
./scripts/build.sh        # clona OpenGMK si falta, aplica el parche y compila (requiere cargo)
./scripts/jugar.sh        # jugar con ventana y audio (controles en game/ReadMe.txt: Shift salta, Z dispara)
./scripts/decompilar.sh   # regenera decompiled/i_wanna_be_the_GB.gmk
uv run python -I tools/export_gml.py game/i_wanna_be_the_GB.exe decompiled/gml   # código GML como texto
uv sync && uv run python examples/random_agent.py                                  # entorno RL
uv run python examples/ver_agente.py                                               # ver jugar al agente
```

## Estructura

| Ruta | Contenido |
|---|---|
| `game/` | Ejecutable original, ReadMe (Shift-JIS) y `save/` (partidas guardadas que crea el juego) |
| `decompiled/i_wanna_be_the_GB.gmk` | Proyecto editable en formato GameMaker 8.0 |
| `decompiled/gml/` | Todo el código del juego en texto UTF-8 (`scripts/`, `objects/`, `rooms/`, `timelines/`) |
| `vendor/OpenGMK/` | Clon de OpenGMK (commit `9e4282e`) con el parche aplicado |
| `patches/opengmk-gm7-y-rl.patch` | Todos los cambios a OpenGMK |
| `iwbtgb_rl/` | Paquete Python: cliente del emulador y entorno Gymnasium |
| `examples/` | Agente aleatorio, entornos en paralelo y visualización/grabación del agente |
| `tools/export_gml.py` | Exportador de GML, que lee el `.exe` GM7 directamente |
| `docs/formato-gm7.md` | **Detalle de la ingeniería inversa**: formato, cifrados y direcciones del runner |
| `docs/entorno-rl.md` | **Guía del entorno RL**: parámetros, protocolo, rendimiento y pendientes |
| `ga/` | Algoritmo genético: entrenamiento, videos, gráficos y una interfaz Qt (`uv run python -m ga.app`) |
| `corridas/` | Resultados de cada entrenamiento (no versionados; enlace al disco de datos) |

## Qué se hizo

1. **Identificación**: el `.exe` es el runner de GameMaker 7 con el juego anexado en el offset 1980000.
   Herramientas como REA, Ghidra o IDA no aplican directamente, porque la lógica no es código nativo sino
   datos.
2. **Descifrado del contenedor GM7** (`gm8exe/src/gm7.rs`). Los assets van en un blob zlib cifrado con
   GMKrypt. Los scripts tienen un segundo cifrado con seed fija 12345, encontrado al desensamblar el runner con
   `objdump`. Los sprites vienen sin alpha ni máscaras de colisión, y se reconstruyen. Ver
   `docs/formato-gm7.md`.
3. **Compatibilidad de API**: `sprite_replace` con la firma de 9 argumentos de GM7. Sin esto, el juego se
   trababa en el título en cuanto existía una partida guardada.
4. **Modo RL** en el emulador (`--rl-socket`) y entorno `IWannaBeTheGB-v0`. Ver `docs/entorno-rl.md`.

## Estado

- **El juego funciona en Linux**: título, menú, partidas guardadas y niveles. No se ha jugado completo, así
  que podría haber otras diferencias GM7/GM8 en partes no recorridas.
- **El entorno RL funciona**: es determinista, detecta la muerte, admite paralelismo y video.
- **Algoritmo genético**: dos experimentos en `rStage01` (exploración y distancia a la salida); ninguno llegó
  todavía a la salida. Detalle en `docs/entorno-rl.md`.
- **Edición**: el `.gmk` se puede abrir con [LateralGM](https://github.com/IsmAvatar/LateralGM) o con
  GameMaker 8.0 vía Wine, pero no se ha probado. El emulador solo ejecuta `.exe`, así que para jugar una
  versión editada hay que compilarla con GameMaker 8.0. Otra opción es modificar el juego en caliente con
  `exec()` (ver la guía RL).

## Nota legal

*I wanna be the GB* es freeware de terceros. Este trabajo es para uso personal, investigación e
interoperabilidad. No redistribuyas el ejecutable ni sus assets o código decompilados sin permiso del autor.
