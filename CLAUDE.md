# CLAUDE.md

Port a Linux de *I wanna be the GB* (fangame en **GameMaker 7.0**, no el IWBTG original) sobre OpenGMK,
con un entorno RL Gymnasium y un algoritmo genético para un video blog de YouTube. Responder al usuario en
español.

La app Android de fangames (hub con rankings, *I Wanna be the Guy: Remastered*) se separó el 2026-10-23 a su
propio repo: `~/Programacion/FangameHub`. Allí se trabaja en otra sesión. Cada repo tiene su propia copia de
OpenGMK y su propio parche: un cambio al emulador que sirva a ambos hay que llevarlo a mano al otro.

## Estado (2026-10-25)

- El GB corre nativo (`./scripts/jugar.sh`). El RL con PPO nunca se entrenó; el trabajo actual es un algoritmo
  genético (`ga/`) para un video de YouTube. El usuario lo maneja desde la app Qt (`ga.app`) y lanza sus propias
  corridas.
- Mecánica actual del GA (detalle en `docs/entorno-rl.md`, sección "Algoritmo genético"):
  - genoma TAS de genes `(acción, frames)`, evaluado por familias con savestates;
  - fitness por distancia a la salida por el laberinto (Dijkstra) y nichos de 64 px;
  - zona prohibida: la sala de portales de `rStage01` cuenta como muerte;
  - reintentos: R → último save, `resets`;
  - premio por save point nuevo (`r_save`);
  - save heredado: los hijos parten del save que activó su padre (`prob_desde_save`), y el video del mejor
    muestra el recorrido completo por tramos.
- Corridas: `corridas/` es un enlace a `/mnt/datos/iwbtgb/corridas` (disco de datos). Estado y lo aprendido de
  cada una en `docs/entorno-rl.md` → "Estado de las corridas". El tope del ~20,5 % era un error del emulador
  (máscara rectangular de las púas, corregido el 2026-10-25); con la corrección, una prueba llegó al 75,7 %.
  Las corridas anteriores a la corrección se repiten con el binario viejo:
  `OPENGMK_EMULATOR=/mnt/datos/iwbtgb/emulador_antiguo/gm8emulator` (o `OPENGMK_GM7_CLAVE_ESQUINA=1`).
  No borrar ni modificar corridas del usuario. Para pruebas, crear una aparte (`_prueba_*`) y borrarla al
  terminar.
- Compatibilidad: cada regla nueva tiene un valor por defecto "apagado" para corridas viejas
  (`almacen.cargar_config`, `setdefault`). La app abierta usa el código con el que se abrió. Los entrenamientos
  que lanza son procesos nuevos, así que usan el código del disco. Tras cambiar `Config`, hay que cerrar y
  reabrir la app.
- Informe del experimento (artifact): https://claude.ai/artifact/XBoYLppCkJP2CkYwzmkP8E. Cubre los exp. 1–2;
  aún no incluye reintentos ni save heredado.
- **Proyecto aparte: redes neuronales (`redes/`)**, para un modelo que generalice a otros niveles y fangames.
  Comparte el emulador y `iwbtgb_rl` con el GA, pero tiene sus propias corridas (`corridas_redes/`), app y
  videos: no mezclarlo con `ga/`. Plan y estado en `docs/redes.md`. Hecho: vista semántica (`OP_GRID`), núcleo de entrenamiento (Deep
  GA) y app (`redes.app`: entrenar, probar en otras salas, videos, gráficos y en vivo). Prototipo de gimnasio
  (`redes/gimnasio.py`): salas generadas dentro del juego real (en `rStageBace`), ~6× más rápidas, para entrenar
  antes de los niveles. Se entrena en él con `--entorno gimnasio` y se sigue en un nivel con `--desde <corrida>`
  (la app: «Crear y entrenar» → «Dónde practica» y «Partir de la red de»; ver y probar en «Videos y en vivo» y
  «Probar»). Los videos de una red la dibujan por dentro (`redes/dibujo_red.py`).
- Detalles del formato GM7 en `docs/formato-gm7.md`; guía y protocolo RL en `docs/entorno-rl.md`. Leerlos
  antes de tocar el lector GM7 o el modo RL.

## Comandos

```bash
./scripts/build.sh                     # cargo build --release de gm8emulator + gm8decompiler
cargo build -p gm8emulator --manifest-path vendor/OpenGMK/Cargo.toml   # build debug, con símbolos
./scripts/jugar.sh                     # jugar (ejecuta desde game/; el juego escribe en game/save/)
./scripts/decompilar.sh                # -> decompiled/i_wanna_be_the_GB.gmk
uv run python -I tools/export_gml.py game/i_wanna_be_the_GB.exe decompiled/gml
uv run python examples/random_agent.py # prueba rápida del entorno
uv run python examples/ver_agente.py [--video x.mp4]
uv run python -m ga.app                                           # interfaz (Qt); también en el menú de apps (~/.local/share/applications/iwbtgb-ga.desktop)
uv run python -m ga.entrenar --nombre salida01 [--continuar]       # algoritmo genético (opciones = campos de Config)
uv run python -m ga.visualizar todo --corrida salida01 --cada 10   # videos y gráficos de la corrida
uv run python -m ga.visualizar vivo --corrida salida01             # ventana (ffplay) con la última generación, mientras entrena
uv run python -m ga.explorar --nombre ge01 --workers 4 --iteraciones 2000   # Go-Explore (archivo de celdas); --continuar
uv run python -m redes.vista video --corrida GA_beta_corregida    # redes: vista semántica junto al juego
uv run python -m redes.entrenar --nombre red01 --workers 10        # redes: neuroevolución; --continuar
uv run python -m redes.visualizar mejor --corrida red01            # redes: video de la mejor red
uv run python -m redes.app                                         # redes: interfaz (Qt); en el menú: iwbtgb-redes.desktop
uv run python -m redes.probar --corrida red01 --sala todas         # redes: probar la mejor red en salas que no vio
uv run python -m redes.visualizar vivo --corrida red05             # redes: todas las redes juntas, como el GA (ventana)
uv run python -m redes.visualizar vivo-red --corrida red01         # redes: la mejor red jugando en tiempo real (ventana)
uv run python -m redes.gimnasio_ver muestras --dificultad -1 --verificar   # gimnasio: salas generadas (imagen)
uv run python -m redes.gimnasio_ver vivo --agente red07            # gimnasio: una red (o azar/solucion) en salas nuevas
uv run python -m redes.gimnasio_ver entreno --agente gym02 --quien todas   # gimnasio: el entrenamiento en vivo (mejor/todas)
```

Rust: se usa el rustc del sistema (1.99, sin rustup). `rust-toolchain.toml` pide 1.77, pero sin rustup se
ignora. Python: `uv` con `.venv` (numpy, gymnasium, pillow, matplotlib y pyside6-essentials).

## Cambios a OpenGMK (`vendor/OpenGMK`, base `9e4282e`)

| Archivo | Cambio |
|---|---|
| `gm8exe/src/gm7.rs` (nuevo) | Lector GM7. Devuelve `GameAssets` marcados como `GameMaker8_0`. Color clave corregido para sprites con fila de relleno (`padding_row_key`; regla vieja con `OPENGMK_GM7_CLAVE_ESQUINA`) |
| `gm8exe/src/reader.rs`, `lib.rs` | Detecta GM7 (offset 1980000, versión 700) antes de buscar GM8 |
| `gm8emulator/src/game/rl.rs` (nuevo) | Servidor RL: STEP (con trazas por frame, `no_draw`, corte al morir y grilla), SAVE, LOAD, OBS, QUERY, INFO, EXEC, TRACK, GRID_CONFIG, GRID y QUIT |
| `gm8emulator/src/game.rs` | `PlayType::Rl`: sin audio, ventana oculta salvo con `OPENGMK_RL_SHOW` |
| `gm8emulator/src/main.rs` | Opciones `--rl-socket` y `--rl-seed` |
| `gm8emulator/src/gml/kernel.rs` | `randomize()` y `sleep()` deterministas en modo RL; `sprite_replace` acepta la firma de 9 argumentos de GM7; `file_find_first` convierte `\` a `/` |
| `external/emulated.rs`, `external/fmod.rs`, `audio.rs`, `savestate.rs` | DLL y FMOD emulados para el Remastered (vienen del hub; el GB no los usa). Detalle en el `CLAUDE.md` del hub |

**Tras modificar OpenGMK, regenerar el parche**:
`cd vendor/OpenGMK && git add -N <archivos nuevos> && git diff > ../../patches/opengmk-gm7-y-rl.patch`.
`vendor/OpenGMK` es un clon aparte (ignorado por este repo): no hacer commit dentro de él.

## Trampas conocidas

- **Rutas con `\`**: toda función de archivo del kernel debe pasar la ruta por `file::to_path`.
- **`PlayType` tiene `match` exhaustivos** en `game.rs` y `gml/kernel.rs` (`randomize`, `sleep`). Al agregar
  una variante hay que cubrirlos todos.
- **Savestates**: el renderer reserva 2 texturas para la UI del TAS (`stock_atlas_count = n + 2` en
  `render/opengl.rs`). Cualquier modo que no sea grabación debe subir 2 sprites vacíos antes de `init()`,
  como hacen `replay()` y `rl_server()`. Si no, `dump_dynamic_textures` falla con *capacity overflow*.
- **El framebuffer de `get_pixels` ya viene con la fila superior primero**: no hay que invertirlo.
- **Los panics en release no traen símbolos** (`strip = true`). Para obtener un backtrace hay que usar el
  build debug con `RUST_BACKTRACE=1`.
- **El emulador loguea por stdout**, por eso el protocolo RL usa un socket Unix. Las rutas tienen un máximo
  de 108 caracteres: no crear el socket dentro del scratchpad.
- **`/tmp` es tmpfs (RAM)**: cada emulador crea `/tmp/iwbtgb_*` (8,3 MB, copia del exe). Si su proceso muere sin
  `close()`, la carpeta queda ocupando RAM sin que aparezca en ningún proceso. Los workers del GA la cierran al
  salir (`multiprocessing.util.Finalize` + SIGTERM en `iniciar_worker`) y cada `Emulator` nuevo borra las
  huérfanas (`limpiar_huerfanos`). El sistema no tiene earlyoom ni systemd-oomd: sin RAM se congela en vez de
  matar un proceso (probable causa del congelamiento del 2026-10-24, con `300A_Stage01_01` y el en vivo
  abiertos).
- **Savestates y archivos**: el emulador no guarda los archivos que escribe el juego (`save/`). El cliente
  (`Emulator.save/load`) los copia junto a cada savestate; sin eso, R tras un `load` va al save equivocado.
- **`player` es persistente**: con `room_goto` conserva sus coordenadas. Hay que reubicarlo en
  `playerStart + (17, 23)`.
- **Diferencias de API GM7 → GM8**: se revisaron las aridades de todas las llamadas del juego contra
  `kernel.rs`, y `sprite_replace` era la única distinta. Si aparece `wrong argument count` en otra función,
  adaptarla igual, por aridad. Hay candidatas listadas en `docs/formato-gm7.md`.
- **Cadenas en Shift-JIS**: el emulador usa `encoding_rs::SHIFT_JIS`, y los exportadores decodifican con
  `shift_jis`.
- **`game/save/`** contiene las partidas reales del usuario. No borrarlas. Para pruebas, copiar `game/` a
  otro directorio.
- **Contenido del juego**: es freeware de terceros. No pegar en el chat bloques largos de su código ni de sus
  assets, y no versionarlos (`game/`, `decompiled/` y el zip están en `.gitignore`).

## Datos útiles del juego

- Rooms en orden: `rInit`, `rTitle`, `rMenu`, `rSelectStage` (dificultades), `rKeyConfig`, `rStageBace`,
  `rStage01`–`17`, `rBossRush`, `rBoss01`–`08`, `rBossEnd` y `rHideArea01`–`11` (44 existentes, 152 ids).
- Objetos clave: `player`, `playerStart`, `warpStart` (portales de dificultad), `GAMEOVER`, `block` y
  `spikeUp`/`Down`/`Left`/`Right`. La room funciona a 50 FPS y la vista es de 800×600.
- Controles (según `game/ReadMe.txt`, en Shift-JIS): ← → para moverse, Shift confirma/salta (doble salto),
  Z dispara, R reintenta, Q suicidio, S salta los diálogos, F12 vuelve al inicio, F2 vuelve al título, M
  volumen, W tamaño de ventana, F9 captura y B *boss rush* (tras completar el juego). Para guardar, se dispara
  al botón rojo "SAVE", que pasa a verde.
