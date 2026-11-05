# El emulador: formato GM7, cambios a OpenGMK y modo RL

El juego corre en [OpenGMK](https://github.com/OpenGMK/OpenGMK) (runtime de GameMaker 8 en Rust, clon en
`vendor/OpenGMK`, base `9e4282e`) con el parche `patches/opengmk-gm7-y-rl.patch`. `vendor/OpenGMK` es un clon aparte,
ignorado por este repo: no se hace commit dentro de él. **Tras modificar OpenGMK, regenerar el parche**:
`cd vendor/OpenGMK && git add -N <archivos nuevos> && git diff > ../../patches/opengmk-gm7-y-rl.patch`.

La app Android de fangames (`~/Programacion/FangameHub`, otro repo) tiene su propia copia de OpenGMK y su propio
parche: un cambio al emulador que sirva a ambos hay que llevarlo a mano al otro.

```bash
./scripts/build.sh                     # cargo build --release de gm8emulator + gm8decompiler
cargo build -p gm8emulator --manifest-path vendor/OpenGMK/Cargo.toml   # build debug, con símbolos
```

Rust: el rustc del sistema (1.99, sin rustup; `rust-toolchain.toml` pide 1.77 pero sin rustup se ignora).

## Cambios a OpenGMK

| Archivo | Cambio |
|---|---|
| `gm8exe/src/gm7.rs` (nuevo) | Lector GM7. Devuelve `GameAssets` marcados como `GameMaker8_0`. Color clave corregido para sprites con fila de relleno (`padding_row_key`; regla vieja con `OPENGMK_GM7_CLAVE_ESQUINA=1`) |
| `gm8exe/src/reader.rs`, `lib.rs` | Detecta GM7 (offset 1980000, versión 700) antes de buscar GM8 |
| `gm8emulator/src/game/rl.rs` (nuevo) | Servidor RL (ver «Protocolo») |
| `gm8emulator/src/game.rs` | `PlayType::Rl`: sin audio, ventana oculta salvo con `OPENGMK_RL_SHOW` |
| `gm8emulator/src/main.rs` | Opciones `--rl-socket` y `--rl-seed` |
| `gm8emulator/src/gml/kernel.rs` | `randomize()` y `sleep()` deterministas en modo RL; `sprite_replace` acepta la firma de 9 argumentos de GM7; `file_find_first` convierte `\` a `/` |
| `external/emulated.rs`, `external/fmod.rs`, `audio.rs`, `savestate.rs` | DLL y FMOD emulados para el *I Wanna be the Guy: Remastered* (vienen del hub; el GB no los usa). Detalle en el `CLAUDE.md` del hub |

## Formato del ejecutable GameMaker 7.0 (ingeniería inversa)

El `.exe` es el runner GM7 con el juego anexado como datos. Implementado en `gm8exe/src/gm7.rs` (Rust) y en
`iwbtgb/gm7.py` (Python: `uv run python -m iwbtgb.gm7 game/i_wanna_be_the_GB.exe decompiled/gml` exporta todo el GML
a texto UTF-8; `./scripts/decompilar.sh` genera el `.gmk` editable con `gm8decompiler`). Herramientas como REA,
Ghidra o IDA no aplican directamente, porque la lógica no es código nativo sino datos.

Convenciones: enteros de 32 bits little-endian; `str` = `u32 largo + bytes` (Shift-JIS); `blob` = `u32 largo + bytes`.

### Fuentes y método

- Docs de LateralGM: <http://ismavatar.com/lgm/formats/> (`gmformat7.txt` describe el `.gmk` de GM7, y
  `gmkrypt1.html` el cifrado). El exe se parece al `.gmk`, pero no es idéntico.
- Desensamblado con `objdump -d -M intel`. Las secciones son las de un PE de Delphi: `CODE` empieza en VMA
  `0x401000`, offset de archivo `0x400`. Para convertir: `VMA = 0x401000 + (offset - 0x400)`.
- Para encontrar el loader, buscar las constantes en el código: `91 D5 12 00` (1234321), `60 36 1E 00` (1980000) y
  `0xFA`/`0xFE`/`0x2711` juntas (generación de la tabla de sustitución).

| Dirección (VMA) | Qué es |
|---|---|
| `0x547858` | Verifica que el tamaño sea ≥ 1980000 y hace seek fijo a 1980000 (no escanea) |
| `0x547570` | Lee el header: magic, versión 700, debug, settings, DLL y blob principal |
| `0x5470f0` | Carga todas las secciones en orden (cada llamada va precedida del mensaje de progreso `0x546284`) |
| `0x4abcbc` | Sección de paths (versión 420). La de scripts es `0x52cc90` |
| `0x4b1e4c` | Carga de un script: versión 400 y un stream comprimido que se guarda tal cual |
| `0x4b1f1c` | Decodificación del script: llama a `0x485968` con seed `0x3039` (12345), offset 0, todo el stream |
| `0x4858b4` | Genera la tabla de sustitución a partir de un seed |
| `0x485968` | Aplica la sustitución inversa (sin restar la posición) |
| `0x4857cc` | Lee `u32 largo + datos zlib` y devuelve un stream en memoria |

### Contenedor

```
offset 1980000:
  u32 1234321        magic
  u32 700            versión
  u32 debug
  u32 702            versión de settings
  settings (sin comprimir, ver abajo)
  str  "D3DX8.dll"
  blob dll
  blob datos         zlib -> GMKrypt -> datos de assets (llega exactamente al final del archivo)
```

**Settings 702**, todos `u32` y en este orden: fullscreen, interpolate, dont_draw_border, display_cursor, scaling
(i32), allow_resize, window_on_top, clear_colour, set_resolution, colour_depth, resolution, frequency,
dont_show_buttons, vsync, f4, f1, esc, f5f6, f9, close_as_esc, priority, freeze_on_lose_focus y loading_bar. Si
`loading_bar != 0` siguen 2 imágenes opcionales (`u32 existe [+ blob zlib]`). Después: imagen de carga propia
(opcional, mismo formato), transparent, translucency, scale_progress_bar, show_errors, log_errors, always_abort,
zero_uninit y **un u32 extra de significado desconocido**. A diferencia de GM8, no existe `disable_screensaver`.

**GMKrypt** (sobre el blob ya inflado):

```
u32 bill, u32 fred, u32 basura[bill], u32 seed, u32 basura[fred], 1 byte en claro, resto cifrado
tabla: a = 6 + seed % 250; b = seed / 250; t = identidad
       for i in 1..=10000: j = 1 + (i*a + b) % 254; swap(t[j], t[j+1])
       rev[t[i]] = i
byte_claro[pos] = (rev[byte[pos]] - pos) & 0xFF      (pos = offset dentro del blob inflado)
```

### Datos de assets (descifrados)

```
u32 ?? (12525)  u32 game_id  u32 guid[4]
u32 700  u32 n_extensiones (0; no soportado)
```

Cada sección es `u32 versión_sección, u32 n`, y luego `n` entradas de la forma `u32 existe` + datos. **Nada va
comprimido por asset**, salvo lo que se indica. En GM8, en cambio, cada asset es un bloque zlib.

| Sección | Ver. | Entrada | Diferencia con el exe de GM8 |
|---|---|---|---|
| sonidos | 400 | nombre, 600, kind, ext, fuente, `existe [blob]`, efectos, f64 vol, f64 pan, preload | **los datos de audio van en zlib** |
| sprites | 400 | nombre, 542, w, h, bbox l, r, b, t, transparent, smooth, preload, bbox_mode, precise, xorig, yorig, n_frames, frames | sin máscaras de colisión |
| frame/imagen | — | 540, `i32 existe (-1 = no)`, w, h, blob zlib | píxeles **BGR + 4.º byte** (0 en el fondo, 255 en el resto; GM7 no lo usa como alfa); la transparencia es por color clave (ver abajo) |
| fondos | 400 | nombre, 543, w, h, transparent, smooth, preload, `existe [imagen]` | igual que la imagen anterior |
| paths | 420 | nombre, 530, kind, closed, precision, n, (f64 x, y, speed)×n | igual a GM8 |
| scripts | 400 | nombre, 400, blob | **blob = zlib → sustitución con seed 12345 → `str`** |
| fuentes | 540 | nombre, 540, nombre_sist, size, bold, italic, rango_ini, rango_fin, u32×0x600, atlas w, h, blob | atlas en zlib (1 byte alpha por pixel) |
| timelines | 500 | nombre, 500, n, (u32 momento, acciones)×n | igual |
| objetos | 400 | nombre, 430, sprite, solid, visible, depth, persistent, parent, mask, **u32 10**, listas | 11 listas (eventos 0..10), no hay *trigger*; cada lista es `(i32 sub, acciones)*` y termina en `-1` |
| rooms | 420 | nombre, 541, caption, w, h, speed, persistent, colour, draw_colour, código, 8 fondos ×10 u32, views_enabled, 8 vistas ×14 u32, instancias (x, y, obj, id, código), tiles ×9 u32 | igual al exe de GM8.0 |

**Color clave.** El pixel inferior izquierdo, salvo en los sprites con una fila extra abajo (`sprSpikeUp` y
`sprSpikeRight` de 32×33, `maskSpikeShort`, `sprSpikeUpMini`, `sprSpikeUpSuperBig`, `sprBoundSpike`, …): ahí esa
fila es oscura, el pixel inferior izquierdo borra el contorno y deja opaco el fondo turquesa (192,192,0), y la máscara
de colisión queda rectangular. El 4.º byte sí marca el fondo real. Regla (`padding_row_key`; también
`gm7.color_clave`): si la fila inferior entera tiene ese byte en 0, la clave es el color más común entre los píxeles
con byte 0. Con `OPENGMK_GM7_CLAVE_ESQUINA=1` se vuelve a la regla vieja (para repetir corridas anteriores a la
corrección del 2026-10-25). La regla también cambia `sprLife`, `sprCommand`, `sprSpringLeft[4]`, `sprWarning03` y
`sprMoguraAna`; no se revisó si esos cambios son correctos.

**Acciones** (en eventos y timelines): `u32 400, u32 n`, y por cada acción: `440, lib, id, kind, may_be_relative,
question, applies, exe_type, str función, str código, u32 argc, u32 8, u32 tipos[8], i32 aplica_a, u32 relativa, u32
8, str args[8], u32 not`. "Ejecutar código" es lib 1, acción 603, con el código en `args[0]`.

**Cola**: `i32 último_id_instancia, i32 último_id_tile`; `620, n_includes` (0; no soportado); diálogo de ayuda
`600`: bg_colour, new_window, `str` caption, left, top, w, h, border, resizable, on_top, freeze, `str` RTF (sin
zlib); `500, n, str[n]` (código de inicio de librerías); `700, n, i32[n]` (orden de rooms). Después quedan ~6 KB de
basura que el runner ignora.

### Diferencias de runtime GM7 → GM8 que importan

- El lector entrega el juego como **GameMaker 8.0**. El runtime de OpenGMK es el de GM8.
- `sprite_replace`: en GM7 recibe 9 argumentos `(ind, fname, imgnumb, precise, transparent, smooth, preload, xorig,
  yorig)` y en GM8, 7. Lo usa `scripts/initExe` para las capturas de las partidas guardadas: sin la adaptación, el
  juego se trababa en el título apenas existía `save/save1`.
- Se verificó la cantidad de argumentos de todas las llamadas del juego contra las 847 funciones de aridad fija de
  OpenGMK: no hay otras discrepancias (puede haber cambios de semántica con la misma aridad). Si aparece `wrong
  argument count` en otra función, adaptarla igual. Candidatas si otro juego las usa: `sprite_add` (8 → 6),
  `background_add` (4 → 3), `background_replace`, `sprite_create_from_screen` (9 → 8).

### Simplificaciones del lector

- No se cargan las imágenes de la barra de carga ni la de carga propia (son cosméticas y vienen en otro formato).
- Se ignoran "smooth edges" y "preload".
- Máscaras de colisión: si el sprite es *precise*, los píxeles opacos dentro de la caja de colisión, una por frame;
  si no, la caja completa, compartida por todos los frames.
- No hay soporte para extensiones ni includes de GM7 (este juego no los usa).

## El modo RL

`gm8emulator <exe> --rl-socket RUTA [--rl-seed N]` corre sin ventana ni audio y atiende a un cliente por un socket
Unix. El cliente es `iwbtgb.emulador.Emulator`: lanza el emulador sobre una copia del `.exe` en `/tmp/iwbtgb_*` (el
juego guarda sus partidas junto al ejecutable y varias instancias se pisarían) y habla el protocolo.
`emulador.arrancar(dificultad, sala)` lo deja al comienzo de una sala con el savestate 0 ahí: título → menú →
`rSelectStage` → portal de la dificultad (medium (96, 384), hard (672, 384), veryhard (64, 224), impossible
(704, 224)) → rStage01 y, con una sala, `room_goto` y el jugador en `playerStart + (17, 23)` (`player` es persistente:
conserva las coordenadas de la room anterior, que pueden quedar dentro de un muro).

```python
from iwbtgb.emulador import ACTIONS, arrancar
emu = arrancar("medium")                 # al comienzo de rStage01, savestate 0
emu.track("player")
r = emu.step(("right", "shift"), frames=10, obs=False, trace=True, no_draw=True, stop_on_gone=True)
emu.load(0); emu.query("player"); emu.exec("return global.death"); emu.close()
```

Acciones (`ACTIONS`, 12): nada, ←, →, Shift, ←+Shift, →+Shift, Z, ←+Z, →+Z, Shift+Z, ←+Shift+Z y →+Shift+Z. El
salto sale al *apretar* Shift (su altura depende de cuánto se mantiene) y volver a apretarlo en el aire hace el doble
salto. La room funciona a 50 FPS y la vista es de 800×600 (el objeto `camera` la mueve en pasos de 800×608).

### Protocolo del socket

Documentado en la cabecera de `rl.rs`. Todo es little-endian y cada petición empieza con 1 byte de opcode.

| Op | Petición | Respuesta |
|---|---|---|
| 1 STEP | `u8 n, u8 vk[n], u32 frames, u8 flags` | `u8 status [, str error], i32 room, u64 frame [, obs] [, traza] [, grilla]` |
| 2 SAVE / 3 LOAD | `u32 slot` | `u8 status` |
| 4 OBS | — | `obs` = `u32 w, u32 h, RGB w*h*3` (fila superior primero) |
| 5 QUERY | `u16 len, nombre del objeto` | `u32 n, (i32 id, f64 x, y, hspeed, vspeed)×n` (incluye objetos hijos) |
| 6 INFO | — | `i32 room, u32 w, h, speed, u16 len, nombre, u64 frame` |
| 7 EXEC | `u32 len, GML` | `u8 status`, y luego `u8 0 + f64` o `u8 1 + str` (valor de `return`) o el error |
| 8 TRACK | `u16 len, nombre del objeto` | `u8 status`: elige el objeto que describen las trazas |
| 9 GRID_CONFIG | celda, ancho, alto, muestras, canales (objetos, excluidos, solo visibles), campos | `u8 status, u16 n, n × nombre de objeto que falta` |
| 10 GRID | — | `u8 status` + la grilla (y los rayos, si están configurados) |
| 11 RAYS_CONFIG | `u16 n, u8 paso px, u8 c, c × u8 banderas` | `u8 status` (2 = `c` no es la cantidad de canales de la grilla) |
| 255 QUIT | — | — |

- Flags de STEP: bit 0 = obs, bit 1 = `skip_draw` (dibuja solo el último frame: los eventos Draw intermedios no
  corren), bit 2 = traza, bit 3 = `no_draw` (no dibuja ninguno), bit 4 = `stop_on_gone` (corta el paso cuando
  desaparece el objeto seguido, p. ej. al morir) y bit 5 = grilla. La traza es `u32 n` y un registro por frame:
  `i32 room, u8 alive, f32 x, y, i32 sprite_index, f32 image_index, f32 image_xscale` (`emulador.TRACE_DTYPE`).
- `status`: 0 = ok, 1 = el juego terminó (se reinicia) y 2 = error GML. Con un error, STEP manda igual el resto de la
  respuesta; el cliente la lee entera antes de lanzar `EmulatorError`.
- Las teclas son por nivel: las que no vienen en un STEP se sueltan. Cada savestate guarda también las teclas que
  estaban presionadas. Los savestates (~14 ms, ~11 MB cada uno) viven en el emulador hasta que se pisa su slot.
- El emulador no guarda los archivos que escribe el juego (`save/`): `Emulator.save/load` los copian junto a cada
  savestate (salvo los `.bmp`). Sin eso, cargar un savestate y apretar R llevaría al save de otra partida.
- `randomize()` deriva la semilla nueva de la actual, así que todo es reproducible desde `--rl-seed`.

### La grilla semántica (op 9–10)

Una grilla de `ancho`×`alto` celdas centrada en el objeto seguido, con un canal por grupo de objetos (con sus hijos,
salvo los excluidos): cada celda vale la fracción (0–255) de sus `muestras`×`muestras` puntos dentro de la **máscara de
colisión** de alguna instancia (las púas son triángulos). Más hspeed, vspeed y las variables pedidas del objeto (p.
ej. `djump`). Cuesta ~0,45 ms por llamada en rStage01. En el cliente: `grid_config`, `grid` o `step(grid=True)`.

### Los rayos (op 11)

`n` rayos desde el centro de la caja del jugador hasta el borde de la pantalla (la vista 0 si las vistas están
activas, o la room), con los canales de la grilla: hay que configurarla antes (puede ser de 0×0, y no se paga su
costo) y volver a configurar los rayos si se cambia. Banderas por canal: 1 = corta el rayo, 2 = se ve. Vienen al
final de cada respuesta de la grilla: origen, `image_xscale` y, por rayo, un `u16` por canal (px hasta el primer
píxel de ese canal) más uno para el borde (`0xFFFF` = no alcanzado). El rayo `i` apunta a `i·2π/n`, en sentido
antihorario empezando hacia la derecha.

- Cada rayo se interseca con la caja de cada instancia en pantalla y solo ahí se prueba la máscara exacta cada `paso`
  px: el costo no depende del área de la pantalla.
- Precisión, comparada con `collision_point` del juego (2.505 rayos): error medio 0,7 px.
- Velocidad en el gimnasio decidiendo cada 2 frames: 3.010 frames/s con 64 rayos y paso 2.

### Verificado

- **Determinismo**: la misma secuencia de acciones desde un savestate produce frames idénticos (por hash), y jugar
  desde un savestate da la misma trayectoria que jugar desde el inicio.
- **Dibujar no cambia la lógica**: las trazas con `no_draw`, `skip_draw` y dibujando cada frame son idénticas, y
  dibujar no consume el azar del juego.
- **Rendimiento**: cada frame cuesta ~1–1,4 ms en rStage01 (~970 instancias, 722 `block`: OpenGMK busca colisiones
  recorriéndolas todas), ~0,2 ms en una sala del gimnasio (~120 instancias) y ~9–11 ms con el jugador muerto (la
  sangre). Dibujar cuesta ~1,8 ms por frame; `observe()` ~3 ms.

## Entorno Gymnasium (`iwbtgb/entorno.py`)

`IWannaBeTheGB-v0` (se registra al importar `iwbtgb.entorno`), para RL por gradiente (p. ej. PPO de
Stable-Baselines3 con `CnnPolicy`, `SubprocVecEnv` y `VecFrameStack(4)`). No se ha entrenado nada con él.

```python
import gymnasium as gym
import iwbtgb.entorno
env = gym.make("IWannaBeTheGB-v0", obs_type="gray", frameskip=4)
obs, info = env.reset()
```

| Parámetro | Defecto | Nota |
|---|---|---|
| `difficulty` / `start_room` | `"medium"` / `None` | sala inicial, p. ej. `"rStage05"` o `"rBoss01"` |
| `frameskip` / `skip_draw` | 4 / False | |
| `obs_type` / `downscale` | `"rgb"` / 4 | `rgb` (150×200×3), `gray` (150×200×1) o `state` (`[x, y, hspeed, vspeed, room]`) |
| `max_episode_steps` / `noop_reset_max` | 5000 / 0 | truncado; frames al azar sin acción al hacer reset |
| `cell_size`, `cell_reward`, `room_reward`, `death_penalty` | 32, 0.1, 10.0, 1.0 | la recompensa por defecto (sobreescribir `compute_reward`) |

El episodio termina cuando muere el jugador (su instancia desaparece y aparecen `GAMEOVER` y `PlayerDeathParts`).

## Trampas conocidas

- **Rutas con `\`**: toda función de archivo del kernel debe pasar la ruta por `file::to_path`.
- **`PlayType` tiene `match` exhaustivos** en `game.rs` y `gml/kernel.rs` (`randomize`, `sleep`): al agregar una
  variante hay que cubrirlos todos.
- **Savestates**: el renderer reserva 2 texturas para la UI del TAS (`stock_atlas_count = n + 2` en
  `render/opengl.rs`). Todo modo que no sea grabación debe subir 2 sprites vacíos antes de `init()`, como `replay()`
  y `rl_server()`; si no, `dump_dynamic_textures` falla con *capacity overflow*.
- **El framebuffer de `get_pixels` ya viene con la fila superior primero**: no hay que invertirlo.
- **Los panics en release no traen símbolos** (`strip = true`): para un backtrace, el build debug con
  `RUST_BACKTRACE=1`.
- **El emulador loguea por stdout**: por eso el protocolo usa un socket Unix. Las rutas de socket tienen un máximo de
  108 caracteres: el cliente los crea en `/tmp/iwbtgb_*` (no en el scratchpad).
- **`/tmp` es tmpfs (RAM)**: cada emulador crea `/tmp/iwbtgb_*` (8,3 MB, la copia del exe). Si su proceso muere sin
  `close()`, la carpeta queda ocupando RAM: los workers la cierran al salir (`workers.cerrar_al_salir`) y cada
  `Emulator` nuevo borra las huérfanas (`limpiar_huerfanos`).
- **El azar del juego** lo usan la sangre al morir, jefes (KirbyDDD, Himiko, PocketCameraMario), enemigos
  (MazemonEnemy01–03) y spikeVib; ni el juego ni `game_restart` reinician la semilla. Saltarse frames (p. ej. los de
  muerto) cambia el azar de ahí en adelante: hay que hacerlo igual al entrenar y al mostrar.
- **Cadenas en Shift-JIS**: el emulador usa `encoding_rs::SHIFT_JIS` y el cliente y `gm7.py` decodifican con
  `shift_jis`.

## Datos útiles del juego

- Rooms en orden: `rInit`, `rTitle`, `rMenu`, `rSelectStage` (dificultades), `rKeyConfig`, `rStageBace`,
  `rStage01`–`17`, `rBossRush`, `rBoss01`–`08`, `rBossEnd` y `rHideArea01`–`11` (44 existentes, 152 ids).
- Objetos clave: `player`, `playerStart`, `warpStart` (portales de dificultad), `GAMEOVER`, `block` (la colisión de los
  tiles, invisible), `platform` (hija de `block`, se atraviesa desde abajo), `playerKiller` y `spikeUp`/`Down`/`Left`/
  `Right`, `savePoint`, `roomChanger` (la salida de una sala: se cambia de room al salir de la sala sobre uno).
- Controles (según `game/ReadMe.txt`): ← → para moverse, Shift confirma/salta (doble salto), Z dispara, R reintenta, Q
  suicidio, S salta los diálogos, F12 vuelve al inicio, F2 vuelve al título, M volumen, W tamaño de ventana, F9 captura
  y B *boss rush* (tras completar el juego). Para guardar, se dispara al botón rojo "SAVE", que pasa a verde.
- En `rStage02`, moverse a la izquierda desplazó al jugador 1467 px en 40 frames: no se verificó si es un
  teletransporte propio del nivel.
