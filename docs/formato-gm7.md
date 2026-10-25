# Formato del ejecutable GameMaker 7.0 (ingeniería inversa)

Referencia de todo lo descubierto al desensamblar `game/i_wanna_be_the_GB.exe`, que es el runner GM7 con el
juego anexado como datos. Está implementado en `vendor/OpenGMK/gm8exe/src/gm7.rs` (Rust) y en
`tools/export_gml.py` (Python).

Convenciones: enteros de 32 bits little-endian; `str` = `u32 largo + bytes` (Shift-JIS);
`blob` = `u32 largo + bytes`.

## Fuentes y método

- Docs de LateralGM: <http://ismavatar.com/lgm/formats/> (`gmformat7.txt` describe el `.gmk` de GM7, y
  `gmkrypt1.html` el cifrado). El exe se parece al `.gmk`, pero no es idéntico.
- Desensamblado con `objdump -d -M intel` (no hace falta Ghidra). Las secciones son las de un PE de Delphi:
  `CODE` empieza en VMA `0x401000`, offset de archivo `0x400`. Para convertir:
  `VMA = 0x401000 + (offset - 0x400)`.
- Para encontrar el loader, buscar las constantes en el código: `91 D5 12 00` (1234321), `60 36 1E 00`
  (1980000) y `0xFA`/`0xFE`/`0x2711` juntas (generación de la tabla de sustitución).

Direcciones útiles del runner (VMA):

| Dirección | Qué es |
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

## Contenedor (en el `.exe`)

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

**Settings 702**, todos `u32` y en este orden: fullscreen, interpolate, dont_draw_border, display_cursor,
scaling (i32), allow_resize, window_on_top, clear_colour, set_resolution, colour_depth, resolution, frequency,
dont_show_buttons, vsync, f4, f1, esc, f5f6, f9, close_as_esc, priority, freeze_on_lose_focus y loading_bar.
Si `loading_bar != 0` siguen 2 imágenes opcionales (`u32 existe [+ blob zlib]`). Después: imagen de carga
propia (opcional, mismo formato), transparent, translucency, scale_progress_bar, show_errors, log_errors,
always_abort, zero_uninit y **un u32 extra de significado desconocido**. A diferencia de GM8, no existe
`disable_screensaver`.

**GMKrypt** (sobre el blob ya inflado):

```
u32 bill, u32 fred, u32 basura[bill], u32 seed, u32 basura[fred], 1 byte en claro, resto cifrado
tabla: a = 6 + seed % 250; b = seed / 250; t = identidad
       for i in 1..=10000: j = 1 + (i*a + b) % 254; swap(t[j], t[j+1])
       rev[t[i]] = i
byte_claro[pos] = (rev[byte[pos]] - pos) & 0xFF      (pos = offset dentro del blob inflado)
```

## Datos de assets (descifrados)

```
u32 ?? (12525)  u32 game_id  u32 guid[4]
u32 700  u32 n_extensiones (0; no soportado)
```

Cada sección es `u32 versión_sección, u32 n`, y luego `n` entradas de la forma `u32 existe` + datos.
**Nada va comprimido por asset**, salvo lo que se indica. En GM8, en cambio, cada asset es un bloque zlib.

| Sección | Ver. | Entrada | Diferencia con el exe de GM8 |
|---|---|---|---|
| sonidos | 400 | nombre, 600, kind, ext, fuente, `existe [blob]`, efectos, f64 vol, f64 pan, preload | **los datos de audio van en zlib** |
| sprites | 400 | nombre, 542, w, h, bbox l, r, b, t, transparent, smooth, preload, bbox_mode, precise, xorig, yorig, n_frames, frames | sin máscaras de colisión |
| frame/imagen | — | 540, `i32 existe (-1 = no)`, w, h, blob zlib | píxeles **BGR + 4.º byte** (0 en el fondo, 255 en el resto; GM7 no lo usa como alfa); la transparencia es por color clave: pixel inferior izquierdo, salvo la excepción de abajo |
| fondos | 400 | nombre, 543, w, h, transparent, smooth, preload, `existe [imagen]` | igual que la imagen anterior |
| paths | 420 | nombre, 530, kind, closed, precision, n, (f64 x, y, speed)×n | igual a GM8 |
| scripts | 400 | nombre, 400, blob | **blob = zlib → sustitución con seed 12345 → `str`** |
| fuentes | 540 | nombre, 540, nombre_sist, size, bold, italic, rango_ini, rango_fin, u32×0x600, atlas w, h, blob | atlas en zlib (1 byte alpha por pixel) |
| timelines | 500 | nombre, 500, n, (u32 momento, acciones)×n | igual |
| objetos | 400 | nombre, 430, sprite, solid, visible, depth, persistent, parent, mask, **u32 10**, listas | 11 listas (eventos 0..10), no hay *trigger*; cada lista es `(i32 sub, acciones)*` y termina en `-1` |
| rooms | 420 | nombre, 541, caption, w, h, speed, persistent, colour, draw_colour, código, 8 fondos ×10 u32, views_enabled, 8 vistas ×14 u32, instancias (x, y, obj, id, código), tiles ×9 u32 | igual al exe de GM8.0 |

**Color clave con fila de relleno.** Algunos sprites tienen una fila extra abajo (`sprSpikeUp` y `sprSpikeRight`
de 32×33, `maskSpikeShort`, `sprSpikeUpMini`, `sprSpikeUpSuperBig`, `sprBoundSpike`, …). En el exe esa fila es
oscura, así que el pixel inferior izquierdo borra el contorno y deja opaco el fondo turquesa (192,192,0): la
máscara de colisión queda rectangular. El 4.º byte sí marca el fondo real. Regla (`padding_row_key` en
`gm7.rs`): si la fila inferior entera tiene ese byte en 0, la clave es el color más común entre los píxeles con
byte 0. Con `OPENGMK_GM7_CLAVE_ESQUINA=1` se vuelve a la regla vieja (para repetir corridas anteriores). La
regla también cambia `sprLife`, `sprCommand`, `sprSpringLeft[4]`, `sprWarning03` y `sprMoguraAna`; no se
revisó si esos cambios son correctos.

**Acciones** (en eventos y timelines): `u32 400, u32 n`, y por cada acción:
`440, lib, id, kind, may_be_relative, question, applies, exe_type, str función, str código, u32 argc,
u32 8, u32 tipos[8], i32 aplica_a, u32 relativa, u32 8, str args[8], u32 not`. "Ejecutar código" es lib 1,
acción 603, con el código en `args[0]`.

**Cola**: `i32 último_id_instancia, i32 último_id_tile`; `620, n_includes` (0; no soportado);
diálogo de ayuda `600`: bg_colour, new_window, `str` caption, left, top, w, h, border, resizable, on_top,
freeze, `str` RTF (sin zlib); `500, n, str[n]` (código de inicio de librerías); `700, n, i32[n]` (orden de
rooms). Después quedan ~6 KB de basura que el runner ignora.

## Diferencias de runtime GM7 → GM8 que importan

- El lector entrega el juego como **GameMaker 8.0**. El runtime de OpenGMK es el de GM8.
- `sprite_replace`: en GM7 recibe 9 argumentos `(ind, fname, imgnumb, precise, transparent, smooth, preload,
  xorig, yorig)` y en GM8, 7. Se agregó compatibilidad en `gml/kernel.rs`. Lo usa `scripts/initExe` para
  mostrar las capturas de las partidas guardadas, así que el juego se trababa en el título apenas existía
  `save/save1`.
- Se verificó la cantidad de argumentos de todas las llamadas del juego contra las 847 funciones de aridad
  fija de OpenGMK: no hay otras discrepancias. Puede haber cambios de semántica con la misma aridad que esta
  revisión no detecta. Funciones de GM7 que podrían necesitar la misma adaptación si otro juego las usa:
  `sprite_add` (8 → 6), `background_add` (4 → 3), `background_replace`, `sprite_create_from_screen` (9 → 8).

## Simplificaciones del lector

- No se cargan las imágenes de la barra de carga ni la de carga propia (son cosméticas y vienen en otro
  formato).
- Se ignoran "smooth edges" y "preload".
- Las máscaras de colisión se generan así: si el sprite es *precise*, son los píxeles opacos dentro de la caja
  de colisión, uno por frame; si no, es la caja completa, compartida por todos los frames.
- No hay soporte para extensiones ni includes de GM7 (este juego no los usa).
