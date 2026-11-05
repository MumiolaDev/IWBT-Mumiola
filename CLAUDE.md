# CLAUDE.md

Port a Linux de *I wanna be the GB* (fangame en **GameMaker 7.0**, no el IWBTG original) sobre OpenGMK, y algoritmos
que aprenden a jugarlo: un GA con TAS para un video blog de YouTube y redes neuronales que buscan generalizar (con una
grilla y con rayos). El objetivo final es transmitir en vivo al algoritmo entrenando y pasándose el juego. Responder al
usuario en español.

Todo el código Python es un solo paquete, `iwbtgb/`, y se usa como `python -m iwbtgb.<línea> <comando>`. Uso y
estructura en `README.md`; el resto en `docs/`:

- `docs/emulador.md`: formato GM7, cambios a OpenGMK, protocolo RL, trampas conocidas del emulador y datos del juego.
  **Leerlo antes de tocar el lector GM7 o el modo RL.**
- `docs/ga.md`: GA con TAS y Go-Explore, y el estado de sus corridas.
- `docs/redes.md`: neuroevolución (`iwbtgb/evolucion`, común a `iwbtgb/redes` y `iwbtgb/rayos`), el gimnasio y las
  corridas. **Leerlo antes de tocar las redes.**

La app Android de fangames (hub con rankings, *I Wanna be the Guy: Remastered*) se separó el 2026-10-23 a su propio
repo: `~/Programacion/FangameHub`, donde se trabaja en otra sesión. Cada repo tiene su propia copia de OpenGMK y su
propio parche: un cambio al emulador que sirva a ambos hay que llevarlo a mano al otro.

## Estado (2026-11-05)

- El GB corre nativo (`./scripts/jugar.sh`). Con RL por gradiente (PPO) no se entrenó nada (`iwbtgb/entorno.py`).
- **GA con TAS** (`iwbtgb.ga`): en pausa desde el 2026-10-26. Llegó al 75,7 % del camino de rStage01 con save
  heredado y al ~47 % en partida continua (`GA_beta_corregida`).
- **Redes** (`iwbtgb.redes`, la grilla; `iwbtgb.rayos`, los rayos): en rStage01 todas se estancan en el 7–8,9 %, en la
  trampa del pozo derecho. Los rayos generalizan mejor en el gimnasio (24 de 40 salas nuevas contra 15).
- El 2026-11-05 se unificó el código en un paquete (antes `iwbtgb_rl`, `ga`, `redes` y `rayos` sueltos, con mucho
  código copiado), se arreglaron cuatro bugs (el protocolo se desfasaba tras un error GML en STEP, las opciones
  booleanas no se podían apagar, los videos de una red con reintentos desalineaban el azar del juego, `resolver`
  llegaba a ~1,6 GB por emulador) y se unificó la documentación. Las corridas no cambian:
  todo se lee igual y los resultados son idénticos (comprobado con entrenamientos cortos antes y después). Respaldo del
  código anterior en `/mnt/datos/iwbtgb/respaldos/codigo_antes_de_unificar_2026-11-05.tar.gz`.
- Informe del GA (artifact): https://claude.ai/artifact/XBoYLppCkJP2CkYwzmkP8E. Cubre los exp. 1–2.

## Reglas

- **Corridas** (`corridas/`, `corridas_redes/`, `corridas_rayos/`: enlaces a `/mnt/datos/iwbtgb/`): no borrar ni
  modificar las del usuario. Para pruebas, crear una aparte (`_prueba_*`) y borrarla al terminar. Las corridas
  anteriores a la corrección de las púas (2026-10-25) se repiten con el binario viejo:
  `OPENGMK_EMULATOR=/mnt/datos/iwbtgb/emulador_antiguo/gm8emulator` (o `OPENGMK_GM7_CLAVE_ESQUINA=1`).
- **Compatibilidad**: cada regla nueva tiene un valor por defecto "apagado" para corridas viejas (GA:
  `ga/almacen.cargar_config`, `setdefault`; redes: `Config.ANTERIORES`). Cambios al generador de salas del gimnasio:
  siempre detrás de una regla con versión guardada en cada generación (cambiar el catálogo cambia el azar de todas las
  salas y rompería el en vivo de las generaciones guardadas).
- **Las dos redes comparten `iwbtgb/evolucion/`**: un cambio ahí hay que probarlo en las dos. Cada línea solo define lo
  que ve y su red (`Linea` en su `__init__.py`). No copiar código entre líneas.
- **Apps**: la app abierta usa el código con el que se abrió; los procesos que lanza son nuevos y usan el del disco.
  Tras cambiar una config o una app, cerrarla y reabrirla. Sus accesos directos están en
  `~/.local/share/applications/iwbtgb-{ga,redes,rayos}.desktop` (llaman a `python -m iwbtgb.<línea> app`).
- **RAM**: 16 GB y sin OOM killer (sin RAM la máquina se congela). Cada emulador del gimnasio usa ~150 MB más ~11 MB por
  savestate (las salas armadas, hasta 8; `resolver`, hasta 32). Antes de lanzar pruebas junto a un entrenamiento del
  usuario, mirar `MemAvailable`.
- **OpenGMK**: tras modificarlo, regenerar el parche (`docs/emulador.md`). `vendor/OpenGMK` es un clon aparte: no hacer
  commit dentro de él.
- **`game/save/`** contiene las partidas reales del usuario: no borrarlas (para pruebas, copiar `game/` a otro lado).
- **Contenido del juego**: es freeware de terceros. No pegar en el chat bloques largos de su código ni de sus assets, y
  no versionarlos (`game/`, `decompiled/` y el zip están en `.gitignore`).

## Comandos

```bash
./scripts/build.sh                     # compila gm8emulator y gm8decompiler (cargo, rustc del sistema)
./scripts/jugar.sh                     # jugar (ejecuta desde game/; el juego escribe en game/save/)
./scripts/decompilar.sh                # -> decompiled/i_wanna_be_the_GB.gmk
uv run python -I -m iwbtgb.gm7 game/i_wanna_be_the_GB.exe decompiled/gml   # el GML como texto
uv run python -m iwbtgb.ga             # lista los comandos del GA (entrenar, explorar, visualizar, app)
uv run python -m iwbtgb.rayos          # y los de las redes (entrenar, probar, visualizar, transmitir, app)
```

Python: `uv` con `.venv` (numpy, gymnasium, pillow, matplotlib y pyside6-essentials). No hay tests automáticos: para
verificar un cambio que no debería alterar resultados, comparar los `gen_*.npz` de un entrenamiento corto antes y
después.
