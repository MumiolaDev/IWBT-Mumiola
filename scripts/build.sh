#!/usr/bin/env bash
# Clona OpenGMK (si falta), aplica el parche GM7 + RL y compila el emulador y el decompilador.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENGMK="$ROOT/vendor/OpenGMK"
COMMIT=9e4282e6d19f4dd00961811840ddc3e11da6dd5a

if [ ! -d "$OPENGMK" ]; then
    git clone https://github.com/OpenGMK/OpenGMK.git "$OPENGMK"
    git -C "$OPENGMK" checkout "$COMMIT"
    git -C "$OPENGMK" apply "$ROOT/patches/opengmk-gm7-y-rl.patch"
fi
cargo build --release --manifest-path "$OPENGMK/Cargo.toml" -p gm8emulator -p gm8decompiler
