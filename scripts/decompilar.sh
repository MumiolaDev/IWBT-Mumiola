#!/usr/bin/env bash
# Genera el proyecto editable decompiled/i_wanna_be_the_GB.gmk (formato GameMaker 8.0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/decompiled"
"$ROOT/vendor/OpenGMK/target/release/gm8decompiler" "$ROOT/game/i_wanna_be_the_GB.exe" \
    -o "$ROOT/decompiled/i_wanna_be_the_GB.gmk" "$@"
