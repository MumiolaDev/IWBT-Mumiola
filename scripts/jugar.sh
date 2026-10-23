#!/usr/bin/env bash
# Juega normalmente (con ventana y audio) usando el emulador nativo de Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/game"
exec "$ROOT/vendor/OpenGMK/target/release/gm8emulator" i_wanna_be_the_GB.exe "$@"
