"""Ver jugar a un agente: en vivo (ventana del juego) o grabando un video .mp4.

    uv run python examples/ver_agente.py                          # ventana, velocidad real
    uv run python examples/ver_agente.py --velocidad 0.5          # en cámara lenta
    uv run python examples/ver_agente.py --video videos/aleatorio.mp4 --episodios 5

Por defecto juega un agente aleatorio. Para ver tu propio agente, reemplaza `politica()`.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym

import iwbtgb_rl  # noqa: F401  (registra IWannaBeTheGB-v0)

GAME_FPS = 50


def politica(obs, env):
    """Devuelve la acción a tomar. Ejemplo con un modelo de Stable-Baselines3 ya entrenado:

        action, _ = modelo.predict(obs, deterministic=True)
        return int(action)
    """
    return env.action_space.sample()


class GrabadorVideo:
    """Envía frames RGB a ffmpeg y genera un .mp4 H.264."""

    def __init__(self, ruta: Path, ancho: int, alto: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg no está instalado")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ancho}x{alto}", "-r", f"{fps}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(ruta)],
            stdin=subprocess.PIPE,
        )

    def escribir(self, frame) -> None:
        self.proc.stdin.write(frame.tobytes())

    def cerrar(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, help="grabar a este .mp4 en vez de mostrar la ventana")
    p.add_argument("--episodios", type=int, default=3)
    p.add_argument("--velocidad", type=float, default=1.0, help="1 = tiempo real (solo modo ventana)")
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--start-room", default=None, help="p. ej. rStage03")
    p.add_argument("--dificultad", default="medium")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    en_vivo = args.video is None
    env = gym.make(
        "IWannaBeTheGB-v0",
        obs_type="gray",
        frameskip=args.frameskip,
        start_room=args.start_room,
        difficulty=args.dificultad,
        seed=args.seed,
        show_window=en_vivo,
    )
    obs, info = env.reset(seed=args.seed)

    grabador = None
    if not en_vivo:
        frame = env.render()
        grabador = GrabadorVideo(args.video, frame.shape[1], frame.shape[0], GAME_FPS / args.frameskip)
        grabador.escribir(frame)

    duracion_paso = args.frameskip / GAME_FPS / args.velocidad
    episodio, total = 1, 0.0
    try:
        while episodio <= args.episodios:
            inicio = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(politica(obs, env))
            total += reward
            if grabador:
                grabador.escribir(env.render())
            else:
                time.sleep(max(0.0, duracion_paso - (time.perf_counter() - inicio)))

            if terminated or truncated:
                print(f"episodio {episodio}: {info['steps']} pasos, recompensa {total:.1f}, "
                      f"{'murió' if terminated else 'truncado'}, rooms {info['rooms_visited']}")
                if en_vivo:
                    time.sleep(1.0)  # pausa para ver la muerte antes del reset
                episodio, total = episodio + 1, 0.0
                obs, info = env.reset()
                if grabador and episodio <= args.episodios:
                    grabador.escribir(env.render())
    except KeyboardInterrupt:
        pass
    finally:
        if grabador:
            grabador.cerrar()
            print(f"video guardado en {args.video}")
        env.close()


if __name__ == "__main__":
    main()
