"""Varios emuladores en paralelo (un proceso por entorno) con AsyncVectorEnv."""

import os
import time

import gymnasium as gym

import iwbtgb_rl  # noqa: F401

if __name__ == "__main__":
    n = int(os.environ.get("N_ENVS", min(8, os.cpu_count() or 1)))
    envs = gym.vector.AsyncVectorEnv(
        [lambda i=i: gym.make("IWannaBeTheGB-v0", obs_type="gray", seed=i, skip_draw=True) for i in range(n)]
    )
    obs, info = envs.reset(seed=0)
    print(f"{n} entornos, observaciones {obs.shape}")
    steps = 500
    start = time.time()
    for _ in range(steps):
        obs, rewards, terminated, truncated, info = envs.step(envs.action_space.sample())
    elapsed = time.time() - start
    print(f"{steps * n} pasos en {elapsed:.1f}s -> {steps * n / elapsed:.0f} pasos/s en total")
    envs.close()
