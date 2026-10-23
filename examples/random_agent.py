"""Agente aleatorio: verifica el entorno y mide su velocidad."""

import time

import gymnasium as gym

import iwbtgb_rl  # noqa: F401  (registra IWannaBeTheGB-v0)

env = gym.make("IWannaBeTheGB-v0", obs_type="gray", frameskip=4)
obs, info = env.reset(seed=0)
print("observación:", obs.shape, obs.dtype, "| acciones:", env.action_space)

steps, episodes, total = 0, 0, 0.0
start = time.time()
while episodes < 5:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    steps += 1
    total += reward
    if terminated or truncated:
        episodes += 1
        print(f"episodio {episodes}: {info['steps']} pasos, recompensa acumulada {total:.1f}, "
              f"{'murió' if terminated else 'truncado'}, rooms {info['rooms_visited']}")
        total = 0.0
        obs, info = env.reset()
elapsed = time.time() - start
print(f"{steps} pasos en {elapsed:.1f}s -> {steps / elapsed:.0f} pasos/s ({steps * 4 / elapsed:.0f} frames/s)")
env.close()
