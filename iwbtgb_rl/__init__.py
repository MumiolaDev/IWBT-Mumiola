"""I wanna be the GB como entorno de aprendizaje reforzado (Linux, vía OpenGMK)."""

from gymnasium.envs.registration import register

from .client import Emulator, EmulatorError
from .env import ACTIONS, IWannaBeTheGBEnv

register(id="IWannaBeTheGB-v0", entry_point="iwbtgb_rl.env:IWannaBeTheGBEnv")

__all__ = ["ACTIONS", "Emulator", "EmulatorError", "IWannaBeTheGBEnv"]
