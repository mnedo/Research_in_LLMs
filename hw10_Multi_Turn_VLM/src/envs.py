import numpy as np
import gymnasium as gym


ACTION2TXT = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}


def build_env(size: int = 4, slippery: bool = False):
    env = gym.make(
        "FrozenLake-v1",
        map_name=f"{size}x{size}",
        is_slippery=slippery,
        render_mode="rgb_array",
    )
    return env


def obs_to_image(env) -> np.ndarray:
    img = env.render()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def parse_state(obs) -> int:
    if isinstance(obs, tuple):
        obs = obs[0]
    return int(obs)

