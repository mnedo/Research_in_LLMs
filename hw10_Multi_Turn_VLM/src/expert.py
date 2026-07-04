import random
from dataclasses import dataclass

import numpy as np

from .envs import obs_to_image, parse_state


@dataclass
class StepRecord:
    image: np.ndarray
    state: int
    action: int
    reason: str
    reward: float
    done: bool


def _q_iteration(env, gamma: float = 0.99, iters: int = 300):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    q = np.zeros((n_s, n_a), dtype=np.float32)
    for _ in range(iters):
        new_q = np.copy(q)
        for s in range(n_s):
            for a in range(n_a):
                val = 0.0
                for p, ns, r, done in env.unwrapped.P[s][a]:
                    boot = 0.0 if done else gamma * np.max(q[ns])
                    val += p * (r + boot)
                new_q[s, a] = val
        q = new_q
    return q


def _reason(a: int) -> str:
    if a == 0:
        return "иду влево к безопасной клетке"
    if a == 1:
        return "иду вниз к цели"
    if a == 2:
        return "иду вправо, обхожу яму"
    return "иду вверх, не рискую"


def collect_expert_trajectories(env, n_traj: int = 1000, eps: float = 0.08, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    q = _q_iteration(env)
    pack = []

    for _ in range(n_traj):
        obs, _ = env.reset()
        done = False
        while not done:
            s = parse_state(obs)
            if random.random() < eps:
                a = int(env.action_space.sample())
            else:
                a = int(np.argmax(q[s]))

            img = obs_to_image(env)
            nxt, r, term, trunc, _ = env.step(a)
            done = term or trunc
            pack.append(
                StepRecord(
                    image=img,
                    state=s,
                    action=a,
                    reason=_reason(a),
                    reward=float(r),
                    done=done,
                )
            )
            obs = nxt
    return pack

