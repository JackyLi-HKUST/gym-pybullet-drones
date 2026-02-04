import time
import numpy as np

from gym_pybullet_drones.envs.wave_pad_env import WavePadHoverEnv


def get_hover_action(env):
    """
    Different versions/configs expose different action conventions.
    We'll try a few common ones safely.
    """
    n = env.NUM_DRONES

    # Common in gym-pybullet-drones: HOVER_RPM exists
    if hasattr(env, "HOVER_RPM"):
        return np.ones((n, 4)) * float(env.HOVER_RPM)

    # Sometimes it's named HOVER_RPM or HOVER_RPMs; try a fallback
    if hasattr(env, "HOVER_RPMs"):
        hr = np.array(env.HOVER_RPMs).reshape(1, -1)
        if hr.shape[1] == 4:
            return np.repeat(hr, n, axis=0)

    # If unknown, return zeros (drone may drop, but sim still shows pad)
    return np.zeros((n, 4))


if __name__ == "__main__":
    initial_xyzs = np.array([[0.0, 0.0, 1.0]])
    initial_rpys = np.array([[0.0, 0.0, 0.0]])

    env = WavePadHoverEnv(
        gui=True,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
    )

    obs, info = env.reset()

    for _ in range(20000):
        action = get_hover_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if _ % 120 == 0:
            print("pad center:", info["pad_top_center_world"])
            print("pad rpy:", info["pad_rpy_world"])
        # Slow down to real-time so you can see the motion
        time.sleep(env.CTRL_TIMESTEP)

       # if terminated or truncated:
        #    obs, info = env.reset()

    env.close()
