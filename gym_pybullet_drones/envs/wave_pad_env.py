import os
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.HoverAviary import HoverAviary


class WavePad:
    """Deterministic wave motion generator for a landing pad."""

    def __init__(
        self,
        base_pos=(0.0, 0.0, 0.05),
        heave_amp=0.05,     # meters
        heave_freq=0.3,     # Hz
        roll_amp_deg=5.0,   # degrees
        roll_freq=0.2,      # Hz
        pitch_amp_deg=3.0,  # degrees
        pitch_freq=0.25,    # Hz
        drift_amp=0.0,      # meters (optional x/y drift)
        drift_freq=0.05     # Hz
    ):
        self.base_pos = np.array(base_pos, dtype=float)

        self.heave_amp = float(heave_amp)
        self.heave_w = 2.0 * np.pi * float(heave_freq)

        self.roll_amp = np.deg2rad(float(roll_amp_deg))
        self.roll_w = 2.0 * np.pi * float(roll_freq)

        self.pitch_amp = np.deg2rad(float(pitch_amp_deg))
        self.pitch_w = 2.0 * np.pi * float(pitch_freq)

        self.drift_amp = float(drift_amp)
        self.drift_w = 2.0 * np.pi * float(drift_freq)

    def pose(self, t: float):
        """Return (pos, quat) at time t."""
        x0, y0, z0 = self.base_pos

        # Heave (z)
        z = z0 + self.heave_amp * np.sin(self.heave_w * t)

        # Optional drift (x,y)
        x = x0 + self.drift_amp * np.sin(self.drift_w * t)
        y = y0 + self.drift_amp * np.cos(self.drift_w * t)

        # Roll / Pitch (radians)
        roll = self.roll_amp * np.sin(self.roll_w * t)
        pitch = self.pitch_amp * np.sin(self.pitch_w * t)
        yaw = 0.0

        quat = p.getQuaternionFromEuler([roll, pitch, yaw])
        return (x, y, z), quat


class WavePadHoverEnv(HoverAviary):
    """
    HoverAviary + a moving/tilting pad (kinematic object controlled by resetBasePositionAndOrientation).
    """

    def __init__(self, *args, pad_urdf_path=None, **kwargs):
        # Defensive: ignore older args you might accidentally pass
        kwargs.pop("obstacles", None)
        kwargs.pop("num_drones", None)

        # ---- IMPORTANT: define these BEFORE super().__init__()
        # because BaseAviary.__init__ will call self._addObstacles()
        self._pad_id = None
        self._t = 0.0

        # You placed URDF at: gym_pybullet_drones/assets/wave_pad.urdf
        if pad_urdf_path is None:
            pkg_root = os.path.dirname(os.path.dirname(__file__))  # -> gym_pybullet_drones/
            pad_urdf_path = os.path.join(pkg_root, "assets", "wave_pad.urdf")

        self._pad_urdf_path = os.path.abspath(pad_urdf_path)
        if not os.path.isfile(self._pad_urdf_path):
            raise FileNotFoundError(f"wave pad URDF not found at: {self._pad_urdf_path}")

        self._wave = WavePad(
            base_pos=(0.0, 0.0, 0.10),
            heave_amp=0.25,      # 25 cm (very obvious)
            heave_freq=0.8,      # faster
            roll_amp_deg=25.0,   # very obvious tilt
            roll_freq=0.6,
            pitch_amp_deg=15.0,
            pitch_freq=0.7,
            drift_amp=0.20,      # x/y drift
            drift_freq=0.2
        )

        # -----------------------------------------------------

        super().__init__(*args, **kwargs)

    def _addObstacles(self):
        """
        Called inside BaseAviary initialization. We add our pad here.
        Keeping super()._addObstacles() is usually safe (it may add other standard obstacles).
        If you want ONLY the pad, comment out the super() line.
        """
        super()._addObstacles()

        pos0, quat0 = self._wave.pose(0.0)
        self._pad_id = p.loadURDF(
            self._pad_urdf_path,
            basePosition=pos0,
            baseOrientation=quat0,
            useFixedBase=True,
            physicsClientId=self.CLIENT
        )

        # Make deck less slippery
        p.changeDynamics(self._pad_id, -1, lateralFriction=1.0, physicsClientId=self.CLIENT)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._t = 0.0
        self._update_pad()
        return obs, info

    def step(self, action):
        self._t += float(self.CTRL_TIMESTEP)
        self._update_pad()
        return super().step(action)

    def _update_pad(self):
        if self._pad_id is None:
            return
        pos, quat = self._wave.pose(self._t)
        p.resetBasePositionAndOrientation(self._pad_id, pos, quat, physicsClientId=self.CLIENT)
