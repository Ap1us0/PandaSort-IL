"""Selected functions from the sibling trajectory-planning project. See docs/code-source.json."""

from copy import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from .position import require_position_servo


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))


TCP_OFFSET_M = np.array([0.0, 0.0, 0.1034])


JOINT_MARGIN_RAD = 0.05


ORIENTATION_WEIGHT = 0.25  # meters per radian in the weighted pose error


IK_DAMPING = 0.01


SINGULAR_VALUE_THRESHOLD = 0.06


NEAR_SINGULAR_DAMPING = 0.12


IK_TOLERANCE = 1e-8


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Arm:
    hand: int
    qpos: NDArray[np.intp]
    dofs: NDArray[np.intp]
    actuators: NDArray[np.intp]
    joint_range: FloatArray
    ctrl_range: FloatArray
    torque_range: FloatArray
    kp: FloatArray
    kd: FloatArray


def read_arm(model: mujoco.MjModel) -> Arm:
    joints = np.array([model.joint(name).id for name in JOINT_NAMES])
    actuators = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
    for joint, actuator in zip(joints, actuators):
        require_position_servo(model, model.joint(int(joint)).name, model.actuator(int(actuator)).name)
    if not (
        model.jnt_limited[joints].all()
        and model.actuator_ctrllimited[actuators].all()
        and model.actuator_forcelimited[actuators].all()
    ):
        raise ValueError("All seven arm joints must have position, control, and output limits")
    return Arm(
        model.body("hand").id,
        model.jnt_qposadr[joints],
        model.jnt_dofadr[joints],
        actuators,
        model.jnt_range[joints],
        model.actuator_ctrlrange[actuators],
        model.actuator_forcerange[actuators],
        model.actuator_gainprm[actuators, 0],
        -model.actuator_biasprm[actuators, 2],
    )


def tcp_position(data: mujoco.MjData, arm: Arm, offset: FloatArray = TCP_OFFSET_M) -> FloatArray:
    return data.xpos[arm.hand] + data.xmat[arm.hand].reshape(3, 3) @ offset


def unit_quaternion(orientation: FloatArray) -> FloatArray:
    quaternion = np.asarray(orientation, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.any(quaternion):
        raise ValueError("Orientation must be a finite nonzero wxyz quaternion")
    scaled = quaternion / np.max(np.abs(quaternion))
    return scaled / np.linalg.norm(scaled)


def rotation_error(target: FloatArray, actual: FloatArray) -> FloatArray:
    """Shortest rotation vector from actual to target, in world axes (radians)."""
    inverse = actual * (1, -1, -1, -1)
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, target, inverse)
    if relative[0] < 0:
        relative *= -1
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, relative, 1.0)
    return error


def damped_step(jacobian: FloatArray, error: FloatArray) -> FloatArray:
    u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    proximity = max(0.0, 1.0 - singular_values[-1] / SINGULAR_VALUE_THRESHOLD)
    damping = IK_DAMPING + NEAR_SINGULAR_DAMPING * proximity**2
    gains = singular_values / (singular_values**2 + damping**2)
    return vt.T @ (gains * (u.T @ error))


def check_contacts(model: mujoco.MjModel, data: mujoco.MjData, stage: str) -> None:
    for contact in data.contact:
        if contact.dist < -1e-6:
            bodies = [model.body(int(model.geom_bodyid[geom])).name for geom in contact.geom]
            raise RuntimeError(f"{stage}: collision {bodies[0]} / {bodies[1]} (dist={contact.dist:.6f} m)")


class InverseKinematics:
    def __init__(self, model: mujoco.MjModel, home: mujoco.MjData, arm: Arm):
        self.model = model
        self.arm = arm
        self.tcp_offset = TCP_OFFSET_M
        # Candidate poses live in separate data. Only mj_step moves the simulated arm.
        self.scratch = copy(home)
        self.home_orientation = home.xquat[arm.hand].copy()
        self.lower = np.maximum(arm.joint_range[:, 0], arm.ctrl_range[:, 0]) + JOINT_MARGIN_RAD
        self.upper = np.minimum(arm.joint_range[:, 1], arm.ctrl_range[:, 1]) - JOINT_MARGIN_RAD
        self.jacp = np.empty((3, model.nv))
        self.jacr = np.empty((3, model.nv))

    def pose_error(
        self, angles: FloatArray, target: FloatArray, orientation: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        self.scratch.qpos[self.arm.qpos] = angles
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        point = tcp_position(self.scratch, self.arm, self.tcp_offset)
        angular_error = rotation_error(orientation, self.scratch.xquat[self.arm.hand])
        error = np.concatenate((target - point, ORIENTATION_WEIGHT * angular_error))
        return point, error

    def solve(
        self, target: FloatArray, seed: FloatArray, *,
        orientation: FloatArray | None = None, stage: str = "IK",
    ) -> FloatArray:
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("IK target must be a finite world XYZ vector")
        if seed.shape != self.lower.shape or not np.isfinite(seed).all():
            raise ValueError("IK seed must contain seven finite joint angles")
        orientation = self.home_orientation if orientation is None else unit_quaternion(orientation)
        angles, residual = self.iterate(target, seed, orientation, 120, IK_TOLERANCE)
        if residual >= IK_TOLERANCE:
            raise ValueError(f"{stage}: IK did not converge, weighted residual {residual:.6f}")
        mujoco.mj_fwdPosition(self.model, self.scratch)
        check_contacts(self.model, self.scratch, stage)
        return angles

    def iterate(self, target, seed, orientation, max_iterations, tolerance):
        angles = np.clip(seed, self.lower, self.upper)
        point, error = self.pose_error(angles, target, orientation)
        norm = np.linalg.norm(error)
        for _ in range(max_iterations):
            if norm < tolerance:
                break
            mujoco.mj_jac(self.model, self.scratch, self.jacp, self.jacr, point, self.arm.hand)
            jac = np.vstack((self.jacp[:, self.arm.dofs], ORIENTATION_WEIGHT * self.jacr[:, self.arm.dofs]))
            delta = damped_step(jac, error)
            delta *= min(1.0, 0.12 / max(0.12, np.max(np.abs(delta))))
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(angles + fraction * delta, self.lower, self.upper)
                trial_point, trial_error = self.pose_error(trial, target, orientation)
                trial_norm = np.linalg.norm(trial_error)
                if trial_norm < norm:
                    angles, point, error, norm = trial, trial_point, trial_error, trial_norm
                    break
            else:
                break
        return angles, norm


def joint_controls(data: mujoco.MjData, arm: Arm, angle: FloatArray, velocity: FloatArray) -> FloatArray:
    position = data.qpos[arm.qpos]
    speed = data.qvel[arm.dofs]
    torque = arm.kp * (angle - position) + arm.kd * (velocity - speed) + data.qfrc_bias[arm.dofs]
    torque = np.clip(torque, arm.torque_range[:, 0], arm.torque_range[:, 1])
    # Invert the verified general actuator law: tau = kp * (ctrl - q) - kd * dq.
    control = position + (torque + arm.kd * speed) / arm.kp
    return np.clip(control, arm.ctrl_range[:, 0], arm.ctrl_range[:, 1])

