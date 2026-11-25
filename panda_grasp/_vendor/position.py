"""Selected model utilities from 轨迹规划/src/panda_position.py."""

from pathlib import Path

import mujoco

SCENE = Path(__file__).resolve().parents[1] / "model" / "scene.xml"


def load_panda_spec(scene: Path = SCENE) -> mujoco.MjSpec:
    # Python handles Unicode Windows paths. MuJoCo resolves the relative model
    # resources from memory, independently of the current working directory.
    directory = SCENE.parent
    assets = {
        f"assets/{file.name}": file.read_bytes()
        for file in (directory / "assets").iterdir() if file.is_file()
    }
    return mujoco.MjSpec.from_string(
        scene.read_text(encoding="utf-8"),
        include={name: (directory / name).read_bytes() for name in ("panda.xml", "scene.xml")},
        assets=assets,
    )


def require_position_servo(model: mujoco.MjModel, joint_name: str, actuator_name: str) -> None:
    joint = model.joint(joint_name)
    actuator = model.actuator(actuator_name)
    kp = actuator.gainprm[0]
    bias_offset, position_bias, velocity_bias = actuator.biasprm[:3]
    if not (
        joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        and actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
        and actuator.trnid[0] == joint.id
        and actuator.gear[0] == 1.0
        and actuator.dyntype[0] == mujoco.mjtDyn.mjDYN_NONE
        and actuator.gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED
        and actuator.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
        and kp > 0
        and bias_offset == 0
        and position_bias == -kp
        and velocity_bias <= 0
    ):
        raise ValueError(f"{actuator_name} must be a position servo for {joint_name} with a 1:1 transmission")
