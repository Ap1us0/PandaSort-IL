"""The MuJoCo Panda lift benchmark."""

from gymnasium.envs.registration import register

register(id="PandaGrasp-v0", entry_point="panda_grasp.env:PandaGraspEnv")

register(id="PandaPickPlace-v0", entry_point="panda_grasp.place:PandaPickPlaceEnv")
register(id="PandaObjectPlace-v0", entry_point="panda_grasp.place_objects:ObjectPlacementEnv")
