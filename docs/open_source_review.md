# 开源源码调研

源码来自官方仓库的本地检出。表中的链接固定到本次实际阅读的版本。三套上游实现用于对照任务、控制和评估设计。本项目直接调用固定版本的 SB3，物理模型及专家例程来源见本页末尾。

| 项目 | Commit | 许可 |
|---|---|---|
| ManiSkill | `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` | [Apache-2.0](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/LICENSE) |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | [MIT](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/LICENSE) |
| Stable-Baselines3 v2.7.1 | `8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458` | [MIT](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/LICENSE) |

## ManiSkill PickCube-v1

- `PickCubeEnv` 注册的时限是 50 步。Panda 配置为边长 4 cm 的方块，初始 XY 和绕 Z 的角度随机，目标可位于空中。成功条件是物体中心进入目标 2.5 cm 范围且机械臂静止。抓持标记参与观测和奖励。密集奖励把 TCP 接近、双指抓持、抓持后的物体到目标接近和到位后的减速相加，成功覆写为 5，归一版除以 5。[pick_cube.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/pick_cube.py) · [pick_cube_cfgs.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py)
- Panda 控制器提供 `pd_ee_delta_pos` 和带姿态增量的 `pd_ee_delta_pose`，都可拼接单一联动夹爪动作。前者使用三维末端平移，Panda 配置将分量限制在 ±0.1 m。控制器根据当前或缓存目标生成 EE 目标，求 IK 并发送关节目标。Panda 的 `is_grasping` 同时检查双指各自的接触力范数与方向，默认阈值 0.5 N、85°。[panda.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/agents/robots/panda/panda.py) · [pd_ee_pose.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/agents/controllers/pd_ee_pose.py)
- 官方 Panda 专家以物体 OBB 生成抓取位姿，依次运动到预抓取点和抓取点、闭爪，再移动到目标位姿。运行器以 `pd_joint_pos` 生成经 `RecordEpisode` 保存的演示，支持只保留成功样本。轨迹工具支持控制模式转换和通过动作重放验证成功。转换后的动作需要与学习环境对应。[solutions/pick_cube.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/solutions/pick_cube.py) · [run.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/run.py) · [replay_trajectory.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/trajectory/replay_trajectory.py)
- BC 源码按 HDF5 和 JSON 读取轨迹，配对 `obs[:-1]` 与 actions，使用两层 256 单元 ReLU MLP 和动作 MSE。评估按 `success_once`、`success_at_end` 记录，并分别选择最佳 checkpoint。SAC 示例使用 replay buffer、tanh Gaussian actor 和双 Q，评估使用确定性动作和独立评估环境。[bc.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/bc/bc.py) · [sac.py](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/sac/sac.py)

## robosuite Lift 与 PickPlace

- `Lift._check_success` 使用方块中心高度大于桌面高度加 0.04 m。奖励成功时为 2.25，否则在开启 shaping 时叠加 `1 - tanh(10d)` 和双指抓住时的 0.25，最后乘 `reward_scale / 2.25`。本提交的成功条件没有持续时间要求。通用 `_check_grasp` 按左右 fingerpad geom 组分别查接触。[lift.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/lift.py) · [manipulation_env.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/manipulation_env.py)
- `PickPlace` 检查物体位于指定 bin 内，且抓手远到由距离计算的 `r_reach < 0.6`，单物体模式与全物体模式分别检查任一个或全部。奖励以正确入 bin 的数量为稀疏项，可再加接近、抓住、抬升、悬停四项的最大值。抬升项取决于抓持，悬停项按物体是否处于目标 bin 上方区分。[pick_place.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/pick_place.py)
- `OSC` 在仅控制位置与同时控制姿态时分别为三维、六维固定阻抗手臂指令，并可按阻抗模式扩展。增量先经 `scale_action` 再更新目标。Gym 包装的 Box 取实际 `action_spec`，这取决于机器人、手臂与夹爪配置。基础环境的 `done` 依据 horizon，Gym 包装将它作为 terminated 返回并令 truncated 为 False。评估任务成功需另读成功判定。[osc.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/controllers/parts/arm/osc.py) · [gym_wrapper.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/wrappers/gym_wrapper.py) · [base.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/base.py)
- 人工演示 wrapper 记录 MuJoCo 展平状态和动作，收集脚本将成功轨迹、模型 XML 与环境元数据打包为 HDF5。回放脚本既能逐条设置状态供查看，也能从初态开始开环执行动作并比较实际状态。本项目的评估和视频使用后者对应的物理动作重放思路。[data_collection_wrapper.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/wrappers/data_collection_wrapper.py) · [collect_human_demonstrations.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/scripts/collect_human_demonstrations.py) · [playback_demonstrations_from_hdf5.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/scripts/playback_demonstrations_from_hdf5.py)

## Stable-Baselines3 SAC

- `SAC.train` 从 replay buffer 采样，最小化双 critic 的 TD 误差，目标 Q 使用 target critics 中较小的值并加入熵；actor 优化目标为 `alpha * log_prob - min(Q1, Q2)`。支持自动学习熵系数与 Polyak 更新。off-policy 基类在 `learning_starts` 以前均匀采样，存 Box 缩放后的动作，将终止观测与重置后观测区分处理。[sac.py](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/sac/sac.py) · [off_policy_algorithm.py](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/off_policy_algorithm.py)
- 默认非 gSDE actor 包含特征提取、`latent_pi`、`mu`、独立 `log_std`，形成 squash 后的 Gaussian。确定性输出是 tanh 后的均值。确定性 BC 缺少方差头和 SAC critic，因此本项目显式复制兼容的隐藏层和均值头，再单独初始化方差头。[policies.py](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/sac/policies.py)
- `evaluate_policy` 默认确定性预测并汇总各回合回报/长度。 `EvalCallback` 可通过 `info["is_success"]` 记录成功率。项目使用同样的预测模式，额外逐场景保存成功、抓持、掉落与稳定性。[evaluation.py](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/evaluation.py) · [callbacks.py](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/callbacks.py)

## 本地复用

- `轨迹规划/models/franka_emika_panda/` 提供 MJCF 与网格。模型来源记录对应 Menagerie `370c43965323c0944b433b202aac5058b5f907a9`。随项目保留原始模型许可证。[Menagerie Panda 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda)
- `夹取/src/panda_pick_place.py` 提供联动夹爪、法向力接触检测和原专家的 approach、descend、grasp、lift、hold 次序及 11.5 cm 预抓取、9 cm 提升航点。原控制器跟踪规划后的关节轨迹。移植版将这三段航点改为对实时 TCP 计算四维动作，闭爪时序缩短，且在单方块任务完成后结束。接触检测从源状态机中解耦，清除阶段和被动方块过滤。`夹取/models/franka_emika_panda/pick_place.xml` 提供方块、桌与摩擦设置。
- `强化学习物体操作/panda_push/panda_cartesian.py` 提供六维迭代 IK、位姿误差和关节伺服，引入固定抓取姿态与工作区。模型加载和执行器检查来自 `轨迹规划/src/panda_position.py`。本地脚本按工作区内复用。[原文件 SHA-256](local-source-sha256.json) 可定位实际输入。
