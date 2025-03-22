# 接收目标位置：目标位置以 pin.SE3 对象的形式存储在 goals 队列中，包含探测点的位置和方向。
# 队列处理：通过状态机和插值器 (interp)，按顺序处理 goals 中的目标位姿，驱动机械臂完成探测任务。
# 探索：利用搜索算法 (search.next()) 生成新的探测点 (palp_pt 和 surf_normal)，并根据探测结果（如硬度 stiffness）更新策略，实现对未知区域的探索。


# 在机械臂通过位置控制靠近目标点 (palp_se3) 并接触表面到一定程度（力或深度达到阈值）后，切换到力控制模式。
# 在此模式下，机械臂在 X-Y 平面内进行振荡运动，同时沿 Z 方向施加恒定下压力，计算刚度 (stiffness)。
# 当探测深度达到最大值或持续时间超过限制时，退出力控制，切换回位置控制并执行下一步动作（例如撤回）。



# 具体流程
# 目标生成：
#   当 goals 队列为空且插值完成时，调用 palp_pt, surf_normal = search.next() 生成新的探测点。
#   palpate(palp_pt, surf_normal) 将探测点转化为三个目标位姿（悬停 above_se3、探测 palp_se3、复位 reset_pose），加入 goals 队列。
# 队列处理：
#   主循环检测 len(goals) > 0 and interp.done，调用 state_transition() 从 goals 中取出下一个目标。
#   插值器 (interp) 生成平滑轨迹，机械臂按顺序移动到每个目标位姿。
# 探索与反馈：
#   在探测状态 (PalpateState.PALPATE) 下，机械臂施加力并测量硬度 (stiffness)。
#   硬度值通过 search.update_outcome(stiffness) 反馈给搜索算法，影响后续探测点的选择（例如，贝叶斯优化会优先选择硬度异常区域）。

import time
from collections import deque
import click
from pathlib import Path
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import open3d as o3d
import pinocchio as pin
# 机器人动力学库，SE3位姿等。
from scipy.spatial.transform import Rotation
# scipy 提供的旋转计算工具，处理四元数、旋转矩阵等。

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.transform_utils import quat2axisangle, quat2mat
from deoxys.utils import YamlConfig


from seebelow.algorithms.grid import SurfaceGridMap
from seebelow.algorithms.search import (
    RandomSearch,
    SearchHistory,
    Search,
    ActiveSearchWithRandomInit,
    ActiveSearchAlgos,
)
# 搜索算法: 生成待探测目标：
# 随机搜索 (RandomSearch)
# 主动搜索 (ActiveSearchWithRandomInit)



from seebelow.utils.control_utils import generate_joint_space_min_jerk
# 平滑运动轨迹生成器
from seebelow.utils.data_utils import DatasetWriter
# 数据写入器，存储实验数据。
from seebelow.utils.devices import ForceSensor
# 读取机械臂末端的力。
from seebelow.utils.interpolator import Interpolator, InterpType
# 插值器，用于平滑控制目标。
from seebelow.utils.time_utils import Ratekeeper
# 保持程序循环频率（如 50Hz）
from seebelow.utils.proc_utils import RingBuffer, RunningStats
# 环形缓冲区，实时保存最近的数据。
import seebelow.utils.constants as seebelow_const
from seebelow.utils.constants import PALP_CONST
from seebelow.utils.constants import PalpateState
from seebelow.utils.pcd_utils import scan2mesh, mesh2roi
from tfvis.visualizer import RealtimeVisualizer


def main_ctrl(shm_buffer, stop_event: mp.Event, save_folder: Path, search: Search):
# 子进程运行函数，控制机械臂运动。
# shm_buffer: 共享内存名称，和主进程交换数据。
# stop_event: 停止信号。
# save_folder: 数据保存路径。
# search: 搜索算法对象，提供目标位置。
    existing_shm = shared_memory.SharedMemory(name=shm_buffer)
    data_buffer = np.ndarray(1, dtype=seebelow_const.PALP_DTYPE, buffer=existing_shm.buf)
    # 绑定共享内存，data_buffer 用于实时存储机械臂位置、力、状态等。
    
    np.random.seed(PALP_CONST.seed)
#     设置 numpy 的随机数生成器的种子，保证算法运行的随机性可控、结果可复现。
#     PALP_CONST.seed 来自全局配置文件，通常在主函数里根据命令行参数设置：
    goals = deque([])
# goals：存储机器人执行过程中需要依次到达的目标点（包括过渡、探测点等）。
# deque（双向队列）：支持从头和尾高效插入、弹出操作，非常适合动态路径管理。
                # goals.appendleft(goal_pose)  # 插入新目标
                # next_goal = goals.pop()     # 取出下一个目标



    force_buffer = RingBuffer(PALP_CONST.buffer_size)
    pos_buffer = RingBuffer(PALP_CONST.buffer_size)
# force_buffer: 存储最近一段时间的力量值（Fz）。
# pos_buffer: 存储探测过程中的机械臂末端位置变化。
# 两个都是RingBuffer对象 固定长度，超过部分自动覆盖，保存最新的数据。长度由PALP_CONST.buffer_size 定义



    running_stats = RunningStats()
# 创建一个 RunningStats 实例，用于实时计算力量和位移数据的均值和方差。
# 用于平滑力传感器和位置信息，判断探测是否稳定。
# RunningStats 内部使用 滑动窗口统计，不会存储所有数据，而是增量更新均值和标准差。



    palp_state = PalpateState()
# 创建一个 PalpateState 状态机，用于管理 机械臂的探测流程（palpation）。
# 控制触觉探测的不同阶段：
# INIT → 机械臂移动到初始位置
# APPROACH → 机械臂靠近目标点
# PALPATE → 进行触觉探测（按压）
# RETRACT → 机械臂离开目标
# TERMINATE → 探测完成，退出流程



    curr_pose_se3 = pin.SE3.Identity()
# pin.SE3.Identity() 表示一个 单位变换矩阵（机械臂当前的初始位姿）。
# SE3 是 刚体运动中的 6D 变换矩阵，包含：
# 平移 (translation) → 机械臂的 XYZ 位置
# 旋转 (rotation) → 机械臂的方向




    using_force_control_flag = False
    collect_points_flag = False
    curr_eef_pose = None



    F_norm = 0.0
    Fxyz = np.zeros(3)

    palp_pt = None
    surf_normal = None
# palp_pt	当前要探测的目标点	由搜索算法 search.next() 生成
# surf_normal	目标点的表面法线	机械臂会沿着这个方向进行探测



    palp_progress = 0.0  # [0, 1]
    palp_id = 0
    stiffness = 0.0




    search_history = SearchHistory()
# Search 是 基类（抽象类），定义了 搜索算法必须实现的方法：
# next() 选择下一个探测点。
# update_outcome(prev_value) 更新网格状态。

    CF_start_time = None
    oscill_start_time = None
    start_angles = np.full(2, -PALP_CONST.angle_oscill)  # theta, phi

# theta (θ) → 仰角，控制偏离法线方向的“倾斜角度”如果 theta = 0，那方向就是法线方向。
# phi (φ) → 方位角，控制绕法线旋转的方向，如果 phi 变化，就是绕着法线在平面上转一圈。


    end_angles = np.full(2, PALP_CONST.angle_oscill)
# 假设：PALP_CONST.angle_oscill = 15 
# 则：
# start_angles = [-15, -15]
# end_angles = [15, 15]



    max_cf_time = 0.1 if PALP_CONST.discrete_only else PALP_CONST.max_cf_time

# 生成轨迹traj = generate_joint_space_min_jerk(start, goal, time_to_go, dt)
# start	起始点（通常是角度向量）	[θ₀, φ₀]
# goal	终点（目标角度）	[θ₁, φ₁]
# time_to_go	这段运动的总时间（秒）	1.0
# dt	每一步的时间间隔（秒）	0.01（100Hz 控制频率）
    force_oscill_out = generate_joint_space_min_jerk(
        start_angles,
        end_angles,
        PALP_CONST.t_oscill / 2,
        1 / PALP_CONST.ctrl_freq,
    )


    
    force_oscill_in = generate_joint_space_min_jerk(
        end_angles,
        start_angles,
        PALP_CONST.t_oscill / 2,
        1 / PALP_CONST.ctrl_freq,
    )


    force_oscill_traj = force_oscill_out + force_oscill_in
# 这里的轨迹字典都是归一化后的，实际上的位移还要乘以振幅。这里的振荡策略是用球坐标，给定一个振荡角度
# force_oscill_traj = [
    # {
    #     "time_from_start": float,从轨迹开始以来的相对时间
    #     "position": np.array([theta, phi]),当前角度偏移量（θ, φ） → 控制方向的微调
    #     "velocity": np.array([...]),角速度
    #     "acceleration": np.array([...])角加速度
    # },


    # 注意这里其实是个字典“列表”
    # force_oscill_traj = [
#     {"position": [...], "velocity": [...], "acceleration": [...]},
#     {"position": [...], "velocity": [...], "acceleration": [...]},
#     {"position": [...], ...},
#     ...
#       ]
# 

    force_cap = ForceSensor()
    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG),
        use_visualizer=False,
        control_freq=80,
    )





    force_ctrl_cfg = YamlConfig(str(seebelow_const.FORCE_CTRL_CFG)).as_easydict()
    
    
    osc_abs_ctrl_cfg = YamlConfig(str(seebelow_const.OSC_ABSOLUTE_CFG)).as_easydict()
    robot_interface._state_buffer = []
    interp = Interpolator(interp_type=InterpType.SE3)
    # 创建一个轨迹插值器对象 interp，用于在给定起点/终点之间生成平滑的运动轨迹。

    def palpate(pos, O_surf_norm_unit=np.array([0, 0, 1])):
# pos	np.array([x, y, z])	探测点的三维位置
# O_surf_norm_unit	np.array([x, y, z])	探测点表面的单位法向量（默认是 z 轴方向）


        assert np.isclose(np.linalg.norm(O_surf_norm_unit), 1)

        # Uses Kabasch algo get rotation that aligns the eef tip -z_axis and
        # the normal vector, and secondly ensure that the y_axis is aligned
# O_surf_norm_unit是一个目标方向，是当前皮肤表面的法线方向，目的是构造一个旋转矩阵让机械臂末端对准这个方向
# Rotation.align_vectors(A, B, weights)计算一个最优旋转 R，使得 A 中的每个向量尽量对齐 B 中的对应向量。
# A 是一组源向量，B 是目标向量，weights 表示每对向量的重要程度。

# 使用 Scipy 的 align_vectors 进行姿态对齐。
# 目标是让末端执行器的 -z 轴 对齐到 surf_normal。
# 同时让末端的 y 轴尽量对齐与全局 y 轴方向一致，保证姿态不歪斜。
        R = Rotation.align_vectors(
            np.array([O_surf_norm_unit, np.array([0, -1, 0])]),
            # A[0]：你希望对齐的方向（表面法线） A[1]：末端原始 Y 轴（反向 Y，用于辅助）
            np.array([[0, 0, -1], np.array([0, 1, 0])]),
            # B[0]：-Z 轴（表示力探测器的朝向） B[1]：全局 Y 轴
            weights=np.array([10, 0.1]),
# 第一个向量（法线）非常重要 → 权重 = 10
# 第二个辅助向量（y 轴）不重要 → 权重 = 0.1
        )[0].as_matrix()
# 得到一个 3×3 的旋转矩阵 R

# 下压的探测位置姿态
        palp_se3 = pin.SE3.Identity()
# pin.SE3.Identity() 表示：
# 位置是 [0, 0, 0]
# 方向是单位旋转矩阵（无旋转）
        palp_se3.translation = pos - PALP_CONST.palpate_depth * O_surf_norm_unit
# pos	当前探测点在空间中的位置（如肿瘤表面上的一个点）
# O_surf_norm_unit	表面在该点的法向量（单位向量，表示垂直方向）
# PALP_CONST.palpate_depth	探测的“下压深度”，单位通常是米或毫米
#           pos = [0.5, 0.2, 0.1]         # 表面点
#           O_surf_norm_unit = [0, 0, 1]  # 法线朝上
#           palpate_depth = 0.01          # 下压 1cm
# 则palp_se3.translation = [0.5, 0.2, 0.1 - 0.01] = [0.5, 0.2, 0.09]
        palp_se3.rotation = R
# 刚刚对齐得到的旋转矩阵R，使得机械臂末端朝向和表面法向垂直


# 悬停的探测姿态
        above_se3 = pin.SE3.Identity()
        above_se3.translation = pos + PALP_CONST.above_height * O_surf_norm_unit
# 同上面，只是把高度换成了在皮肤表面以上的高度
        above_se3.rotation = R
# 依然对其法向



        reset_pose = pin.SE3.Identity()
        reset_pose.translation = seebelow_const.RESET_PALP_POSE[:3]
# seebelow_const.RESET_PALP_POSE 是一个长度为 7 的数组：，前三个是平移向量，后四个是旋转四元数



        reset_pose.rotation = quat2mat(seebelow_const.RESET_PALP_POSE[3:7])

        goals.appendleft(above_se3)
        goals.appendleft(palp_se3)
        goals.appendleft(reset_pose)


# goals 是一个 deque 类型的双端队列，在主控制循环中用来存储即将执行的机械臂目标姿态。
# 使用 appendleft() 意味着：新目标放在队列头部，先加入的会最后执行

    def state_transition():
# “根据当前控制状态、目标队列 goals、插值器进度，判断是否该进入下一阶段，比如从 reset → approach → palpate → retract”
# 在当前一个目标姿态执行完毕后，切换到下一个目标，并初始化下一段插值轨迹。

        palp_state.next()
        # 调用状态机 palp_state 的 .next() 方法
# 进入下一个状态（如从 INIT → APPROACH → PALPATE → RETRACT）

        steps = seebelow_const.STEP_FAST
        # 切换状态用的步数，快速移动
        if palp_state.state == PalpateState.PALPATE:
            steps = seebelow_const.STEP_SLOW
            # 如果下一步还是触诊则慢速移动
        pose_goal = goals.pop()
        # pop方法会倒着读取append的目标
        interp.init(curr_pose_se3, pose_goal, steps=steps)
# interp是插值器，给定两个目标和步数进行插值


    while len(robot_interface._state_buffer) == 0:
        # 直到机器人控制接口开始正常运行，主程序才会继续往下执行。
        continue

    start_pose = pin.SE3.Identity()
    start_pose.translation = seebelow_const.GT_SCAN_POSE[:3]
    start_pose.rotation = quat2mat(seebelow_const.GT_SCAN_POSE[3:7])
    goals.appendleft(start_pose)

    curr_eef_pose = robot_interface.last_eef_rot_and_pos
    curr_pose_se3.rotation = curr_eef_pose[0]
    curr_pose_se3.translation = curr_eef_pose[1]


    pose_goal = goals.pop()


    interp.init(curr_pose_se3, pose_goal, steps=seebelow_const.STEP_FAST)
    try:
        while not stop_event.is_set():
            # 获取当前位置
            curr_eef_pose = robot_interface.last_eef_rot_and_pos
            curr_pose_se3.rotation = curr_eef_pose[0]
            curr_pose_se3.translation = curr_eef_pose[1]
            # 读取力传感器数据
            Fxyz_temp = force_cap.read()

            if Fxyz_temp is not None:
                Fxyz = Fxyz_temp
                F_norm = np.sqrt(np.sum(Fxyz**2))

            force_buffer.append(F_norm)

            # 记录位移
            if palp_pt is not None:
                pos_buffer.append(np.linalg.norm(palp_pt - curr_pose_se3.translation))

            if force_buffer.overflowed():
                running_stats.update(force_buffer.buffer)

            # terminate palpation and reset goals

# 械臂处于力控制模式 (using_force_control_flag)，并且满足以下任一情况：
# 探测进度达到或超过最大深度 (palp_progress >= 1)。
# 力控制模式运行时间过长 (time.time() - CF_start_time > max_cf_time)。
            if (using_force_control_flag
                    and (palp_progress >= 1 or time.time() - CF_start_time > max_cf_time)
                    # and pos_buffer.std < PALP_CONST.pos_stable_thres
                ):
                print("palpation done")
                collect_points_flag = False
                using_force_control_flag = False
                state_transition()
                palp_id += 1
                if palp_id == PALP_CONST.max_palpations:
                    print("terminate")
                    palp_state.state = PalpateState.TERMINATE
                    goals.clear()
                    interp.init(curr_pose_se3, start_pose, steps=seebelow_const.STEP_FAST)

            # initiate palpate
            if len(goals) > 0 and interp.done:
                state_transition()
            elif len(goals) == 0 and interp.done:
                if palp_state.state == PalpateState.TERMINATE:
                    print("breaking")
                    break
                palp_pt, surf_normal = search.next()
                search_history.add(*search.grid_estimate)
                palpate(palp_pt, surf_normal)

            # start palpation
            stiffness = 0.0
            if palp_state.state == PalpateState.PALPATE:
                assert palp_pt is not None



                # update stiffness
                palp_disp = curr_pose_se3.translation - palp_pt
# 计算机械臂末端当前位置 (curr_pose_se3.translation) 与探测目标点 (palp_pt) 的位移向量。
# 类型为 np.ndarray，形状为 (3,)。

                # 将位移向量投影到负表面法向量方向（探测通常是向下压）。再除以最大允许位移量
                #      # between 0 and 1
                palp_progress = (np.dot(palp_disp, -surf_normal) / PALP_CONST.max_palp_disp)
                
                # 如果力或深度达到阈值且未进入力控制模式，准备切换到力控制。
                if (Fxyz[2] >= PALP_CONST.max_Fz #z轴的力是否大于最大允许力
                        or palp_progress >= 1.0) and not using_force_control_flag:
                  


                    CF_start_time = time.time()
                    # 记录力控制模式的时间戳、





                    # 进入轮廓跟随模式
                    print("CONTOUR FOLLOWING!")
                    stiffness = Fxyz[2] / (np.linalg.norm(curr_pose_se3.translation - palp_pt) +
                                           1e-6)
                    # 计算硬度：力 (Fxyz[2]) 除以位移距离。
                    # 1e-6 避免除以零。
                    stiffness /= PALP_CONST.stiffness_normalization#硬度归一化处理
                    using_force_control_flag = True
                    collect_points_flag = True
                    oscill_start_time = time.time()
                    print("STIFFNESS: ", stiffness)
                    search.update_outcome(stiffness)



            # 力控制模式control: force
            if using_force_control_flag:
                if time.time() - oscill_start_time > PALP_CONST.t_oscill:
                    # 检查振荡周期是否完成，超过振荡周期三预设时间（常量）则更新，开始新一轮振荡
                    oscill_start_time = time.time()
                
                idx = int((time.time() - oscill_start_time) / (1 / PALP_CONST.ctrl_freq))
#计算当前振荡轨迹的索引。
# time.time() - oscill_start_time: 当前振荡周期内经过的时间。
# 1 / PALP_CONST.ctrl_freq: 控制循环的时间步长（例如，若 ctrl_freq = 80 Hz，则步长为 1/80 = 0.0125 秒）。
# int(...): 将时间转换为轨迹数组中的整数索引。
                action = np.zeros(9)

                # force_oscill_traj是预先计算好的轨迹字典“列表” 选择了第idx帧的position，格式是x,y
                oscill_pos = force_oscill_traj[idx]["position"]
                action[0] = oscill_pos[0]
                action[1] = oscill_pos[1]
                action[2] = -0.005


                robot_interface.control(
                    controller_type=seebelow_const.FORCE_CTRL_TYPE,
                    action=action,
                    controller_cfg=force_ctrl_cfg,
                )



            # control: OSC  Operational Space Control  位置控制模式
            else:
                action = np.zeros(7)
                next_se3_pose = interp.next()

# 调用.next方法会从
# self._trajectory = [
#     SE3(x0, R0),
#     SE3(x1, R1),
#     SE3(x2, R2),
#     ...
# ]选取下一个插值点
                target_xyz_quat = pin.SE3ToXYZQUAT(next_se3_pose)
# 将 SE3 位姿转换为 XYZ + 四元数格式。
# 返回一个 7 维数组：[x, y, z, qx, qy, qz, qw]。
                axis_angle = quat2axisangle(target_xyz_quat[3:7])
# 将四元数 (target_xyz_quat[3:7]) 转换为轴角表示 (axis_angle)。
# 轴角是一个 3 维向量，表示旋转轴和旋转角度。
                # print(se3_pose)
                action[:3] = target_xyz_quat[:3]
                action[3:6] = axis_angle
                # print(action)

                robot_interface.control(
                    controller_type=seebelow_const.OSC_CTRL_TYPE,
                    action=action,
                    controller_cfg=osc_abs_ctrl_cfg,
                )

                
            q, p = robot_interface.last_eef_quat_and_pos
            # quat and position
            save_action = np.zeros(9)
            save_action[:len(action)] = action
            data_buffer[0] = (
                Fxyz,
                q.flatten(),  # quat
                p.flatten(),  # pos
                target_xyz_quat[3:7],#目标四元数
                target_xyz_quat[:3],#目标位置
                save_action,
                palp_progress,
                palp_pt if palp_pt is not None else np.zeros(3),
                surf_normal if surf_normal is not None else np.zeros(3),
                palp_id,
                palp_state.state,
                stiffness,
                using_force_control_flag,
                collect_points_flag,
            )

    except KeyboardInterrupt:
        pass

    # stop
    robot_interface.close()
    search_history.save(save_folder)
    print("history saved")
    stop_event.set()
    existing_shm.close()


@click.command()
@click.option(
    "--tumor",
    "-t",
    type=str,
    help="tumor type [crescent,hemisphere]",
    default="hemisphere",
)
@click.option("--algo", "-a", type=str, help="algorithm [bo, random]", default="random")
@click.option("--select_bbox", "-b", type=bool, help="choose bounding box", default=False)
@click.option("--max_palpations", "-m", type=int, help="max palpations", default=60)
@click.option("--autosave", "-s", type=bool, help="autosave", default=False)
@click.option("--seed", "-e", type=int, help="seed", default=None)
@click.option("--debug", "-d", type=bool, help="runs visualizations", default=False)
@click.option("--discrete_only", "-s", type=bool, help="discrete probing only", default=False)





def main(tumor, algo, select_bbox, max_palpations, autosave, seed, debug, discrete_only):
    pcd = o3d.io.read_point_cloud(str(seebelow_const.SURFACE_SCAN_PATH))
# 使用 open3d 库读取点云文件。
# seebelow_const.SURFACE_SCAN_PATH: 表面扫描数据的文件路径（常量）。
# pcd: 返回一个 open3d.geometry.PointCloud 对象，表示表面点云。
    surface_mesh = scan2mesh(pcd)#点云转化为网格(open3d.geometry.TriangleMesh)
    PALP_CONST.max_palpations = max_palpations
    PALP_CONST.algo = algo
    PALP_CONST.seed = np.random.randint(1000) if seed is None else seed
    PALP_CONST.tumor_type = tumor
    PALP_CONST.discrete_only = discrete_only

    if PALP_CONST.tumor_type == "hemisphere":
        seebelow_const.BBOX_DOCTOR_ROI = seebelow_const.ROI_HEMISPHERE
    elif PALP_CONST.tumor_type == "crescent":
        seebelow_const.BBOX_DOCTOR_ROI = seebelow_const.ROI_CRESCENT



    bbox_roi = seebelow_const.BBOX_DOCTOR_ROI
    if select_bbox:
        bbox_roi = None



    roi_pcd = mesh2roi(surface_mesh, bbox_pts=bbox_roi)
    print("here")



    surface_grid_map = SurfaceGridMap(roi_pcd, grid_size=seebelow_const.PALP_CONST.grid_size)
    if debug:
        surface_grid_map.visualize()
        # ROI点云划分成网格
    if algo == "bo":
        search = ActiveSearchWithRandomInit(
            ActiveSearchAlgos.BO,
            surface_grid_map,#上一步生成的网格
            kernel_scale=seebelow_const.PALP_CONST.kernel_scale,#核函数设置
            random_sample_count=seebelow_const.PALP_CONST.random_sample_count,#初始随即采样的次数
        )
    elif algo == "random":
        search = RandomSearch(surface_grid_map)
   
   
    # search.grid.visualize()
    dataset_writer = DatasetWriter(prefix=f"{tumor}_{algo}", print_hz=False)
# 创建 DatasetWriter 对象，用于保存实验数据。
# 
# prefix=f"{tumor}_{algo}": 文件名前缀，例如 "hemisphere_random"。
# print_hz=False: 不打印数据写入频率。
    data_buffer = np.zeros(1, dtype=seebelow_const.PALP_DTYPE)
    # 创建一个长度为 1 的 NumPy 数组，数据类型为 PALP_DTYPE（结构化数组，包含力、位置等字段）。
    shm = shared_memory.SharedMemory(create=True, size=data_buffer.nbytes)#创建共享内存块
    # 绑定databuffer到共享内存
    data_buffer = np.ndarray(data_buffer.shape, dtype=data_buffer.dtype, buffer=shm.buf)
    stop_event = mp.Event()
    ctrl_process = mp.Process(
        target=main_ctrl,#子进程的目标函数
        args=(shm.name, stop_event, dataset_writer.dataset_folder, search),
    )#创建一个子进程用于搜索
    ctrl_process.start()

    subsurface_pts = []

    rk = Ratekeeper(50, name="data_collect")

    rtv = RealtimeVisualizer()
    rtv.add_frame("BASE")
    rtv.set_frame_tf("BASE", np.eye(4))
    rtv.add_frame("EEF", "BASE")
    try:
        while not stop_event.is_set():
            if np.all(data_buffer["O_q_EE"] == 0):
                # print("Waiting for deoxys...")
                continue

            O_p_EE = data_buffer["O_p_EE"].flatten()
            O_p_EE_target = data_buffer["O_p_EE_target"].flatten()
            O_q_EE_target = data_buffer["O_q_EE_target"].flatten()
            O_q_EE = data_buffer["O_q_EE"].flatten()

            if data_buffer["collect_points_flag"]:
                subsurface_pts.append(O_p_EE)

            print(data_buffer)
            # print("Pos ERROR: ", np.linalg.norm(O_p_EE - O_p_EE_target))
            # print("Rot ERROR: ", np.linalg.norm(O_q_EE - O_q_EE_target))

            dataset_writer.add_sample(data_buffer.copy())
            ee_pos = np.array(O_p_EE)
            ee_rmat = quat2mat(O_q_EE)
            O_T_E = np.eye(4)
            O_T_E[:3, :3] = ee_rmat
            O_T_E[:3, 3] = ee_pos
            rtv.set_frame_tf("EEF", O_T_E)
            rk.keep_time()
    except KeyboardInterrupt:
        pass
    stop_event.set()
    print("CTRL STOPPED!")

    while ctrl_process.is_alive():
        continue
    ctrl_process.join()

    dataset_writer.save_subsurface_pcd(np.array(subsurface_pts).squeeze())
    dataset_writer.save_roi_pcd(roi_pcd)
    dataset_writer.save_grid_pcd(surface_grid_map.grid_pcd)
    dataset_writer.save(autosave)
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    main()
