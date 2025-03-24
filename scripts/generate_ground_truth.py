# 生成了三个目标姿态，分别拍摄并且裁剪点云，最后相加获取整体视角

import argparse
import multiprocessing as mp
from datetime import datetime
from multiprocessing import shared_memory
import numpy as np
import open3d as o3d
import pinocchio as pin
import yaml
from tfvis.visualizer import RealtimeVisualizer

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.transform_utils import quat2mat, quat2axisangle
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.pcd_utils import pick_surface_bbox
from seebelow.utils.time_utils import Ratekeeper
from seebelow.utils.transform_utils import euler2mat
from seebelow.utils.interpolator import Interpolator, InterpType
import seebelow.utils.constants as seebelow_const




def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    O_T_EE_posquat = np.ndarray(7, dtype=np.float32, buffer=existing_shm.buf)


    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG), use_visualizer=False, control_freq=80
    )

    osc_absolute_ctrl_cfg = YamlConfig(str(seebelow_const.OSC_ABSOLUTE_CFG)).as_easydict()
    interp = Interpolator(interp_type=InterpType.SE3)

    goals = []

    O_T_P = np.eye(4) #O到phantom的坐标系转化，eye是个单位阵
    O_T_P[:3, 3] = seebelow_const.BBOX_PHANTOM.mean(axis=0)
    # 将 O_T_P 的平移部分（矩阵的前 3 行，第 4 列）设置为 BBOX_PHANTOM 的平均位置（seebelow_const.BBOX_PHANTOM.mean(axis=0)）。
# [[1, 0, 0, 0],
#  [0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]
#变化为
# [[1, 0, 0, x],
#  [0, 1, 0, y],
#  [0, 0, 1, z],
#  [0, 0, 0, 1]]
    P_T_O = np.linalg.inv(O_T_P)


    O_T_E = np.eye(4)#机器人到末端执行器
    O_T_E[:3, :3] = quat2mat(seebelow_const.GT_SCAN_POSE[3:7])
    #预设好的一个姿态，包含位置和四元数的一个七元素数组
# [[1, 0, 0, 0],
#  [0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]
#变化为
# [[r11, r12, r13, 0],
#  [r21, r22, r23, 0],
#  [r31, r32, r33, 0],
#  [  0,   0,   0, 1]]
    O_T_E[:3, 3] = seebelow_const.GT_SCAN_POSE[:3]
# 变化为：
# [[r11, r12, r13, x],
#  [r21, r22, r23, y],
#  [r31, r32, r33, z],
#  [  0,   0,   0, 1]]
# O_T_E 现在完全定义了末端执行器的初始位姿（位置和朝向），用预设好的姿态作为扫描的起点。

    goals.append(O_T_E)#第一个位置

    P_T_E = np.matmul(P_T_O, O_T_E)


    for ang in np.linspace(np.radians(5), np.radians(20), 3):
#模拟三个5-20度内均匀分布的角度

        Ry = np.eye(4)
        Ry[:3, :3] = euler2mat(np.array([0, ang, 0]))
#将欧拉角（X=0, Y=ang, Z=0）转换为 3x3 旋转矩阵
#  [[cos(ang),  0, sin(ang)],
#  [      0,   1,       0],
#  [-sin(ang), 0, cos(ang)]]
# Ry变为：
# [[cos(ang),  0, sin(ang), 0],
#  [      0,   1,       0,  0],
#  [-sin(ang), 0, cos(ang), 0],
#  [      0,   0,       0,  1]]
        goals.append(O_T_P @ Ry @ P_T_E)
# P_T_E：末端执行器相对于 phantom 的位姿。
# Ry @ P_T_E：在 phantom 坐标系中绕 Y 轴旋转后的末端位姿。
# O_T_P @ (Ry @ P_T_E)：将结果转换回世界坐标系。


    while len(robot_interface._state_buffer) == 0:
        continue

    while len(goals) != 0 or not interp.done:
        q, p = robot_interface.last_eef_quat_and_pos
# 回一个元组 (eef_quat, eef_pos)，其中包含两部分：
# eef_quat：一个 4 维的四元数数组 [x, y, z, w]，表示末端执行器的方向。
# eef_pos：一个 3 维的位置向量 [x, y, z]，表示末端执行器在世界坐标系中的位置。
        O_T_EE_posquat[:3] = p.flatten()
        O_T_EE_posquat[3:7] = q.flatten()
# 展平，赋值后，O_T_EE_posquat 格式为 [x, y, z, w, x, y, z]。


#将当前位姿转化为se3（供插值使用）
        curr_eef_pose = robot_interface.last_eef_rot_and_pos
        curr_pose_se3 = pin.SE3.Identity()
        curr_pose_se3.rotation = curr_eef_pose[0]
        curr_pose_se3.translation = curr_eef_pose[1]
# 将目标位姿转化为SE3对象（供插值使用）
        if interp.done:
            pose_goal = goals.pop()
            goal_pose_se3 = pin.SE3.Identity()
            goal_pose_se3.rotation = pose_goal[:3, :3]
            goal_pose_se3.translation = pose_goal[:3, 3]
# 规划运动轨迹
            interp.init(
                curr_pose_se3, goal_pose_se3, steps=int(seebelow_const.STEP_FAST / 1)
            )
# 第一个姿态
        action = np.zeros(7)
        next_se3_pose = interp.next()
        xyz_quat = pin.SE3ToXYZQUAT(next_se3_pose)
        axis_angle = quat2axisangle(xyz_quat[3:7])

        action[:3] = xyz_quat[:3]
        action[3:6] = axis_angle

        robot_interface.control(
            controller_type=seebelow_const.OSC_CTRL_TYPE,
            action=action,
            controller_cfg=osc_absolute_ctrl_cfg,
        )

    # stop
    stop_event.set()
    robot_interface.close()

    existing_shm.close()
    existing_shm.unlink()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
# --interface-cfg：机器人接口配置文件，默认值 "pan-pan-force.yml"。
    argparser.add_argument("--interface-cfg", type=str, default="pan-pan-force.yml")
# --calibration-cfg：相机标定配置文件，默认值 "camera_calibration_12-16-2023_14-48-12"。
    argparser.add_argument(
        "--calibration-cfg", type=str, default="camera_calibration_12-16-2023_14-48-12"
    )
# --cam-name：相机名称，默认值 "wrist_d415"（可能是 RealSense D415 型号）。
    argparser.add_argument("--cam-name", type=str, default="wrist_d415")
    args = argparser.parse_args()

    O_T_EE_posquat = np.zeros(7, dtype=np.float32)

    shm = shared_memory.SharedMemory(create=True, size=O_T_EE_posquat.nbytes)

    O_T_EE_posquat = np.ndarray(
        O_T_EE_posquat.shape, dtype=O_T_EE_posquat.dtype, buffer=shm.buf
    )

    stop_event = mp.Event()
    ctrl_process = mp.Process(target=deoxys_ctrl, args=(shm.name, stop_event))
    ctrl_process.start()

    while np.all(O_T_EE_posquat == 0):
        continue

    # print(np_to_constant("GT_SCAN_POSE", O_T_EE_posquat))


# 获取E2C
# 加载相机相对于末端执行器的外参（extrinsic parameters）
    with open(
        str(seebelow_const.SEEBELOW_CFG_PATH / args.calibration_cfg / "extrinsics.yaml"), "r"
    ) as file:
        
        calib_cfg = yaml.safe_load(file)
# 将 YAML 文件内容解析为 Python 数据结构。
# yaml.safe_load 读取文件并返回字典。
# calib_cfg 可能包含多个相机的标定数据，例如：

# wrist_d415:
#   - 0.1  # x
#   - 0.2  # y
#   - 0.3  # z
#   - 1.0  # qw
#   - 0.0  # qx
#   - 0.0  # qy
#   - 0.0  # qz




        xyzxyzw = calib_cfg[args.cam_name]
        ee_pos = np.array(xyzxyzw[:3])
        ee_rot = quat2mat(xyzxyzw[-4:])
        E_T_C = np.eye(4)
        E_T_C[:3, :3] = ee_rot
        E_T_C[:3, 3] = ee_pos



    rs = RealsenseCapture()
    pcd = o3d.geometry.PointCloud()

    rk = Ratekeeper(1)

    rtv = RealtimeVisualizer()
    rtv.add_frame("BASE")
    rtv.set_frame_tf("BASE", np.eye(4))
    rtv.add_frame("EEF", "BASE")
    rtv.add_frame("EEF_45", "EEF")
    rot_45 = np.eye(4)
    rot_45[:3, :3] = euler2mat([0, 0, -np.pi / 4])
    rtv.set_frame_tf("EEF_45", rot_45)
    rtv.add_frame("CAM", "EEF")
    rtv.add_frame("TUMOR", "BASE")

    selected_bbox = seebelow_const.BBOX_PHANTOM

    while not stop_event.is_set():
        # _, new_pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
        _, new_pcd = rs.read()
# rs.read() 是 RealsenseCapture 的方法，返回一个元组（可能包括颜色帧和点云）。
# 被注释掉的行使用了掩码（get_mask），可能是基于 HSV 颜色阈值过滤点云，但当前版本直接读取完整点云。
# _ 表示忽略第一个返回值（可能是颜色帧），new_pcd 是新的点云对象（o3d.geometry.PointCloud）。



        ee_pos = np.array(O_T_EE_posquat[:3])
        # 获取执行器位置xyz
        ee_mat = quat2mat(O_T_EE_posquat[3:7])
        # 末端执行器四元数转化为旋转矩阵，朝向
        O_T_E = np.eye(4)
        O_T_E[:3, :3] = ee_mat
        O_T_E[:3, 3] = ee_pos


        O_T_C = O_T_E @ E_T_C
        new_pcd.transform(O_T_C)
        # 将新采集的点云从相机坐标系转换到世界坐标

        bbox = pick_surface_bbox(new_pcd, bbox_pts=selected_bbox)
        # print(array2constant("BBOX_ROI", np.asarray(bbox.get_box_points())))

        # 剪裁点云
        selected_bbox = np.asarray(bbox.get_box_points())
        new_pcd = new_pcd.crop(bbox)
        # 剪裁后的点云叠加回去
        pcd += new_pcd

# 完整点云数据转化为坐标
        tumor_pts = np.asarray(pcd.points)
        tumor_mean = tumor_pts.mean(axis=0)
        O_T_TUM = np.eye(4)
        O_T_TUM[:3, 3] = tumor_mean
        # 平移到肿瘤中心的转换矩阵

        # tf visualizer
        rtv.set_frame_tf("EEF", O_T_E)
        rtv.set_frame_tf("CAM", E_T_C)
        rtv.set_frame_tf("TUMOR", O_T_TUM)

        rk.keep_time()

    ctrl_process.join()
    print("CTRL STOPPED!")

    now_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    save_pth = str(seebelow_const.SEEBELOW_MESH_PATH / f"tumors_gt_{now_str}.ply")

    print(f"saving to {save_pth}")

    o3d.io.write_point_cloud(save_pth, pcd)

    shm.close()
