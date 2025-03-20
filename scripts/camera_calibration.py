"""Generate camera calibration dataset for https://github.com/ToyotaResearchInstitute/handical"""

# 手眼标定，求解相机和机械臂末端之间的固定变换矩阵（外参），让相机和机械臂的坐标系关联起来。
# 但这份代码只是用于生成机械臂位置姿态与相机图像对应的数据集

# 生成用于手眼标定的数据集，包含：

# 相机拍摄的棋盘格图像。
# 机械臂末端执行器的实时位姿（即棋盘格在空间中的真实位姿）。
# 最终用于推算：相机 -> 机械臂末端 的固定空间变换矩阵。






import argparse
# 用途：在程序运行时，可以通过命令行传递参数（如配置文件路径）。
# 有这句话才可以：python camera_calibration.py --interface-cfg my_config.yml

import multiprocessing as mp
# 启动多进程
from multiprocessing import shared_memory
# 多个进程共享数据
import cv2
import numpy as np
import yaml

from deoxys.franka_interface import FrankaInterface
# 控制机械臂运动，获取机械臂的位姿（四元数和位置）。
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
# 输入设备（如 SpaceMouse） 的操作转换成机械臂动作。
# 实现 人机交互控制，例如用鼠标推动机械臂。
from deoxys.utils.io_devices import SpaceMouse
# 空间鼠标（3D 控制器）接口。
# 用来 手动控制机械臂运动，进行示教或标定过程中的位置调整。


import seebelow.utils.constants as seebelow_const
from seebelow.utils.data_utils import CalibrationWriter
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.keystroke_counter import KeyCode, KeystrokeCounter
from seebelow.utils.time_utils import Ratekeeper

PROPRIO_DIM = 7  # pos: x,y,z, rot: x,y,z,w


def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    # 根据共享内存的名字 连接已经存在的共享内存，确保机械臂进程与主进程共享一块数据区（O_T_EE）。
    O_T_EE = np.ndarray(PROPRIO_DIM, dtype=np.float32, buffer=existing_shm.buf)
    # 将共享内存的缓冲区 映射成 numpy 数组。
    # 类型 float32，7 维，保存实时的机械臂末端 位置 + 姿态。

    
    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG),
        use_visualizer=False,
        control_freq=20,
        # 初始化机械臂控制接口：
        # PAN_PAN_FORCE_CFG：加载机械臂控制的 YAML 配置文件路径。
        # use_visualizer=False：不显示图形可视化。
        # control_freq=20：设置控制频率为 20Hz（每秒发送 20 次指令）。
    )

    osc_delta_ctrl_cfg = YamlConfig(str(seebelow_const.OSC_DELTA_CFG)).as_easydict()
    
    
    device = SpaceMouse(
        vendor_id=seebelow_const.SPACEM_VENDOR_ID, product_id=seebelow_const.SPACEM_PRODUCT_ID
    )


    device.start_control()

    while len(robot_interface._state_buffer) == 0:
        continue

    while not stop_event.is_set():
        q, p = robot_interface.last_eef_quat_and_pos
        # 写入共享内存，供主进程读取。
        O_T_EE[:3] = p.flatten()
        # [:3] 是位置。
        O_T_EE[3:7] = q.flatten()
        # [3:7] 是四元数姿态。


        action, grasp = input2action(
            device=device,
            controller_type=seebelow_const.OSC_CTRL_TYPE,
        )
        # 从空间鼠标读取动作，转换成机械臂运动指令。
        # action：机械臂运动指令（如移动或旋转）。
        # grasp：夹爪控制（抓取动作）。

        robot_interface.control(
            controller_type=seebelow_const.OSC_CTRL_TYPE,
            action=action,
            controller_cfg=osc_delta_ctrl_cfg,
        )

    robot_interface.control(
        controller_type=seebelow_const.OSC_CTRL_TYPE,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=osc_delta_ctrl_cfg,
        termination=True,
    )

    # 停止循环后，发送一个全零动作（不再移动），加 termination=True 让控制器优雅退出。
# 最后的 [1.0] 通常是让夹爪松开。
    robot_interface.close()

    existing_shm.close()
    existing_shm.unlink()


if __name__ == "__main__":
# Python 脚本的标准入口。确保以下代码只在 直接运行这个脚本 时执行，而在被其他模块导入时不会执行。
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--interface-cfg", type=str, default="pan-pan-force.yml")
    args = argparser.parse_args()

    O_T_EE = np.zeros(PROPRIO_DIM, dtype=np.float32)
    # 创建一个 全 0 的 numpy 数组，用于存放机械臂位姿（7 维 float32）。
    shm = shared_memory.SharedMemory(create=True, size=O_T_EE.nbytes)
    # 创建一块共享内存（大小与 O_T_EE 相同）。
    O_T_EE = np.ndarray(O_T_EE.shape, dtype=O_T_EE.dtype, buffer=shm.buf)
    # 把共享内存映射成 numpy 数组 O_T_EE，供主进程和子进程实时读写。


    stop_event = mp.Event()
    # 用于将来通知子进程停止运行
    ctrl_process = mp.Process(target=deoxys_ctrl, args=(shm.name, stop_event))
    # 创建子进程，目标函数是 deoxys_ctrl，传递共享内存名字和停止信号。
    ctrl_process.start()

# 初始化相机和标定写入器
    rs = RealsenseCapture()
    calibration_writer = CalibrationWriter()

# 读取标定配置文件
    with open(str(seebelow_const.BASE_CALIB_FOLDER / "config.yaml"), "r") as file:
        calib_cfg = yaml.safe_load(file)

# 获取棋盘格的行数和列数，用于棋盘检测（如 7x9）。
    cb_size = (
        calibration_writer.calib_cfg["board"]["nrows"],
        calibration_writer.calib_cfg["board"]["ncols"],
    )

# 从 RealSense 相机读取一帧图像和点云（虽然这里只用了图像）。
    im, pcd = rs.read()
# 创建一个 速率保持器，目标频率为 30Hz，确保后面循环以稳定速度运行（避免 CPU 占用过高）。
    rk = Ratekeeper(30)
# 启动 按键监听器，自动检测键盘输入（比如按 r 进行数据采集）。
    with KeystrokeCounter() as key_counter:
        try:
            while 1:
                if np.all(O_T_EE == 0):
                    print("Waiting for pose...")
                    continue
                # 如果共享内存里的位姿数据全为 0，说明还没准备好，等待机械臂初始化。
                
                
                im, new_pcd = rs.read()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="r"):
                        # 如果用户按下 r，进入数据采集流程。
                        ret, corners = cv2.findChessboardCorners(
                            im,
                            cb_size,
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK
                            + cv2.CALIB_CB_NORMALIZE_IMAGE,
                        )
                        # 用 OpenCV 查找棋盘格。 如果找到 ret == True，就继续保存图像和位姿。

                        if ret:
                            print(f"added #{len(calibration_writer.images)} data point")
                            print(f"O_T_EE: {O_T_EE}")
                            calibration_writer.add(im, O_T_EE.copy())
                        else:
                            print(f"nope!")
                rk.keep_time()
        except KeyboardInterrupt:
            pass
        stop_event.set()
        ctrl_process.join()
        print("CTRL STOPPED!")

        calibration_writer.write()

        shm.close()
        shm.unlink()
