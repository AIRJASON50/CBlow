import argparse

import cv2
import numpy as np
import open3d as o3d

from seebelow.utils.constants import *
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.segmentation_utils import get_color_mask, get_hsv_threshold

argparser = argparse.ArgumentParser()
argparser.add_argument("--rgb", type=bool, default= False)
argparser.add_argument("--tumors-only", type=bool, default=False)
args = argparser.parse_args()

rs = RealsenseCapture()
# 创建 RealsenseCapture 对象，与 RealSense 相机连接。配置相机参数，并启动捕获。

im, pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
    # rs.read(): 捕获 RGB 和深度帧。

    # 使用 HSV 阈值过滤图像，仅保留感兴趣区域（如肿瘤相关像素）。
    # 返回值：im: RGB 图像数据（过滤后的图像）。pcd: 基于深度帧生成的点云。



# 如果 --rgb 为 False，初始化 Open3D 的可视化窗口：
# Visualizer: 创建一个可交互窗口。
# create_window: 打开显示窗口。
# add_geometry: 将点云添加到可视化场景中。

if not args.rgb:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # 检查深度图范围
    depth_tensor = rs.rs.capture_frame().depth.as_tensor()
    depth_np = depth_tensor.numpy()
    print(f"Depth shape: {depth_np.shape}, min: {depth_np.min()}, max: {depth_np.max()}")

# 检查点云生成
    print(f"Generated point cloud: {len(pcd.points)} points") 


# 如果 --tumors-only 为 True，仅捕获肿瘤相关的数据帧。
# 否则，捕获完整帧数据。
while 1:  # Set the number of frames to display the image
    if args.tumors_only:
        im, new_pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
    else:
        im, new_pcd = rs.read()
# 如果rgb==true
# 使用 OpenCV 显示 RGB 图像。
# cv2.waitKey(1) 确保窗口实时更新。
    if args.rgb:
        cv2.imshow("Image", np.asarray(im))
        cv2.waitKey(1)  # Wait for 1 millisecond




# 如果 --rgb 为 False：
# 更新 Open3D 点云：
# pcd.points 和 pcd.colors：更新点云的点和颜色数据。
# 更新 Open3D 窗口：
# update_geometry：刷新点云数据。
# poll_events 和 update_renderer：渲染窗口。
    else:
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        import time
        time.sleep(0.01)

        vis.update_renderer()
vis.destroy_window()



