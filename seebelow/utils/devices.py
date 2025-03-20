import numpy as np
import open3d as o3d
import open3d.core as o3c
import serial


class RealsenseCapture:
    
    def __init__(self):
        # self.rs_cfg = o3d.t.io.RealSenseSensorConfig(
        #     {
        #         "serial": "",
        #         "color_format": "RS2_FORMAT_RGB8",
        #         "color_resolution": "0,480",
        #         "depth_format": "RS2_FORMAT_Z16",
        #         "depth_resolution": "0,480",
        #         "fps": "60",
        #         "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE",
        #     }
        # )
                # 打印设备列表
        print("Checking available RealSense devices...")
        device_list = o3d.t.io.RealSenseSensor.list_devices()
        print(f"Raw output from list_devices: {device_list} (type: {type(device_list)})")
        if device_list is True:  # 如果返回布尔值，假设设备可用
            print("Device detected but unable to retrieve detailed list.")
            device_list = ["D435I"]  # 使用占位符表示检测到的设备
        elif not device_list or not isinstance(device_list, list):
            raise RuntimeError("No RealSense devices detected or unexpected return type from list_devices!")
        else:
            print(f"Detected {len(device_list)} RealSense device(s):")
            for device in device_list:
                print(f" - {device}")



        self.rs_cfg = o3d.t.io.RealSenseSensorConfig(
    {
        "serial": "140122071098",  # 可选：指定设备序列号
        "color_format": "RS2_FORMAT_RGB8",  # 彩色格式
        "color_resolution": "640,480",  # 彩色分辨率
        "depth_format": "RS2_FORMAT_Z16",  # 深度格式
        "depth_resolution": "640,480",  # 深度分辨率
        "fps": "30",  # 确保帧率同步
    }
)



        
        print("Using the following configuration for RealSense sensor:")
        print(self.rs_cfg)

        self.rs = o3d.t.io.RealSenseSensor()

        try:
            self.rs.init_sensor(self.rs_cfg,0)
            print("RealSense sensor initialized successfully.")
        except RuntimeError as e:
            print("Failed to initialize RealSense sensor.")
            print(f"Configuration: {self.rs_cfg}")
            print(f"Error: {e}")
            raise
        try:
            metadata = self.rs.get_metadata()
            print(f"RealSense metadata: {metadata}")
            print(f"Intrinsics matrix: {metadata.intrinsics.intrinsic_matrix}")
            self.intrinsics = o3c.Tensor(metadata.intrinsics.intrinsic_matrix)
        except Exception as e:
            print(f"Failed to retrieve sensor metadata: {e}")
            raise
        self.rs.start_capture(True)  # true: start recording with capture
        self.intrinsics = o3c.Tensor(self.rs.get_metadata().intrinsics.intrinsic_matrix)

    def read(self, get_mask=None):
        im_rgbd = self.rs.capture_frame(True, True)  # wait for frames and align them

        color_tensor = im_rgbd.color.as_tensor()
        color_np = color_tensor.numpy()
        print(f"Color frame shape: {color_np.shape}, min: {color_np.min()}, max: {color_np.max()}")

    # 检查深度数据
        depth_tensor = im_rgbd.depth.as_tensor()
        depth_np = depth_tensor.numpy()
        print(f"Depth frame shape: {depth_np.shape}, min: {depth_np.min()}, max: {depth_np.max()}")

        if get_mask is not None:
            mask = get_mask(color_np.copy())
            depth_tensor = im_rgbd.depth.as_tensor()
            depth_np = depth_tensor.numpy()
            depth_np[mask == 0] = 0

        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            im_rgbd, self.intrinsics
        ).to_legacy()
        print(f"Generated point cloud with {len(pcd.points)} points.")
        return color_np, pcd


class ForceSensor:
    def __init__(self, port="/dev/ttyACM0", baudrate="115200"):
        self.serial = serial.Serial(port=port)
        self.serial.baudrate = baudrate

        while True:
            while self.serial.inWaiting() == 1:
                pass
            bytes_data = self.serial.readline()
            print(str(bytes_data, encoding="utf-8"))
            if bytes_data.startswith(bytes("T:", "utf-8")):
                print("initialized")
                break
            else:
                self.serial.write(bytes("ACK", "utf-8"))

    def read(self):
        if self.serial.inWaiting() == 0:
            return
        else:
            while self.serial.inWaiting() == 2:
                continue
            self.data_stream = self.serial.readline()
        self.data_string = str(self.data_stream)
        self.elements = self.data_string.split(",")
        Fxyz = np.array(
            [
                (float)(self.elements[1]),
                (float)(self.elements[2]),
                (float)(self.elements[3]),
            ],
            dtype=np.double,
        )
        return Fxyz
