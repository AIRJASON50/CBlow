import serial
import struct
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import deque

# 设置字体支持
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def setup_serial(port='/dev/ttyUSB0', baudrate=115200):
    """
    初始化串口连接
    """
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
    if ser.is_open:
        print(f"串口 {port} 已打开")
    else:
        print(f"无法打开串口 {port}")
    return ser

def send_command(ser, command):
    """
    发送命令到传感器
    """
    if not command.endswith('\r\n'):
        command += '\r\n'
    ser.write(command.encode())
    print(f"发送: {command.strip()}")

def reverse_bytes(data):
    """
    按字节反转顺序
    """
    return bytes(data[::-1])

def parse_sensor_data(data):
    """
    解析传感器返回的六通道力数据
    """
    try:
        # 检查帧头
        if data[:2] != b'\xAA\x55':
            print("数据头无效")
            print(f"原始数据: {data.hex()}")  # 打印原始数据用于调试
            return None, None

        # 获取长度和包编号
        package_no = struct.unpack('>H', data[4:6])[0]

        # 按需反转每个通道的数据字节
        forces = []
        for i in range(6):
            raw_bytes = data[6 + i * 4:10 + i * 4]  # 每个通道 4 字节
            reversed_bytes = reverse_bytes(raw_bytes)
            force = struct.unpack('>f', reversed_bytes)[0]
            forces.append(force)

        return package_no, forces
    except Exception as e:
        print(f"解析错误: {e}")
        print(f"原始数据: {data.hex()}")  # 打印原始数据用于调试
        return None, None

def sync_data(ser):
    """
    数据同步：找到帧头 AA 55
    """
    while True:
        byte = ser.read(1)  # 逐字节读取
        if byte == b'\xAA':
            next_byte = ser.read(1)
            if next_byte == b'\x55':
                return b'\xAA\x55'  # 找到帧头

def is_valid_force(force, min_val=-100, max_val=100):
    """
    校验力值是否在合理范围内
    """
    return min_val <= force <= max_val

def main():
    # 设置串口参数
    ser = setup_serial(port='/dev/ttyUSB0', baudrate=115200)

    # 初始化实时绘图
    plt.ion()
    fig, ax = plt.subplots()
    lines = []
    data_buffers = [deque(maxlen=100) for _ in range(6)]  # 每个通道保存最近 100 个数据点
    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']  # 通道对应的图例名称
    for i in range(6):
        line, = ax.plot([], [], label=labels[i])
        lines.append(line)

    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 10)  # 根据数据范围调整 Y 轴
    ax.set_title("Real-Time 6D Force Visualization")
    ax.set_xlabel("Time (Frames)")
    ax.set_ylabel("Force/Moment")
    ax.legend()

    try:
        # 设置采样频率
        send_command(ser, 'AT+SMPF=300')
        time.sleep(0.5)
        send_command(ser, 'AT+GSD')

        # 实时读取数据并更新绘图
        while True:
            # 数据同步
            sync_data(ser)
            raw_data = ser.read(29)  # 帧头后面 29 字节
            if len(raw_data) < 29:
                print("数据包不完整")
                continue

            raw_data = b'\xAA\x55' + raw_data  # 补充帧头
            package_no, forces = parse_sensor_data(raw_data)
            if forces:
                # 校验每个通道的力值
                filtered_forces = [force if is_valid_force(force) else None for force in forces]

                # 忽略完全无效的数据包
                if all(f is None for f in filtered_forces):
                    print(f"无效数据包: 包编号 {package_no}, 原始力值: {forces}")
                    continue

                # 打印有效力值
                valid_forces = [f if f is not None else 0 for f in filtered_forces]
                print(f"Package No: {package_no}, Filtered Forces: {valid_forces}")

                # 更新数据缓冲区
                for i, force in enumerate(valid_forces):
                    data_buffers[i].append(force)

                # 更新曲线数据
                for i, line in enumerate(lines):
                    line.set_data(range(len(data_buffers[i])), list(data_buffers[i]))

                # 更新绘图
                ax.set_xlim(0, max(100, len(data_buffers[0])))
                plt.pause(0.01)
    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        send_command(ser, 'AT+STOP')  # 停止连续传输
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
