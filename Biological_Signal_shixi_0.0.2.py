# -*- coding: utf-8 -*-

import sys
import numpy as np
import threading
import pyqtgraph as pg
import serial
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QTextCharFormat, QFont
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QLineEdit, \
    QPushButton, QTextBrowser, QApplication, QGroupBox,QCheckBox
import datetime
import os  # 导入os模块用于文件路径操作

import time
from PyQt5.QtCore import QThread, pyqtSignal



# 多线程通信
class SerialThread(QThread):
    """串口 数据接收线程"""
    data_received = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baudrate=115200, buffer_threshold=60):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.is_recording = False
        self.buffer = bytearray()
        self.buffer_threshold = buffer_threshold
        self.packet_size = 6  # 0xAA + EMG(L,H) + ECG(L,H) + 0x55
        self.max_buffer_size = 4096
        
        # 数据积累器
        self.accumulate_threshold = 10
        self.accumulated_ecg = []
        self.accumulated_emg = []

        
        # 异常包计数器和最后一个正常包的备份
        self.abnormal_packet_count = 0
        self.last_valid_data = None  # 保存最后一组正常数据

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.ser.set_buffer_size(rx_size=65536, tx_size=4096)
            self.running = True
            self.error_occurred.emit(f"串口 {self.port} 已连接，开始预采集")
            
            while self.running:
                if self.ser.in_waiting:
                    # 读取所有可用数据
                    data = self.ser.read(self.ser.in_waiting)
                    if data:
                        self.buffer.extend(data)
                        
                        if len(self.buffer) >= self.buffer_threshold:
                            self.process_buffer_data()

                else:
                    # 即使没有新数据，也处理现有缓冲区
                    if len(self.buffer) >= self.packet_size:
                        self.process_buffer_data()
                    self.msleep(1)
                    
        except Exception as e:
            self.error_occurred.emit(f"串口错误: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def process_buffer_data(self):
        """优化的缓冲区数据处理，增强异常包处理"""
        buffer_len = len(self.buffer)
        packet_size = self.packet_size
        
        processed_bytes = 0
        i = 0
        
        # 尽可能处理所有完整的数据包
        while i <= buffer_len - packet_size:
            if self.buffer[i] == 0xAA:
                # 检查是否有完整数据包
                if i + packet_size <= buffer_len and self.buffer[i + packet_size - 1] == 0x55:
                    # 找到完整的正常数据包
                    data = self.buffer[i:i + packet_size]
                    
                    try:
                        # 解析数据并直接添加到积累器
                        emg = (data[2] << 8) | data[1]
                        ecg = (data[4] << 8) | data[3]

                        
                        # 保存最后一组正常数据用于异常恢复
                        self.last_valid_data = (emg, ecg)
                        
                        # 添加到积累器
                        self.accumulated_ecg.append(ecg)
                        self.accumulated_emg.append(emg)

                        
                        # 检查是否达到积累阈值
                        if len(self.accumulated_ecg) >= self.accumulate_threshold:
                            self.send_accumulated_data()
                            self.reset_accumulator()
                        
                        i += packet_size
                        processed_bytes = i
                        
                    except Exception as e:
                        self.error_occurred.emit(f"数据解析错误: {e}")
                        i += 1
                else:
                    # 包头正确但包尾错误的异常包处理
                    self.handle_abnormal_packet_with_correct_header(i, buffer_len)
                    i += 1
            else:
                # 包头错误，计为异常包
                self.abnormal_packet_count += 1
                i += 1
        
        # 优化的缓冲区清理逻辑
        self.cleanup_buffer(processed_bytes, buffer_len)

    def handle_abnormal_packet_with_correct_header(self, header_pos, buffer_len):
        """处理包头正确但包尾异常的数据包"""
        remaining_bytes = buffer_len - header_pos
        
        if remaining_bytes < self.packet_size:
            # 情况1: 剩余字节不足一个完整包，可能是不完整的包
            # 这些数据将在cleanup_buffer中被保留到下次处理
            pass
        else:
            # 情况2: 剩余字节足够一个包但包尾不正确
            # 使用上一个正常数据包的数据进行恢复
            if self.last_valid_data:
                emg, ecg= self.last_valid_data
                
                # 添加到积累器

                self.accumulated_ecg.append(ecg)
                self.accumulated_emg.append(emg)

                
                self.error_occurred.emit(f"检测到异常包，使用上一个正常包数据进行恢复")
            
        self.abnormal_packet_count += 1

    def cleanup_buffer(self, processed_bytes, buffer_len):
        """优化的缓冲区清理逻辑"""
        if processed_bytes > 0:
            # 清理已处理的数据
            self.buffer = self.buffer[processed_bytes:]
        else:
            # 没有处理任何完整包的情况
            # 检查是否有不完整的包头需要保留
            incomplete_packet_start = self.find_incomplete_packet_start()
            
            if incomplete_packet_start is not None:
                # 保留不完整的包数据到缓冲区开头
                self.buffer = self.buffer[incomplete_packet_start:]
            else:
                # 没有找到有效的包头，清理部分无效数据，但保留一些数据以防遗漏
                if len(self.buffer) > self.packet_size * 2:
                    # 如果缓冲区太大，保留最后packet_size长度的数据
                    self.buffer = self.buffer[-self.packet_size:]
        
        # 防止缓冲区过大
        if len(self.buffer) > self.max_buffer_size:
            # 只保留最后一部分数据
            self.buffer = self.buffer[-self.max_buffer_size//2:]
            self.error_occurred.emit("缓冲区过大，已清理部分数据")

    def find_incomplete_packet_start(self):
        """查找可能的不完整包开始位置"""
        buffer_len = len(self.buffer)
        
        # 从缓冲区末尾向前查找最后一个可能的包头
        for i in range(buffer_len - 1, -1, -1):
            if self.buffer[i] == 0xAA:
                remaining_bytes = buffer_len - i
                if remaining_bytes < self.packet_size:
                    # 找到不完整的包头
                    return i
                elif remaining_bytes >= self.packet_size:
                    # 检查这个位置是否有完整的包
                    if self.buffer[i + self.packet_size - 1] != 0x55:
                        # 包头正确但包尾错误，继续查找
                        continue
                    else:
                        # 找到完整包，不需要保留
                        return None
        
        return None

    def send_accumulated_data(self):
        """发送积累的数据到主线程"""
        if self.accumulated_ecg:
            self.data_received.emit(
                np.array(self.accumulated_ecg, dtype=np.uint16),
                np.array(self.accumulated_emg, dtype=np.uint16),
            )

        self.reset_accumulator()
            
    def reset_accumulator(self):
        """重置数据积累器"""

        self.accumulated_ecg.clear()
        self.accumulated_emg.clear()


    def stop(self):
        self.running = False
        self.is_recording = False
        
        # 发送剩余积累的数据
        if self.accumulated_ecg or self.accumulated_emg:
            self.send_accumulated_data()
            self.reset_accumulator()
            
        if self.ser and self.ser.is_open:
            self.ser.close()

        # 报告异常包统计
        self.error_occurred.emit(f"数据采集结束，本次共检测到 {self.abnormal_packet_count} 个异常数据包")

    def send_data(self, data):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data)
            except Exception as e:
                print(f"发送数据错误: {e}")



# 缓存类 - 优化，使用预分配数组和循环缓冲区
class DataBuffer:
    """数据缓冲区（无限长度，动态列表）"""

    def __init__(self):
        self.buffer = []  # 用于存储所有数据
        self.lock = threading.Lock()

    def append_raw(self, data):
        with self.lock:
            self.buffer.extend(data)

    def get_length(self):
        """返回当前缓冲区长度（不做大拷贝）"""
        with self.lock:
            return len(self.buffer)

    def get_last_n(self, n):
        """高效获取最后 n 个样本（直接对 list 切片再转 numpy，避免拷贝全部）"""
        with self.lock:
            if n <= 0:
                return np.array([], dtype=np.uint16)
            buf_len = len(self.buffer)
            if buf_len == 0:
                return np.array([], dtype=np.uint16)
            if n >= buf_len:
                return np.array(self.buffer, dtype=np.uint16)
            # 只把尾部切片转为 numpy，避免全量拷贝
            tail = self.buffer[-n:]
            return np.array(tail, dtype=np.uint16)

    def get_raw_data(self, low_index=None, high_index=None):
        """兼容接口：当指定区间时只转区间部分为 numpy，避免先把整个 list 转为 numpy 再切片"""
        with self.lock:
            buf_len = len(self.buffer)
            # 无参：返回全部（会拷贝全部）
            if low_index is None and high_index is None:
                return np.array(self.buffer, dtype=np.uint16)
            # 规范默认值
            if low_index is None:
                low_index = 0
            if high_index is None:
                high_index = buf_len
            # 验证索引合法性
            if not (0 <= low_index <= high_index <= buf_len):
                return None
            slice_len = high_index - low_index
            if slice_len == 0:
                return np.array([], dtype=np.uint16)
            # 如果请求的是尾部数据，使用切片后转 numpy（高效）
            if low_index >= 0:
                part = self.buffer[low_index:high_index]
                return np.array(part, dtype=np.uint16)
            # 兜底（不太可能到这里）
            return np.array(self.buffer[low_index:high_index], dtype=np.uint16)

    def clear(self):
        with self.lock:
            self.buffer = []

    def has_data(self):
        with self.lock:
            return len(self.buffer) > 0


# GUI设置
class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("生物数据采集与处理系统")
        self.resize(1000, 700)

        # 数据结构和配置参数
        # 正式数据缓冲区
        self.ECG_1_data_ = DataBuffer()
        self.EMG_1_data_ = DataBuffer()


        # 预采集数据缓冲区
        self.pre_ECG_1_data_ = DataBuffer()
        self.pre_EMG_1_data_ = DataBuffer()


        self.serial_thread = None
        self.is_recording = False  # 是否正式记录数据
        self.initialized = False  # 是否已完成初始化

         # marker存储列表
        # self.markers = []  # 存储格式: [(marker_name, elapsed_time), ...]
        # self.current_cycle_markers = []  # 当前循环的marker存储
        # self.current_phase = 1  # 当前实验阶段：1或2
        # self.video_count = 0  # 视频计数器

        # 新增：实验人数设置（1或2）
        self.subject_count = 1  # 默认1人

        self.Heart_rate = 0  # 心率属性
        self.current_phase = 1
        self.current_cycle_markers = []

        # 实验信号采集设置
        self.ecg_channel = False  # ECG通道
        self.emg_channel = False  # EMG通道
        # self.gsr_channel = False  # GSR通道

        # 配置参数 - 与频率设置类似的方式处理实验编号
        self.sampling_rate = 1000  # 采样率属性，固定为1000Hz
        self.experiment_id = ""  # 实验编号属性

        # 优化：调整显示参数，减少UI更新频率以避免阻塞数据采集
        self.display_rate = 100  # Hz
        self.display_points = self.display_rate * 10  # 10秒数据
        self.update_interval = 100  # 毫秒，降低更新频率从50ms到100ms
        self.update_points = int(self.display_rate * (self.update_interval / 1000))  # 10点
        self.draw_index = 99  # 索引起始点

        # 创建UI
        self.setup_ui()


        # 程序启动时自动连接串口，开始预采集
        self.init_serial_connection()

        # 定时器用于更新图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)

        # 心率定时器：滑动窗口 8s，更新频率 0.5s
        self.hr_window_s = 8.0
        self.hr_update_interval_ms = 500  # 0.5s
        self.hr_timer = QTimer(self)
        self.hr_timer.timeout.connect(self._hr_update)
        self.hr_timer.start(self.hr_update_interval_ms)

        


    def save_intermediate_data(self):
        """保存中途数据为_f.csv文件并清理缓存"""
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return
        
        # 确保将串口线程中的积累数据传回主线程，然后再保存数据
        if self.serial_thread and self.is_recording:
            self.serial_thread.send_accumulated_data()
            self.log_message("已传回串口线程积累数据", "info")
            
        self.serial_thread.is_recording = False
        self.is_recording = False
        
        # 创建实验编号文件夹
        data_dir = os.path.join("data", self.experiment_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = os.path.join(data_dir, f"{self.experiment_id}_{self.current_phase}.csv")


        # 保存当前正式采集的数据（第一阶段数据）
        if self._save_data_to_file(filename):
            self.log_message(f"第{self.current_phase}阶段数据已保存到 {filename}", "info")

            # 保存当前阶段marker日志
            self.save_cycle_markers(data_dir)
            
            # 清理所有缓存（生物数据、marker、串口）
            self.clear_all_data_and_markers()
            
            # 清理串口缓存并休眠30秒
            self.clean_serial_and_rest()

            self.current_phase += 1  # 进入下一个阶段
        else:
            self.log_message(f"第{self.current_phase}阶段数据保存失败", "error")







    def save_final_data(self):
        """保存最终数据为_a.csv文件"""
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return
            
        # 创建实验编号文件夹
        data_dir = os.path.join("data", self.experiment_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = os.path.join(data_dir, f"{self.experiment_id}_2.csv")
        
        # 保存当前正式采集的数据（第二阶段数据）
        if self._save_data_to_file(filename, phase="second"):
            self.log_message(f"第二阶段数据已保存到 {filename}", "info")
            # 保存第二阶段marker日志
            self.save_second_phase_markers(data_dir)
        else:
            self.log_message(f"第二阶段数据保存失败", "error")


    # def save_resting_state_markers(self, data_dir):
    #     """保存静息态marker日志"""
    #     if not self.current_cycle_markers:
    #         self.log_message("没有静息态marker信息可保存", "warning")
    #         return

    #     marker_filename = os.path.join(data_dir, f"{self.experiment_id}_resting_state_marker_log.txt")

    #     try:
    #         with open(marker_filename, 'w', encoding='utf-8') as f:
    #             for marker_code, elapsed, timestamp in self.current_cycle_markers:
    #                 # 格式：marker数字 | 时间差 | 时间戳
    #                 line = f"{marker_code} | {elapsed:.3f} | {timestamp}\n"
    #                 f.write(line)
    #         self.log_message(f"静息态的Marker信息已保存到 {marker_filename}", "info")
    #     except Exception as e:
    #         self.log_message(f"静息态的marker文件失败: {str(e)}", "error")

    # def end_experiment(self):
    #     """结束实验：保存数据，断开串口连接，清理缓存"""
    #     self.log_message("实验结束", "info")

    # def save_second_phase_markers(self, data_dir):
    #     """保存第二阶段marker日志"""
    #     if not self.second_phase_markers:
    #         self.log_message("没有第二阶段marker信息可保存", "warning")
    #         return

    #     marker_filename = os.path.join(data_dir, f"{self.experiment_id}_2_marker_log.txt")

    #     try:
    #         with open(marker_filename, 'w', encoding='utf-8') as f:
    #             for marker_code, elapsed, timestamp in self.second_phase_markers:
    #                 # 格式：marker数字 | 时间戳
    #                 line = f"{marker_code} | {timestamp}\n"
    #                 f.write(line)
    #         self.log_message(f"第二阶段Marker信息已保存到 {marker_filename}", "info")
    #     except Exception as e:
    #         self.log_message(f"保存第二阶段marker文件失败: {str(e)}", "error")

    def restart_collection(self):
        """重新开始采集：清除预采集数据，开始正式采集下个阶段"""

        # 关键优化：清空串口线程的积累缓存，确保正式采集数据准确
        if self.serial_thread:
            self.serial_thread.reset_accumulator()

        # 开始正式采集
        self.serial_thread.is_recording = True
        self.is_recording = True

        # 清除预采集数据
        self.clear_pre_buffers()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 更新图表标题
        self.ecg_plot.setTitle(f'ECG信号 (正式采集模式 - 第{self.current_phase}阶段)')
        self.emg_plot.setTitle(f'EMG信号 (正式采集模式 - 第{self.current_phase}阶段)')

        self.log_message(f"{self.experiment_id} 实验第{self.current_phase}阶段开始", "info")

    def clean_serial_and_rest(self):
        """清理串口缓存并休眠30秒，然后重新开始预采集"""
        # 停止当前正式采集，但保持串口连接

        
        # 清理串口内部缓存
        if self.serial_thread and self.serial_thread.ser and self.serial_thread.ser.is_open:
            try:
                self.serial_thread.reset_accumulator()
                self.log_message("串口缓存已清理", "info")
            except Exception as e:
                self.log_message(f"清理串口缓存失败: {e}", "warning")
        
        # 更新图表标题为预采集模式
        self.ecg_plot.setTitle('ECG信号 (预采集模式 )')
        self.emg_plot.setTitle('EMG信号 (预采集模式 )')
        # GSR 已移除
        
        

    def init_serial_connection(self):
        """初始化串口连接，启动后立即开始预采集"""
        port = self.port_select.currentText()
        # 使用优化后的较小阈值
        self.serial_thread = SerialThread(port, baudrate=115200, buffer_threshold=60)
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.error_occurred.connect(self.handle_serial_error)
        self.serial_thread.start()

    def handle_serial_error(self, msg):
        """处理串口错误错误信息"""
        level = "error" if "错误" in msg else "info"
        self.log_message(msg, level)
        # 检查是否已初始化，再决定是否启用开始按钮
        self.start_button.setEnabled("错误" not in msg and self.initialized)

    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 串口控制区
        control_layout = QHBoxLayout()

        # 串口选择
        self.port_label = QLabel("选择串口:")
        self.port_select = QComboBox()
        self.port_select.addItem("COM10")
        self.port_select.addItem("COM9")
        self.port_select.addItem("COM8")
        self.port_select.addItem("COM7")
        self.port_select.addItem("COM6")
        self.port_select.addItem("COM5")
        self.port_select.addItem("COM4")
        self.port_select.addItem("COM3")
        self.port_select.addItem("COM2")
        self.port_select.addItem("COM1")
        self.port_select.setCurrentIndex(5)  # 默认选择第一个串口

        # # 新增：实验人数选择
        # self.subject_label = QLabel("采集人数:")
        # self.subject_select = QComboBox()
        # self.subject_select.addItem("1人")
        # self.subject_select.addItem("2人")
        # self.subject_select.setCurrentIndex(1)  # 默认选择2人

        # 实验编号输入框
        self.exp_id_label = QLabel("编号:")
        self.exp_id_edit = QLineEdit()
        self.exp_id_edit.setPlaceholderText("输入编号，如101、102")
        self.exp_id_edit.setFixedWidth(120)

        # 添加初始化按钮
        self.init_button = QPushButton("初始化设置")
        self.init_button.clicked.connect(self.initialize_settings)

        # 串口通信选择回调
        self.port_select.currentTextChanged.connect(self.select_port)

        # 按钮
        self.start_button = QPushButton("开始采集")
        self.start_button.clicked.connect(self.start_collection)
        self.start_button.setEnabled(False)  # 初始禁用，等待初始化和串口连接成功

        self.stop_button = QPushButton("停止采集")
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setEnabled(False)#开始采集数据后才能选择停止采集

        self.save_button = QPushButton("保存数据")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)  # 停止采集后才能选择保存数据

        control_layout.addWidget(self.port_label)
        control_layout.addWidget(self.port_select)
        # control_layout.addWidget(self.subject_label)  
        # control_layout.addWidget(self.subject_select)  
        control_layout.addWidget(self.exp_id_label)
        control_layout.addWidget(self.exp_id_edit)
        control_layout.addWidget(self.init_button)  
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)




        # 模式选择勾选框
        module_layout = QHBoxLayout()

        self.ecg_checkbox = QCheckBox("ECG")
        self.emg_checkbox = QCheckBox("EMG")
        # self.gsr_checkbox = QCheckBox("GSR")



        self.ecg_checkbox.setChecked(False)
        self.emg_checkbox.setChecked(False)
        # self.gsr_checkbox.setChecked(False)

        self.ecg_checkbox.stateChanged.connect(self.ECG_channel_changed)
        self.emg_checkbox.stateChanged.connect(self.EMG_channel_changed)
        # self.gsr_checkbox.stateChanged.connect(self.GSR_channel_changed)


        # 心率显示标签
        self.heart_rate_qlabel = QLabel(f"心率:{self.Heart_rate} bpm")
        self.heart_rate_qlabel.setFixedWidth(100)


        # 添加到模块布局
        module_layout.addWidget(self.ecg_checkbox)
        module_layout.addWidget(self.emg_checkbox)
        module_layout.addWidget(self.heart_rate_qlabel)
        # module_layout.addWidget(self.gsr_checkbox)


        # 状态栏
        self.messagelabel = QTextBrowser()

        # 波形显示
        self.create_groupboxes()

        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(module_layout)
        main_layout.addWidget(self.widget)
        main_layout.addWidget(self.messagelabel)

        self.log_message(f"应用已启动，开始预采集信号...", "info")
        self.log_message(f"采样频率: 1000Hz", "info")  # 启动时显示采样频率
        self.log_message(f"正在连接串口：{self.port_select.currentText()};波特率：500000", "info")
        self.log_message(f"请设置实验编号，然后点击初始化按钮", "warning")

    # 新增：初始化设置方法
    def initialize_settings(self):
        """处理初始化设置，将文本框内容赋值到属性并验证"""
        # 获取实验编号
        exp_id = self.exp_id_edit.text().strip()
        if not exp_id:
            self.log_message(f"错误: 实验编号不能为空", "error")
            self.start_button.setEnabled(False)
            self.initialized = False
            return

        # 单人模式：固定为1人
        self.experiment_id = exp_id
        self.subject_count = 1
        self.initialized = True

        # 禁用设置选项，防止运行时更改
        # self.ecg_checkbox.setEnabled(False)
        # self.emg_checkbox.setEnabled(False)
        # self.gsr_checkbox.setEnabled(False)

        # 更新曲线可见性
        self.update_curve_visibility()

        # 在状态栏显示信息
        self.log_message(f"初始化成功 - 实验编号: {self.experiment_id}", "info")

       # 新增：汇总并显示当前选择的通道
        selected_channels = []
        if self.ecg_checkbox.isChecked():
            selected_channels.append("ECG")
        if self.emg_checkbox.isChecked():
            selected_channels.append("EMG")
        # if self.gsr_checkbox.isChecked():
        #     selected_channels.append("GSR")
        channels_text = "、".join(selected_channels) if selected_channels else "无"
        self.log_message(f"已选择通道: {channels_text}", "info")

        # 检查串口连接状态，启用开始按钮
        if hasattr(self, 'serial_thread') and self.serial_thread and self.serial_thread.running:
            self.start_button.setEnabled(True)

    # 新增：更新曲线可见性
    def update_curve_visibility(self):
        # 单人模式：只显示第一组曲线
        self.ecg_curve1.setVisible(self.ecg_channel)
        self.emg_curve1.setVisible(self.emg_channel)
        try:
            self.ecg_curve2.setVisible(False)
            self.emg_curve2.setVisible(False)
        except Exception:
            pass


    def create_groupboxes(self):
        self.widget = QWidget()
        self.verticalLayout = QVBoxLayout(self.widget)

        # ECG GroupBox
        self.ecgBox = QGroupBox("ECG信号", self.widget)
        self.ecg_layout = QVBoxLayout(self.ecgBox)
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setRange(yRange=[200, 600])
        self.ecg_plot.setDownsampling(mode='peak')
        self.ecg_plot.setClipToView(True)
        self.ecg_plot.setBackground('w')
        self.ecg_plot.setLabel('left', '幅度', units='mV')
        self.ecg_plot.setLabel('bottom', '时间', units='s')
        self.ecg_plot.showGrid(x=True, y=True)
        self.ecg_plot.setTitle('ECG信号 (预采集模式)')
        self.ecg_layout.addWidget(self.ecg_plot)
        self.verticalLayout.addWidget(self.ecgBox)

        self.ecg_plot.addLegend()
        # 只保留一条 ECG 曲线，名称为 "ECG"
        self.ecg_curve1 = self.ecg_plot.plot(pen=pg.mkPen(color='r', width=2), name='ECG')

        # EMG GroupBox
        self.emgBox = QGroupBox("EMG信号", self.widget)
        self.emg_layout = QVBoxLayout(self.emgBox)
        self.emg_plot = pg.PlotWidget()
        self.emg_plot.setBackground('w')
        self.emg_plot.setYRange(0, 600)
        self.emg_plot.setDownsampling(mode='peak')
        self.emg_plot.setClipToView(True)
        self.emg_plot.setLabel('left', '幅度', units='mV')
        self.emg_plot.setLabel('bottom', '时间', units='s')
        self.emg_plot.showGrid(x=True, y=True)
        self.emg_plot.setTitle('EMG信号 (预采集模式)')
        self.emg_layout.addWidget(self.emg_plot)
        self.verticalLayout.addWidget(self.emgBox)

        self.emg_plot.addLegend()
        # 只保留一条 EMG 曲线，名称为 "EMG"
        self.emg_curve1 = self.emg_plot.plot(pen=pg.mkPen(color='r', width=2), name='EMG')

        # # GSR GroupBox
        # self.gsrBox = QGroupBox("GSR信号", self.widget)
        # self.gsr_layout = QVBoxLayout(self.gsrBox)
        # self.gsr_plot = pg.PlotWidget()
        # self.gsr_plot.setBackground('w')
        # self.gsr_plot.setYRange(0, 600)
        # self.gsr_plot.setDownsampling(mode='mean')
        # self.gsr_plot.setClipToView(True)
        # self.gsr_plot.setLabel('left', '幅度', units='mV')
        # self.gsr_plot.setLabel('bottom', '时间', units='s')
        # self.gsr_plot.showGrid(x=True, y=True)
        # self.gsr_plot.setTitle('GSR信号 (预采集模式)')
        # self.gsr_layout.addWidget(self.gsr_plot)
        # self.verticalLayout.addWidget(self.gsrBox)

        # self.gsr_plot.addLegend()
        # self.gsr_curve1 = self.gsr_plot.plot(pen=pg.mkPen(color='r', width=2), name='GSR_1')
        # self.gsr_curve2 = self.gsr_plot.plot(pen=pg.mkPen(color='b', width=2), name='GSR_2')

        # 初始更新曲线可见性
        self.update_curve_visibility()


    # 通道选择回调
    def ECG_channel_changed(self, state):
        self.ecg_channel = (state == Qt.Checked)
        if self.ecg_channel:
            self.log_message("ECG通道已启用", "info")
        else:
            self.log_message("ECG通道已禁用", "info")
     
    def EMG_channel_changed(self, state):
        self.emg_channel = (state == Qt.Checked)
        if self.emg_channel:
            self.log_message("EMG通道已启用", "info")
        else:
            self.log_message("EMG通道已禁用", "info")

    # def GSR_channel_changed(self, state):
    #     self.gsr_channel = (state == Qt.Checked)
    #     if self.gsr_channel:
    #         self.log_message("GSR通道已启用", "info")
    #     else:
    #         self.log_message("GSR通道已禁用", "info")



    # 串口选择回调
    def select_port(self):
        # 停止当前串口线程
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()

        # 连接新串口，继续预采集
        port = self.port_select.currentText()
        self.log_message(f"切换到串口：{port}，继续预采集", "info")
        # 优化：使用相同的优化设置
        self.serial_thread = SerialThread(port, baudrate=115200, buffer_threshold=140)
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.error_occurred.connect(self.handle_serial_error)
        self.serial_thread.start()
        # 检查是否已初始化
        self.start_button.setEnabled(self.initialized)

    # 在MainWindow类中修改start_collection方法
    def start_collection(self):
        # 新增：检查是否已初始化
        if not self.initialized:
            self.log_message("错误：请先完成初始化设置再开始采集", "error")
            return  # 未初始化则直接返回，不执行后续采集逻辑

        # 关键优化：清空串口线程的积累缓存，确保正式采集数据准确
        if self.serial_thread:
            self.serial_thread.reset_accumulator()

        # 清除缓冲区并开始记录数据
        self.clear_pre_buffers()


        self.serial_thread.is_recording = True
        self.is_recording = True

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)

        # 更新图表标题为正式采集模式
        self.ecg_plot.setTitle('ECG信号 (正式采集模式)')
        self.emg_plot.setTitle('EMG信号 (正式采集模式)')


        # 显示实验编号开始信息
        self.log_message(f"{self.experiment_id} 实验正式开始", "info")

    def handle_serial_data(self, emg, ecg):
        """处理串口接收的数据，根据状态决定存入预采集还是正式数据缓冲区"""
        if self.is_recording:
            # 正式采集状态，存入正式数据缓冲区
            self.ECG_1_data_.append_raw(ecg)
            self.EMG_1_data_.append_raw(emg)



        else:
            # 预采集状态，存入预采集数据缓冲区
            self.pre_ECG_1_data_.append_raw(ecg)
            self.pre_EMG_1_data_.append_raw(emg)


    def stop_collection(self):
        """停止正式数据采集（保持串口连接，继续预采集）"""
        if self.is_recording:
            self.serial_thread.is_recording = False
            self.is_recording = False

            # 关键优化：采集结束时，将串口线程积累区剩余数据全部传回主线程
            if self.serial_thread:
                self.serial_thread.send_accumulated_data()

                # 显示异常包数量
                self.display_abnormal_packet_count(self.serial_thread.abnormal_packet_count)

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(self.ECG_1_data_.has_data())
            
            # 恢复图表标题为预采集模式
            self.ecg_plot.setTitle('ECG信号 (预采集模式)')
            self.emg_plot.setTitle('EMG信号 (预采集模式)')
            # GSR 已移除

            # 显示实验编号结束信息
            self.log_message(f"{self.experiment_id} 停止采集", "info")
            
    def display_abnormal_packet_count(self, count):
        """在状态栏显示异常包数量"""
        self.log_message(f"本次采集异常包数量: {count}", "warning")

    # 预采集缓冲区清理方法在文件下方已定义，此处保留占位以避免重复定义

    def clear_all_data_and_markers(self):
        """清除所有数据缓冲区和marker缓存（在第一阶段保存后调用）"""
        # 清除正式数据缓冲区
        self.clear_formal_buffers()
        self.current_cycle_markers = []
        
        self.log_message("所有数据和marker缓存已清理", "info")

    def clear_formal_buffers(self):
        """清除正式数据缓冲区"""
        self.ECG_1_data_.clear()

        self.EMG_1_data_.clear()

        self.log_message("正式数据缓存已清理", "info")

    def clear_pre_buffers(self):
        """清除预采集数据缓冲区"""
        self.pre_ECG_1_data_.clear()

        self.pre_EMG_1_data_.clear()

        self.log_message("预采集数据已清理", "info")



    def experiment_complete(self):
        """实验完全结束"""
        self.stop_collection()
        self.is_experiment_started = False
        self.current_phase = 0
        self.log_message(f"{self.experiment_id} 实验完全结束", "info")

    def update_plot(self):
        # 根据当前状态选择要显示的数据缓冲区（只取最新数据，避免全量拷贝）
        if self.is_recording:
            ecg_buf = self.ECG_1_data_
            emg_buf = self.EMG_1_data_
        else:
            ecg_buf = self.pre_ECG_1_data_
            emg_buf = self.pre_EMG_1_data_

        # 要显示的点数（最多 display_points）
        n_display = int(self.display_points)
        # 获取各通道当前长度（无拷贝）
        len_ecg = ecg_buf.get_length()
        len_emg = emg_buf.get_length()

        # 如果都没有数据则直接返回
        if not (len_ecg or len_emg):
            return

        # 优先使用可选通道的最大可用点数（确保时间轴长度一致）
        if self.ecg_channel and self.emg_channel:
            n = min(n_display, len_ecg, len_emg)
        elif self.ecg_channel:
            n = min(n_display, len_ecg)
        elif self.emg_channel:
            n = min(n_display, len_emg)
        else:
            return  # 没有选中通道

        if n <= 0:
            return

        # 高效获取尾部数据（只拷贝需要的 n 个样本）
        ecg_tail = ecg_buf.get_last_n(n) if self.ecg_channel else np.zeros(n, dtype=np.uint16)
        emg_tail = emg_buf.get_last_n(n) if self.emg_channel else np.zeros(n, dtype=np.uint16)

        # 使用真实采样率生成时间轴（最近的数据在最后）
        time_axis = np.linspace(- (n - 1) / float(self.sampling_rate), 0, n)

        # 更新曲线（只调用一次 setData，避免重复读取）
        try:
            if self.ecg_channel:
                self.ecg_curve1.setData(time_axis, ecg_tail)
            else:
                self.ecg_curve1.setData([], [])

            if self.emg_channel:
                self.emg_curve1.setData(time_axis, emg_tail)
            else:
                self.emg_curve1.setData([], [])
        except Exception as e:
            self.log_message(f"绘图更新错误: {e}", "warning")

    def save_data(self):
        """保存原始数据到CSV文件（只保存正式采集的数据）- 保持原有功能不变"""
        # 使用类属性中的实验编号
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return

        # 确保data文件夹存在
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 使用类属性中的实验编号作为文件名
        default_filename = os.path.join(data_dir, f"bio_data_{self.experiment_id}")

        # 打开文件对话框
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", default_filename, "CSV文件 (*.csv);;所有文件 (*)"
        )

        if filename:
            if self._save_data_to_file(filename):
                self.log_message(f"正式采集数据已保存到 {filename}", "info")
                self.save_markers_to_txt()

    def _save_data_to_file(self, filename):
        """内部方法：将数据保存到指定文件"""
        try:
            # === 第一步：根据 subject_count 和复选框状态，构建 (列名, buffer) 列表 ===
            channel_info = []  # 存储元组: (列名字符串, 数据buffer)

            # 单人模式，仅保留 ECG 和 EMG 两个通道
            mapping = [
                (self.ecg_channel, "ECG_1", self.ECG_1_data_),
                (self.emg_channel, "EMG_1", self.EMG_1_data_),
            ]
            for is_selected, col_name, buffer in mapping:
                if is_selected:
                    channel_info.append((col_name, buffer))

            # 检查是否有选中通道
            if not channel_info:
                self.log_message("没有选择任何通道进行保存", "warning")
                return False

            # 提取 buffer 列表（用于计算长度）
            data_buffers = [buf for _, buf in channel_info]

            # === 第二步：计算最大长度并生成时间、marker ===
            data_lengths = [len(buf.get_raw_data()) for buf in data_buffers]
            max_length = max(data_lengths) if data_lengths else 0

            if max_length == 0:
                self.log_message("没有正式采集的数据可保存", "warning")
                return False

            # 生成时间列
            time = np.arange(max_length) / self.sampling_rate

            # 生成 marker 列
            marker_column = ["0"] * max_length
            for marker, elapsed, _ in self.current_cycle_markers:
                row_index = int(round(elapsed * self.sampling_rate))
                if 0 <= row_index < max_length:
                    marker_column[row_index] = str(marker)
                else:
                    self.log_message(f"Marker {marker} 位置超出数据范围", "warning")

            # === 第三步：动态构建 DataFrame ===
            df_dict = {'Time(s)': time}

            # 为每个选中的通道添加列
            for col_name, buffer in channel_info:
                raw_data = buffer.get_raw_data()
                # 补齐到 max_length（如果某些通道数据较短）
                if len(raw_data) < max_length:
                    raw_data = np.pad(raw_data, (0, max_length - len(raw_data)), constant_values=0)
                elif len(raw_data) > max_length:
                    raw_data = raw_data[:max_length]  # 截断（理论上不应发生）
                df_dict[col_name] = raw_data.astype(int)

            df_dict['marker'] = marker_column

            # 创建 DataFrame 并保存
            raw_df = pd.DataFrame(df_dict)
            if not filename.endswith('.csv'):
                filename += '.csv'
            raw_df.to_csv(filename, index=False)
            return True

        except Exception as e:
            self.log_message(f"保存文件失败: {str(e)}", "error")
            return False

    def log_message(self, message, level="info"):
        """记录消息到状态栏"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"

        # 根据日志级别设置不同的颜色
        if level == "info":
            self.set_text_color(QColor(0, 0, 0))  # 黑色
        elif level == "warning":
            self.set_text_color(QColor(255, 140, 0))  # 橙色
        elif level == "error":
            self.set_text_color(QColor(255, 0, 0))  # 红色

        # 添加消息到状态栏
        self.messagelabel.append(log_entry)

        # 滚动到底部
        self.messagelabel.verticalScrollBar().setValue(
            self.messagelabel.verticalScrollBar().maximum()
        )

    def set_text_color(self, color):
        """设置文本颜色"""
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor = self.messagelabel.textCursor()
        cursor.setCharFormat(fmt)
        self.messagelabel.setTextCursor(cursor)

    def closeEvent(self, event):
        """窗口关闭时确保串口线程正确停止"""
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        # if self.marker_server and self.marker_server.isRunning():
        #     self.marker_server.stop()
        #     self.marker_server.wait()
        event.accept()

    def _hr_update(self):
        """心率滑动窗口更新回调（由 hr_timer 每 0.5s 调用）

        使用 `self.ECG_1_data_.get_raw_data()` 获取最新数据，窗口长度为 `self.hr_window_s` 秒。
        调用已集成的 neurokit2 静态方法进行 R 波检测与心率计算，然后更新 `self.Heart_rate` 和 UI 标签。
        """
        try:
            ecg_all = self.ECG_1_data_.get_raw_data()
            # 
            if ecg_all is None or ecg_all.size == 0:
                self.Heart_rate = 0.0
                self.heart_rate_qlabel.setText(f"心率:{self.Heart_rate:.1f} bpm")
                return

            window_samples = int(self.sampling_rate * float(self.hr_window_s))
            if len(ecg_all) >= window_samples:
                segment = ecg_all[-window_samples:]
            else:
                segment = ecg_all

            bpm_mean, rpeaks, bpm_inst = self.compute_heart_rate_with_neurokit(segment, sampling_rate=self.sampling_rate)

            # 更新内部状态与 UI
            self.Heart_rate = float(bpm_mean)
            self.heart_rate_qlabel.setText(f"心率:{self.Heart_rate:.1f} bpm")

        except ImportError as ie:
            # neurokit2 未安装或导入失败，记录并停止定时器以避免重复报错
            self.log_message(str(ie), "error")
            try:
                self.hr_timer.stop()
            except Exception:
                pass
            self.heart_rate_qlabel.setText("心率: N/A")
        except Exception as e:
            # 其它运行时错误，记录但不停止定时器
            self.log_message(f"心率计算错误: {e}", "warning")
            self.heart_rate_qlabel.setText("心率: 0.0 bpm")
            self.Heart_rate = 0.0


    @staticmethod
    def compute_heart_rate_with_neurokit(ecg_signal, sampling_rate=1000):
        """
        使用 neurokit2 从原始 ECG 信号计算心率。

        参数:
        - ecg_signal: 1D 可迭代对象或 numpy 数组，原始 ECG 样本（任意数值类型）
        - sampling_rate: 采样率（Hz），默认 1000

        返回:
        - bpm_mean: float，平均心率（bpm）。当 R 波数 < 2 时返回 0.0。
        - rpeaks: 1D numpy 整数数组，R 波样本索引（以样点为单位）。
        - bpm_inst: 1D numpy 浮点数组，瞬时心率（bpm），长度为 len(rpeaks)-1。


        """
        try:
            import neurokit2 as nk
        except Exception as e:
            raise ImportError("neurokit2 未安装或导入失败。请运行 `pip install neurokit2`。原始错误: " + str(e))

        if ecg_signal is None:
            return 0.0, np.array([], dtype=int), np.array([], dtype=float)

        ecg = np.asarray(ecg_signal, dtype=float)
        if ecg.size < 10:
            return 0.0, np.array([], dtype=int), np.array([], dtype=float)

        try:
            # nk.ecg_peaks 返回 (signals, info)，info 中通常包含 'ECG_R_Peaks'
            signals, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)

            rpeaks = info.get('ECG_R_Peaks', None)

            # neurokit2 有时返回二值向量或索引列表，统一转换为索引数组
            if rpeaks is None:
                rpeaks_idx = np.array([], dtype=int)
            else:
                rpeaks_arr = np.asarray(rpeaks)
                # 如果是与信号等长的二值向量
                if rpeaks_arr.ndim == 1 and rpeaks_arr.size == ecg.size and set(np.unique(rpeaks_arr)).issubset({0, 1}):
                    rpeaks_idx = np.where(rpeaks_arr)[0].astype(int)
                else:
                    # 尝试把它当成索引列表
                    rpeaks_idx = rpeaks_arr.astype(int)

            if rpeaks_idx.size < 2:
                return 0.0, rpeaks_idx, np.array([], dtype=float)

            # 计算 RR 间期（秒）和瞬时心率（bpm）
            rr_intervals = np.diff(rpeaks_idx) / float(sampling_rate)
            bpm_inst = 60.0 / rr_intervals
            bpm_mean = float(np.mean(bpm_inst)) if bpm_inst.size > 0 else 0.0

            return bpm_mean, rpeaks_idx, bpm_inst

        except Exception as e:
            raise RuntimeError("neurokit2 在处理 ECG 信号时出错: " + str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())