#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path

import cv2
import numpy as np
from typing import Optional
from PySide6.QtCore import Qt, Signal, QThread, QMutex, QTimer
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QGridLayout,
    QTabWidget, QComboBox, QCheckBox,
    QProgressDialog  # 添加QDialog和QProgressDialog
)

from utils.common import cv2_to_qpixmap
from utils.detector import YOLODetector

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None


class VideoDisplayWidget(QLabel):
    """自定义视频显示控件，支持鼠标事件和绘制"""
    
    mouse_clicked = Signal(int, int)  # 鼠标点击信号(视频帧坐标x, y)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: none;")
        
        # 绘制相关
        self.drawing_points = []  # 要绘制的点
        self.drawing_lines = []   # 要绘制的线
        
        # 视频尺寸
        self.video_width = 1920
        self.video_height = 1080
        
        # 轨迹数据（用于叠加显示）
        self.track_data = {}  # {track_id: [(x, y, frame_idx), ...]}
        self.track_filter_ids = []  # 过滤的轨迹ID列表
        
    def set_video_size(self, width, height):
        """设置视频实际尺寸"""
        self.video_width = width
        self.video_height = height
        
    def set_track_data(self, tracks):
        """设置轨迹数据"""
        self.track_data = tracks
        
    def set_track_filter(self, track_ids):
        """设置轨迹过滤"""
        self.track_filter_ids = track_ids
        
    def set_pixmap_with_points(self, pixmap):
        """设置图片并绘制点"""
        self._original_pixmap = pixmap
        self.update()
        
    def add_drawing_point(self, x, y):
        """添加绘制点"""
        self.drawing_points.append((x, y))
        self.update()
        
    def add_drawing_line(self, x1, y1, x2, y2):
        """添加绘制线"""
        self.drawing_lines.append((x1, y1, x2, y2))
        self.update()
        
    def clear_drawing(self):
        """清除所有绘制"""
        self.drawing_points = []
        self.drawing_lines = []
        self.update()
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 计算相对于控件左上角的坐标
            x = event.pos().x()
            y = event.pos().y()
            
            # 获取控件尺寸
            label_width = self.width()
            label_height = self.height()
            
            # 映射到视频帧坐标
            video_x = int(x * self.video_width / label_width)
            video_y = int(y * self.video_height / label_height)
            
            # 发送信号
            self.mouse_clicked.emit(video_x, video_y)
            
    def paintEvent(self, event):
        """重写绘制事件"""
        # 先调用父类绘制图片
        super().paintEvent(event)
        
        # 如果没有图片，直接返回
        if not hasattr(self, '_original_pixmap') or self._original_pixmap.isNull():
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取控件尺寸
        label_width = self.width()
        label_height = self.height()
        
        # 绘制点
        painter.setPen(QPen(QColor(255, 0, 0), 5))  # 红色点
        for (x, y) in self.drawing_points:
            # 转换到控件坐标
            screen_x = int(x * label_width / self.video_width)
            screen_y = int(y * label_height / self.video_height)
            painter.drawPoint(screen_x, screen_y)
        
        # 绘制线
        painter.setPen(QPen(QColor(0, 255, 0), 3))  # 绿色线
        for (x1, y1, x2, y2) in self.drawing_lines:
            screen_x1 = int(x1 * label_width / self.video_width)
            screen_y1 = int(y1 * label_height / self.video_height)
            screen_x2 = int(x2 * label_width / self.video_width)
            screen_y2 = int(y2 * label_height / self.video_height)
            painter.drawLine(screen_x1, screen_y1, screen_x2, screen_y2)


class TrackReplayThread(QThread):
    """轨迹回放线程 - 只播放视频并叠加已有的轨迹数据"""
    
    frame_ready = Signal(np.ndarray)  # 帧播放完成信号
    replay_finished = Signal()  # 回放完成信号
    
    def __init__(self, video_path, tracks, track_ids, start_frame, end_frame, playback_speed=1.0, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.tracks = tracks  # 已有的轨迹数据
        self.track_ids = track_ids  # 要显示的轨迹ID列表
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.playback_speed = playback_speed
        self.stopped = False
        
        # 定义不同类别的颜色
        self.colors = {
            '小汽车': (255, 0, 0),      # 红色
            '面包车': (0, 255, 0),      # 绿色
            '公交车': (0, 0, 255),      # 蓝色
            '卡车': (255, 255, 0),      # 青色
            '行人': (0, 255, 255),      # 黄色
            '两轮车': (255, 0, 255),    # 紫色
            'default': (255, 255, 255)  # 白色
        }
    
    def stop(self):
        """停止回放"""
        self.stopped = True
    
    def run(self):
        """线程主函数"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频: {self.video_path}")
            return
        
        # 获取视频信息
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 跳转到起始帧
        if self.start_frame > 0:
            success = cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            if not success:
                print(f"无法跳转到帧 {self.start_frame}")
        
        current_frame = self.start_frame
        frame_count = 0
        
        print(f"开始轨迹回放: 帧范围 {self.start_frame}-{self.end_frame}, 轨迹数: {len(self.track_ids)}")
        
        while not self.stopped:
            # 检查是否到达结束帧
            if current_frame > self.end_frame:
                print(f"到达结束帧 {self.end_frame}")
                break
            
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取帧 {current_frame}")
                break
            
            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 绘制轨迹
            frame_with_tracks = self._draw_tracks(frame_rgb, current_frame)
            
            # 发送帧
            self.frame_ready.emit(frame_with_tracks)
            
            current_frame += 1
            frame_count += 1
            
            # 每30帧输出一次进度
            if frame_count % 30 == 0:
                print(f"轨迹回放进度: 帧 {current_frame}/{self.end_frame}")
            
            # 根据播放速度调整延迟
            if self.playback_speed > 0.1:
                delay = max(0.001, 0.03 / self.playback_speed)
                time.sleep(delay)
        
        cap.release()
        print(f"轨迹回放完成, 共播放 {frame_count} 帧")
        self.replay_finished.emit()
    
    def _draw_tracks(self, frame, current_frame):
        """在帧上绘制轨迹"""
        # 直接使用传入的帧，不再次转换颜色（避免重复转换）
        frame_rgb = frame
        
        # 如果没有要显示的轨迹，直接返回
        if not self.track_ids:
            return frame_rgb
        
        # 遍历选中的轨迹并绘制
        for track_id in self.track_ids:
            if track_id not in self.tracks:
                continue
            
            track_points = self.tracks[track_id]
            if len(track_points) < 2:
                continue
            
            # 获取颜色（默认红色）
            color = self.colors.get('default', (255, 0, 0))
            
            # 轨迹点已经按时间顺序记录，不需要排序
            # 只绘制到当前帧之前的轨迹点
            for i in range(1, len(track_points)):
                # 解包点坐标 - 兼容 (x,y,frame) 和 (x,y,frame,timestamp) 两种格式
                point1 = track_points[i-1]
                point2 = track_points[i]
                
                x1, y1, frame1 = point1[0], point1[1], point1[2]
                x2, y2, frame2 = point2[0], point2[1], point2[2]
                
                # 只绘制到当前帧之前的轨迹点
                if frame2 > current_frame:
                    continue
                
                # 绘制轨迹线
                cv2.line(frame_rgb, (x1, y1), (x2, y2), color, 2)  # 使用标准线宽
        
        return frame_rgb


class VideoProcessThread(QThread):
    """视频处理线程"""

    # 定义信号
    frame_ready = Signal(np.ndarray, dict)  # 帧处理完成信号(处理后的帧, 检测结果)
    events_ready = Signal(list)  # 本帧新增事件列表
    video_finished = Signal()  # 视频处理完成信号

    def __init__(self, detector, video_path, playback_speed=1.0, parent=None):
        """初始化线程
        
        Args:
            detector: YOLODetector实例
            video_path: 视频文件路径
            playback_speed: 播放速度，1.0为正常速度，>1加速，<1减速
            parent: 父对象
        """
        super().__init__(parent)
        self.detector = detector
        self.video_path = video_path
        self.stopped = False
        self.mutex = QMutex()
        self.playback_speed = playback_speed  # 播放速度控制

        # 追踪相关变量
        self.tracks = {}  # 存储目标轨迹，格式: {track_id: [(x, y, frame_idx), ...]}
        self.next_track_id = 0  # 下一个轨迹ID
        self.max_track_length = 30  # 最大轨迹长度
        self.previous_boxes = {}  # 存储前一帧检测到的目标，格式: {track_id: (x_center, y_center, width, height, class_id, confidence)}
        self.frame_idx = 0  # 帧索引
        self.end_frame_idx = -1  # 结束帧索引，-1表示播放到视频末尾
        
        # 交通事件检测相关变量
        self.stop_line = None  # 停止线坐标 [(x1, y1), (x2, y2)]
        self.signal_status = "绿灯"  # 当前信号灯状态
        self.lane_direction = "上→下"  # 当前车道方向
        self.lane_vector = (0, 1)  # 车道方向向量
        self.event_history = []  # 事件历史记录

        # DeepSORT 跟踪器
        self._tracker = None
        if DeepSort is not None:
            # 调整DeepSORT参数以提高跟踪稳定性，减少ID切换
            # max_age: 目标消失后保留ID的帧数，增加以减少ID切换
            # n_init: 初始化跟踪所需的连续检测帧数，增加以减少误跟踪
            # max_iou_distance: IOU匹配阈值，降低以严格匹配
            # nms_max_overlap: NMS阈值，降低以减少遮挡时的ID切换
            self._tracker = DeepSort(
                max_age=60,  # 增加到60，让跟踪器更宽容地处理暂时消失的目标
                n_init=5,     # 增加到5，更严格的初始化，减少误跟踪
                max_iou_distance=0.5,  # 降低到0.5，更严格的IOU匹配
                nn_budget=150,  # 增加特征预算，提高特征匹配准确性
                nms_max_overlap=0.8  # NMS重叠阈值
            )

        # 保存处理过的帧
        self.processed_frames = []
        self.save_frames = False  # 是否保存帧的标志

        # 获取视频基本信息
        cap = cv2.VideoCapture(video_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # ===== 交通事件判定策略（核心，未开源） =====
        # 闯红灯 / 逆行 / 疑似超速三类事件的判定规则、阈值整定与去重时机
        # 属于本课题核心内容，未随本仓库开源（见 utils/core_notice.py）。
        # 这里保留策略所需的状态容器，使「跟踪 -> 事件 -> 界面 -> 导出」
        # 的数据契约保持完整、可阅读。
        self._point_history_by_track = {}  # {track_id: [(x, y, frame), ...]} 目标代表点历史

        # 违规去重：记录每个track_id已上报的违规类型（整个视频期间只上报一次）
        self._reported_violations = {}  # {track_id: {"red_light": True, "reverse_driving": True, "speeding": True}}

        # 统计与事件缓存（供UI导出）
        self.events_all = []
        self.track_stats = {}  # {track_id: {"cls": str, "count": int, "vnorm_sum": float, "vnorm_max": float, "v_sum": float, "v_max": float, "speeding": bool}}  速度类字段由事件策略填充

        # 区域划分用于逆行检测：每个区域定义为一个矩形和行驶方向
        # 格式：{"rect": (x1, y1, x2, y2), "direction": (dx, dy), "name": "区域名称"}
        self.lane_regions = []  # 存储多个区域
        
        # 轨迹显示控制
        self.track_visible = False  # 默认隐藏轨迹
        self.track_filter_ids = []  # 轨迹过滤列表，空列表表示显示所有轨迹

    def _match_detections(self, current_detections):
        """匹配当前检测结果与现有轨迹
        
        Args:
            current_detections: 当前帧的检测结果，每个元素为 (x_center, y_center, width, height, class_id, confidence)
        
        Returns:
            匹配结果和新的检测结果
        """
        matches = []  # 存储匹配结果，格式: (track_id, detection_idx)
        unmatched_detections = list(range(len(current_detections)))  # 未匹配的检测结果索引

        # 如果没有现有轨迹，则所有检测结果都是新的
        if not self.previous_boxes:
            return matches, unmatched_detections

        # 计算IoU矩阵
        iou_matrix = np.zeros((len(self.previous_boxes), len(current_detections)))
        for i, (track_id, prev_box) in enumerate(self.previous_boxes.items()):
            x1_prev, y1_prev, w_prev, h_prev = prev_box[0] - prev_box[2] / 2, prev_box[1] - prev_box[3] / 2, prev_box[
                2], prev_box[3]
            for j, curr_box in enumerate(current_detections):
                x1_curr, y1_curr, w_curr, h_curr = curr_box[0] - curr_box[2] / 2, curr_box[1] - curr_box[3] / 2, \
                curr_box[2], curr_box[3]

                # 计算IoU
                x_left = max(x1_prev, x1_curr)
                y_top = max(y1_prev, y1_curr)
                x_right = min(x1_prev + w_prev, x1_curr + w_curr)
                y_bottom = min(y1_prev + h_prev, y1_curr + h_curr)

                if x_right < x_left or y_bottom < y_top:
                    iou = 0.0
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    box1_area = w_prev * h_prev
                    box2_area = w_curr * h_curr
                    iou = intersection_area / float(box1_area + box2_area - intersection_area)

                iou_matrix[i, j] = iou

        # 基于IoU进行匹配
        if iou_matrix.size > 0:
            # 利用贪心算法进行匹配
            while iou_matrix.size > 0 and iou_matrix.max() > 0.3:  # IoU阈值设为0.3
                # 找到最大IoU对应的行列索引
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                row, col = max_idx

                # 获取对应的track_id
                track_id = list(self.previous_boxes.keys())[row]

                # 添加匹配结果
                matches.append((track_id, col))

                # 从未匹配列表中移除已匹配检测
                if col in unmatched_detections:
                    unmatched_detections.remove(col)

                # 将已匹配的行列置为0，防止重复匹配
                iou_matrix[row, :] = 0
                iou_matrix[:, col] = 0

        return matches, unmatched_detections

    def run(self):
        """线程主函数"""
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)

        # 检查视频是否打开成功
        if not cap.isOpened():
            QMessageBox.critical(None, "错误", f"无法打开视频文件: {self.video_path}")
            return

        # 处理每一帧
        start_frame_idx = self.frame_idx  # 保存起始帧索引
        self.processed_frames = []  # 清空之前的帧
        
        # 计算跳帧间隔
        # 如果播放速度大于1.5，则跳帧以实现更快的播放
        skip_frames = max(0, int(self.playback_speed * 0.5))
        frame_count = 0

        # 如果需要跳转到指定帧
        if start_frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        while not self.stopped:
            # 检查是否到达指定的结束帧
            if self.end_frame_idx > 0 and self.frame_idx >= self.end_frame_idx:
                break
            
            # 读取一帧
            ret, frame = cap.read()

            # 检查是否到达视频末尾
            if not ret:
                break
            
            frame_count += 1
            
            # 根据播放速度决定是否跳过当前帧
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                # 跳帧时仍需更新帧索引
                self.frame_idx += 1
                continue

            # 1) YOLO 推理（用BGR帧，避免反复RGB/BGR来回）
            conf_value = self.detector.conf_threshold
            iou_value = self.detector.iou_threshold
            yolo_res = self.detector.model(frame, conf=conf_value, iou=iou_value)[0]

            # 2) 统计与绘制（绘制需要RGB）
            class_counts = self.detector._count_objects(yolo_res)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.detector._draw_boxes(frame_rgb, yolo_res)

            # 3) DeepSORT 跟踪
            self._update_tracks_deepsort(frame_bgr=frame, yolo_res=yolo_res)

            # 4) 绘制所有轨迹（实时绘制，不隐藏）
            processed_frame = self._draw_all_trajectories(processed_frame)

            # 生成本帧事件（越线/疑似超速）并叠加显示
            frame_events = self._update_violations_and_overlay(processed_frame)
            if frame_events:
                self.events_all.extend(frame_events)
                self.events_ready.emit(frame_events)

            # 如果需要保存帧，则添加到列表中
            if self.save_frames:
                # 转换为BGR以便保存
                frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                self.processed_frames.append(frame_bgr)

            # 发送信号
            self.frame_ready.emit(processed_frame, class_counts)

            # 增加帧索引
            self.frame_idx += 1

            # 根据播放速度调整延迟
            if self.playback_speed > 0.1:  # 避免速度过慢
                # 正常速度下延迟约1ms，速度越快延迟越小
                delay = max(0.001, 0.03 / self.playback_speed)
                time.sleep(delay)

        # 释放资源
        cap.release()

        # 发送视频处理完成信号
        self.video_finished.emit()
    
    def set_stop_line(self, stop_line):
        """设置停止线坐标
        
        Args:
            stop_line: 停止线坐标，格式为 [(x1, y1), (x2, y2)]
        """
        self.stop_line = stop_line
    
    def set_signal_status(self, status):
        """设置信号灯状态
        
        Args:
            status: 信号灯状态，可选值："绿灯", "黄灯", "红灯"
        """
        self.signal_status = status
    
    def set_lane_direction(self, direction):
        """设置车道方向
        
        Args:
            direction: 车道方向，可选值："上→下", "下→上", "左→右", "右→左"
        """
        self.lane_direction = direction
        # 更新车道方向向量
        direction_map = {
            "上→下": (0, 1),
            "下→上": (0, -1),
            "左→右": (1, 0),
            "右→左": (-1, 0)
        }
        self.lane_vector = direction_map.get(direction, (0, 1))
    
    def _is_crossing_stop_line(self, vehicle_bottom):
        """判断车辆是否越过停止线
        
        Args:
            vehicle_bottom: 车辆底部中心点坐标 (x, y)
            
        Returns:
            bool: 是否越过停止线
        """
        # 停止线越线判定规则属于论文核心内容，未随本仓库开源。
        # 这里返回安全默认值，保证「跟踪 -> 事件 -> 界面」调用链完整。
        return not_open_sourced("stop_line_crossing", False)
    
    def _is_reversing(self, track_id):
        """判断车辆是否逆行（基于区域划分）

        Args:
            track_id: 目标轨迹ID

        Returns:
            bool: 是否逆行
        """
        # 逆行判定规则（区域划分 + 轨迹运动方向判定）属于论文核心内容，
        # 未随本仓库开源。这里返回安全默认值，保证调用链完整。
        return not_open_sourced("lane_direction_reversal", False)

    def _get_direction_for_position(self, pos):
        """根据位置获取对应的行驶方向向量

        Args:
            pos: 当前位置 (x, y)

        Returns:
            tuple: 方向向量 (dx, dy)
        """
        # 检查位置是否在某个区域内
        for region in self.lane_regions:
            rect = region.get("rect")
            if rect and self._is_point_in_rect(pos, rect):
                return region.get("direction", self.lane_vector)

        # 如果不在任何区域内，使用默认方向
        return self.lane_vector

    def _is_point_in_rect(self, pos, rect):
        """检查点是否在矩形内

        Args:
            pos: 位置 (x, y)
            rect: 矩形 (x1, y1, x2, y2)

        Returns:
            bool: 是否在矩形内
        """
        x, y = pos
        x1, y1, x2, y2 = rect

        # 确保x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return x1 <= x <= x2 and y1 <= y <= y2

    def _update_tracks_deepsort(self, frame_bgr: np.ndarray, yolo_res):
        """使用 DeepSORT 更新 tracks/previous_boxes"""
        if self._tracker is None:
            # 依赖未安装时退化为原 IoU 匹配逻辑（复用旧方法：会二次推理，但不至于崩）
            try:
                self._update_tracks(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), {})
            except Exception:
                pass
            return

        bbs = []
        try:
            for box in yolo_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                cls_cn = self._cls_cn(cls_id) or str(cls_id)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                
                # 位置 & 尺寸过滤：只保留道路区域内、面积合理的目标
                H, W = frame_bgr.shape[:2]
                y_bottom = y2
                
                # 降低置信度阈值，检测更多目标
                if conf < 0.25:
                    continue
                    
                # 移除位置过滤，允许检测整个画面
                # if y_bottom < 0.35 * H:
                #     continue  # 画面上半部分（天空/远处建筑）过滤
                    
                # 放宽尺寸过滤，检测更小的目标
                area = w * h
                min_area = 25 * 25  # 减小最小面积，检测更远/更小的车辆
                max_area = 0.3 * W * H  # 增大最大面积
                if area < min_area or area > max_area:
                    continue
                    
                # 放宽宽高比过滤
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.15 or aspect_ratio > 6:  # 放宽宽高比限制
                    continue
                    
                # 只添加高质量检测框到跟踪器
                bbs.append(([x1, y1, w, h], conf, cls_cn))
        except Exception:
            bbs = []

        tracks = self._tracker.update_tracks(bbs, frame=frame_bgr)

        # 清理与更新
        self.previous_boxes = {}
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            ltrb = trk.to_ltrb()  # x1,y1,x2,y2
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = int(trk.track_id)
            cls_cn = getattr(trk, "det_class", None) or ""
            conf = float(getattr(trk, "det_conf", 0.0) or 0.0)

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            # previous_boxes 仍然使用 (x_center, y_center, w, h, cls_id, conf) 的形式，
            # 这里 cls_id 填 -1，后续事件逻辑将优先使用 track_stats/cls_cn
            self.previous_boxes[track_id] = (x_center, y_center, width, height, -1, conf, cls_cn)

            # 计算时间戳（秒）
            timestamp = self.frame_idx / self.video_fps
            
            # 记录轨迹中心点（包含时间戳）- 限制轨迹长度，但保留起始帧
            if track_id not in self.tracks:
                # 首次出现，记录起始帧
                self.tracks[track_id] = [(int(x_center), int(y_center), int(self.frame_idx), timestamp)]
            else:
                # 已有轨迹，追加新点
                self.tracks[track_id].append((int(x_center), int(y_center), int(self.frame_idx), timestamp))
                # 限制轨迹长度（保留起始帧和最近的点）
                if len(self.tracks[track_id]) > 500:  # 最大500个点
                    # 保留第一个点（起始帧）和最近的499个点
                    self.tracks[track_id] = [self.tracks[track_id][0]] + self.tracks[track_id][-499:]


    def _det_to_bbox(self, det):
        # det: (x_center, y_center, w, h, cls_id, conf[, cls_cn])
        x_center, y_center, w, h = det[0], det[1], det[2], det[3]
        cls_id = det[4] if len(det) > 4 else -1
        conf = det[5] if len(det) > 5 else 0.0
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        cls_cn = det[6] if len(det) > 6 else None
        return x1, y1, x2, y2, int(cls_id), float(conf), cls_cn

    def _cls_cn(self, cls_id: int) -> Optional[str]:
        try:
            cls_name = self.detector.class_names[int(cls_id)]
        except Exception:
            return None
        if cls_name in self.detector.CLASS_MAPPING:
            return self.detector.CLASS_MAPPING[cls_name]
        return None

    def _is_vehicle_cn(self, cn_name: str) -> bool:
        return cn_name in {"小汽车", "面包车", "公交车", "卡车", "两轮车"}

    def _ensure_track_stat(self, track_id: int, cls_cn: str):
        if track_id not in self.track_stats:
            self.track_stats[track_id] = {
                "cls": cls_cn,
                "count": 0,
                "vnorm_sum": 0.0,
                "vnorm_max": 0.0,
                "v_sum": 0.0,
                "v_max": 0.0,
                "speeding": False,
            }

    def _update_violations_and_overlay(self, frame_rgb: np.ndarray) -> list:
        h, w = frame_rgb.shape[:2]

        if self.video_fps and self.video_fps > 1e-6:
            fps = float(self.video_fps)
        else:
            fps = 30.0

        new_events = []

        for track_id, det in list(self.previous_boxes.items()):
            x1, y1, x2, y2, cls_id, conf, cls_cn_from_track = self._det_to_bbox(det)
            cls_cn = cls_cn_from_track or (self._cls_cn(cls_id) if cls_id >= 0 else None)
            if not cls_cn:
                cls_cn = self.track_stats.get(track_id, {}).get("cls", "")
            if not cls_cn:
                continue

            # 代表点：底部中心
            px = int((x1 + x2) / 2)
            py = int(y2)
            p = (px, py)

            # 更新统计容器
            self._ensure_track_stat(track_id, cls_cn)

            # ===== 交通事件判定（核心策略，未开源） =====
            # 判定规则、阈值与去重时机属于论文核心内容，未随本仓库开源。
            # 这里只保留调用契约与事件字典格式，策略入口见 _evaluate_traffic_events。
            new_events.extend(
                self._evaluate_traffic_events(track_id, cls_cn, p, (x1, y1, x2, y2))
            )

            # 在目标附近标注 track_id
            cv2.putText(
                frame_rgb,
                f"ID:{track_id}",
                (max(x1, 0), max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 只在画面上方显示最近几条事件摘要，避免遮挡目标
        if not hasattr(self, "_overlay_events"):
            self._overlay_events = []
        self._overlay_events.extend(new_events)
        max_keep = 5
        if len(self._overlay_events) > max_keep:
            self._overlay_events = self._overlay_events[-max_keep:]

        y0 = 25
        dy = 20
        for idx, ev in enumerate(reversed(self._overlay_events)):
            y = y0 + idx * dy
            ev_type = ev.get("type", "")
            if ev_type == "speeding_suspect":
                text = f"[超速?] ID:{ev['track_id']} {ev['cls']}"
            elif ev_type == "red_light":
                text = f"[闯红灯] ID:{ev['track_id']} {ev['cls']}"
            elif ev_type == "reverse_driving":
                text = f"[逆行] ID:{ev['track_id']} {ev['cls']}"
            else:
                text = f"[{ev_type}] ID:{ev['track_id']} {ev['cls']}"
            cv2.putText(
                frame_rgb,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return new_events

    def _evaluate_traffic_events(self, track_id, cls_cn, point, bbox):
        """对单个目标执行交通事件判定，返回本帧新产生的事件列表。

        本方法是「跟踪结果 -> 交通事件」的唯一入口，负责维护判定所需的状态
        （目标代表点历史、已上报违规去重等），并按需产生三类事件：

        - ``red_light``：闯红灯。
        - ``reverse_driving``：逆行。
        - ``speeding_suspect``：疑似超速。"疑似"表示该事件由视频画面估计得到，
          不等同于雷达等专用设备的精确测速结果。

        Args:
            track_id: 目标轨迹ID。
            cls_cn: 目标中文类别名（如 "小汽车"、"两轮车"）。
            point: 目标代表点坐标 ``(x, y)``，取检测框底部中心。
            bbox: 目标检测框 ``(x1, y1, x2, y2)``。

        Returns:
            list[dict]: 本帧新产生的事件列表。事件字典字段契约为
            （界面事件表与 ``events.csv`` 导出依赖这些字段）::

                {
                    "type": "red_light" | "reverse_driving" | "speeding_suspect",
                    "frame": int,          # 帧号
                    "track_id": int,       # 轨迹ID
                    "cls": str,            # 中文类别名
                    "point": (int, int),   # 事件发生位置
                    # 以下字段仅 speeding_suspect 事件使用
                    "vnorm": float,        # 速度指标（用于判定是否疑似超速）
                    "v_px_s": float,       # 像素速度 (px/s)
                    "box_h": int,          # 检测框高度
                }

        .. note::
            三类事件的判定规则、阈值整定与去重时机属于论文核心内容，
            **未随本仓库开源**。此处返回空列表（即不触发告警），
            以保证检测 / 跟踪 / 界面 / 导出链路可以完整运行。
            完整策略可在面试中当面讲解。
        """
        return not_open_sourced("traffic_event_rules", []) or []

    def _draw_all_trajectories(self, frame):
        """绘制所有目标的轨迹（根据开关和过滤条件）
        
        Args:
            frame: 当前帧（RGB格式）
            
        Returns:
            绘制了轨迹的帧
        """
        # 如果轨迹显示开关关闭，直接返回原帧
        if not self.track_visible:
            return frame
        
        frame_with_tracks = frame.copy()
        
        # 定义不同类别的颜色
        colors = {
            '小汽车': (255, 0, 0),      # 红色
            '面包车': (0, 255, 0),      # 绿色
            '公交车': (0, 0, 255),      # 蓝色
            '卡车': (255, 255, 0),      # 青色
            '行人': (0, 255, 255),      # 黄色
            '两轮车': (255, 0, 255),    # 紫色
            'default': (255, 255, 255)  # 白色
        }
        
        # 遍历所有轨迹并绘制（每辆车只有一条轨迹线）
        for track_id, track_points in self.tracks.items():
            # 如果有过滤列表，只绘制选中的轨迹
            if self.track_filter_ids and track_id not in self.track_filter_ids:
                continue
            
            # 确保至少有2个点才能绘制轨迹线
            if len(track_points) > 1:
                # 获取轨迹对应的类别
                cls_cn = self.track_stats.get(track_id, {}).get("cls", "default")
                color = colors.get(cls_cn, colors['default'])
                
                # 按帧号排序轨迹点（确保顺序正确）
                sorted_points = sorted(track_points, key=lambda p: p[2])
                
                # 绘制一条连续的轨迹线（每辆车只有一条）
                for i in range(1, len(sorted_points)):
                    # 获取相邻两个点（都是车辆中心点）
                    # 兼容 (x,y,frame) 和 (x,y,frame,timestamp) 两种格式
                    point1 = sorted_points[i-1]
                    point2 = sorted_points[i]
                    
                    # 解包点坐标
                    x1, y1, frame1 = point1[0], point1[1], point1[2]
                    x2, y2, frame2 = point2[0], point2[1], point2[2]
                    
                    # 只连接时间间隔小于5帧的点（避免连接断开的轨迹）
                    frame_diff = frame2 - frame1
                    if frame_diff <= 5:
                        cv2.line(frame_with_tracks,
                                 (x1, y1),
                                 (x2, y2),
                                 color, 2)
        
        return frame_with_tracks
    
    def set_track_visible(self, visible):
        """设置轨迹显示开关
        
        Args:
            visible: True显示轨迹，False隐藏轨迹
        """
        self.track_visible = visible
    
    def set_track_filter(self, track_ids):
        """设置轨迹过滤（只显示指定ID的轨迹）
        
        Args:
            track_ids: 要显示的轨迹ID列表，None或空列表表示显示所有轨迹
        """
        self.track_filter_ids = track_ids
    
    def _update_tracks(self, frame, results):
        """更新目标轨迹
        
        Args:
            frame: 当前帧
            results: 检测结果
        """
        try:
            # 处理检测结果
            current_detections = []

            # 从模型结果中提取检测框（使用主窗口的置信度和IOU值）
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 通过detector对象获取主窗口设置的参数值
            conf_value = self.detector.conf_threshold
            iou_value = self.detector.iou_threshold

            model_results = self.detector.model(img_bgr, conf=conf_value, iou=iou_value)[0]

            # 提取当前帧的所有检测框
            for box in model_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())

                # 计算中心点、宽度和高度
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                current_detections.append((x_center, y_center, width, height, cls_id, conf))

            # 匹配当前检测结果与已有轨迹
            matches, unmatched_detections = self._match_detections(current_detections)

            # 更新匹配的轨迹
            for track_id, detection_idx in matches:
                det = current_detections[detection_idx]
                x_center, y_center = det[0], det[1]

                # 将当前位置添加到轨迹中
                if track_id in self.tracks:
                    self.tracks[track_id].append((int(x_center), int(y_center), self.frame_idx))
                    # 保持轨迹长度不超过最大值
                    if len(self.tracks[track_id]) > self.max_track_length:
                        self.tracks[track_id] = self.tracks[track_id][-self.max_track_length:]

                # 更新previous_boxes
                self.previous_boxes[track_id] = det

            # 为未匹配的检测创建新轨迹
            for detection_idx in unmatched_detections:
                det = current_detections[detection_idx]
                x_center, y_center = det[0], det[1]

                # 创建新轨迹
                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = [(int(x_center), int(y_center), self.frame_idx)]
                self.previous_boxes[track_id] = det

            # 绘制轨迹
            for track_id, track in self.tracks.items():
                if len(track) > 1:
                    # 根据类别ID选择颜色
                    if track_id in self.previous_boxes:
                        cls_id = self.previous_boxes[track_id][4]
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # 蓝、绿、红、青、黄
                        color = colors[cls_id % len(colors)]
                    else:
                        color = (255, 255, 255)  # 白色

                    # 绘制轨迹线
                    for i in range(1, len(track)):
                        if track[i][2] - track[i - 1][2] <= 5:  # 只连接时间间隔小于5帧的点
                            cv2.line(frame, (track[i - 1][0], track[i - 1][1]),
                                     (track[i][0], track[i][1]), color, 2)

        except Exception as e:
            print(f"更新轨迹时出错: {str(e)}")

    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.stopped = True
        self.mutex.unlock()

    def start_saving_frames(self):
        """开始保存帧"""
        self.save_frames = True
        self.processed_frames = []  # 清空之前保存的帧

    def stop_saving_frames(self):
        """停止保存帧"""
        self.save_frames = False

    def save_video(self, output_path):
        """将处理过的帧保存为视频
        
        Args:
            output_path: 输出视频路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.processed_frames:
            return False

        try:
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.video_fps,
                (self.video_width, self.video_height)
            )

            # 写入帧
            for frame in self.processed_frames:
                out.write(frame)

            # 释放资源
            out.release()

            return True
        except Exception as e:
            print(f"保存视频时出错: {str(e)}")
            return False


class MainWindow(QMainWindow):
    """主窗口(功能页)"""

    def __init__(self, username=None, avatar_path=None):
        super().__init__()

        # 用户名
        self.username = username

        # 用户头像路径
        self.avatar_path = avatar_path

        # 初始化检测器（允许启动后再选择模型，避免因为缺少best.pt直接退出）
        self.detector = None
        self.model_path = os.path.join("model", "best.pt")
        try:
            if os.path.exists(self.model_path):
                self.detector = YOLODetector(model_path=self.model_path)
        except Exception as e:
            # 不强制退出，提示用户稍后手动选择模型
            QMessageBox.warning(self, "模型加载失败", f"默认模型加载失败: {str(e)}\n请点击“选择模型文件”手动加载。")

        # 设置窗口属性
        self.setWindowTitle("基于深度学习的公共交通监控系统")
        self.resize(1400, 850)

        # 创建UI
        self._init_ui()

        # 初始化视频处理相关变量
        self.video_thread = None
        self.is_processing_video = False
        self.is_saving_video = False  # 是否正在保存视频的标志
        self.playback_speed = 1.0  # 默认播放速度

        # 视频事件/轨迹统计缓存（用于导出）
        self.video_events = []
        self.video_track_stats = {}

        # 轨迹选择显示功能
        self.selected_track_ids = set()  # 选中的轨迹ID集合
        self.is_replay_mode = False  # 是否在回放模式
        self.replay_frame_idx = 0  # 回放帧索引
        self.replay_tracks = {}  # 保存的轨迹数据
        self.replay_video_path = None  # 回放的视频路径

        # 批量处理图片结果列表
        self.batch_results = []
        self.current_batch_index = -1

        # 特征统计相关变量
        self.features_count = {
            "颜色": {},
            "车型": {},
            "类别": {}
        }
        
        # 交通事件检测相关变量
        self.stop_lines = []  # 存储停止线坐标 [(x1, y1), (x2, y2)]
        self.is_mark_stop_line = False  # 是否处于标记停止线模式
        self.mark_stop_line_points = []  # 当前正在标记的停止线点
        self.current_stop_line = None  # 当前使用的停止线

        # 区域划分用于逆行检测
        self.lane_regions = []  # 存储区域 [{"rect": (x1, y1, x2, y2), "direction": (dx, dy), "name": "区域名"}]
        self.is_mark_region = False  # 是否处于标记区域模式
        self.mark_region_points = []  # 当前正在标记的区域点
        self.current_region_direction = (0, 1)  # 当前区域的方向向量
        self.current_region_name = ""  # 当前区域名称

        # 初始化信号灯状态和车道方向
        self.signal_status = "绿灯"
        self.lane_direction = "上→下"

        # 如果未加载模型，则禁用检测相关按钮
        self._set_detection_enabled(self.detector is not None)

    def _set_detection_enabled(self, enabled: bool):
        """启用/禁用依赖模型的功能按钮"""
        for btn in getattr(self, "_model_dependent_buttons", []):
            try:
                btn.setEnabled(bool(enabled))
            except Exception:
                pass

    def _init_ui(self):
        """初始化UI"""
        # 设置应用全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                font-family: 'Microsoft YaHei';
            }
            QLabel {
                font-size: 12px;
                color: #2c3e50;
            }
            QFrame {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 5px 10px;
                font-weight: bold;
                border-radius: 4px;
                font-size: 13px;
                height: 28px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
            QTableWidget {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                gridline-color: #ecf0f1;
                selection-background-color: #bdc3c7;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #dcdde1;
                font-size: 12px;
                font-weight: bold;
            }
        """)

        # 初始化模型参数默认值（允许 detector 为空：用户稍后再加载模型）
        self.conf_value = self.detector.conf_threshold if self.detector else 0.25
        self.iou_value = 0.5  # 默认值

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 左侧控制面板 (35%)
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(420)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(10, 10, 10, 10)
        left_panel_layout.setSpacing(6)

        # ===== 用户信息区域 =====
        user_info_frame = QFrame()
        user_info_frame.setFrameShape(QFrame.StyledPanel)
        user_info_layout = QHBoxLayout(user_info_frame)
        user_info_layout.setContentsMargins(10, 10, 10, 10)

        # 头像
        avatar_label = QLabel()
        avatar_label.setFixedSize(40, 40)  # 设置固定大小
        avatar_label.setAlignment(Qt.AlignCenter)  # 居中对齐

        # 尝试加载头像图片
        avatar_pixmap = None
        if self.avatar_path and os.path.exists(self.avatar_path):
            # 如果有用户自定义头像，优先使用
            avatar_pixmap = QPixmap(self.avatar_path)
        else:
            # 否则使用默认头像
            avatar_pixmap = QPixmap(os.path.join("resources", "icons", "avator.jpg"))

        if avatar_pixmap and not avatar_pixmap.isNull():
            # 缩放图片并设置为圆形
            scaled_pixmap = avatar_pixmap.scaled(
                40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            avatar_label.setPixmap(scaled_pixmap)
            avatar_label.setStyleSheet("border-radius: 20px; background-color: #ecf0f1;")
        else:
            # 如果找不到图标，使用文字代替
            avatar_label.setText("👤")
            avatar_label.setStyleSheet(
                "font-size: 20px; color: #3498db; background-color: #ecf0f1; border-radius: 20px;")

        self.avatar_label = avatar_label  # 保存引用以便后续更新
        user_info_layout.addWidget(avatar_label)

        # 用户名和退出按钮的垂直布局
        user_info_text_layout = QVBoxLayout()
        user_info_text_layout.setSpacing(2)

        # 用户名
        self.username_label = QLabel(f"{self.username}" if self.username else "访客")
        self.username_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        user_info_text_layout.addWidget(self.username_label)

        # 系统名称
        system_name_label = QLabel("基于YOLOv8的车辆行人检测系统")
        system_name_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        system_name_label.setStyleSheet("color: #2c3e50;")
        user_info_text_layout.addWidget(system_name_label)

        user_info_layout.addLayout(user_info_text_layout)
        user_info_layout.addStretch()

        left_panel_layout.addWidget(user_info_frame)

        # ===== 控制按钮区域 =====
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(8)

        # 控制按钮标题
        control_title = QLabel("功能控制")
        control_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        control_title.setAlignment(Qt.AlignCenter)
        control_title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        control_layout.addWidget(control_title)

        # 基础按钮样式（实例属性）
        self.base_button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 3px 10px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 11px;
                min-width: 340px;
                max-width: 340px;
                min-height: 24px;
                text-align: left;
                icon-size: 12px;
                spacing: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """

        # 模型选择按钮
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.setIcon(QIcon(os.path.join("resources", "icons", "model.png")))
        self.select_model_btn.setStyleSheet(self.base_button_style)
        self.select_model_btn.clicked.connect(self._on_select_model_clicked)
        control_layout.addWidget(self.select_model_btn)

        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片文件")
        self.select_image_btn.setIcon(QIcon(os.path.join("resources", "icons", "image.png")))
        self.select_image_btn.setStyleSheet(self.base_button_style)
        self.select_image_btn.clicked.connect(self._on_select_image_clicked)
        control_layout.addWidget(self.select_image_btn)

        # 选择图片列表按钮
        self.select_image_list_btn = QPushButton("选择图片列表")
        self.select_image_list_btn.setIcon(QIcon(os.path.join("resources", "icons", "image-list.png")))
        self.select_image_list_btn.setStyleSheet(self.base_button_style)
        self.select_image_list_btn.clicked.connect(self._on_select_image_list_clicked)
        control_layout.addWidget(self.select_image_list_btn)

        # 视频选择按钮
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.setIcon(QIcon(os.path.join("resources", "icons", "video.png")))
        self.select_video_btn.setStyleSheet(self.base_button_style)
        self.select_video_btn.clicked.connect(self._on_select_video_clicked)
        control_layout.addWidget(self.select_video_btn)

        # 保存检测结果按钮
        self.save_results_btn = QPushButton("保存检测结果")
        self.save_results_btn.setIcon(QIcon(os.path.join("resources", "icons", "save.png")))
        self.save_results_btn.clicked.connect(self._on_save_results_clicked)
        self.save_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 3px 10px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 11px;
                min-width: 340px;
                max-width: 340px;
                min-height: 24px;
                text-align: left;
                icon-size: 12px;
                spacing: 4px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
        """)
        control_layout.addWidget(self.save_results_btn)
        
        # 另存为按钮
        self.save_as_results_btn = QPushButton("另存为")
        self.save_as_results_btn.setIcon(QIcon(os.path.join("resources", "icons", "save-as.png")))
        self.save_as_results_btn.clicked.connect(self._on_save_as_results_clicked)
        self.save_as_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 3px 10px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 11px;
                min-width: 340px;
                max-width: 340px;
                min-height: 24px;
                text-align: left;
                icon-size: 12px;
                spacing: 4px;
            }
            QPushButton:hover {
                background-color: #f1c40f;
            }
            QPushButton:pressed {
                background-color: #f39c12;
            }
        """)
        control_layout.addWidget(self.save_as_results_btn)

        # 播放速度控制
        speed_layout = QHBoxLayout()
        speed_label = QLabel("播放速度:")
        speed_label.setStyleSheet("font-weight: bold;")
        speed_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "1.5x", "2x", "3x", "5x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setFixedWidth(80)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_combo)
        speed_layout.addStretch(1)
        control_layout.addLayout(speed_layout)
        
        # 交通事件检测控制
        event_control_label = QLabel("交通事件检测")
        event_control_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        event_control_label.setAlignment(Qt.AlignCenter)
        event_control_label.setStyleSheet("color: #2c3e50; margin-top: 2px;")
        control_layout.addWidget(event_control_label)
        
        # 停止线标记按钮
        self.mark_stop_line_btn = QPushButton("标记停止线")
        self.mark_stop_line_btn.clicked.connect(self._on_mark_stop_line_clicked)
        self.mark_stop_line_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 3px 10px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 11px;
                min-width: 340px;
                max-width: 340px;
                min-height: 24px;
                text-align: left;
                icon-size: 12px;
                spacing: 4px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #9b59b6;
            }
        """)
        control_layout.addWidget(self.mark_stop_line_btn)

        # 区域划分用于逆行检测
        region_label = QLabel("区域划分(逆行检测):")
        region_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        control_layout.addWidget(region_label)

        region_btn_layout = QHBoxLayout()
        region_btn_layout.setSpacing(10)
        
        self.mark_region_btn = QPushButton("绘制区域")
        self.mark_region_btn.clicked.connect(self._on_mark_region_clicked)
        self.mark_region_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                padding: 2px 8px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 10px;
                min-width: 130px;
                max-width: 130px;
                min-height: 20px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:pressed {
                background-color: #e67e22;
            }
        """)
        region_btn_layout.addWidget(self.mark_region_btn)

        self.region_direction_combo = QComboBox()
        self.region_direction_combo.addItems(["上→下", "下→上", "左→右", "右→左"])
        self.region_direction_combo.setFixedWidth(100)
        self.region_direction_combo.setStyleSheet("""
            QComboBox {
                padding: 6px 8px;
                font-size: 11px;
                border-radius: 4px;
                border: 1px solid #bdc3c7;
            }
        """)
        region_btn_layout.addWidget(self.region_direction_combo)
        control_layout.addLayout(region_btn_layout)

        self.clear_regions_btn = QPushButton("清除区域")
        self.clear_regions_btn.clicked.connect(self._on_clear_regions_clicked)
        self.clear_regions_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 3px 10px;
                font-weight: bold;
                border-radius: 3px;
                font-size: 11px;
                min-width: 340px;
                max-width: 340px;
                min-height: 24px;
                text-align: left;
                icon-size: 12px;
                spacing: 4px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #95a5a6;
            }
        """)
        control_layout.addWidget(self.clear_regions_btn)
        
        # 信号灯状态设置
        signal_layout = QHBoxLayout()
        signal_label = QLabel("信号灯状态:")
        signal_layout.addWidget(signal_label)
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["绿灯", "黄灯", "红灯"])
        self.signal_combo.setFixedWidth(80)
        self.signal_combo.currentTextChanged.connect(self._on_signal_status_changed)
        signal_layout.addWidget(self.signal_combo)
        signal_layout.addStretch(1)
        control_layout.addLayout(signal_layout)
        
        # 车道方向设置
        lane_layout = QHBoxLayout()
        lane_label = QLabel("车道方向:")
        lane_layout.addWidget(lane_label)
        self.lane_combo = QComboBox()
        self.lane_combo.addItems(["上→下", "下→上", "左→右", "右→左"])
        self.lane_combo.setFixedWidth(100)
        self.lane_combo.currentTextChanged.connect(self._on_lane_direction_changed)
        lane_layout.addWidget(self.lane_combo)
        lane_layout.addStretch(1)
        control_layout.addLayout(lane_layout)

        # 需要模型才能使用的按钮集合（选择模型始终可用）
        self._model_dependent_buttons = [
            self.select_image_btn,
            self.select_image_list_btn,
            self.select_video_btn,
            self.save_results_btn,
            self.save_as_results_btn,
        ]

        left_panel_layout.addWidget(control_frame)

        # ===== 统计信息区域 =====
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(8)

        # 目标数统计标题
        stats_title = QLabel("目标数据:")
        stats_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        stats_title.setAlignment(Qt.AlignCenter)
        stats_title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        stats_layout.addWidget(stats_title)

        # 目标数量网格布局
        target_grid = QGridLayout()
        target_grid.setVerticalSpacing(8)
        target_grid.setHorizontalSpacing(12)

        # 各类目标数量标签
        target_types = ["行人", "小汽车", "两轮车", "公交车", "卡车"]
        target_colors = ["#9b59b6", "#3498db", "#e74c3c", "#1abc9c", "#f1c40f"]

        # 创建目标类型标签和计数标签
        self.target_count_labels = {}

        for i, (target_type, color) in enumerate(zip(target_types, target_colors)):
            # 类别标签
            type_label = QLabel(target_type)
            type_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            target_grid.addWidget(type_label, i, 0)

            # 计数标签
            count_label = QLabel("0")
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            target_grid.addWidget(count_label, i, 1)

            # 进度条
            progress_bar = QProgressBar()
            progress_bar.setMaximum(100)
            progress_bar.setValue(0)
            progress_bar.setFixedHeight(12)
            progress_bar.setTextVisible(False)
            progress_bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
            target_grid.addWidget(progress_bar, i, 2)

            # 保存引用
            self.target_count_labels[target_type] = (count_label, progress_bar)

        stats_layout.addLayout(target_grid)
        left_panel_layout.addWidget(stats_frame)

        # ===== 配置信息区域 =====
        conf_frame = QFrame()
        conf_frame.setFrameShape(QFrame.StyledPanel)
        conf_layout = QVBoxLayout(conf_frame)
        conf_layout.setContentsMargins(10, 10, 10, 10)
        conf_layout.setSpacing(8)

        # 模型参数行
        conf_row = QHBoxLayout()

        # CONF标签
        self.conf_value_label = QLabel(f"CONF: {self.conf_value:.2f}")
        self.conf_value_label.setFont(QFont("Microsoft YaHei", 10))
        conf_row.addWidget(self.conf_value_label)

        # CONF进度条
        self.conf_progress = QProgressBar()
        self.conf_progress.setMinimum(0)
        self.conf_progress.setMaximum(100)
        self.conf_progress.setValue(int(self.conf_value * 100))
        self.conf_progress.setFixedHeight(12)
        self.conf_progress.setTextVisible(False)
        self.conf_progress.valueChanged.connect(self._on_conf_changed)
        # 启用鼠标拖动功能
        self.conf_progress.setMouseTracking(True)
        self.conf_progress.mousePressEvent = self._on_conf_progress_click
        conf_row.addWidget(self.conf_progress)

        conf_layout.addLayout(conf_row)

        # IOU行
        iou_row = QHBoxLayout()

        # IOU标签
        self.iou_value_label = QLabel(f"IOU: {self.iou_value:.2f}")
        self.iou_value_label.setFont(QFont("Microsoft YaHei", 10))
        iou_row.addWidget(self.iou_value_label)

        # IOU进度条
        self.iou_progress = QProgressBar()
        self.iou_progress.setMinimum(0)
        self.iou_progress.setMaximum(100)
        self.iou_progress.setValue(int(self.iou_value * 100))
        self.iou_progress.setFixedHeight(12)
        self.iou_progress.setTextVisible(False)
        self.iou_progress.valueChanged.connect(self._on_iou_changed)
        # 启用鼠标拖动功能
        self.iou_progress.setMouseTracking(True)
        self.iou_progress.mousePressEvent = self._on_iou_progress_click
        iou_row.addWidget(self.iou_progress)

        conf_layout.addLayout(iou_row)

        # 用时信息
        time_row = QHBoxLayout()

        # 时钟图标
        time_icon = QLabel("⏱")
        time_icon.setFont(QFont("Arial", 12))
        time_row.addWidget(time_icon)

        # 用时标签
        time_label = QLabel("用时:")
        time_label.setFont(QFont("Microsoft YaHei", 10))
        time_row.addWidget(time_label)

        # 用时值
        self.time_value_label = QLabel("0 s")
        self.time_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        time_row.addWidget(self.time_value_label)

        conf_layout.addLayout(time_row)

        # 目标数目
        target_row = QHBoxLayout()

        # 目标图标
        target_icon = QLabel("🎯")
        target_icon.setFont(QFont("Arial", 12))
        target_row.addWidget(target_icon)

        # 目标标签
        target_label = QLabel("目标数目:")
        target_label.setFont(QFont("Microsoft YaHei", 10))
        target_row.addWidget(target_label)

        # 目标值
        self.target_value_label = QLabel("0")
        self.target_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        target_row.addWidget(self.target_value_label)

        conf_layout.addLayout(target_row)

        left_panel_layout.addWidget(conf_frame)

        # 让统计信息区域占用剩余空间
        left_panel_layout.addStretch(1)

        # ===== 右侧显示区域 =====
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # 主显示区域
        display_frame = QFrame()
        display_frame.setFrameShape(QFrame.StyledPanel)
        display_frame.setStyleSheet("background-color: #282c34;")  # 暗色背景
        display_layout = QVBoxLayout(display_frame)

        # 显示标签 - 使用自定义的视频显示控件
        self.display_label = VideoDisplayWidget()
        
        # 连接鼠标点击信号
        self.display_label.mouse_clicked.connect(self._on_video_display_clicked)

        # 默认显示占位图
        placeholder_path = os.path.join("resources", "images", "placeholder.png")
        if os.path.exists(placeholder_path):
            placeholder_pixmap = QPixmap(placeholder_path)
            self.display_label.setPixmap(placeholder_pixmap.scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.display_label.setText("请选择图像或视频文件进行检测")
            self.display_label.setStyleSheet("color: white; font-size: 16px;")

        display_layout.addWidget(self.display_label)

        # 添加图片导航控制按钮
        nav_layout = QHBoxLayout()

        # 上一张按钮
        self.prev_image_btn = QPushButton("上一张")
        self.prev_image_btn.setIcon(QIcon(os.path.join("resources", "icons", "previous.png")))
        self.prev_image_btn.clicked.connect(self._on_prev_image_clicked)
        self.prev_image_btn.setEnabled(False)
        self.prev_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        nav_layout.addWidget(self.prev_image_btn)

        # 图片索引指示器
        self.image_index_label = QLabel("0/0")
        self.image_index_label.setAlignment(Qt.AlignCenter)
        self.image_index_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        self.image_index_label.setMinimumWidth(100)
        nav_layout.addWidget(self.image_index_label)

        # 下一张按钮
        self.next_image_btn = QPushButton("下一张")
        self.next_image_btn.setIcon(QIcon(os.path.join("resources", "icons", "next.png")))
        self.next_image_btn.clicked.connect(self._on_next_image_clicked)
        self.next_image_btn.setEnabled(False)
        self.next_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        nav_layout.addWidget(self.next_image_btn)

        display_layout.addLayout(nav_layout)
        right_layout.addWidget(display_frame, 7)  # 分配70%的空间

        # 检测结果表格
        table_frame = QFrame()
        table_frame.setFrameShape(QFrame.StyledPanel)
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(5, 5, 5, 5)

        # Tab：检测结果 / 事件告警 / 研究数据
        tabs = QTabWidget()

        # 1) 检测结果
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(0, 0, 0, 0)

        self.result_table = QTableWidget(0, 5)  # 序号, 画面标识, 结果, 位置, 置信度
        self.result_table.setHorizontalHeaderLabels(["序号", "画面标识", "结果", "位置", "置信度"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 不可编辑
        self.result_table.setAlternatingRowColors(True)  # 交替行颜色
        self.result_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
            }
        """)
        result_layout.addWidget(self.result_table)

        # 2) 事件告警
        event_tab = QWidget()
        event_layout = QVBoxLayout(event_tab)
        event_layout.setContentsMargins(0, 0, 0, 0)

        self.event_table = QTableWidget(0, 6)  # 帧号, 时间, 类型, 轨迹ID, 类别, 数值
        self.event_table.setHorizontalHeaderLabels(["帧号", "时间(s)", "事件类型", "轨迹ID", "类别", "数值"])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
            }
        """)
        event_layout.addWidget(self.event_table)

        # 3) 研究数据
        research_tab = QWidget()
        research_layout = QVBoxLayout(research_tab)
        research_layout.setContentsMargins(0, 0, 0, 0)

        # 添加研究数据统计标签
        stats_label = QLabel("目标特征统计")
        stats_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        stats_label.setAlignment(Qt.AlignCenter)
        stats_label.setStyleSheet("color: #2c3e50; margin: 5px 0;")
        research_layout.addWidget(stats_label)

        # 添加特征统计表格
        self.features_table = QTableWidget(0, 3)  # 特征类型, 特征值, 数量
        self.features_table.setHorizontalHeaderLabels(["特征类型", "特征值", "数量"])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.features_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
            }
        """)
        research_layout.addWidget(self.features_table)

        # 添加轨迹分析标签
        trajectory_label = QLabel("轨迹分析")
        trajectory_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        trajectory_label.setAlignment(Qt.AlignCenter)
        trajectory_label.setStyleSheet("color: #2c3e50; margin: 5px 0;")
        research_layout.addWidget(trajectory_label)

        # 添加轨迹统计表格（6列：选择、轨迹ID、类别、轨迹长度、最大速度、时间范围）
        self.trajectory_table = QTableWidget(0, 6)
        self.trajectory_table.setHorizontalHeaderLabels(["选择", "轨迹ID", "类别", "轨迹长度", "最大速度(px/s)", "时间范围(s)"])
        self.trajectory_table.setColumnWidth(0, 50)  # 选择列宽度
        self.trajectory_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.trajectory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.trajectory_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.trajectory_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.trajectory_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.trajectory_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trajectory_table.setAlternatingRowColors(True)
        self.trajectory_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
            }
        """)
        self.trajectory_table.cellClicked.connect(self._on_trajectory_cell_clicked)
        research_layout.addWidget(self.trajectory_table)

        # 添加轨迹显示控制按钮
        trajectory_btn_layout = QHBoxLayout()
        
        # 轨迹显示开关
        self.track_visible_checkbox = QCheckBox("显示轨迹")
        self.track_visible_checkbox.setChecked(False)
        self.track_visible_checkbox.stateChanged.connect(self._on_track_visible_changed)
        self.track_visible_checkbox.setStyleSheet("font-size: 11px;")
        trajectory_btn_layout.addWidget(self.track_visible_checkbox)
        
        self.show_selected_track_btn = QPushButton("显示选中轨迹")
        self.show_selected_track_btn.setStyleSheet("""
            background-color: #27ae60;
            color: white;
            border: none;
            padding: 2px 8px;
            font-weight: bold;
            border-radius: 3px;
            font-size: 10px;
            min-height: 20px;
        """)
        self.show_selected_track_btn.clicked.connect(self._on_show_selected_track_clicked)
        self.show_all_tracks_btn = QPushButton("显示全部轨迹")
        self.show_all_tracks_btn.setStyleSheet("""
            background-color: #3498db;
            color: white;
            border: none;
            padding: 2px 8px;
            font-weight: bold;
            border-radius: 3px;
            font-size: 10px;
            min-height: 20px;
        """)
        self.show_all_tracks_btn.clicked.connect(self._on_show_all_tracks_clicked)
        self.clear_track_selection_btn = QPushButton("清除选择")
        self.clear_track_selection_btn.setStyleSheet("""
            background-color: #95a5a6;
            color: white;
            border: none;
            padding: 2px 8px;
            font-weight: bold;
            border-radius: 3px;
            font-size: 10px;
            min-height: 20px;
        """)
        self.clear_track_selection_btn.clicked.connect(self._on_clear_track_selection_clicked)
        trajectory_btn_layout.addWidget(self.show_selected_track_btn)
        trajectory_btn_layout.addWidget(self.show_all_tracks_btn)
        trajectory_btn_layout.addWidget(self.clear_track_selection_btn)
        research_layout.addLayout(trajectory_btn_layout)

        tabs.addTab(result_tab, "检测结果")
        tabs.addTab(event_tab, "事件告警")
        tabs.addTab(research_tab, "研究数据")  # 添加研究数据标签页
        table_layout.addWidget(tabs)
        right_layout.addWidget(table_frame, 3)  # 分配30%的空间

        # 将左右面板添加到主布局
        main_layout.addWidget(left_panel, 2)  # 左侧占比约40%
        main_layout.addWidget(right_panel, 3)  # 右侧占比约60%

    def _on_select_image_clicked(self):
        """选择图片按钮点击处理"""
        if self.detector is None:
            QMessageBox.warning(self, "提示", "请先点击“选择模型文件”加载YOLO模型。")
            return
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if not file_path:
            return

        try:
            # 记录开始时间
            start_time = time.time()

            # 检测图片，使用当前设置的置信度和IOU阈值
            original_img = cv2.imread(file_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # 执行检测，使用当前设置的置信度和IOU阈值
            results = self.detector.model(original_img, conf=self.conf_value, iou=self.iou_value)[0]

            # 统计各类别数量
            class_counts = self.detector._count_objects(results)

            # 在图像上绘制边界框和标签
            processed_img = self.detector._draw_boxes(original_img.copy(), results)

            # 计算用时
            elapsed_time = time.time() - start_time

            # 更新用时显示
            self.time_value_label.setText(f"{elapsed_time:.2f} s")

            # 更新目标数目
            total_objects = sum(class_counts.values())
            self.target_value_label.setText(str(total_objects))

            # 更新统计数据
            for target_type, (count_label, progress_bar) in self.target_count_labels.items():
                count_label.setText(str(class_counts.get(target_type, 0)))
                progress_bar.setValue(0)

            # 更新特征统计信息
            self._update_features_statistics(class_counts)
            
            # 显示处理后的图片
            h, w = processed_img.shape[:2]
            # 计算合适的显示尺寸，保持纵横比
            display_w = self.display_label.width()
            display_h = int(display_w * h / w)

            pixmap = cv2_to_qpixmap(processed_img, (display_w, display_h))
            self.display_label.setPixmap(pixmap)

            # 更新检测结果表格
            self._update_result_table(file_path, processed_img)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片时出错: {str(e)}")

    def _on_select_video_clicked(self):
        """选择视频按钮点击处理"""
        if self.detector is None:
            QMessageBox.warning(self, "提示", "请先点击“选择模型文件”加载YOLO模型。")
            return
        # 如果已经在处理视频，则停止
        if self.is_processing_video:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait()

            self.is_processing_video = False
            self.select_video_btn.setText("选择视频文件")
            self.select_video_btn.setStyleSheet(self.base_button_style)
            # 缓存最后一次统计（用于保存）
            if self.video_thread:
                self.video_track_stats = getattr(self.video_thread, "track_stats", {}) or {}
            return

        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        try:
            # 停止所有正在运行的播放任务（包括轨迹回放）
            self._stop_all_playback()
            
            # 清空本次视频的事件与统计
            self.video_events = []
            self.video_track_stats = {}
            if hasattr(self, "event_table"):
                self.event_table.setRowCount(0)

            # 创建并启动视频处理线程，传递播放速度参数
            self.video_thread = VideoProcessThread(self.detector, file_path, playback_speed=self.playback_speed)

            # 设置交通事件检测参数
            if self.current_stop_line:
                self.video_thread.set_stop_line(self.current_stop_line)
            self.video_thread.set_signal_status(self.signal_combo.currentText())
            self.video_thread.set_lane_direction(self.lane_combo.currentText())
            # 同步区域划分数据
            if self.lane_regions:
                self.video_thread.lane_regions = list(self.lane_regions)
            
            # 连接信号槽
            self.video_thread.frame_ready.connect(self._update_video_frame)
            self.video_thread.events_ready.connect(self._on_events_ready)
            self.video_thread.video_finished.connect(self._on_video_finished)
            
            # 启动线程
            self.video_thread.start()

            # 更新状态
            self.is_processing_video = True
            self.select_video_btn.setText("停止视频")
            self.select_video_btn.setStyleSheet("""
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 12px;
                font-weight: bold;
                border-radius: 4px;
                font-size: 12px;
                min-width: 150px;
            """)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理视频时出错: {str(e)}")

    def _update_video_frame(self, processed_frame, class_counts):
        """更新视频帧
        
        Args:
            processed_frame: 处理后的帧
            class_counts: 各类别数量统计
        """
        # 更新统计数据
        for target_type, (count_label, progress_bar) in self.target_count_labels.items():
            count_label.setText(str(class_counts.get(target_type, 0)))
            progress_bar.setValue(0)

        # 更新目标数目
        total_objects = sum(class_counts.values())
        self.target_value_label.setText(str(total_objects))

        # 更新特征统计信息
        self._update_features_statistics(class_counts)
        
        # 更新轨迹统计信息
        if self.is_processing_video and self.video_thread:
            self._update_trajectory_statistics(self.video_thread.tracks, self.video_thread.track_stats)

        # 显示处理后的帧
        h, w = processed_frame.shape[:2]
        
        # 设置视频尺寸到显示控件
        self.display_label.set_video_size(w, h)
        
        # 记录当前帧的实际尺寸（用于绘制坐标映射）
        self.current_frame_size = (w, h)
        
        # 使用显示标签的实际尺寸，让图片填满整个区域
        display_w = self.display_label.width()
        display_h = self.display_label.height()

        pixmap = cv2_to_qpixmap(processed_frame, (display_w, display_h))
        self.display_label.setPixmap(pixmap)

        # 更新检测结果表格，传递轨迹信息
        if self.is_processing_video and self.video_thread:
            self._update_result_table("视频帧", processed_frame, self.video_thread.tracks,
                                      self.video_thread.previous_boxes)
        else:
            self._update_result_table("视频帧", processed_frame)

    def _on_speed_changed(self, speed_text):
        """播放速度变更处理"""
        try:
            # 从字符串中提取速度值，如 "1.5x" -> 1.5
            speed = float(speed_text.replace("x", ""))
            self.playback_speed = speed
            
            # 如果当前正在播放视频，需要重启视频线程以应用新速度
            if self.is_processing_video and self.video_thread and self.video_thread.isRunning():
                # 停止当前视频线程
                self.video_thread.stop()
                self.video_thread.wait()
                
                # 重新启动视频线程，使用新的播放速度
                # 注意：这里需要保存当前视频路径，以便重启
                video_path = self.video_thread.video_path
                self.video_thread = VideoProcessThread(self.detector, video_path, playback_speed=self.playback_speed)
                # 同步区域划分数据
                if self.lane_regions:
                    self.video_thread.lane_regions = list(self.lane_regions)
                self.video_thread.frame_ready.connect(self._update_video_frame)
                self.video_thread.events_ready.connect(self._on_events_ready)
                self.video_thread.video_finished.connect(self._on_video_finished)
                self.video_thread.start()
        except ValueError:
            QMessageBox.warning(self, "错误", "无效的播放速度值")
    
    def _on_video_finished(self):
        """视频处理完成处理"""
        # 更新状态
        self.is_processing_video = False
        self.select_video_btn.setText("选择视频文件")
        self.select_video_btn.setStyleSheet(self.base_button_style)

        # 缓存轨迹统计，便于导出
        if self.video_thread:
            self.video_track_stats = getattr(self.video_thread, "track_stats", {}) or {}

        # 保存轨迹数据用于回放
        if self.video_thread:
            self.replay_tracks = dict(self.video_thread.tracks)
            self.replay_video_path = self.video_thread.video_path
        
        # 处理结束后的运行摘要（输出真实统计，不使用预置数值）
        fps = float(self.video_thread.video_fps or 0.0)
        print("\n[视频处理完成] 运行摘要")
        print(f"  视频文件    : {self.video_thread.video_path}")
        print(f"  分辨率/帧率 : {self.video_thread.video_width}x"
              f"{self.video_thread.video_height} @ {fps:.2f} FPS")
        print(f"  处理帧数    : {self.video_thread.frame_idx}")
        print(f"  检测参数    : conf={self.conf_value}  iou={self.iou_value}")
        print(f"  跟踪算法    : DeepSORT")
        print(f"  信号灯/车道 : {self.signal_status} / {self.lane_direction}")
        if self.current_stop_line:
            print(f"  停止线坐标  : {self.current_stop_line}")
        print(f"  交通事件数  : {len(self.video_events)}")
        print("  说明        : 论文中的精度与性能指标见 performance_output.py")

        QMessageBox.information(self, "处理完成", "视频处理已完成")
    
    def _update_features_statistics(self, class_counts):
        """更新特征统计信息"""
        # 清空之前的统计
        self.features_count = {
            "颜色": {},
            "车型": {},
            "类别": {}
        }
        
        # 更新类别统计
        for cls, count in class_counts.items():
            self.features_count["类别"][cls] = count
        
        # 更新颜色和车型统计（简化实现，实际项目中可以根据检测结果提取）
        # 这里使用模拟数据作为示例
        self.features_count["颜色"] = {
            "黑色": 12,
            "白色": 8,
            "红色": 5,
            "蓝色": 3,
            "银色": 7
        }
        
        self.features_count["车型"] = {
            "小汽车": 25,
            "公交车": 3,
            "卡车": 2,
            "两轮车": 5,
            "行人": 10
        }
        
        # 更新特征统计表格
        self._update_features_table()
    
    def _update_features_table(self):
        """更新特征统计表格"""
        # 清空表格
        self.features_table.setRowCount(0)
        
        # 添加统计数据
        row = 0
        for feature_type, feature_data in self.features_count.items():
            for feature_value, count in feature_data.items():
                self.features_table.insertRow(row)
                self.features_table.setItem(row, 0, QTableWidgetItem(feature_type))
                self.features_table.setItem(row, 1, QTableWidgetItem(feature_value))
                self.features_table.setItem(row, 2, QTableWidgetItem(str(count)))
                row += 1
    
    def _update_trajectory_statistics(self, tracks, track_stats):
        """更新轨迹统计信息"""
        # 清空轨迹统计表格
        self.trajectory_table.setRowCount(0)
        # 清空选中的轨迹ID
        self.selected_track_ids.clear()

        # 添加轨迹统计数据
        for track_id, stats in track_stats.items():
            row = self.trajectory_table.rowCount()
            self.trajectory_table.insertRow(row)

            # 添加复选框
            checkbox = QCheckBox()
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(lambda state, tid=track_id: self._on_track_checkbox_changed(tid, state))
            self.trajectory_table.setCellWidget(row, 0, checkbox)

            # 添加轨迹数据
            self.trajectory_table.setItem(row, 1, QTableWidgetItem(str(track_id)))
            self.trajectory_table.setItem(row, 2, QTableWidgetItem(stats.get("cls", "")))
            self.trajectory_table.setItem(row, 3, QTableWidgetItem(str(len(tracks.get(track_id, [])))))
            self.trajectory_table.setItem(row, 4, QTableWidgetItem(f"{stats.get('v_max', 0):.1f}"))
            
            # 添加时间范围
            track_points = tracks.get(track_id, [])
            if track_points:
                start_time = track_points[0][3] if len(track_points[0]) > 3 else 0
                end_time = track_points[-1][3] if len(track_points[-1]) > 3 else 0
                time_range = f"{start_time:.1f}-{end_time:.1f}"
            else:
                time_range = "-"
            self.trajectory_table.setItem(row, 5, QTableWidgetItem(time_range))

    def _on_track_checkbox_changed(self, track_id, state):
        """轨迹复选框状态变化处理"""
        if state == 2:  # Qt.Checked
            self.selected_track_ids.add(track_id)
        else:
            self.selected_track_ids.discard(track_id)
    
    def _on_track_visible_changed(self, state):
        """轨迹显示开关状态变化处理"""
        # 检查是否是 VideoProcessThread 类型
        if self.video_thread and hasattr(self.video_thread, 'set_track_visible'):
            visible = state == 2  # Qt.Checked
            self.video_thread.set_track_visible(visible)
            
            # 如果关闭轨迹显示，清空过滤列表
            if not visible and hasattr(self.video_thread, 'set_track_filter'):
                self.video_thread.set_track_filter([])

    def _on_trajectory_cell_clicked(self, row, col):
        """轨迹表格单元格点击处理"""
        if col == 0:  # 点击复选框列
            return
        
        # 点击时间范围列时跳转到对应时间点
        if col == 5:  # 时间范围列
            track_id_item = self.trajectory_table.item(row, 1)
            if track_id_item:
                track_id = int(track_id_item.text())
                self._jump_to_track_time(track_id)
            return
        
        # 点击其他列时切换该行的复选框状态
        checkbox = self.trajectory_table.cellWidget(row, 0)
        if checkbox:
            checkbox.setChecked(not checkbox.isChecked())
    
    def _jump_to_track_time(self, track_id):
        """跳转到轨迹对应的时间点播放 - 带清理前面播放的逻辑"""
        if not self.video_thread or not self.video_thread.tracks:
            QMessageBox.warning(self, "提示", "没有可回放的视频数据，请先处理视频")
            return
        
        track_points = self.video_thread.tracks.get(track_id, [])
        if not track_points:
            QMessageBox.warning(self, "提示", "该轨迹没有数据")
            return
        
        # 获取轨迹起始时间
        start_time = track_points[0][3] if len(track_points[0]) > 3 else 0
        
        # 停止所有播放任务（清理前面的播放）
        self._stop_all_playback()
        
        # 重新启动视频线程，从指定时间点开始
        self.is_processing_video = True
        self.video_thread = VideoProcessThread(self.detector, self.video_thread.video_path, 
                                               playback_speed=self.playback_speed)
        
        # 设置起始帧
        self.video_thread.frame_idx = int(start_time * self.video_thread.video_fps)
        
        # 设置交通事件检测参数
        if self.current_stop_line:
            self.video_thread.set_stop_line(self.current_stop_line)
        self.video_thread.set_signal_status(self.signal_combo.currentText())
        self.video_thread.set_lane_direction(self.lane_combo.currentText())
        if self.lane_regions:
            self.video_thread.lane_regions = list(self.lane_regions)
        
        # 连接信号槽
        self.video_thread.frame_ready.connect(self._update_video_frame)
        self.video_thread.events_ready.connect(self._on_events_ready)
        self.video_thread.video_finished.connect(self._on_video_finished)
        
        # 启动线程
        self.video_thread.start()
        
        QMessageBox.information(self, "提示", f"已跳转到轨迹 {track_id} 的起始时间: {start_time:.1f}秒")

    def _on_show_selected_track_clicked(self):
        """显示选中轨迹按钮点击处理 - 从车辆出现帧到消失帧播放并绘制轨迹"""
        print("=== 开始轨迹回放 ===")
        
        if not self.selected_track_ids:
            QMessageBox.warning(self, "提示", "请先在轨迹表格中选择要显示的轨迹")
            return
        print(f"选中的轨迹ID: {self.selected_track_ids}")

        # 检查是否有缓存的轨迹数据
        if not hasattr(self, 'replay_tracks') or not self.replay_tracks:
            QMessageBox.warning(self, "提示", "没有轨迹数据，请先处理视频")
            return
        print(f"缓存的轨迹数量: {len(self.replay_tracks)}")

        # 停止所有播放任务（清理前面的播放）
        self._stop_all_playback()
        print("已停止所有播放任务")

        # 获取选中轨迹的时间范围
        min_frame = float('inf')
        max_frame = -float('inf')
        for track_id in self.selected_track_ids:
            track_points = self.replay_tracks.get(track_id, [])
            if track_points:
                # 获取该轨迹的第一帧和最后一帧
                frames = [p[2] for p in track_points]
                min_frame = min(min_frame, min(frames))
                max_frame = max(max_frame, max(frames))
                print(f"轨迹 {track_id}: 帧数范围 {min(frames)}-{max(frames)}, 点数: {len(track_points)}")
        
        if min_frame == float('inf') or max_frame == -float('inf'):
            QMessageBox.warning(self, "提示", "无法获取轨迹的时间范围")
            return
        
        # 检查帧范围是否有效
        if min_frame > max_frame:
            QMessageBox.warning(self, "提示", "帧范围无效")
            return
        print(f"播放范围: 帧 {min_frame} - {max_frame}")
        
        # 检查视频路径
        if not hasattr(self, 'replay_video_path') or not self.replay_video_path:
            QMessageBox.warning(self, "提示", "没有视频路径")
            return
        print(f"视频路径: {self.replay_video_path}")
        
        # 创建轨迹回放线程（使用已有的轨迹数据，不重新检测和跟踪）
        self.track_replay_thread = TrackReplayThread(
            video_path=self.replay_video_path,
            tracks=self.replay_tracks,
            track_ids=list(self.selected_track_ids),
            start_frame=min_frame,
            end_frame=max_frame,
            playback_speed=self.playback_speed
        )
        
        # 连接信号
        self.track_replay_thread.frame_ready.connect(self._update_video_frame_replay)
        self.track_replay_thread.replay_finished.connect(self._on_track_replay_finished)
        
        # 启动线程
        print("启动轨迹回放线程...")
        self.track_replay_thread.start()
        self.is_processing_video = True
        
        # 不使用阻塞对话框，避免卡死
        print(f"轨迹回放已启动，播放范围: 帧 {min_frame} - {max_frame}")
    
    def _stop_all_playback(self):
        """停止所有播放任务"""
        # 停止轨迹回放线程（优先）
        if hasattr(self, 'track_replay_thread') and self.track_replay_thread and self.track_replay_thread.isRunning():
            self.track_replay_thread.stop()
            self.track_replay_thread.wait()
            self.is_processing_video = False
            self.select_video_btn.setText("选择视频文件")
            self.select_video_btn.setStyleSheet(self.base_button_style)
        
        # 停止视频处理线程
        elif self.is_processing_video and self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            self.is_processing_video = False
            self.select_video_btn.setText("选择视频文件")
            self.select_video_btn.setStyleSheet(self.base_button_style)
        
        # 停止轨迹回放
        if hasattr(self, 'replay_timer') and self.replay_timer:
            self.replay_timer.stop()
            self.replay_timer = None
        if hasattr(self, 'replay_cap') and self.replay_cap:
            self.replay_cap.release()
            self.replay_cap = None
    
    def _update_video_frame_replay(self, frame):
        """更新轨迹回放的视频帧显示"""
        h, w = frame.shape[:2]
        
        # 设置视频尺寸到显示控件
        self.display_label.set_video_size(w, h)
        
        # 使用显示标签的实际尺寸，让图片填满整个区域
        display_w = self.display_label.width()
        display_h = self.display_label.height()
        
        pixmap = cv2_to_qpixmap(frame, (display_w, display_h))
        self.display_label.setPixmap(pixmap)
        # 强制刷新显示
        self.display_label.repaint()
    
    def _on_track_replay_finished(self):
        """轨迹回放完成处理"""
        self.is_processing_video = False
        self.select_video_btn.setText("选择视频文件")
        self.select_video_btn.setStyleSheet(self.base_button_style)

    def _on_show_all_tracks_clicked(self):
        """显示全部轨迹按钮点击处理 - 只在当前帧显示轨迹，不播放视频"""
        self.is_replay_mode = False
        self.selected_track_ids.clear()
        
        if not self.video_thread or not self.video_thread.tracks:
            QMessageBox.warning(self, "提示", "没有轨迹数据，请先处理视频")
            return
        
        # 确保轨迹显示开关已开启
        if not self.track_visible_checkbox.isChecked():
            self.track_visible_checkbox.setChecked(True)
        
        # 取消轨迹过滤，显示所有轨迹
        self.video_thread.set_track_filter([])
        
        # 选中所有轨迹ID
        self.selected_track_ids.update(self.video_thread.tracks.keys())
        
        # 更新表格中的复选框状态
        for row in range(self.trajectory_table.rowCount()):
            checkbox = self.trajectory_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)
        
        QMessageBox.information(self, "提示", f"已设置显示所有 {len(self.selected_track_ids)} 条轨迹")

    def _on_clear_track_selection_clicked(self):
        """清除选择按钮点击处理"""
        self.selected_track_ids.clear()
        self.is_replay_mode = False
        # 取消所有复选框
        for row in range(self.trajectory_table.rowCount()):
            checkbox = self.trajectory_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
        QMessageBox.information(self, "提示", "已清除轨迹选择")

    def _start_trajectory_replay(self):
        """开始轨迹回放"""
        if not self.replay_video_path or not os.path.exists(self.replay_video_path):
            return

        # 打开视频
        self.replay_cap = cv2.VideoCapture(self.replay_video_path)
        if not self.replay_cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件进行回放")
            return

        # 设置回放定时器
        self.replay_timer = QTimer()
        self.replay_timer.timeout.connect(self._update_trajectory_replay_frame)
        fps = self.replay_cap.get(cv2.CAP_PROP_FPS)
        self.replay_timer.start(int(1000 / fps))

    def _update_trajectory_replay_frame(self):
        """更新轨迹回放帧"""
        if not hasattr(self, 'replay_cap') or not self.replay_cap:
            return

        ret, frame = self.replay_cap.read()
        if not ret:
            self.replay_cap.release()
            if hasattr(self, 'replay_timer'):
                self.replay_timer.stop()
            QMessageBox.information(self, "回放结束", "轨迹回放已完成")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 绘制轨迹
        self._draw_selected_tracks(frame_rgb)
        # 显示帧
        h, w = frame_rgb.shape[:2]
        qt_img = QPixmap.fromImage(QImage(frame_rgb.data, w, h, w * 3, QImage.Format_RGB888))
        self.display_label.setPixmap(qt_img.scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _on_video_display_clicked(self, x, y):
        """视频显示控件点击处理"""
        # 停止线绘制模式
        if self.is_mark_stop_line:
            self.mark_stop_line_points.append((x, y))
            
            # 显示点击提示
            if len(self.mark_stop_line_points) == 1:
                QMessageBox.information(self, "提示", f"第一个点已标记 ({x}, {y})\n请点击第二个点完成停止线绘制")
            elif len(self.mark_stop_line_points) >= 2:
                self.is_mark_stop_line = False
                self.mark_stop_line_btn.setText("标记停止线")
                self.mark_stop_line_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #9b59b6;
                        color: white;
                        border: none;
                        padding: 3px 10px;
                        font-weight: bold;
                        border-radius: 3px;
                        font-size: 11px;
                        min-width: 340px;
                        max-width: 340px;
                        min-height: 24px;
                        text-align: left;
                        icon-size: 12px;
                    }
                    QPushButton:hover {
                        background-color: #8e44ad;
                    }
                """)
                # 保存停止线
                self.stop_lines.append((self.mark_stop_line_points[0], self.mark_stop_line_points[1]))
                self.current_stop_line = (self.mark_stop_line_points[0], self.mark_stop_line_points[1])
                
                # 在视频控件上绘制线
                self.display_label.add_drawing_line(
                    self.mark_stop_line_points[0][0], 
                    self.mark_stop_line_points[0][1],
                    self.mark_stop_line_points[1][0], 
                    self.mark_stop_line_points[1][1]
                )
                
                QMessageBox.information(self, "提示", f"停止线标记完成！\n起点: {self.mark_stop_line_points[0]}\n终点: {self.mark_stop_line_points[1]}")
                self.mark_stop_line_points = []
        
        # 区域绘制模式
        elif self.is_mark_region:
            self.mark_region_points.append((x, y))
            
            # 显示点击提示
            point_count = len(self.mark_region_points)
            if point_count == 1:
                QMessageBox.information(self, "提示", f"第1个点已标记 ({x}, {y})\n还需点击3次完成矩形绘制")
            elif point_count == 2:
                QMessageBox.information(self, "提示", f"第2个点已标记 ({x}, {y})\n还需点击2次完成矩形绘制")
            elif point_count == 3:
                QMessageBox.information(self, "提示", f"第3个点已标记 ({x}, {y})\n还需点击1次完成矩形绘制")
            elif point_count == 4:
                self.is_mark_region = False
                self.mark_region_btn.setText("绘制区域")
                self.mark_region_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e67e22;
                        color: white;
                        border: none;
                        padding: 2px 8px;
                        font-weight: bold;
                        border-radius: 3px;
                        font-size: 10px;
                        min-width: 130px;
                        max-width: 130px;
                        min-height: 20px;
                        text-align: center;
                    }
                    QPushButton:hover {
                        background-color: #d35400;
                    }
                """)
                
                # 从四个点计算矩形边界
                xs = [p[0] for p in self.mark_region_points]
                ys = [p[1] for p in self.mark_region_points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                # 获取车道方向向量 - 直接从主窗口的lane_combo获取用户选择的方向
                direction_map = {
                    "上→下": (0, 1),
                    "下→上": (0, -1),
                    "左→右": (1, 0),
                    "右→左": (-1, 0)
                }
                # 使用用户当前选择的方向
                direction = direction_map.get(self.lane_combo.currentText(), (0, 1))
                
                # 保存区域
                region_name = f"区域{len(self.lane_regions) + 1}"
                self.lane_regions.append({
                    "rect": (x1, y1, x2, y2),
                    "direction": direction,
                    "name": region_name
                })
                
                # 同步到视频线程
                if self.video_thread:
                    self.video_thread.lane_regions = list(self.lane_regions)
                
                # 在视频控件上绘制矩形
                self.display_label.add_drawing_line(x1, y1, x2, y1)  # 上边
                self.display_label.add_drawing_line(x2, y1, x2, y2)  # 右边
                self.display_label.add_drawing_line(x2, y2, x1, y2)  # 下边
                self.display_label.add_drawing_line(x1, y2, x1, y1)  # 左边
                
                QMessageBox.information(self, "提示", f"{region_name}标记完成，方向: {self.lane_combo.currentText()}")
                self.mark_region_points = []
    
    def _draw_selected_tracks(self, frame):
        """绘制选中的轨迹"""
        if not self.selected_track_ids:
            return

        for track_id in self.selected_track_ids:
            track = self.replay_tracks.get(track_id, [])
            if len(track) < 2:
                continue

            # 获取该车辆的颜色
            if self.video_thread and track_id in self.video_thread.previous_boxes:
                cls_id = self.video_thread.previous_boxes[track_id][4]
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                color = colors[cls_id % len(colors)]
            else:
                color = (255, 0, 0)  # 默认红色

            # 绘制轨迹线
            for i in range(1, len(track)):
                if track[i][2] - track[i - 1][2] <= 5:
                    cv2.line(frame, (track[i - 1][0], track[i - 1][1]),
                             (track[i][0], track[i][1]), color, 3)

                    # 绘制起点和终点
                    if i == 1:
                        cv2.circle(frame, (track[0][0], track[0][1]), 5, (0, 255, 0), -1)
                    if i == len(track) - 1:
                        cv2.circle(frame, (track[i][0], track[i][1]), 5, (0, 0, 255), -1)

            # 在轨迹终点标注ID
            if track:
                last_point = track[-1]
                cv2.putText(frame, f"ID:{track_id}", (last_point[0], last_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _on_signal_status_changed(self, status):
        """信号灯状态变化处理"""
        if self.is_processing_video and self.video_thread:
            self.video_thread.set_signal_status(status)
    
    def _on_lane_direction_changed(self, direction):
        """车道方向变化处理"""
        if self.is_processing_video and self.video_thread:
            self.video_thread.set_lane_direction(direction)
    
    def _on_mark_stop_line_clicked(self):
        """标记停止线按钮点击处理"""
        self.is_mark_stop_line = not self.is_mark_stop_line
        if self.is_mark_stop_line:
            self.mark_stop_line_btn.setText("结束标记停止线")
            self.mark_stop_line_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    font-weight: bold;
                    border-radius: 4px;
                    font-size: 12px;
                    min-width: 340px;
                    max-width: 340px;
                    min-height: 38px;
                    text-align: left;
                    icon-size: 16px;
                }
            """)
            self.mark_stop_line_points = []  # 清空之前的点
            QMessageBox.information(self, "提示", "请在视频画面上点击两次，绘制停止线")
        else:
            self.mark_stop_line_btn.setText("标记停止线")
            self.mark_stop_line_btn.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    font-weight: bold;
                    border-radius: 4px;
                    font-size: 12px;
                    min-width: 340px;
                    max-width: 340px;
                    min-height: 38px;
                    text-align: left;
                    icon-size: 16px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
            """)
            if len(self.mark_stop_line_points) >= 2:
                # 保存停止线
                self.stop_lines.append((self.mark_stop_line_points[0], self.mark_stop_line_points[1]))
                self.current_stop_line = (self.mark_stop_line_points[0], self.mark_stop_line_points[1])
                QMessageBox.information(self, "提示", "停止线标记完成")
            else:
                QMessageBox.warning(self, "提示", "停止线标记未完成，请点击两次绘制停止线")
            self.mark_stop_line_points = []
    
    def mousePressEvent(self, event):
        """鼠标点击事件处理"""
        if self.is_mark_stop_line:
            # 检查点击位置是否在显示标签内
            display_rect = self.display_label.geometry()
            if display_rect.contains(event.pos()):
                # 计算相对位置（相对于显示标签左上角）
                relative_x = event.pos().x() - display_rect.x()
                relative_y = event.pos().y() - display_rect.y()
                
                # 获取显示标签的实际尺寸
                label_width = self.display_label.width()
                label_height = self.display_label.height()
                
                # 获取当前显示的视频实际尺寸
                if hasattr(self, "current_frame_size"):
                    frame_width, frame_height = self.current_frame_size
                elif self.video_thread and hasattr(self.video_thread, 'video_width'):
                    frame_width = self.video_thread.video_width
                    frame_height = self.video_thread.video_height
                else:
                    # 使用默认尺寸
                    frame_width, frame_height = 1920, 1080
                
                # 直接映射坐标（图片已填满整个显示区域）
                actual_x = int(relative_x * frame_width / label_width)
                actual_y = int(relative_y * frame_height / label_height)
                
                # 确保坐标在有效范围内
                actual_x = max(0, min(actual_x, frame_width - 1))
                actual_y = max(0, min(actual_y, frame_height - 1))
                
                # 添加到点列表
                self.mark_stop_line_points.append((actual_x, actual_y))

                # 显示点击提示
                if len(self.mark_stop_line_points) == 1:
                    QMessageBox.information(self, "提示", f"第一个点已标记 ({actual_x}, {actual_y})\n请点击第二个点完成停止线绘制")
                elif len(self.mark_stop_line_points) >= 2:
                    self.is_mark_stop_line = False
                    self.mark_stop_line_btn.setText("标记停止线")
                    self.mark_stop_line_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #9b59b6;
                            color: white;
                            border: none;
                            padding: 3px 10px;
                            font-weight: bold;
                            border-radius: 3px;
                            font-size: 11px;
                            min-width: 340px;
                            max-width: 340px;
                            min-height: 24px;
                            text-align: left;
                            icon-size: 12px;
                        }
                        QPushButton:hover {
                            background-color: #8e44ad;
                        }
                    """)
                    # 保存停止线
                    self.stop_lines.append((self.mark_stop_line_points[0], self.mark_stop_line_points[1]))
                    self.current_stop_line = (self.mark_stop_line_points[0], self.mark_stop_line_points[1])
                    QMessageBox.information(self, "提示", f"停止线标记完成！\n起点: {self.mark_stop_line_points[0]}\n终点: {self.mark_stop_line_points[1]}")
                    self.mark_stop_line_points = []

        elif self.is_mark_region:
            # 区域标记模式
            display_rect = self.display_label.geometry()
            if display_rect.contains(event.pos()):
                # 计算相对位置（相对于显示标签左上角）
                relative_x = event.pos().x() - display_rect.x()
                relative_y = event.pos().y() - display_rect.y()

                # 获取显示标签的实际尺寸
                label_width = self.display_label.width()
                label_height = self.display_label.height()

                # 获取当前显示的视频实际尺寸
                if hasattr(self, "current_frame_size"):
                    frame_width, frame_height = self.current_frame_size
                elif self.video_thread and hasattr(self.video_thread, 'video_width'):
                    frame_width = self.video_thread.video_width
                    frame_height = self.video_thread.video_height
                else:
                    frame_width, frame_height = 1920, 1080

                # 直接映射坐标（图片已填满整个显示区域）
                actual_x = int(relative_x * frame_width / label_width)
                actual_y = int(relative_y * frame_height / label_height)
                
                # 确保坐标在有效范围内
                actual_x = max(0, min(actual_x, frame_width - 1))
                actual_y = max(0, min(actual_y, frame_height - 1))

                # 添加到点列表
                self.mark_region_points.append((actual_x, actual_y))

                # 显示点击提示
                point_count = len(self.mark_region_points)
                if point_count == 1:
                    QMessageBox.information(self, "提示", f"第1个点已标记 ({actual_x}, {actual_y})\n还需点击3次完成矩形绘制")
                elif point_count == 2:
                    QMessageBox.information(self, "提示", f"第2个点已标记 ({actual_x}, {actual_y})\n还需点击2次完成矩形绘制")
                elif point_count == 3:
                    QMessageBox.information(self, "提示", f"第3个点已标记 ({actual_x}, {actual_y})\n还需点击1次完成矩形绘制")
                elif point_count == 4:
                    QMessageBox.information(self, "提示", f"第4个点已标记 ({actual_x}, {actual_y})\n矩形区域绘制完成！")

                # 如果已经有四个点（矩形需要四个顶点），自动结束标记
                if len(self.mark_region_points) >= 4:
                    self.is_mark_region = False
                    self.mark_region_btn.setText("绘制区域")
                    self.mark_region_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #e67e22;
                            color: white;
                            border: none;
                            padding: 2px 8px;
                            font-weight: bold;
                            border-radius: 3px;
                            font-size: 10px;
                            min-width: 130px;
                            max-width: 130px;
                            min-height: 20px;
                            text-align: center;
                        }
                        QPushButton:hover {
                            background-color: #d35400;
                        }
                    """)

                    # 获取当前选择的方向
                    direction_text = self.region_direction_combo.currentText()
                    direction_map = {
                        "上→下": (0, 1),
                        "下→上": (0, -1),
                        "左→右": (1, 0),
                        "右→左": (-1, 0)
                    }
                    direction = direction_map.get(direction_text, (0, 1))

                    # 从四个点计算矩形边界
                    xs = [p[0] for p in self.mark_region_points]
                    ys = [p[1] for p in self.mark_region_points]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)

                    # 保存区域
                    region_name = f"区域{len(self.lane_regions) + 1}"
                    self.lane_regions.append({
                        "rect": (x1, y1, x2, y2),
                        "direction": direction,
                        "name": region_name
                    })

                    # 同步到视频线程
                    if self.video_thread:
                        self.video_thread.lane_regions = list(self.lane_regions)

                    QMessageBox.information(self, "提示", f"{region_name}标记完成，方向: {direction_text}")
                    self.mark_region_points = []

    def _on_mark_region_clicked(self):
        """绘制区域按钮点击处理"""
        self.is_mark_region = not self.is_mark_region
        if self.is_mark_region:
            self.mark_region_btn.setText("结束绘制")
            self.mark_region_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 2px 8px;
                    font-weight: bold;
                    border-radius: 3px;
                    font-size: 10px;
                    min-width: 130px;
                    max-width: 130px;
                    min-height: 20px;
                    text-align: center;
                }
            """)
            self.mark_region_points = []  # 清空之前的点
            QMessageBox.information(self, "提示", "请在视频画面上点击四次，绘制矩形区域的四个顶点")
        else:
            self.mark_region_btn.setText("绘制区域")
            self.mark_region_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e67e22;
                    color: white;
                    border: none;
                    padding: 2px 8px;
                    font-weight: bold;
                    border-radius: 3px;
                    font-size: 10px;
                    min-width: 130px;
                    max-width: 130px;
                    min-height: 20px;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: #d35400;
                }
            """)
            if len(self.mark_region_points) < 4:
                QMessageBox.information(self, "提示", "区域标记未完成，请点击四次绘制矩形区域")
            self.mark_region_points = []

    def _on_clear_regions_clicked(self):
        """清除区域按钮点击处理"""
        self.lane_regions = []
        # 同步到视频线程
        if self.video_thread:
            self.video_thread.lane_regions = []
        QMessageBox.information(self, "提示", "已清除所有区域划分")
    
    def _get_signal_status(self):
        """获取当前信号灯状态"""
        return self.signal_combo.currentText()
    
    def _get_lane_direction(self):
        """获取当前车道方向"""
        return self.lane_combo.currentText()
    
    def _convert_lane_direction_to_vector(self, direction):
        """将车道方向转换为向量"""
        direction_map = {
            "上→下": (0, 1),
            "下→上": (0, -1),
            "左→右": (1, 0),
            "右→左": (-1, 0)
        }
        return direction_map.get(direction, (0, 1))

    def _on_events_ready(self, events: list):
        """接收并展示本帧事件"""
        if not events:
            return
        self.video_events.extend(events)

        # 最多保留最近500条，避免表格无限增长
        max_keep = 500
        if len(self.video_events) > max_keep:
            self.video_events = self.video_events[-max_keep:]
            # 简单起见：重建表格
            self.event_table.setRowCount(0)
            for ev in self.video_events:
                self._append_event_row(ev)
            return

        for ev in events:
            self._append_event_row(ev)

    def _append_event_row(self, ev: dict):
        """往事件表追加一行"""
        try:
            row = self.event_table.rowCount()
            self.event_table.insertRow(row)

            frame_idx = int(ev.get("frame", -1))
            fps = 30.0
            if self.video_thread and getattr(self.video_thread, "video_fps", 0):
                fps = float(self.video_thread.video_fps) or 30.0
            t_sec = frame_idx / fps if frame_idx >= 0 else 0.0

            ev_type = ev.get("type", "")
            if ev_type == "speeding_suspect":
                ev_type_cn = "疑似超速"
                value_text = f"vN={ev.get('vnorm', 0):.2f}  v={ev.get('v_px_s', 0):.0f}px/s"
            elif ev_type == "red_light":
                ev_type_cn = "闯红灯"
                point = ev.get('point', (0, 0))
                value_text = f"位置: ({point[0]}, {point[1]})"
            elif ev_type == "reverse_driving":
                ev_type_cn = "逆行"
                point = ev.get('point', (0, 0))
                value_text = f"位置: ({point[0]}, {point[1]})"
            else:
                ev_type_cn = str(ev_type)
                value_text = ""

            self.event_table.setItem(row, 0, QTableWidgetItem(str(frame_idx)))
            self.event_table.setItem(row, 1, QTableWidgetItem(f"{t_sec:.2f}"))
            self.event_table.setItem(row, 2, QTableWidgetItem(ev_type_cn))
            self.event_table.setItem(row, 3, QTableWidgetItem(str(ev.get("track_id", ""))))
            self.event_table.setItem(row, 4, QTableWidgetItem(str(ev.get("cls", ""))))
            self.event_table.setItem(row, 5, QTableWidgetItem(value_text))
        except Exception as e:
            print(f"追加事件行失败: {e}")

    def _update_result_table(self, source_name, processed_img, tracks=None, previous_boxes=None):
        """更新检测结果表格
        
        Args:
            source_name: 图像来源名称
            processed_img: 处理后的图像
            tracks: 目标轨迹字典，可选
            previous_boxes: 上一帧检测框字典，可选
        """
        # 修改表格列数，增加轨迹ID列
        if self.result_table.columnCount() == 5:
            self.result_table.setColumnCount(6)  # 序号, 轨迹ID, 画面标识, 结果, 位置, 置信度
            self.result_table.setHorizontalHeaderLabels(["序号", "轨迹ID", "画面标识", "结果", "位置", "置信度"])

        # 清空表格
        self.result_table.setRowCount(0)

        # 重新获取检测结果
        try:
            # 将RGB图像转换为BGR，以便与YOLO模型兼容
            img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

            # 执行检测，使用当前设置的置信度和IOU阈值
            results = self.detector.model(img_bgr, conf=self.conf_value, iou=self.iou_value)[0]

            # 获取所有检测框
            row_index = 0

            # 检查是否有轨迹数据
            track_id_map = {}
            if tracks and previous_boxes:
                # 构建中心点到轨迹ID的映射
                for track_id, box_data in previous_boxes.items():
                    x_center, y_center = int(box_data[0]), int(box_data[1])
                    # 使用位置作为键，可能存在多个目标位置相同的情况
                    track_id_map[(x_center, y_center)] = track_id

            for box in results.boxes:
                self.result_table.insertRow(row_index)

                # 获取坐标和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                cls_name = self.detector.class_names[cls_id]

                # 计算中心点（用于匹配轨迹ID）
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # 查找轨迹ID
                track_id = -1
                if track_id_map:
                    # 寻找最近的轨迹点
                    min_distance = float('inf')
                    for pos, tid in track_id_map.items():
                        distance = ((pos[0] - x_center) ** 2 + (pos[1] - y_center) ** 2) ** 0.5
                        if distance < min_distance and distance < 30:  # 设置距离阈值为30像素
                            min_distance = distance
                            track_id = tid

                # 仅处理已映射的类别
                if cls_name in self.detector.CLASS_MAPPING:
                    cn_name = self.detector.CLASS_MAPPING[cls_name]

                    # 序号
                    self.result_table.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))

                    # 轨迹ID
                    track_id_item = QTableWidgetItem(str(track_id) if track_id >= 0 else "无")
                    if track_id >= 0:
                        # 为不同轨迹ID设置不同的背景颜色
                        colors = [QColor(255, 200, 200), QColor(200, 255, 200),
                                  QColor(200, 200, 255), QColor(255, 255, 200),
                                  QColor(255, 200, 255), QColor(200, 255, 255)]
                        track_id_item.setBackground(colors[track_id % len(colors)])
                    self.result_table.setItem(row_index, 1, track_id_item)

                    # 画面标识
                    source_item = QTableWidgetItem(
                        os.path.basename(str(source_name)) if isinstance(source_name, str) else str(source_name))
                    self.result_table.setItem(row_index, 2, source_item)

                    # 结果
                    self.result_table.setItem(row_index, 3, QTableWidgetItem(cn_name))

                    # 位置
                    position = f"({x1},{y1})-({x2},{y2})"
                    self.result_table.setItem(row_index, 4, QTableWidgetItem(position))

                    # 置信度
                    conf_str = f"{conf:.2f}"
                    conf_item = QTableWidgetItem(conf_str)
                    self.result_table.setItem(row_index, 5, conf_item)

                    row_index += 1
        except Exception as e:
            print(f"更新检测结果表格时出错: {str(e)}")

    def _on_save_results_clicked(self):
        """保存检测结果按钮点击处理"""
        # 检查是否有检测结果
        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有检测结果可保存！")
            return

        # 创建保存目录
        save_dir = os.path.join("results", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 保存结果数据到CSV文件
            csv_path = os.path.join(save_dir, "detection_results.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入CSV头部
                headers = []
                for col in range(self.result_table.columnCount()):
                    headers.append(self.result_table.horizontalHeaderItem(col).text())
                f.write(",".join(headers) + "\n")

                # 写入每一行数据
                for row in range(self.result_table.rowCount()):
                    row_data = []
                    for col in range(self.result_table.columnCount()):
                        item = self.result_table.item(row, col)
                        if item is not None:
                            # 确保CSV中的文本字段被引号包围，避免逗号导致的分隔问题
                            text = item.text().replace('"', '""')  # 转义双引号
                            row_data.append(f'"{text}"')
                        else:
                            row_data.append('""')
                    f.write(','.join(row_data) + '\n')

            # ===== 事件与轨迹统计导出（视频模式） =====
            track_stats = self.video_track_stats or {}
            if not track_stats and self.video_thread:
                track_stats = getattr(self.video_thread, "track_stats", {}) or {}

            if self.video_events:
                events_csv = os.path.join(save_dir, "events.csv")
                with open(events_csv, "w", encoding="utf-8") as f:
                    f.write("type,frame,time_s,track_id,cls,vnorm,v_px_s,point_x,point_y\n")
                    fps = 30.0
                    if self.video_thread and getattr(self.video_thread, "video_fps", 0):
                        fps = float(self.video_thread.video_fps) or 30.0
                    for ev in self.video_events:
                        frame_idx = int(ev.get("frame", -1))
                        t_sec = frame_idx / fps if frame_idx >= 0 else 0.0
                        px, py = ev.get("point", ("", ""))
                        f.write(
                            f"{ev.get('type','')},{frame_idx},{t_sec:.3f},{ev.get('track_id','')},"
                            f"{ev.get('cls','')},{ev.get('vnorm','')},{ev.get('v_px_s','')},{px},{py}\n"
                        )

            if track_stats:
                tracks_csv = os.path.join(save_dir, "tracks.csv")
                with open(tracks_csv, "w", encoding="utf-8") as f:
                    f.write("track_id,cls,count,avg_vnorm,max_vnorm,avg_v_px_s,max_v_px_s,speeding\n")
                    for tid, st in track_stats.items():
                        cnt = int(st.get("count", 0) or 0)
                        avg_vn = (float(st.get("vnorm_sum", 0.0)) / cnt) if cnt > 0 else 0.0
                        avg_v = (float(st.get("v_sum", 0.0)) / cnt) if cnt > 0 else 0.0
                        f.write(
                            f"{tid},{st.get('cls','')},{cnt},{avg_vn:.6f},{float(st.get('vnorm_max',0.0)):.6f},"
                            f"{avg_v:.3f},{float(st.get('v_max',0.0)):.3f},{int(bool(st.get('speeding')))}\n"
                        )

            # 如果正在处理视频，询问是否要保存视频
            if self.is_processing_video and self.video_thread and self.video_thread.isRunning():
                reply = QMessageBox.question(
                    self,
                    "保存视频",
                    "是否要保存当前的视频处理结果？\n(这可能需要一些时间)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    # 使用QProgressDialog替代QMessageBox作为进度显示
                    progress_dialog = QProgressDialog("正在准备保存视频...", "取消", 0, 100, self)
                    progress_dialog.setWindowTitle("保存中")
                    progress_dialog.setMinimumDuration(0)
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setValue(10)
                    progress_dialog.setCancelButton(None)  # 禁用取消按钮，防止用户中断

                    # 开始保存帧
                    self.video_thread.start_saving_frames()

                    # 等待几秒钟收集足够的帧
                    seconds_to_wait = 5
                    for i in range(seconds_to_wait):
                        # 更新进度条
                        progress_percent = 10 + int((i + 1) / seconds_to_wait * 40)  # 10-50%
                        progress_dialog.setValue(progress_percent)
                        progress_dialog.setLabelText(f"正在收集视频帧... ({i + 1}/{seconds_to_wait})")

                        QApplication.processEvents()  # 保持UI响应
                        time.sleep(1)

                    # 停止保存帧
                    self.video_thread.stop_saving_frames()

                    # 保存视频
                    progress_dialog.setValue(60)
                    progress_dialog.setLabelText("正在保存视频文件...")
                    QApplication.processEvents()

                    video_path = os.path.join(save_dir, "detection_video.mp4")
                    save_success = self.video_thread.save_video(video_path)

                    # 无论成功与否，都确保进度对话框完成并关闭
                    progress_dialog.setValue(100)
                    progress_dialog.close()

                    if save_success:
                        QMessageBox.information(
                            self,
                            "保存成功",
                            f"检测结果已保存至:\n{save_dir}\n\n包含CSV文件和检测视频。"
                        )
                    else:
                        QMessageBox.warning(
                            self,
                            "视频保存失败",
                            f"视频保存失败，但CSV文件已保存至:\n{save_dir}"
                        )
                else:
                    # 只保存当前显示的图像
                    if self.display_label.pixmap() and not self.display_label.pixmap().isNull():
                        image_path = os.path.join(save_dir, "detection_image.png")
                        self.display_label.pixmap().save(image_path, "PNG")

                    QMessageBox.information(
                        self,
                        "保存成功",
                        f"检测结果已保存至:\n{save_dir}\n\n包含CSV文件和当前帧图像。"
                    )
            else:
                # 保存当前显示的图像（非视频模式）
                if self.display_label.pixmap() and not self.display_label.pixmap().isNull():
                    image_path = os.path.join(save_dir, "detection_image.png")
                    self.display_label.pixmap().save(image_path, "PNG")

                QMessageBox.information(
                    self,
                    "保存成功",
                    f"检测结果已保存至:\n{save_dir}\n\n包含CSV文件和检测图像。"
                )

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存检测结果时出错: {str(e)}")

    def _on_save_as_results_clicked(self):
        """另存为检测结果按钮点击处理"""
        # 检查是否有检测结果
        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有检测结果可保存！")
            return

        # 选择保存目录
        save_dir = QFileDialog.getExistingDirectory(
            self, 
            "选择保存目录", 
            "", 
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not save_dir:
            return  # 用户取消了

        try:
            # 保存结果数据到CSV文件
            csv_path = os.path.join(save_dir, "detection_results.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入CSV头部
                headers = []
                for col in range(self.result_table.columnCount()):
                    headers.append(self.result_table.horizontalHeaderItem(col).text())
                f.write(",".join(headers) + "\n")

                # 写入每一行数据
                for row in range(self.result_table.rowCount()):
                    row_data = []
                    for col in range(self.result_table.columnCount()):
                        item = self.result_table.item(row, col)
                        if item is not None:
                            # 确保CSV中的文本字段被引号包围，避免逗号导致的分隔问题
                            text = item.text().replace('"', '""')  # 转义双引号
                            row_data.append(f'"{text}"')
                        else:
                            row_data.append('""')
                    f.write(','.join(row_data) + '\n')

            # ===== 事件与轨迹统计导出（视频模式） =====
            track_stats = self.video_track_stats or {}
            if not track_stats and self.video_thread:
                track_stats = getattr(self.video_thread, "track_stats", {}) or {}

            if self.video_events:
                events_csv = os.path.join(save_dir, "events.csv")
                with open(events_csv, "w", encoding="utf-8") as f:
                    f.write("type,frame,time_s,track_id,cls,vnorm,v_px_s,point_x,point_y\n")
                    fps = 30.0
                    if self.video_thread and getattr(self.video_thread, "video_fps", 0):
                        fps = float(self.video_thread.video_fps) or 30.0
                    for ev in self.video_events:
                        frame_idx = int(ev.get("frame", -1))
                        t_sec = frame_idx / fps if frame_idx >= 0 else 0.0
                        px, py = ev.get("point", ("", ""))
                        f.write(
                            f"{ev.get('type','')},{frame_idx},{t_sec:.3f},{ev.get('track_id','')},"
                            f"{ev.get('cls','')},{ev.get('vnorm','')},{ev.get('v_px_s','')},{px},{py}\n"
                        )

            if track_stats:
                tracks_csv = os.path.join(save_dir, "tracks.csv")
                with open(tracks_csv, "w", encoding="utf-8") as f:
                    f.write("track_id,cls,count,avg_vnorm,max_vnorm,avg_v_px_s,max_v_px_s,speeding\n")
                    for tid, st in track_stats.items():
                        cnt = int(st.get("count", 0) or 0)
                        avg_vn = (float(st.get("vnorm_sum", 0.0)) / cnt) if cnt > 0 else 0.0
                        avg_v = (float(st.get("v_sum", 0.0)) / cnt) if cnt > 0 else 0.0
                        f.write(
                            f"{tid},{st.get('cls','')},{cnt},{avg_vn:.6f},{float(st.get('vnorm_max',0.0)):.6f},"
                            f"{avg_v:.3f},{float(st.get('v_max',0.0)):.3f},{int(bool(st.get('speeding')))}\n"
                        )

            # 保存当前显示的图像
            if self.display_label.pixmap() and not self.display_label.pixmap().isNull():
                image_path = os.path.join(save_dir, "detection_image.png")
                self.display_label.pixmap().save(image_path, "PNG")

            QMessageBox.information(
                self,
                "保存成功",
                f"检测结果已保存至:\n{save_dir}"
            )

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"另存检测结果时出错: {str(e)}")

    def _on_conf_progress_click(self, event):
        """置信度进度条点击事件处理"""
        if event.button() == Qt.LeftButton:
            # 计算点击位置对应的数值（0-100）
            width = self.conf_progress.width()
            pos = event.position().x() if hasattr(event, 'position') else event.x()
            value = max(0, min(100, int(pos / width * 100)))
            self.conf_progress.setValue(value)

    def _on_iou_progress_click(self, event):
        """IOU进度条点击事件处理"""
        if event.button() == Qt.LeftButton:
            # 计算点击位置对应的数值（0-100）
            width = self.iou_progress.width()
            pos = event.position().x() if hasattr(event, 'position') else event.x()
            value = max(0, min(100, int(pos / width * 100)))
            self.iou_progress.setValue(value)

    def _on_conf_changed(self, value):
        """置信度值变化处理"""
        # 将进度条值（0-100）转换为置信度（0-1）
        self.conf_value = value / 100.0
        # 更新置信度标签
        self.conf_value_label.setText(f"CONF: {self.conf_value:.2f}")
        # 更新检测器的置信度阈值
        if self.detector is not None:
            self.detector.conf_threshold = self.conf_value

    def _on_iou_changed(self, value):
        """IOU值变化处理"""
        # 将进度条值（0-100）转换为IOU（0-1）
        self.iou_value = value / 100.0
        # 更新IOU标签
        self.iou_value_label.setText(f"IOU: {self.iou_value:.2f}")
        # 更新检测器的NMS IOU阈值
        if self.detector is not None and hasattr(self.detector.model, 'iou'):
            self.detector.model.iou = self.iou_value

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 如果正在处理视频，则停止线程
        if self.is_processing_video and self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        event.accept()

    def _on_select_model_clicked(self):
        """选择模型按钮点击处理"""
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型",
            "",
            "模型文件 (*.pt *.pth)"
        )

        if not file_path:
            return

        try:
            # Ultralytics YOLO 推荐使用 runs/train/weights/best.pt 这类 .pt 权重；
            # .pth 常见为 state_dict，ultralytics 可能无法直接加载并报出晦涩错误
            if not file_path.lower().endswith(".pt"):
                QMessageBox.warning(
                    self,
                    "模型格式提示",
                    "建议选择 Ultralytics YOLO 导出的 `.pt` 权重文件（如 `runs/train/weights/best.pt`）。\n"
                    "你当前选择的可能是 `.pth/state_dict`，可能无法加载。\n\n"
                    f"当前文件：{file_path}"
                )
                return

            # 确认是否切换模型
            reply = QMessageBox.question(
                self,
                "确认切换模型",
                f"确定加载模型：{file_path}？\n加载新模型可能需要几秒钟的时间。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # 显示加载进度对话框
                progress_dialog = QProgressDialog("正在加载模型...", "取消", 0, 0, self)
                progress_dialog.setWindowTitle("加载模型")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.show()

                # 创建新的检测器
                QApplication.processEvents()
                self.detector = YOLODetector(model_path=file_path,
                                             conf_threshold=self.conf_value,
                                             iou_threshold=self.iou_value)
                self.model_path = file_path
                self._set_detection_enabled(True)

                # 关闭进度对话框
                progress_dialog.close()

                # 显示成功消息
                QMessageBox.information(
                    self,
                    "模型加载成功",
                    f"模型 {Path(file_path).name} 已成功加载。"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "加载模型失败",
                f"加载YOLO模型失败: {str(e)}\n\n"
                "排查建议：\n"
                "1) 确认你选择的是 Ultralytics 导出的 `.pt` 权重（如 `runs/train/weights/best.pt`）\n"
                "2) 在项目虚拟环境中执行：`python -c \"import ultralytics; print(ultralytics.__version__)\"`\n"
                "3) 如版本较旧/异常：`pip install -U ultralytics` 后重试"
            )

    def _on_select_image_list_clicked(self):
        """选择图片列表按钮点击处理"""
        if self.detector is None:
            QMessageBox.warning(self, "提示", "请先点击“选择模型文件”加载YOLO模型。")
            return
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择包含图片的文件夹",
            ""
        )

        if not folder_path:
            return

        try:
            # 搜索文件夹中的所有图片文件
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
            file_paths = []

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_formats):
                        file_paths.append(os.path.join(root, file))

            # 如果没有找到图片文件
            if not file_paths:
                QMessageBox.warning(
                    self,
                    "未找到图片",
                    f"在选择的文件夹中未找到支持的图片文件。\n支持的格式: {', '.join(supported_formats)}"
                )
                return

            # 创建批量处理进度对话框
            progress_dialog = QProgressDialog("正在处理图片...", "取消", 0, len(file_paths), self)
            progress_dialog.setWindowTitle("批量处理图片")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()

            # 记录开始时间
            start_time = time.time()

            # 清空之前的批量处理结果
            self.batch_results = []
            self.current_batch_index = -1

            # 创建保存结果的目录
            batch_save_dir = os.path.join("results",
                                          f"批量检测_{Path(folder_path).name}_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(batch_save_dir, exist_ok=True)

            # 创建CSV文件记录所有检测结果
            csv_path = os.path.join(batch_save_dir, "检测结果统计.csv")
            with open(csv_path, 'w', encoding='utf-8-sig') as csv_file:
                # 写入CSV头部
                csv_file.write("文件名,行人,小汽车,两轮车,公交车,卡车,总目标数\n")

                # 批量处理图片
                for i, file_path in enumerate(file_paths):
                    # 更新进度
                    progress_dialog.setValue(i)
                    progress_dialog.setLabelText(f"正在处理图片 {i + 1}/{len(file_paths)}: {Path(file_path).name}")
                    QApplication.processEvents()

                    # 检查是否取消
                    if progress_dialog.wasCanceled():
                        break

                    try:
                        # 执行检测
                        original_img = cv2.imread(file_path)
                        if original_img is None:
                            continue  # 跳过无法读取的图片

                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                        # 执行检测，使用当前设置的置信度和IOU阈值
                        results = self.detector.model(original_img, conf=self.conf_value, iou=self.iou_value)[0]

                        # 统计各类别数量
                        class_counts = self.detector._count_objects(results)

                        # 在图像上绘制边界框和标签
                        processed_img = self.detector._draw_boxes(original_img.copy(), results)

                        # 获取相对路径和文件名
                        rel_path = os.path.relpath(file_path, folder_path)
                        file_name = Path(file_path).name

                        # 创建保存子目录结构（如果需要）
                        rel_dir = os.path.dirname(rel_path)
                        if rel_dir:
                            os.makedirs(os.path.join(batch_save_dir, rel_dir), exist_ok=True)

                        # 保存处理后的图片
                        out_path = os.path.join(batch_save_dir, rel_dir, f"{Path(file_name).stem}_检测结果.jpg")
                        cv2.imwrite(out_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

                        # 写入CSV记录
                        total_count = sum(class_counts.values())
                        csv_line = f'"{rel_path}",'
                        csv_line += f'{class_counts.get("行人", 0)},'
                        csv_line += f'{class_counts.get("小汽车", 0)},'
                        csv_line += f'{class_counts.get("两轮车", 0)},'
                        csv_line += f'{class_counts.get("公交车", 0)},'
                        csv_line += f'{class_counts.get("卡车", 0)},'
                        csv_line += f'{total_count}\n'
                        csv_file.write(csv_line)

                        # 保存结果到批量处理列表中
                        self.batch_results.append({
                            'path': file_path,
                            'processed_img': processed_img,
                            'class_counts': class_counts,
                            'output_path': out_path,
                            'rel_path': rel_path
                        })

                    except Exception as e:
                        print(f"处理图片 {file_path} 时出错: {str(e)}")
                        continue  # 跳过处理失败的图片

            # 完成进度
            progress_dialog.setValue(len(file_paths))

            # 计算总用时
            elapsed_time = time.time() - start_time
            self.time_value_label.setText(f"{elapsed_time:.2f} s")

            # 如果有任何处理成功的图片
            successfully_processed = len(self.batch_results)
            if successfully_processed > 0:
                # 更新批量处理索引到第一张图片
                self.current_batch_index = 0
                self._update_batch_image_display()

                # 更新导航按钮状态
                self._update_navigation_buttons()

                # 显示处理成功的消息
                QMessageBox.information(
                    self,
                    "批量处理完成",
                    f"已成功处理 {successfully_processed}/{len(file_paths)} 张图片。\n\n"
                    f"所有检测结果已自动保存至：\n{batch_save_dir}\n\n"
                    f"使用上一张/下一张按钮可以浏览所有处理结果。"
                )
            else:
                QMessageBox.warning(
                    self,
                    "处理失败",
                    "没有成功处理任何图片，请检查图片格式或模型设置。"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量处理图片时出错: {str(e)}")

    def _on_prev_image_clicked(self):
        """上一张图片按钮点击处理"""
        if not self.batch_results or self.current_batch_index <= 0:
            return

        self.current_batch_index -= 1
        self._update_batch_image_display()
        self._update_navigation_buttons()

    def _on_next_image_clicked(self):
        """下一张图片按钮点击处理"""
        if not self.batch_results or self.current_batch_index >= len(self.batch_results) - 1:
            return

        self.current_batch_index += 1
        self._update_batch_image_display()
        self._update_navigation_buttons()

    def _update_batch_image_display(self):
        """更新批量处理图片显示"""
        if not self.batch_results or self.current_batch_index < 0 or self.current_batch_index >= len(
                self.batch_results):
            return

        # 获取当前索引的处理结果
        result = self.batch_results[self.current_batch_index]

        # 更新索引标签
        self.image_index_label.setText(f"{self.current_batch_index + 1}/{len(self.batch_results)}")

        # 更新统计数据
        class_counts = result['class_counts']
        for target_type, (count_label, progress_bar) in self.target_count_labels.items():
            count = class_counts.get(target_type, 0)
            count_label.setText(str(count))

            # 更新进度条（相对于最大值计算百分比）
            max_count = max(max(class_counts.values()), 1)  # 避免除以零
            percentage = int((count / max_count) * 100)
            progress_bar.setValue(percentage)

        # 更新目标数目
        total_objects = sum(class_counts.values())
        self.target_value_label.setText(str(total_objects))

        # 显示处理后的图片
        processed_img = result['processed_img']
        h, w = processed_img.shape[:2]
        # 计算合适的显示尺寸，保持纵横比
        display_w = self.display_label.width()
        display_h = int(display_w * h / w)

        pixmap = cv2_to_qpixmap(processed_img, (display_w, display_h))
        self.display_label.setPixmap(pixmap)

        # 更新检测结果表格
        self._update_result_table(result['rel_path'], processed_img)

    def _update_navigation_buttons(self):
        """更新导航按钮状态"""
        # 启用/禁用上一张按钮
        self.prev_image_btn.setEnabled(self.batch_results and self.current_batch_index > 0)

        # 启用/禁用下一张按钮
        self.next_image_btn.setEnabled(self.batch_results and self.current_batch_index < len(self.batch_results) - 1)
