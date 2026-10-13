#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
from pathlib import Path
try:
    from ultralytics import YOLO
except Exception as e:  # ultralytics 未安装或环境异常
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = e
from PIL import Image


class YOLODetector:
    """YOLOv8目标检测器类"""
    
    # 类别映射（英文到中文）
    CLASS_MAPPING = {
        'Pedestrian': "行人", 
        'Van': '面包车', 
        'Car': '小汽车', 
        'Two-wheeler': '两轮车', 
        'Bus': '公交车', 
        'Truck': '卡车'
    }
    
    # 类别颜色映射（用于显示不同的边界框颜色）
    COLOR_MAPPING = {
        'Pedestrian': (168, 100, 253),  # 紫色
        'Van': (0, 0, 255),             # 红色
        'Car': (254, 138, 0),           # 蓝色 
        'Two-wheeler': (255, 102, 178), # 粉色
        'Bus': (0, 130, 255),           # 橙色
        'Truck': (0, 255, 0)            # 绿色
    }
    
    def __init__(self, model_path=None, conf_threshold=0.4, iou_threshold=0.6):
        """初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值，提高以减少低质量检测框
            iou_threshold: IOU阈值，提高以减少重叠框
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载模型
        if YOLO is None:
            raise ImportError(
                "未能导入 `ultralytics`，请在当前虚拟环境中安装/修复。\n"
                "建议执行：pip install -U ultralytics\n"
                f"导入错误：{_ULTRALYTICS_IMPORT_ERROR}"
            )

        if not model_path:
            raise ValueError("未指定模型路径（model_path=None）")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        if not str(model_path).lower().endswith(".pt"):
            raise ValueError(
                f"模型文件格式可能不兼容：{model_path}\n"
                "建议选择 Ultralytics 导出的 `.pt` 权重（如 runs/train/weights/best.pt）。"
            )

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Ultralytics YOLO 加载失败：{model_path}\n"
                "请确认该文件是 Ultralytics 训练导出的 `.pt` 权重（非 state_dict/.pth）。\n"
                f"原始错误：{e}"
            ) from e
            
        self.class_names = self.model.names
        
    def detect_image(self, image_path):
        """检测图片中的目标
        
        Args:
            image_path: 图片路径
            
        Returns:
            (processed_img, results_dict): (处理后的图片，检测结果)
        """
        # 检查图片是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件 {image_path} 不存在")
            
        # 执行推理
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        # 读取原始图片
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 统计各类别数量
        class_counts = self._count_objects(results)
        
        # 在图像上绘制边界框和标签
        processed_img = self._draw_boxes(original_img, results)
        
        return processed_img, class_counts
    
    def detect_video_frame(self, frame):
        """检测视频帧中的目标
        
        Args:
            frame: 视频帧(RGB格式)
            
        Returns:
            (processed_frame, results_dict): (处理后的帧，检测结果)
        """
        # 执行推理
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        # 统计各类别数量
        class_counts = self._count_objects(results)
        
        # 在图像上绘制边界框和标签
        processed_frame = self._draw_boxes(frame.copy(), results)
        
        return processed_frame, class_counts
    
    def _count_objects(self, results):
        """统计检测到的各类别目标数量
        
        Args:
            results: YOLO检测结果
            
        Returns:
            dict: 各类别数量统计
        """
        # 初始化计数器
        class_counts = {cn: 0 for cn in self.CLASS_MAPPING.values()}
        
        # 统计数量
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = self.class_names[cls_id]
            if cls_name in self.CLASS_MAPPING:
                cn_name = self.CLASS_MAPPING[cls_name]
                class_counts[cn_name] += 1
                
        return class_counts
    
    def _draw_boxes(self, img, results):
        """在图像上绘制检测框和标签
        
        Args:
            img: 原始图片
            results: YOLO检测结果
            
        Returns:
            处理后的图片
        """
        # 复制图像以免修改原图
        img_with_boxes = img.copy()

        H, W = img_with_boxes.shape[:2]
        min_area = 40 * 40  # 增加最小面积，过滤更小的噪声框
        max_area = int(0.2 * W * H)  # 减小最大面积，过滤过大的异常框
        # 移除底部阈值限制，允许检测整个画面
        # bottom_min = int(0.35 * H)

        # 遍历所有检测框（仅绘制边界框，并做位置/尺寸过滤）
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            y_bottom = y2
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h
            conf = float(box.conf.item())  # 获取置信度

            # 过滤低置信度框
            if conf < self.conf_threshold * 0.8:  # 额外的置信度过滤
                continue
            
            # 过滤尺寸异常框，移除底部区域限制，允许检测整个画面
            # if y_bottom < bottom_min:
            #     continue
            if area < min_area or area > max_area:
                continue
            
            # 过滤宽高比异常的框（避免极端瘦长或扁平的框）
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:  # 宽高比限制在0.2-5之间
                continue

            cls_id = int(box.cls.item())
            cls_name = self.class_names[cls_id]

            if cls_name in self.CLASS_MAPPING:
                color = self.COLOR_MAPPING.get(cls_name, (0, 255, 0))
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        return img_with_boxes