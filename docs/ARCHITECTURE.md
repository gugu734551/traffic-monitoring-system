# 系统架构与设计说明

本文档说明系统的分层结构、线程模型、核心数据结构与若干设计取舍。
关于**未开源的核心策略**，见 [README 第七章](../README.md#七关于核心算法未开源说明)。

## 一、分层结构

```text
表现层    ui/           PySide6 窗口、控件、事件绑定、可视化渲染
业务层    utils/        检测器封装、图像工具、事件策略占位
数据层    database/     SQLite 用户系统；CSV 结果导出
支撑      tools/        视频抽帧等离线小工具
```

设计原则是**把「算法」和「工程」解耦**：

- `utils/detector.py` 只负责「给定一帧，返回结构化检测结果并绘制检测框」，
  不关心界面，也不关心事件判定；
- 事件判定被收敛到 `VideoProcessThread._evaluate_traffic_events()` 这一个入口，
  输入是跟踪结果（轨迹 ID、类别、代表点、检测框），输出是事件字典列表。
  这个入口的具体实现即核心策略所在位置，可以整体替换而不触及任何 UI 代码。

## 二、线程模型

界面卡顿是桌面端视觉应用最常见的性能问题。系统把耗时的视频处理放到独立线程，
通过 Qt 信号与主线程通信，主线程只负责渲染与交互。

| 类 | 类型 | 职责 | 对外信号 |
| --- | --- | --- | --- |
| `VideoProcessThread` | `QThread` | 读帧 → YOLO 推理 → DeepSORT 跟踪 → 轨迹绘制 → 事件判定 → 发帧 | `frame_ready`、`events_ready`、`video_finished` |
| `TrackReplayThread` | `QThread` | 按选定轨迹与帧区间回放历史视频 | `frame_ready`、`replay_finished` |
| `VideoDisplayWidget` | `QLabel` | 视频显示与鼠标交互（标记停止线、划车道区域） | `mouse_clicked(x, y)` |
| `MainWindow` | `QMainWindow` | 组装界面、转发参数、维护表格与导出 | — |

关键约定：

- 线程之间**不共享可变图像缓冲**，每帧通过信号传递独立的 `ndarray`，
  避免渲染与推理争用同一块内存；
- 停止播放统一走 `stop()` + `wait()`，保证关闭窗口时线程已退出，
  不会出现「关闭后仍在跑推理」的僵尸线程；
- 参数（置信度 / IOU / 信号灯状态 / 车道方向 / 停止线）采用**运行时下发**：
  界面改动直接写到线程实例的属性上，下一次循环即刻生效，无需重启线程。

## 三、核心数据结构

### 1. 跟踪与轨迹

```python
self.tracks[track_id] = [(x_center, y_center, frame_idx, timestamp), ...]
self.previous_boxes[track_id] = (x_center, y_center, w, h, cls_id, conf, cls_cn)
```

- 轨迹点带帧号与时间戳，既支持按时间回放，也支持导出轨迹级统计；
- 轨迹长度设有上限，超出后**保留首点**（即目标出现帧）再截断，
  这样长视频下内存占用可控，同时不丢失「目标从哪来」的信息。

### 2. 分类统计

```python
self.track_stats[track_id] = {
    "cls": str, "count": int,
    "vnorm_sum": float, "vnorm_max": float,
    "v_sum": float, "v_max": float, "speeding": bool,
}
```

统计容器在目标首次出现时惰性初始化（`_ensure_track_stat`），
使「逐帧累加」的写路径保持简单，也让 `tracks.csv` 的导出字段与内部结构一一对应。

### 3. 事件字典

三类事件共用同一个字典格式，界面事件表与 `events.csv` 都消费这个格式：

```python
{
    "type": "red_light" | "reverse_driving" | "speeding_suspect",
    "frame": int, "track_id": int, "cls": str, "point": (int, int),
    # 仅 speeding_suspect 使用
    "vnorm": float, "v_px_s": float, "box_h": int,
}
```

### 4. 场景配置

```python
self.stop_line    = [(x1, y1), (x2, y2)]        # 由用户在画面上标记
self.signal_status = "红灯" | "黄灯" | "绿灯"     # 由用户选择
self.lane_regions = [{"rect": (...), "direction": (dx, dy), "name": str}, ...]
```

把「场景先验」（停止线在哪、当前什么灯、哪条车道允许往哪个方向走）
交由使用者显式配置，而不是让算法去猜。这是路口高位监控场景下
**用少量人工先验换取判定稳定性**的典型取舍。

## 四、检测与跟踪链路

1. **推理**：直接用 BGR 帧送入 YOLOv8，避免 RGB/BGR 反复转换的额外拷贝；
2. **过滤**：在 `detector._draw_boxes` 与 `_update_tracks_deepsort` 两处做
   面积、宽高比、置信度的多级过滤，抑制远处噪点与异常大框；
3. **跟踪**：把过滤后的检测框整理成 DeepSORT 需要的 `([x, y, w, h], conf, cls)`
   三元组送进跟踪器，只采纳 `is_confirmed()` 的轨迹，避免短暂误检产生的假 ID；
4. **降级**：`deep_sort_realtime` 未安装时自动回退到内置的 IoU 贪心匹配，
   功能降级但不崩溃。

DeepSORT 参数（`max_age` / `n_init` / `max_iou_distance` / `nn_budget`）
针对路口场景做了调优：放宽 `max_age` 以容忍短暂遮挡，提高 `n_init`
以抑制误跟踪，收紧 `max_iou_distance` 以提升匹配精度。

## 五、健壮性设计

| 场景 | 处理方式 |
| --- | --- |
| `ultralytics` 未安装 | `utils/detector.py` 延迟报错并给出安装指引 |
| 模型文件缺失 / 格式不符 | 加载前校验扩展名与存在性，抛出可读的异常信息 |
| 主窗口导入失败 | `main.py` 回退到 `MockMainWindow`，用界面提示替代崩溃 |
| 视频无法打开 | `cv2.VideoCapture` 校验后弹窗提示 |
| 轨迹点数爆炸 | 轨迹长度上限 + 保留首点 |
| 事件重复上报 | 以 `track_id` 为键做违规去重，避免同一目标反复弹告警 |

## 六、可扩展点

- **换检测器**：只要提供 `model`（可调用）与 `CLASS_MAPPING` / `COLOR_MAPPING`，
  即可替换 `YOLODetector`，跟踪与界面无需改动；
- **换跟踪器**：`_update_tracks_deepsort` 是唯一的跟踪适配点，
  换成 ByteTrack / OC-SORT 只需替换该方法内的“检测结果 → 轨迹”转换；
- **换事件策略**：`_evaluate_traffic_events` 是唯一的事件入口，
  保持事件字典契约即可自由替换策略实现；
- **换导出格式**：导出逻辑集中在 `MainWindow._on_save_results_clicked`
  与 `_on_save_as_results_clicked`，改 CSV 为 JSON / Excel 成本很低。
