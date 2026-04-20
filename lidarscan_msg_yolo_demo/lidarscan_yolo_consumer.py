#!/usr/bin/env python3
"""
Subscribe to LidarScan (planar / non-interleaved layout), decode selected
channels to BGR, run Ultralytics YOLO, publish annotated images.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np
import rclpy

try:
    import cv2
except Exception as exc:  # noqa: BLE001 — catch NumPy/OpenCV ABI mismatch
    sys.stderr.write(
        '\nouster_yolo_consumer: importing cv2 failed. This is usually caused by '
        'NumPy 2.x with an OpenCV wheel built for NumPy 1.x.\n'
        'Fix (same venv as this node):\n'
        '  pip install "numpy>=1.23.2,<2.0" --force-reinstall\n'
        '  pip install --force-reinstall "opencv-python-headless>=4.8,<5"\n'
        f'Original error: {exc!r}\n\n'
    )
    raise

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from lidar_msgs.msg import LidarChannel, LidarInfo, LidarScan

# sensor_msgs/msg/PointField.idl numeric types
_POINT_INT8 = 1
_POINT_UINT8 = 2
_POINT_INT16 = 3
_POINT_UINT16 = 4
_POINT_INT32 = 5
_POINT_UINT32 = 6
_POINT_FLOAT32 = 7
_POINT_FLOAT64 = 8

# PointField datatype -> NumPy base dtype character (endianness prepended at use).
_DTYPE_CHAR: Dict[int, str] = {
    _POINT_INT8: 'i1',
    _POINT_UINT8: 'u1',
    _POINT_INT16: 'i2',
    _POINT_UINT16: 'u2',
    _POINT_INT32: 'i4',
    _POINT_UINT32: 'u4',
    _POINT_FLOAT32: 'f4',
    _POINT_FLOAT64: 'f8',
}


def _channel_dtype(datatype: int, is_bigendian: bool) -> np.dtype:
    if datatype not in _DTYPE_CHAR:
        raise ValueError(f'unsupported channel datatype {datatype}')
    endian = '>' if is_bigendian else '<'
    return np.dtype(endian + _DTYPE_CHAR[datatype])


def _extract_channel_2d(
    scan: LidarScan, channel: LidarChannel
) -> np.ndarray:
    """Return an HxW float array for the channel's first sub-element.

    With the planar layout each channel is a contiguous block of
    height * width * count * sizeof(datatype) bytes at LidarChannel.offset, so
    decoding is a single np.frombuffer + reshape — no per-pixel Python loop.
    """
    h, w = int(scan.height), int(scan.width)
    count = max(int(channel.count), 1)
    offset = int(channel.offset)
    dtype = _channel_dtype(int(channel.datatype), bool(scan.is_bigendian))

    block_elems = h * w * count
    block_bytes = block_elems * dtype.itemsize
    if offset < 0 or offset + block_bytes > len(scan.data):
        raise ValueError(
            f'channel "{channel.name}" extends past data[] '
            f'(offset={offset}, needs {block_bytes} B, have {len(scan.data)} B)'
        )

    arr = np.frombuffer(scan.data, dtype=dtype, count=block_elems, offset=offset)
    if count > 1:
        arr = arr.reshape(h, w, count)[..., 0]
    else:
        arr = arr.reshape(h, w)
    return arr.astype(np.float64, copy=False)


def _to_bgr_visual(
    arr: np.ndarray, mode: str, clip_percentile: Tuple[float, float] = (2.0, 98.0)
) -> np.ndarray:
    """Normalize scalar field to uint8 BGR (3-channel) for YOLO and display."""
    flat = arr[np.isfinite(arr)].astype(np.float64)
    if flat.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    lo, hi = np.percentile(flat, clip_percentile)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    if mode == 'range':
        cmap = cv2.COLORMAP_VIRIDIS
    elif mode == 'near_ir':
        cmap = cv2.COLORMAP_INFERNO
    else:
        cmap = cv2.COLORMAP_MAGMA
    color = cv2.applyColorMap(gray, cmap)
    return color


def _bgr_to_ros_image(header: Header, bgr: np.ndarray) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


class YoloLidarConsumer(Node):
    """Runs YOLO on RANGE / NEAR_IR / REFLECTIVITY LidarScan channels."""

    CHANNEL_SPECS = (
        ('range', 'range'),
        ('near_ir', 'near_ir'),
        ('reflectivity', 'reflectivity'),
    )

    def __init__(self) -> None:
        super().__init__('yolo_lidar_consumer')
        self.declare_parameter('model_name', 'yolo11n.pt')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('device', '')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('lidar_info_topic', '/ouster/lidar_info')
        self.declare_parameter('lidar_scan_topic', '/ouster/lidar_scan')

        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self._conf = float(
            self.get_parameter('confidence').get_parameter_value().double_value
        )
        self._device = self.get_parameter('device').get_parameter_value().string_value
        self._imgsz = int(self.get_parameter('imgsz').value)

        self.get_logger().info(f'Loading Ultralytics YOLO model: {model_name}')
        from ultralytics import YOLO

        self._yolo = YOLO(model_name)

        info_topic = self.get_parameter('lidar_info_topic').value
        scan_topic = self.get_parameter('lidar_scan_topic').value

        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._latest_info: Optional[LidarInfo] = None
        self._logged_info_sync = False
        self.create_subscription(LidarInfo, info_topic, self._on_info, info_qos)
        self.create_subscription(LidarScan, scan_topic, self._on_scan, scan_qos)

        self._pubs: Dict[str, Publisher] = {}
        for _, suffix in self.CHANNEL_SPECS:
            topic = f'/ouster_yolo/{suffix}_annotated'
            self._pubs[suffix] = self.create_publisher(
                Image,
                topic,
                QoSProfile(
                    depth=5,
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )

        self.get_logger().info(
            'YOLO lidar consumer ready (RANGE, NEAR_IR, REFLECTIVITY). '
            f'Publishing annotated images under /ouster_yolo/*_annotated'
        )

    def _on_info(self, msg: LidarInfo) -> None:
        self._latest_info = msg

    def _maybe_log_info_sync(self, scan: LidarScan) -> None:
        info = self._latest_info
        if self._logged_info_sync or info is None:
            return
        self._logged_info_sync = True
        va = len(info.vertical_angles)
        ha = len(info.horizontal_angles)
        self.get_logger().info(
            f'LidarInfo received (vertical_angles={va}, horizontal_angles={ha}); '
            f'scan grid {scan.width}x{scan.height}'
        )

    def _find_channels(self, scan: LidarScan) -> Dict[str, LidarChannel]:
        by_name: Dict[str, LidarChannel] = {}
        for ch in scan.channels:
            by_name[ch.name.lower()] = ch
        return by_name

    def _on_scan(self, scan: LidarScan) -> None:
        if scan.height == 0 or scan.width == 0 or not scan.data:
            return
        self._maybe_log_info_sync(scan)
        by_name = self._find_channels(scan)
        header = scan.header

        for logical, suffix in self.CHANNEL_SPECS:
            ch = by_name.get(logical)
            pub = self._pubs[suffix]
            if ch is None:
                self.get_logger().warning(
                    f'Missing LidarScan channel "{logical}"; skipping {suffix}'
                )
                continue
            try:
                arr = _extract_channel_2d(scan, ch)
                bgr = _to_bgr_visual(arr, logical)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'{logical}: decode failed: {exc}')
                continue

            try:
                device = self._device or None
                results = self._yolo.predict(
                    source=bgr,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                    device=device,
                )
                annotated = results[0].plot()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'{logical}: YOLO inference failed: {exc}')
                annotated = bgr

            pub.publish(_bgr_to_ros_image(header, annotated))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloLidarConsumer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
