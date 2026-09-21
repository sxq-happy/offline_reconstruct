"""Small dependency-light live viewer for offline_reconstruct_cpp.

The C++ process publishes a bounded preview.ply and preview_state.json.  This
viewer only reads those files, so it cannot affect registration or fusion.
PyQt5 is used instead of Open3D to keep the viewer usable in the existing
Windows/PyTorch environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ctypes
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def read_ply(path: Path, limit: int = 50000):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    marker = data.find(b"end_header\n")
    if marker < 0:
        return None
    header = data[: marker + len(b"end_header\n")].decode("ascii", "ignore")
    if "format binary_little_endian" not in header:
        return None
    try:
        count = next(
            int(line.split()[-1])
            for line in header.splitlines()
            if line.startswith("element vertex ")
        )
    except (StopIteration, ValueError):
        return None
    raw = data[marker + len(b"end_header\n") :]
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    required = count * dtype.itemsize
    if len(raw) < required:
        return None
    arr = np.frombuffer(raw[:required], dtype=dtype)
    # New snapshots are ordered by a deterministic spatial hash.  If a caller
    # supplies a lower display cap, sample across the complete snapshot rather
    # than taking only its first hash bucket range (which can look like a
    # collection of disconnected scan contours).
    if limit >= 0 and len(arr) > limit:
        indices = np.linspace(0, len(arr) - 1, limit, dtype=np.int64)
        arr = arr[indices]
    out = np.empty((len(arr), 6), dtype=np.float32)
    out[:, :3] = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
    out[:, 3:] = np.stack([arr["r"], arr["g"], arr["b"]], axis=1) / 255.0
    return out


def read_trajectory(path: Path):
    points = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("accepted", "0") == "1":
                    points.append((float(row["x"]), float(row["y"]), float(row["z"])))
    except (OSError, ValueError, KeyError):
        return None
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def committed_path(preview_dir: Path, name, fallback: str):
    """Resolve only a plain filename supplied by the trusted state sidecar."""
    value = str(name or fallback)
    candidate = Path(value)
    if candidate.name != value or candidate.is_absolute():
        return None
    return preview_dir / candidate


def read_pose(path: Path):
    """Read the tiny per-scan pose sidecar published by the C++ solver."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pose = np.asarray(
            [float(payload["x"]), float(payload["y"]), float(payload["z"])],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(pose)):
            return None
        return pose, int(payload.get("scan_index", -1)), str(payload.get("phase", ""))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_committed_snapshot(preview_dir: Path, expected_signature):
    """Load one committed cloud snapshot without blocking Qt's GUI thread."""
    state_path = preview_dir / "preview_state.json"
    try:
        state_stat = state_path.stat()
        signature = (state_stat.st_mtime_ns, state_stat.st_size)
        if signature != expected_signature:
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        ply_path = committed_path(preview_dir, state.get("cloud_file"), "preview.ply")
        trajectory_path = committed_path(
            preview_dir, state.get("trajectory_file"), "preview_trajectory.csv"
        )
        if ply_path is None or trajectory_path is None:
            return None
        points = read_ply(ply_path, limit=50000)
        trajectory = read_trajectory(trajectory_path)
        state_stat_after = state_path.stat()
        signature_after = (state_stat_after.st_mtime_ns, state_stat_after.st_size)
        if points is None or trajectory is None or signature_after != expected_signature:
            return None
        return expected_signature, state, points, trajectory
    except (OSError, ValueError, TypeError):
        return None


def rotation_x(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=np.float32,
    )


def rotation_y(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float32,
    )


def rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class PointView(QtWidgets.QWidget):
    def __init__(self, preview_dir: Path):
        super().__init__()
        self.preview_dir = preview_dir
        self.points = np.empty((0, 6), dtype=np.float32)
        self.trajectory = np.empty((0, 3), dtype=np.float32)
        self.state = {}
        self.pose_start = None
        self.pose_target = None
        self.pose_received_at = 0.0
        self.pose_transition_started = 0.0
        self.pose_transition_duration = 0.18
        self.pose_scan_index = -1
        self.pose_phase = ""
        # A full orientation matrix replaces the old yaw/pitch pair.  The old
        # pitch clamp stopped at +/-83 degrees, making an upside-down or full
        # 360-degree orbit impossible.  Incremental screen-axis rotations have
        # no pole and preserve the current roll naturally.
        self.view_rotation = rotation_x(0.48) @ rotation_z(-0.55)
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=np.float32)
        self.drag_pos = None
        self.drag_button = None
        self.data_signature = None
        self.pose_signature = None
        self.error_signature = None
        self.error_message = ""
        self.snapshot_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="preview-loader"
        )
        self.snapshot_future = None
        self.loading_signature = None
        self.render_dirty = True
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.timeout.connect(self._render_tick)
        self.render_timer.start(16)  # 60 FPS interaction; raster drawing is vectorized
        self.setMinimumSize(860, 560)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def _render_tick(self):
        if self.pose_target is not None and self.pose_start is not None:
            now = time.monotonic()
            alpha = min(
                1.0,
                max(0.0, (now - self.pose_transition_started) /
                    max(1e-3, self.pose_transition_duration)),
            )
            if alpha < 1.0:
                self.render_dirty = True
        if self.render_dirty:
            self.render_dirty = False
            self.update()

    def _display_pose(self):
        if self.pose_target is None or self.pose_start is None:
            return None
        alpha = min(
            1.0,
            max(0.0, (time.monotonic() - self.pose_transition_started) /
                max(1e-3, self.pose_transition_duration)),
        )
        return self.pose_start + (self.pose_target - self.pose_start) * alpha

    def _set_target_pose(self, pose, scan_index, phase):
        now = time.monotonic()
        if self.pose_target is not None:
            current = self._display_pose()
            self.pose_start = current if current is not None else self.pose_target.copy()
        else:
            self.pose_start = pose.copy()
        self.pose_target = pose.copy()
        if self.pose_received_at > 0.0:
            observed = now - self.pose_received_at
            self.pose_transition_duration = min(0.45, max(0.10, observed * 0.90))
        else:
            self.pose_transition_duration = 0.18
        self.pose_received_at = now
        self.pose_transition_started = now
        self.pose_scan_index = scan_index
        self.pose_phase = phase
        self.render_dirty = True

    def update_data(self):
        state_path = self.preview_dir / "preview_state.json"
        try:
            state_stat = state_path.stat()
            signature = (state_stat.st_mtime_ns, state_stat.st_size)
        except OSError:
            signature = (0, 0)
        # Parsing a 50k-point PLY on this timer used to stop the Qt event loop
        # for long enough that mouse move/release events were lost.  Only poll
        # the completed future here; all file I/O and NumPy conversion run on
        # the dedicated loader thread.
        if self.snapshot_future is not None and self.snapshot_future.done():
            try:
                snapshot = self.snapshot_future.result()
            except Exception:
                snapshot = None
            self.snapshot_future = None
            self.loading_signature = None
            if snapshot is not None:
                loaded_signature, state, points, trajectory = snapshot
                self.state = state
                self.points = points
                self.trajectory = trajectory
                self.data_signature = loaded_signature
                self.render_dirty = True
        if (
            signature != (0, 0)
            and signature != self.data_signature
            and signature != self.loading_signature
            and self.snapshot_future is None
        ):
            self.loading_signature = signature
            self.snapshot_future = self.snapshot_executor.submit(
                read_committed_snapshot, self.preview_dir, signature
            )

        pose_path = self.preview_dir / "preview_pose.json"
        try:
            pose_stat = pose_path.stat()
            pose_signature = (str(pose_path), pose_stat.st_mtime_ns, pose_stat.st_size)
        except OSError:
            pose_signature = (str(pose_path), 0, 0)
        if pose_signature != self.pose_signature:
            payload = read_pose(pose_path)
            if payload is not None:
                self.pose_signature = pose_signature
                self._set_target_pose(*payload)

        error_path = self.preview_dir / "preview_error.json"
        try:
            error_stat = error_path.stat()
            error_signature = (error_stat.st_mtime_ns, error_stat.st_size)
        except OSError:
            error_signature = (0, 0)
        if error_signature != self.error_signature:
            if error_signature == (0, 0):
                self.error_message = ""
                self.error_signature = error_signature
            else:
                try:
                    error_payload = json.loads(error_path.read_text(encoding="utf-8"))
                    error_message = str(error_payload.get("message", "unknown error"))
                except (OSError, ValueError, TypeError):
                    pass
                else:
                    self.error_message = error_message
                    self.error_signature = error_signature

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.view_rotation = rotation_y(-0.12) @ self.view_rotation
        elif event.key() == QtCore.Qt.Key_Right:
            self.view_rotation = rotation_y(0.12) @ self.view_rotation
        elif event.key() == QtCore.Qt.Key_Up:
            self.view_rotation = rotation_x(0.10) @ self.view_rotation
        elif event.key() == QtCore.Qt.Key_Down:
            self.view_rotation = rotation_x(-0.10) @ self.view_rotation
        else:
            super().keyPressEvent(event)
            return
        self.render_dirty = True

    def mousePressEvent(self, event):
        self.drag_pos = event.pos()
        self.drag_button = event.button()
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self.grabMouse()
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_pos is None:
            return
        delta = event.pos() - self.drag_pos
        self.drag_pos = event.pos()
        if self.drag_button == QtCore.Qt.LeftButton:
            horizontal = rotation_y(delta.x() * 0.008)
            vertical = rotation_x(delta.y() * 0.008)
            self.view_rotation = vertical @ horizontal @ self.view_rotation
        elif self.drag_button == QtCore.Qt.RightButton:
            self.pan += np.array([delta.x(), delta.y()], dtype=np.float32)
        self.render_dirty = True
        event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_pos = None
        self.drag_button = None
        if QtWidgets.QApplication.mouseButtons() == QtCore.Qt.NoButton:
            self.releaseMouse()
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.render_dirty = True
        event.accept()

    def closeEvent(self, event):
        if self.snapshot_future is not None:
            self.snapshot_future.cancel()
        self.snapshot_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom = max(0.12, min(20.0, self.zoom * (1.18 ** steps)))
        self.render_dirty = True
        event.accept()

    def project(self, xyz, center=None, span=None):
        if len(xyz) == 0:
            return np.empty((0, 2), dtype=np.float32), 1.0
        if center is None:
            center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
        if span is None:
            span = max(float(np.max(xyz.max(axis=0) - xyz.min(axis=0))), 1e-3)
        p = xyz - center
        rotated = p @ self.view_rotation.T
        x = rotated[:, 0]
        y = rotated[:, 1]
        depth = -rotated[:, 2]
        scale = min(self.width(), self.height()) * 0.82 * self.zoom / span
        return np.column_stack((self.width() * 0.5 + self.pan[0] + x * scale,
                                self.height() * 0.5 + self.pan[1] - y * scale, depth)), scale

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#090d12"))
        cloud_xyz = self.points[:, :3] if len(self.points) else np.empty((0, 3), dtype=np.float32)
        display_pose = self._display_pose()
        render_trajectory = self.trajectory
        if display_pose is not None:
            if len(render_trajectory) == 0:
                render_trajectory = display_pose.reshape(1, 3)
            elif np.linalg.norm(render_trajectory[-1] - display_pose) > 1e-5:
                # The CSV trajectory is intentionally low-rate.  Append the
                # independently published current pose so the last segment
                # moves continuously between map snapshots.
                render_trajectory = np.vstack((render_trajectory, display_pose))
        all_xyz = cloud_xyz
        if len(render_trajectory):
            all_xyz = np.vstack((all_xyz, render_trajectory)) if len(all_xyz) else render_trajectory
        shared_center = None
        shared_span = None
        if len(all_xyz):
            shared_center = (all_xyz.min(axis=0) + all_xyz.max(axis=0)) * 0.5
            shared_span = max(float(np.max(all_xyz.max(axis=0) - all_xyz.min(axis=0))), 1e-3)
        if len(self.points):
            # Interactive LOD keeps orbit/pan responsive on integrated GPUs.
            # The full snapshot is restored on mouse release and remains the
            # source for framing, so the view never jumps while dragging.
            render_points = self.points
            if self.drag_pos is not None and len(render_points) > 20000:
                indices = np.linspace(0, len(render_points) - 1, 20000, dtype=np.int64)
                render_points = render_points[indices]
            projected, _ = self.project(render_points[:, :3], shared_center, shared_span)
            # Drawing one QPainter ellipse per point made mouse rotation
            # visibly stutter once the bounded preview reached 50k points.
            # Rasterize all points into one RGBA image with NumPy instead. It
            # keeps the same pseudo-colours and point size but reduces the
            # per-frame work from tens of thousands of Python/Qt calls to a
            # handful of vectorized array operations and one drawImage call.
            width = self.width()
            height = self.height()
            if width > 0 and height > 0:
                screen = np.rint(projected[:, :2]).astype(np.int32)
                valid = (
                    (screen[:, 0] >= 0) & (screen[:, 0] < width) &
                    (screen[:, 1] >= 0) & (screen[:, 1] < height)
                )
                if np.any(valid):
                    screen = screen[valid]
                    colours = np.clip(render_points[valid, 3:] * 255.0, 0.0, 255.0).astype(np.uint8)
                    rgba = np.empty((len(screen), 4), dtype=np.uint8)
                    rgba[:, :3] = colours
                    rgba[:, 3] = 255
                    canvas = np.zeros((height, width, 4), dtype=np.uint8)
                    point_radius = 1 if len(render_points) >= 35000 else 2
                    for dy in range(-point_radius, point_radius + 1):
                        for dx in range(-point_radius, point_radius + 1):
                            xx = screen[:, 0] + dx
                            yy = screen[:, 1] + dy
                            inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                            if np.any(inside):
                                canvas[yy[inside], xx[inside]] = rgba[inside]
                    image = QtGui.QImage(
                        canvas.data,
                        width,
                        height,
                        int(canvas.strides[0]),
                        QtGui.QImage.Format_RGBA8888,
                    )
                    painter.drawImage(0, 0, image)
        if len(render_trajectory) > 1:
            # Use precisely the same center and scale as the cloud.  Applying
            # an independent fit here makes a correct trajectory appear to
            # fly away from the map.
            projected, _ = self.project(render_trajectory, shared_center, shared_span)
            pen = QtGui.QPen(QtGui.QColor("#ff9e2c"), 2)
            painter.setPen(pen)
            for a, b in zip(projected[:-1], projected[1:]):
                painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            painter.setBrush(QtGui.QColor("#7bf0bd"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#07120e"), 1))
            for item in (projected[0], projected[-1]):
                painter.drawEllipse(int(item[0]) - 4, int(item[1]) - 4, 8, 8)
        painter.setPen(QtGui.QColor("#8ea5b8"))
        painter.drawText(18, 28, "LIVE MAP PREVIEW")
        painter.end()


class Window(QtWidgets.QWidget):
    def __init__(self, preview_dir: Path):
        super().__init__()
        self.preview_dir = preview_dir
        self.setWindowTitle("Offline Reconstruction · Live Map Preview")
        self.resize(1220, 760)
        self.setStyleSheet(
            "QWidget{background:#101820;color:#eaf2f8;}"
            "QFrame#side{background:#16232e;border:1px solid #2a4050;border-radius:10px;}"
            "QLabel#phase{font-size:23px;font-weight:700;color:#f2c94c;}"
            "QLabel#caption{color:#9db1bf;font-size:13px;}"
            "QLabel#metric{color:#eef5f8;font-size:18px;font-weight:600;padding:7px 0;}"
            "QProgressBar{height:12px;border:0;border-radius:6px;background:#263642;text-align:center;color:#eaf2f8;}"
            "QProgressBar::chunk{border-radius:6px;background:#e0ad3e;}"
        )
        self.view = PointView(preview_dir)
        self.phase = QtWidgets.QLabel("等待解算启动…")
        self.phase.setObjectName("phase")
        self.info = QtWidgets.QLabel("预览只读，不参与配准")
        self.info.setObjectName("caption")
        self.info.setVisible(False)
        self.frame_info = QtWidgets.QLabel("\u5e27  0 / 0")
        self.frame_info.setObjectName("metric")
        self.point_info = QtWidgets.QLabel("\u9884\u89c8\u70b9  0")
        self.point_info.setObjectName("metric")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.legend = QtWidgets.QLabel("彩色点：深度伪彩色\n橙线：当前位姿轨迹\n绿点：起点 / 当前点")
        self.legend.setObjectName("caption")
        self.legend.setWordWrap(True)
        self.legend.setText("\u70b9\u4e91  \u6df1\u5ea6\u4f2a\u5f69\u8272\n\u8f68\u8ff9  \u914d\u51c6\u4f4d\u59ff\n\u6807\u8bb0  \u8d77\u70b9 / \u5f53\u524d\u70b9")
        self.controls = QtWidgets.QLabel("\u9f20\u6807\u5de6\u952e  \u65cb\u8f6c\u89c6\u89d2\n\u9f20\u6807\u6eda\u8f6e  \u653e\u5927 / \u7f29\u5c0f\n\u9f20\u6807\u53f3\u952e  \u5e73\u79fb\u89c6\u56fe")
        self.controls.setObjectName("caption")
        self.controls.setWordWrap(True)
        panel = QtWidgets.QVBoxLayout()
        panel.setContentsMargins(20, 22, 20, 22)
        panel.setSpacing(12)
        panel.addWidget(self.phase)
        panel.addWidget(self.info)
        panel.addWidget(self.frame_info)
        panel.addWidget(self.point_info)
        panel.addWidget(self.progress)
        panel.addStretch(2)
        panel.addWidget(self.legend)
        panel.addSpacing(14)
        panel.addWidget(self.controls)
        panel.addStretch(1)
        panel_widget = QtWidgets.QFrame()
        panel_widget.setObjectName("side")
        panel_widget.setLayout(panel)
        panel_widget.setMinimumWidth(280)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.view, 1)
        layout.addWidget(panel_widget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        # Consume the lightweight pose sidecar at 10 Hz.  The point-cloud
        # snapshot is still reloaded only when its file signature changes.
        self.timer.start(100)
        self.refresh()

    def refresh(self):
        self.view.update_data()
        state = self.view.state
        phase = "failed" if self.view.error_message else state.get("phase", "waiting")
        progress = float(state.get("progress_percent", 0.0))
        processed = state.get("processed", 0)
        total = state.get("total", 0)
        points = state.get("points", len(self.view.points))
        phase_label = {
            "waiting": "等待数据",
            "registration": "前端配准",
            "fusion": "全量融合",
            "complete": "解算完成",
        }.get(phase, phase)
        phase_label = {"waiting": "\u7b49\u5f85\u6570\u636e", "registration": "\u524d\u7aef\u914d\u51c6", "fusion": "\u5168\u91cf\u878d\u5408", "complete": "\u89e3\u7b97\u5b8c\u6210", "failed": "\u89e3\u7b97\u5931\u8d25"}.get(phase, phase)
        self.phase.setText(f"{phase_label}  {progress:.1f}%")
        self.progress.setValue(max(0, min(1000, int(progress * 10.0))))
        self.frame_info.setText(f"\u5e27  {processed} / {total}")
        self.point_info.setText(f"\u9884\u89c8\u70b9  {points:,}")
        self.info.setText(f"帧: {processed}/{total}    预览点: {points}\n方向键旋转视角 · 预览不参与解算")


class CommercialWindow(QtWidgets.QWidget):
    """Commercial-style shell around the low-overhead point view."""

    def __init__(self, preview_dir: Path):
        super().__init__()
        self.setObjectName("app")
        self.setWindowTitle("Offline Reconstruction · Live Map Preview")
        self.resize(1440, 860)
        self.setStyleSheet(
            "QWidget#app{background:#0b141c;color:#e9f1f6;}"
            "QFrame#sidebar{background:#12212d;border:1px solid #2a4356;border-radius:12px;}"
            "QFrame#card{background:#0d1a24;border:1px solid #223b4d;border-radius:9px;}"
            "QLabel#brand{font-size:21px;font-weight:700;color:#f3f7fa;}"
            "QLabel#eyebrow{font-size:11px;letter-spacing:1px;color:#7f9caf;}"
            "QLabel#phase{font-size:23px;font-weight:700;color:#f0c34e;}"
            "QLabel#percent{font-size:19px;font-weight:700;color:#f0c34e;}"
            "QLabel#status{font-size:12px;color:#82dfb7;}"
            "QLabel#metricTitle{font-size:11px;color:#89a2b3;}"
            "QLabel#metricValue{font-size:19px;font-weight:650;color:#edf5f8;}"
            "QLabel#muted{font-size:12px;color:#91a7b7;}"
            "QLabel#footer{font-size:11px;color:#668092;}"
            "QProgressBar{height:10px;border:0;border-radius:5px;background:#263b49;text-align:center;color:#d9e6ed;}"
            "QProgressBar::chunk{border-radius:5px;background:#e0ae3f;}"
        )
        self.view = PointView(preview_dir)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(22, 22, 22, 20)
        side.setSpacing(12)
        brand = QtWidgets.QLabel("\u79bb\u7ebf\u91cd\u5efa")
        brand.setObjectName("brand")
        eyebrow = QtWidgets.QLabel("LIVE MAP PREVIEW")
        eyebrow.setObjectName("eyebrow")
        side.addWidget(brand)
        side.addWidget(eyebrow)

        status_card = QtWidgets.QFrame()
        status_card.setObjectName("card")
        status_layout = QtWidgets.QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(10)
        phase_row = QtWidgets.QHBoxLayout()
        self.phase = QtWidgets.QLabel("\u7b49\u5f85\u6570\u636e")
        self.phase.setObjectName("phase")
        self.percent = QtWidgets.QLabel("0.0%")
        self.percent.setObjectName("percent")
        phase_row.addWidget(self.phase)
        phase_row.addStretch(1)
        phase_row.addWidget(self.percent)
        status_layout.addLayout(phase_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        status_layout.addWidget(self.progress)
        """
        self.status = QtWidgets.QLabel("●  \u7b49\u5f85\u89e3\u7b97\")
        """
        self.status = QtWidgets.QLabel("\u25cf  \u7b49\u5f85\u89e3\u7b97")
        self.status.setObjectName("status")
        status_layout.addWidget(self.status)
        side.addWidget(status_card)

        metrics = QtWidgets.QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        self.frame_info = self._metric_card("\u5f53\u524d\u5e27")
        self.total_info = self._metric_card("\u603b\u5e27\u6570")
        self.point_info = self._metric_card("\u9884\u89c8\u70b9")
        self.track_info = self._metric_card("\u8f68\u8ff9\u72b6\u6001")
        metrics.addWidget(self.frame_info[0], 0, 0)
        metrics.addWidget(self.total_info[0], 0, 1)
        metrics.addWidget(self.point_info[0], 1, 0)
        metrics.addWidget(self.track_info[0], 1, 1)
        side.addLayout(metrics)

        side.addStretch(1)
        legend = QtWidgets.QFrame()
        legend.setObjectName("card")
        legend_layout = QtWidgets.QVBoxLayout(legend)
        legend_layout.setContentsMargins(16, 13, 16, 13)
        legend_layout.setSpacing(7)
        legend_title = QtWidgets.QLabel("\u663e\u793a\u56fe\u4f8b")
        legend_title.setObjectName("metricTitle")
        legend_layout.addWidget(legend_title)
        legend_layout.addWidget(self._legend_row("#f0a83c", "\u70b9\u4e91  \u6df1\u5ea6\u4f2a\u5f69\u8272"))
        legend_layout.addWidget(self._legend_row("#ff9e2c", "\u6a59\u7ebf  \u914d\u51c6\u8f68\u8ff9"))
        legend_layout.addWidget(self._legend_row("#7bf0bd", "\u7eff\u70b9  \u8d77\u70b9 / \u5f53\u524d\u70b9"))
        side.addWidget(legend)

        controls = QtWidgets.QFrame()
        controls.setObjectName("card")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 13, 16, 13)
        controls_title = QtWidgets.QLabel("\u89c6\u56fe\u64cd\u4f5c")
        controls_title.setObjectName("metricTitle")
        controls_layout.addWidget(controls_title)
        controls_text = QtWidgets.QLabel("\u5de6\u952e\u65cb\u8f6c   ·   \u6eda\u8f6e\u7f29\u653e   ·   \u53f3\u952e\u5e73\u79fb")
        controls_text.setObjectName("muted")
        controls_text.setWordWrap(True)
        controls_layout.addWidget(controls_text)
        # Keep the displayed help text Unicode-safe even when this file is
        # opened by a legacy Windows console using a non-UTF-8 code page.
        controls_text.setText("\u5de6\u952e\u65cb\u8f6c   \u00b7   \u6eda\u8f6e\u7f29\u653e   \u00b7   \u53f3\u952e\u5e73\u79fb")
        side.addWidget(controls)
        footer = QtWidgets.QLabel("\u9884\u89c8\u4ec5\u7528\u4e8e\u67e5\u770b  ·  \u6700\u7ec8\u7ed3\u679c\u4ee5 PLY \u4e3a\u51c6")
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        footer.setText("\u9884\u89c8\u4ec5\u7528\u4e8e\u67e5\u770b  \u00b7  \u6700\u7ec8\u7ed3\u679c\u4ee5 PLY \u4e3a\u51c6")
        side.addWidget(footer)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.view, 1)
        layout.addWidget(sidebar)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(360)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        # Do not read the first (possibly 50k-point) PLY while the native
        # window is still being constructed.  In a long capture that made the
        # preview appear only after many scans had already been processed.
        # The first refresh is scheduled after show() from main().

    @staticmethod
    def _metric_card(title):
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        box = QtWidgets.QVBoxLayout(card)
        box.setContentsMargins(13, 12, 13, 12)
        box.setSpacing(5)
        label = QtWidgets.QLabel(title)
        label.setObjectName("metricTitle")
        value = QtWidgets.QLabel("--")
        value.setObjectName("metricValue")
        box.addWidget(label)
        box.addWidget(value)
        return card, value

    @staticmethod
    def _legend_row(color, text):
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet("color:%s;font-size:13px;" % color)
        label = QtWidgets.QLabel(text)
        label.setObjectName("muted")
        box.addWidget(dot)
        box.addWidget(label)
        box.addStretch(1)
        return row

    def refresh(self):
        self.view.update_data()
        state = self.view.state
        phase = "failed" if self.view.error_message else state.get("phase", "waiting")
        progress = float(state.get("progress_percent", 0.0))
        processed = int(state.get("processed", 0))
        total = int(state.get("total", 0))
        points = int(state.get("points", len(self.view.points)))
        phase_label = {"waiting": "\u7b49\u5f85\u6570\u636e", "registration": "\u524d\u7aef\u914d\u51c6", "fusion": "\u5168\u91cf\u878d\u5408", "complete": "\u89e3\u7b97\u5b8c\u6210", "failed": "\u89e3\u7b97\u5931\u8d25"}.get(phase, phase)
        self.phase.setText(phase_label)
        self.percent.setText(f"{progress:.1f}%")
        self.progress.setValue(max(0, min(1000, int(progress * 10.0))))
        self.frame_info[1].setText(f"{processed:,}")
        self.total_info[1].setText(f"{total:,}")
        self.point_info[1].setText(f"{points:,}")
        track_status = {"complete": "\u5df2\u5b8c\u6210", "failed": "\u5df2\u505c\u6b62"}.get(phase, "\u8ddf\u968f\u66f4\u65b0")
        self.track_info[1].setText(track_status)
        status = {"waiting": "\u7b49\u5f85\u89e3\u7b97", "complete": "\u5df2\u5b8c\u6210", "failed": "\u89e3\u7b97\u5931\u8d25"}.get(phase, "\u6b63\u5728\u8fd0\u884c")
        self.status.setText("●  " + status)
        self.status.setStyleSheet("color:#ff7b72;" if phase == "failed" else "")
        self.status.setToolTip(self.view.error_message)


def main():
    parser = argparse.ArgumentParser(description="view offline_reconstruct_cpp live preview")
    parser.add_argument("preview_dir", type=Path)
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = CommercialWindow(args.preview_dir)
    # A viewer started by ``run_cpp.bat`` is detached from the terminal.  On
    # Windows that means Qt is allowed to create it behind VS Code, or reuse a
    # stale off-screen placement from a previous monitor.  Put the initial
    # window in the centre of the current work area before showing it.  This is
    # only an initial placement; the user can move/resize it normally.
    screen = app.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        width = min(window.width(), max(900, area.width() - 80))
        height = min(window.height(), max(600, area.height() - 80))
        window.resize(width, height)
        window.move(
            area.left() + max(0, (area.width() - width) // 2),
            area.top() + max(0, (area.height() - height) // 2),
        )
    window.show()
    # ``start`` launches the viewer from a background batch process. Windows
    # may therefore create the Qt window behind VS Code or leave it minimized
    # even though the Python process is healthy. Explicitly restore and focus
    # it once after the event loop has created the native window.
    def focus_window():
        window.showNormal()
        window.raise_()
        window.activateWindow()
        if sys.platform.startswith("win"):
            try:
                hwnd = int(window.winId())
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
                # SetForegroundWindow is intentionally conservative for a
                # process launched by ``start``.  A short TOPMOST pulse makes
                # the first frame visible without leaving the app permanently
                # above other applications.
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                SWP_NOSIZE = 0x0001
                SWP_NOMOVE = 0x0002
                SWP_SHOWWINDOW = 0x0040
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
                )
                user32.SetForegroundWindow(hwnd)

                def clear_topmost():
                    user32.SetWindowPos(
                        hwnd,
                        HWND_NOTOPMOST,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
                    )

                QtCore.QTimer.singleShot(700, clear_topmost)
            except (AttributeError, OSError, TypeError, ValueError):
                pass

    def write_startup_status():
        # A tiny diagnostic sidecar makes a detached GUI launch observable
        # from the same terminal without sending any data back to the solver.
        # It is overwritten on every viewer start and is safe to ignore.
        try:
            hwnd = int(window.winId()) if sys.platform.startswith("win") else 0
            status = {
                "pid": os.getpid(),
                "window_handle": hwnd,
                "visible": bool(window.isVisible()),
                "preview_dir": str(args.preview_dir.resolve()),
            }
            (args.preview_dir / "preview_viewer_status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError):
            pass

    focus_window()
    write_startup_status()
    # Let Windows paint the shell immediately; loading the first cloud happens
    # on the next event-loop turn and subsequent snapshots use the 100 ms timer.
    QtCore.QTimer.singleShot(0, window.refresh)
    QtCore.QTimer.singleShot(180, focus_window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
