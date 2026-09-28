from __future__ import annotations

import argparse
import json
import math
import queue
import threading
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

from ultrasound_bin_gui import (
    ProcessParams,
    diffused_region_mask,
    process_frames,
    read_capture_metadata,
    read_frames,
)


APP_TITLE = "三维超声红/蓝区域重建查看器 Image v17"


def resolve_sweep_angles(
    metadata: dict,
    frame_count: int,
    fallback_sweep_deg: float,
) -> tuple[np.ndarray, str]:
    """Use per-frame angle3D values, or distribute frames over a fallback sweep."""
    measured = np.asarray(metadata.get("angle3DValues") or [], dtype=np.float32)
    if len(measured) == frame_count and float(np.ptp(measured)) > 1e-4:
        return measured, "frames.json / angle3D"
    sweep = max(1.0, float(fallback_sweep_deg))
    return np.linspace(-sweep / 2.0, sweep / 2.0, frame_count, dtype=np.float32), "帧数均匀分布（备用扫描角）"


def reconstruct_labeled_point_cloud(
    frames: np.ndarray,
    metadata: dict,
    red_threshold: float = 150.0,
    blue_threshold: float = 225.0,
    diffusion_steps: int = 5,
    fallback_sweep_deg: float = 80.0,
    frame_step: int = 1,
    line_step: int = 2,
    sample_step: int = 2,
    rotation_axis: str = "depth",
    max_points: int = 260_000,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a 3-D cloud whose red and blue classes match the 2-D markings."""
    if frames.ndim != 3 or not frames.size:
        raise ValueError("没有可用于三维重建的帧数据。")
    if not 0 <= red_threshold < 255 or not 0 <= blue_threshold <= 255:
        raise ValueError("红/蓝阈值必须位于 0–255。")
    effective_blue = max(float(blue_threshold), float(red_threshold) + 1.0)
    angles, angle_source = resolve_sweep_angles(metadata, len(frames), fallback_sweep_deg)

    frame_ids = np.arange(0, len(frames), max(1, int(frame_step)), dtype=np.int32)
    line_ids = np.arange(0, frames.shape[1], max(1, int(line_step)), dtype=np.int32)
    sample_ids = np.arange(0, frames.shape[2], max(1, int(sample_step)), dtype=np.int32)

    scan_angle = float(metadata.get("scanAngleRad") or math.radians(60.0))
    dead_radius = float(metadata.get("deadRadius") or 0.0)
    sample_scale = float(metadata.get("sampleScale") or 1.0)
    line_theta = np.linspace(
        -scan_angle / 2.0,
        scan_angle / 2.0,
        frames.shape[1],
        dtype=np.float32,
    )[line_ids]
    radius = (dead_radius + sample_ids.astype(np.float32)) * sample_scale
    lateral = np.sin(line_theta)[:, None] * radius[None, :]
    depth = np.cos(line_theta)[:, None] * radius[None, :]
    center_angle = float(np.median(angles))

    point_blocks: list[np.ndarray] = []
    intensity_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    red_voxels = 0
    blue_voxels = 0
    for order, frame_id in enumerate(frame_ids):
        image = frames[frame_id]
        red_full = diffused_region_mask(image, red_threshold, diffusion_steps)
        blue_full = diffused_region_mask(image, effective_blue, diffusion_steps)
        red = red_full[np.ix_(line_ids, sample_ids)]
        blue = blue_full[np.ix_(line_ids, sample_ids)]
        keep = red | blue
        if np.any(keep):
            x2 = lateral[keep]
            z2 = depth[keep]
            phi = math.radians(float(angles[frame_id] - center_angle))
            if rotation_axis == "depth":
                x = x2 * math.cos(phi)
                y = x2 * math.sin(phi)
                z = z2
            else:
                x = x2
                y = z2 * math.sin(phi)
                z = z2 * math.cos(phi)
            labels = np.where(blue[keep], 2, 1).astype(np.uint8)
            values = image[np.ix_(line_ids, sample_ids)][keep].astype(np.uint8)
            point_blocks.append(np.column_stack((x, y, z)).astype(np.float32))
            intensity_blocks.append(values)
            label_blocks.append(labels)
            red_voxels += int(np.count_nonzero(labels == 1))
            blue_voxels += int(np.count_nonzero(labels == 2))
        if progress and (order % 4 == 0 or order + 1 == len(frame_ids)):
            progress((order + 1) / len(frame_ids), f"正在叠加第 {order + 1}/{len(frame_ids)} 个扫描角")

    if not point_blocks:
        raise ValueError("当前红色阈值过高，没有可用于三维重建的区域。")
    points = np.concatenate(point_blocks, axis=0)
    intensity = np.concatenate(intensity_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    candidate_count = len(points)

    # Reserve display capacity for blue strong echoes so a large red region
    # cannot visually bury clinically interesting high-intensity structures.
    if len(points) > max_points:
        red_idx = np.flatnonzero(labels == 1)
        blue_idx = np.flatnonzero(labels == 2)
        blue_budget = min(len(blue_idx), max_points // 2)
        red_budget = min(len(red_idx), max_points - blue_budget)
        blue_budget = min(len(blue_idx), max_points - red_budget)
        chosen = []
        if red_budget:
            chosen.append(red_idx[np.linspace(0, len(red_idx) - 1, red_budget, dtype=np.int64)])
        if blue_budget:
            chosen.append(blue_idx[np.linspace(0, len(blue_idx) - 1, blue_budget, dtype=np.int64)])
        keep_idx = np.sort(np.concatenate(chosen))
        points, intensity, labels = points[keep_idx], intensity[keep_idx], labels[keep_idx]

    low = np.percentile(points, 1, axis=0)
    high = np.percentile(points, 99, axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    points = (points - center) / max(scale, 1e-6)
    info = {
        "angleSource": angle_source,
        "angleRangeDeg": [float(np.min(angles)), float(np.max(angles))],
        "scanAngleDeg": math.degrees(scan_angle),
        "frameCount": len(frames),
        "usedFrameCount": len(frame_ids),
        "redThreshold": float(red_threshold),
        "blueThreshold": effective_blue,
        "diffusionSteps": int(diffusion_steps),
        "redCandidateCount": red_voxels,
        "blueCandidateCount": blue_voxels,
        "candidatePointCount": candidate_count,
        "displayPointCount": len(points),
        "rotationAxis": rotation_axis,
        "physicalBounds": {"low": low.tolist(), "high": high.tolist(), "sampleScale": sample_scale},
    }
    return points.astype(np.float32), intensity, labels, info


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ rx @ ry


def _dilate_max(layer: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(max(0, iterations)):
        layer = np.maximum.reduce(
            [layer, np.roll(layer, 1, 0), np.roll(layer, -1, 0), np.roll(layer, 1, 1), np.roll(layer, -1, 1)]
        )
    return layer


def render_labeled_point_cloud(
    points: np.ndarray,
    intensity: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
    point_size: int,
) -> Image.Image:
    """Software-render red tissue points and make blue strong echoes dominant."""
    width, height = max(200, width), max(200, height)
    rotated = points @ rotation_matrix(yaw, pitch).T
    base_scale = 0.88 * min(width, height)
    u = np.rint(width / 2.0 + rotated[:, 0] * base_scale * zoom).astype(np.int32)
    v = np.rint(height / 2.0 - rotated[:, 2] * base_scale * zoom).astype(np.int32)
    visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[visible], v[visible]
    values = intensity[visible].astype(np.float32)
    shown_labels = labels[visible]
    depth = rotated[visible, 1]
    if values.size:
        depth_norm = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-6)
        values = np.clip(45.0 + values * (0.55 + 0.35 * depth_norm), 0, 255).astype(np.uint8)

    red_layer = np.zeros((height, width), dtype=np.uint8)
    blue_layer = np.zeros((height, width), dtype=np.uint8)
    red = shown_labels == 1
    blue = shown_labels == 2
    np.maximum.at(red_layer, (v[red], u[red]), values[red])
    np.maximum.at(blue_layer, (v[blue], u[blue]), np.maximum(values[blue], 170))
    red_layer = _dilate_max(red_layer, max(0, int(point_size) - 1))
    blue_layer = _dilate_max(blue_layer, max(1, int(point_size)))

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = red_layer
    rgb[..., 1] = (red_layer.astype(np.float32) * 0.13).astype(np.uint8)
    rgb[..., 2] = (red_layer.astype(np.float32) * 0.10).astype(np.uint8)
    blue_pixels = blue_layer > 0
    # Blue wins on overlap, matching the 2-D overlay and preventing a pink mix.
    rgb[..., 0][blue_pixels] = blue_layer[blue_pixels] // 10
    rgb[..., 1][blue_pixels] = (blue_layer[blue_pixels] * 0.65).astype(np.uint8)
    rgb[..., 2][blue_pixels] = blue_layer[blue_pixels]

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 14, 270, 66), fill=(5, 7, 10), outline=(70, 75, 82))
    draw.line((27, 31, 57, 31), fill=(255, 45, 45), width=4)
    draw.text((66, 22), "Red: overall marked region", fill=(230, 230, 230))
    draw.line((27, 51, 57, 51), fill=(35, 150, 255), width=5)
    draw.text((66, 42), "Blue: strong echo (highlighted)", fill=(230, 230, 230))
    return image


def _params_from_json(text: str | None) -> ProcessParams:
    if not text:
        return ProcessParams()
    values = json.loads(text)
    valid = {item.name for item in fields(ProcessParams)}
    return ProcessParams(**{key: value for key, value in values.items() if key in valid})


class Ultrasound3DViewer(tk.Tk):
    def __init__(self, initial_folder: str | None = None, initial_params: ProcessParams | None = None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1420x920")
        self.minsize(1080, 720)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.task_queue: queue.Queue = queue.Queue()
        self.source_params = initial_params or ProcessParams()
        self.raw_frames: np.ndarray | None = None
        self.processed_frames: np.ndarray | None = None
        self.metadata: dict = {}
        self.points: np.ndarray | None = None
        self.intensity: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.cloud_info: dict = {}
        self.yaw, self.pitch, self.zoom = -25.0, -12.0, 1.0
        self.drag_origin = None
        self.render_job = None
        self.rebuild_job = None
        self.tk_image = None
        self.auto_rotate = False
        self._build_ui()
        self.after(80, self._poll_queue)
        if initial_folder:
            self.folder_var.set(initial_folder)
            self.after(150, self._load_folder)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        panel = ttk.Frame(self, padding=12)
        panel.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(self, padding=(0, 12, 12, 12))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(1, weight=1)

        source = ttk.LabelFrame(panel, text="采集目录", padding=10)
        source.pack(fill="x", pady=(0, 10))
        self.folder_var = tk.StringVar()
        ttk.Entry(source, textvariable=self.folder_var, width=42).pack(fill="x")
        ttk.Button(source, text="浏览并读取目录", command=self._browse_folder).pack(fill="x", pady=(7, 0))
        self.summary_var = tk.StringVar(value="需要 raw_frames_u8.bin 与 frames.json。")
        ttk.Label(source, textvariable=self.summary_var, wraplength=350).pack(fill="x", pady=(7, 0))

        marks = ttk.LabelFrame(panel, text="二维标注继承", padding=10)
        marks.pack(fill="x", pady=(0, 10))
        self.red_threshold_var = tk.DoubleVar(value=self.source_params.red_outline_threshold)
        self.blue_threshold_var = tk.DoubleVar(value=self.source_params.blue_outline_threshold)
        self.diffusion_var = tk.IntVar(value=self.source_params.red_outline_diffusion_steps)
        self._scale(marks, 0, "红色整体区域阈值", self.red_threshold_var, 80, 240, 1)
        self._scale(marks, 1, "蓝色强反声阈值", self.blue_threshold_var, 128, 255, 1)
        self._scale(marks, 2, "连续扩散次数", self.diffusion_var, 0, 12, 1)

        quality = ttk.LabelFrame(panel, text="三维重建参数", padding=10)
        quality.pack(fill="x", pady=(0, 10))
        inherited_sweep = abs(float(self.source_params.dsc_angle_deg))
        self.sweep_var = tk.DoubleVar(value=max(10.0, min(180.0, inherited_sweep)))
        self.frame_step_var = tk.IntVar(value=1)
        self.line_step_var = tk.IntVar(value=2)
        self.sample_step_var = tk.IntVar(value=2)
        self.point_size_var = tk.IntVar(value=2)
        self.axis_var = tk.StringVar(value="depth")
        self._scale(quality, 0, "备用扫描总角度", self.sweep_var, 10, 180, 1)
        self._scale(quality, 1, "帧间隔", self.frame_step_var, 1, 5, 1)
        self._scale(quality, 2, "扫描线间隔", self.line_step_var, 1, 6, 1)
        self._scale(quality, 3, "深度采样间隔", self.sample_step_var, 1, 6, 1)
        self._scale(quality, 4, "显示点大小", self.point_size_var, 1, 4, 1, render_only=True)
        ttk.Label(quality, text="旋转扫描轴").grid(row=5, column=0, sticky="w", pady=4)
        axis = ttk.Combobox(quality, textvariable=self.axis_var, values=("depth", "lateral"), state="readonly", width=12)
        axis.grid(row=5, column=1, sticky="ew", pady=4)
        axis.bind("<<ComboboxSelected>>", lambda _e: self._schedule_rebuild())
        ttk.Label(
            quality,
            text="有 angle3D 时优先逐帧读取；缺失时才按帧数和备用角度均匀分布。",
            wraplength=350,
            foreground="#3a6f8f",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.rebuild_button = ttk.Button(panel, text="生成/更新三维图像", command=self._start_rebuild)
        self.rebuild_button.pack(fill="x")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="重置视角", command=self._reset_view).pack(side="left", expand=True, fill="x")
        self.rotate_button = ttk.Button(row, text="自动旋转", command=self._toggle_auto_rotate)
        self.rotate_button.pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(row, text="保存当前视图", command=self._save_view).pack(side="left", expand=True, fill="x")
        self.progress = ttk.Progressbar(panel, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 3))
        self.status_var = tk.StringVar(value="等待选择目录。")
        ttk.Label(panel, textvariable=self.status_var, wraplength=360).pack(fill="x")

        ttk.Label(view, text="拖动鼠标旋转｜滚轮缩放｜蓝色强反声点会优先显示", anchor="center").grid(row=0, column=0, sticky="ew")
        self.canvas = tk.Canvas(view, background="#050608", highlightthickness=1, highlightbackground="#333333")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(7, 0))
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Double-Button-1>", lambda _e: self._reset_view())
        self.canvas.bind("<Configure>", lambda _e: self._schedule_render())

    def _scale(self, parent, row, label, variable, low, high, resolution, render_only=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        command = (lambda _v: self._schedule_render()) if render_only else (lambda _v: self._schedule_rebuild())
        tk.Scale(
            parent, from_=low, to=high, resolution=resolution, variable=variable,
            orient=tk.HORIZONTAL, length=190, command=command, highlightthickness=0,
        ).grid(row=row, column=1, sticky="ew")

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="选择超声采集目录")
        if folder:
            self.folder_var.set(folder)
            self._load_folder()

    def _load_folder(self):
        self.rebuild_button.configure(state="disabled")
        self.status_var.set("正在读取帧、应用统一调色和去噪…")

        def task():
            try:
                meta = read_capture_metadata(Path(self.folder_var.get()))
                params = replace(
                    self.source_params,
                    width=meta["sampleCount"], height=meta["lineCount"], frame_count=meta["frameCount"],
                    offset_bytes=0, dsc_enabled=False, transpose_output=False,
                )
                raw = read_frames(Path(meta["rawFile"]), params)
                processed, _ = process_frames(
                    raw,
                    params,
                    lambda x, s: self.task_queue.put(("progress", x * 0.45, s)),
                    calibration_frames=raw,
                )
                self.task_queue.put(("loaded", meta, raw, processed))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _schedule_rebuild(self):
        if self.processed_frames is None:
            return
        if self.rebuild_job is not None:
            try:
                self.after_cancel(self.rebuild_job)
            except tk.TclError:
                pass
        self.rebuild_job = self.after(220, self._start_rebuild)

    def _start_rebuild(self):
        if self.processed_frames is None:
            messagebox.showinfo(APP_TITLE, "请先选择并读取采集目录。")
            return
        self.rebuild_job = None
        self.rebuild_button.configure(state="disabled")
        settings = (
            float(self.red_threshold_var.get()), float(self.blue_threshold_var.get()), int(self.diffusion_var.get()),
            float(self.sweep_var.get()), int(self.frame_step_var.get()), int(self.line_step_var.get()),
            int(self.sample_step_var.get()), self.axis_var.get(),
        )
        self.status_var.set("正在按扫描角度和红/蓝区域重建三维图像…")

        def task():
            try:
                points, values, labels, info = reconstruct_labeled_point_cloud(
                    self.processed_frames, self.metadata,
                    red_threshold=settings[0], blue_threshold=settings[1], diffusion_steps=settings[2],
                    fallback_sweep_deg=settings[3], frame_step=settings[4], line_step=settings[5],
                    sample_step=settings[6], rotation_axis=settings[7],
                    progress=lambda x, s: self.task_queue.put(("progress", 0.45 + x * 0.55, s)),
                )
                self.task_queue.put(("cloud", points, values, labels, info))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                item = self.task_queue.get_nowait()
                if item[0] == "progress":
                    self.progress["value"] = item[1] * 100
                    self.status_var.set(item[2])
                elif item[0] == "loaded":
                    self.metadata, self.raw_frames, self.processed_frames = item[1], item[2], item[3]
                    measured = self.metadata.get("angle3DRange")
                    angle_text = f"{measured[0]:.2f}°–{measured[1]:.2f}°" if measured else "缺失（使用备用角度）"
                    integrity = "通过" if self.metadata["integrity"].get("ok") else "未通过/未提供"
                    self.summary_var.set(
                        f'{self.metadata.get("probeType")}｜{self.metadata["frameCount"]}帧｜'
                        f'逐帧角度 {angle_text}｜扇扫 {self.metadata["scanAngleDeg"]:.2f}°｜完整性 {integrity}'
                    )
                    if not measured:
                        self.sweep_var.set(max(10.0, min(180.0, self.metadata["scanAngleDeg"])))
                    self.status_var.set("目录读取完成，正在生成红/蓝三维图像…")
                    self._start_rebuild()
                elif item[0] == "cloud":
                    self.points, self.intensity, self.labels, self.cloud_info = item[1], item[2], item[3], item[4]
                    self.rebuild_button.configure(state="normal")
                    self.progress["value"] = 0
                    self.status_var.set(
                        f'完成：{len(self.points):,} 点｜红 {np.count_nonzero(self.labels == 1):,}｜'
                        f'蓝 {np.count_nonzero(self.labels == 2):,}｜角度来源：{self.cloud_info["angleSource"]}'
                    )
                    self._schedule_render()
                elif item[0] == "error":
                    self.rebuild_button.configure(state="normal")
                    self.progress["value"] = 0
                    self.status_var.set("处理失败。")
                    messagebox.showerror(APP_TITLE, item[1])
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _schedule_render(self):
        if self.points is None:
            return
        if self.render_job is not None:
            try:
                self.after_cancel(self.render_job)
            except tk.TclError:
                pass
        self.render_job = self.after(25, self._render)

    def _render_image(self, width: int, height: int) -> Image.Image:
        return render_labeled_point_cloud(
            self.points, self.intensity, self.labels, width, height,
            self.yaw, self.pitch, self.zoom, int(self.point_size_var.get()),
        )

    def _render(self):
        self.render_job = None
        if self.points is None:
            return
        image = self._render_image(max(200, self.canvas.winfo_width()), max(200, self.canvas.winfo_height()))
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("volume")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw", tags="volume")

    def _save_view(self):
        if self.points is None:
            messagebox.showinfo(APP_TITLE, "请先生成三维图像。")
            return
        name = filedialog.asksaveasfilename(
            title="保存当前三维视图", defaultextension=".png", filetypes=(("PNG 图像", "*.png"),)
        )
        if name:
            self._render_image(1600, 1200).save(name)
            self.status_var.set(f"当前三维视图已保存：{name}")

    def _drag_start(self, event):
        self.drag_origin = (event.x, event.y, self.yaw, self.pitch)

    def _drag_move(self, event):
        if self.drag_origin is None:
            return
        x0, y0, yaw0, pitch0 = self.drag_origin
        self.yaw = yaw0 + (event.x - x0) * 0.45
        self.pitch = max(-89.0, min(89.0, pitch0 + (event.y - y0) * 0.45))
        self._schedule_render()

    def _mousewheel(self, event):
        self.zoom = max(0.25, min(5.0, self.zoom * (1.10 if event.delta > 0 else 1 / 1.10)))
        self._schedule_render()

    def _reset_view(self):
        self.yaw, self.pitch, self.zoom = -25.0, -12.0, 1.0
        self._schedule_render()

    def _toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate
        self.rotate_button.configure(text="停止旋转" if self.auto_rotate else "自动旋转")
        if self.auto_rotate:
            self._auto_rotate_step()

    def _auto_rotate_step(self):
        if not self.auto_rotate:
            return
        self.yaw = (self.yaw + 1.2) % 360.0
        self._schedule_render()
        self.after(45, self._auto_rotate_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--params-json")
    args = parser.parse_args()
    Ultrasound3DViewer(args.folder, _params_from_json(args.params_json)).mainloop()


if __name__ == "__main__":
    main()
