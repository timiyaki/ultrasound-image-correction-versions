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
    add_diffused_dual_outline,
    diffused_region_mask,
    dsc_scan_convert,
    process_frames,
    read_capture_metadata,
    read_frames,
)


APP_TITLE = "三维超声高速连续体查看器 Image v19"


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
    interframe_fill: int = 2,
    interpolation_mode: str = "smooth",
    rotation_axis: str = "depth",
    max_points: int = 360_000,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a filled 3-D cloud whose classes match the 2-D red/blue regions."""
    if frames.ndim != 3 or not frames.size:
        raise ValueError("没有可用于三维重建的帧数据。")
    if not 0 <= red_threshold < 255 or not 0 <= blue_threshold <= 255:
        raise ValueError("红/蓝阈值必须位于 0–255。")
    effective_blue = max(float(blue_threshold), float(red_threshold) + 1.0)
    angles, angle_source = resolve_sweep_angles(metadata, len(frames), fallback_sweep_deg)

    frame_ids = np.arange(0, len(frames), max(1, int(frame_step)), dtype=np.int32)
    if frame_ids[-1] != len(frames) - 1:
        frame_ids = np.append(frame_ids, len(frames) - 1)
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
    source_blocks: list[np.ndarray] = []
    red_voxels = 0
    blue_voxels = 0
    fill_count = max(0, min(8, int(interframe_fill)))
    virtual_slices: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    frame_float = frames.astype(np.float32)
    # Diffusion is the expensive part. Compute it only on acquired frames,
    # then interpolate the resulting region probabilities together with image
    # intensity. This is much faster than diffusing every synthetic slice.
    red_masks: dict[int, np.ndarray] = {}
    blue_masks: dict[int, np.ndarray] = {}
    for mask_order, frame_id_value in enumerate(frame_ids):
        frame_id = int(frame_id_value)
        red_masks[frame_id] = diffused_region_mask(
            frames[frame_id], red_threshold, diffusion_steps
        ).astype(np.float32)
        blue_masks[frame_id] = diffused_region_mask(
            frames[frame_id], effective_blue, diffusion_steps
        ).astype(np.float32)
        if progress and (mask_order % 8 == 0 or mask_order + 1 == len(frame_ids)):
            progress(
                0.25 * (mask_order + 1) / len(frame_ids),
                f"正在计算真实帧区域 {mask_order + 1}/{len(frame_ids)}",
            )

    def blend(p0, p1, p2, p3, t):
        if interpolation_mode == "smooth" and fill_count:
            t2, t3 = t * t, t * t * t
            return 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
        return p1 * (1.0 - t) + p2 * t

    for order in range(len(frame_ids) - 1):
        frame_id = int(frame_ids[order])
        next_id = int(frame_ids[order + 1])
        previous_id = int(frame_ids[max(0, order - 1)])
        following_id = int(frame_ids[min(len(frame_ids) - 1, order + 2)])
        for sub in range(fill_count + 1):
            t = sub / float(fill_count + 1)
            image_f = blend(
                frame_float[previous_id], frame_float[frame_id],
                frame_float[next_id], frame_float[following_id], t,
            )
            red_f = blend(
                red_masks[previous_id], red_masks[frame_id],
                red_masks[next_id], red_masks[following_id], t,
            )
            blue_f = blend(
                blue_masks[previous_id], blue_masks[frame_id],
                blue_masks[next_id], blue_masks[following_id], t,
            )
            image = np.clip(image_f, 0, 255).astype(np.uint8)
            red_full = (red_f >= 0.45) & (image >= max(0.0, red_threshold - 30.0))
            blue_full = (blue_f >= 0.45) & (image >= max(0.0, effective_blue - 30.0))
            source_position = frame_id * (1.0 - t) + next_id * t
            angle = float(angles[frame_id] * (1.0 - t) + angles[next_id] * t)
            virtual_slices.append((image, red_full, blue_full, source_position, angle))
    last_id = int(frame_ids[-1])
    virtual_slices.append(
        (frames[last_id], red_masks[last_id].astype(bool), blue_masks[last_id].astype(bool),
         float(last_id), float(angles[last_id]))
    )

    for order, (image, red_full, blue_full, source_position, angle) in enumerate(virtual_slices):
        red = red_full[np.ix_(line_ids, sample_ids)]
        blue = blue_full[np.ix_(line_ids, sample_ids)]
        keep = red | blue
        if np.any(keep):
            x2 = lateral[keep]
            z2 = depth[keep]
            phi = math.radians(angle - center_angle)
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
            source_blocks.append(np.full(len(labels), source_position, dtype=np.float32))
            red_voxels += int(np.count_nonzero(labels == 1))
            blue_voxels += int(np.count_nonzero(labels == 2))
        if progress and (order % 6 == 0 or order + 1 == len(virtual_slices)):
            progress(
                0.25 + 0.75 * (order + 1) / len(virtual_slices),
                f"正在生成连续体切片 {order + 1}/{len(virtual_slices)}",
            )

    if not point_blocks:
        raise ValueError("当前红色阈值过高，没有可用于三维重建的区域。")
    points = np.concatenate(point_blocks, axis=0)
    intensity = np.concatenate(intensity_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    source_positions = np.concatenate(source_blocks, axis=0)
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
        rng = np.random.default_rng(20260921)
        if red_budget:
            chosen.append(rng.choice(red_idx, size=red_budget, replace=False))
        if blue_budget:
            chosen.append(rng.choice(blue_idx, size=blue_budget, replace=False))
        keep_idx = np.sort(np.concatenate(chosen))
        points, intensity, labels, source_positions = (
            points[keep_idx], intensity[keep_idx], labels[keep_idx], source_positions[keep_idx]
        )

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
        "virtualSliceCount": len(virtual_slices),
        "interframeFill": fill_count,
        "interpolationMode": interpolation_mode,
        "redThreshold": float(red_threshold),
        "blueThreshold": effective_blue,
        "diffusionSteps": int(diffusion_steps),
        "redCandidateCount": red_voxels,
        "blueCandidateCount": blue_voxels,
        "candidatePointCount": candidate_count,
        "displayPointCount": len(points),
        "rotationAxis": rotation_axis,
        "angleCenterDeg": center_angle,
        "angleValues": angles.tolist(),
        "deadRadius": dead_radius,
        "outerRadius": dead_radius + frames.shape[2] - 1.0,
        "normalizationCenter": center.tolist(),
        "normalizationScale": scale,
        "frameStep": int(frame_step),
        "physicalBounds": {"low": low.tolist(), "high": high.tolist(), "sampleScale": sample_scale},
    }
    return points.astype(np.float32), intensity, labels, source_positions, info


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


def _outline_from_region(region: np.ndarray) -> np.ndarray:
    region = region.astype(bool)
    padded = np.pad(region, ((1, 1), (1, 1)), mode="constant")
    eroded = np.ones_like(region)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + region.shape[0], dx : dx + region.shape[1]]
    return region & ~eroded


def _build_dsc_nearest_lookup(
    params: ProcessParams,
    line_count: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute a fast nearest-neighbor mapping for annotation masks."""
    half_angle = math.radians(params.dsc_angle_deg) / 2.0
    outer_radius = params.dsc_inner_radius + sample_count - 1.0
    lateral_limit = outer_radius * math.sin(half_angle)
    z = np.linspace(0.0, outer_radius, params.dsc_output_height, dtype=np.float32)[:, None]
    x = np.linspace(-lateral_limit, lateral_limit, params.dsc_output_width, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x + z * z)
    angle = np.arctan2(x, np.maximum(z, 1e-8))
    source_line = (angle + half_angle) * (line_count - 1) / (2.0 * half_angle)
    source_sample = radius - params.dsc_inner_radius
    valid = (
        (np.abs(angle) <= half_angle)
        & (source_sample >= 0.0)
        & (source_sample <= sample_count - 1.0)
    )
    line_index = np.clip(np.rint(source_line).astype(np.int32), 0, line_count - 1)
    sample_index = np.clip(np.rint(source_sample).astype(np.int32), 0, sample_count - 1)
    return line_index, sample_index, valid


def _project_points(
    points: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
    zoom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotated = points @ matrix.T
    base_scale = 0.88 * min(width, height)
    u = width / 2.0 + rotated[:, 0] * base_scale * zoom
    v = height / 2.0 - rotated[:, 2] * base_scale * zoom
    return rotated, u, v


def _slice_polygon(info: dict, frame_index: int) -> np.ndarray | None:
    angles = np.asarray(info.get("angleValues") or [], dtype=np.float32)
    if not len(angles):
        return None
    index = max(0, min(len(angles) - 1, int(frame_index)))
    phi = math.radians(float(angles[index]) - float(info["angleCenterDeg"]))
    scan_half = math.radians(float(info["scanAngleDeg"])) / 2.0
    inner = float(info["deadRadius"])
    outer = float(info["outerRadius"])
    sample_scale = float(info["physicalBounds"]["sampleScale"])
    theta_outer = np.linspace(-scan_half, scan_half, 48, dtype=np.float32)
    theta_inner = np.linspace(scan_half, -scan_half, 48, dtype=np.float32)
    theta = np.concatenate((theta_outer, theta_inner))
    radius = np.concatenate(
        (np.full_like(theta_outer, outer), np.full_like(theta_inner, inner))
    ) * sample_scale
    lateral = np.sin(theta) * radius
    depth = np.cos(theta) * radius
    if info["rotationAxis"] == "depth":
        physical = np.column_stack(
            (lateral * math.cos(phi), lateral * math.sin(phi), depth)
        )
    else:
        physical = np.column_stack(
            (lateral, depth * math.sin(phi), depth * math.cos(phi))
        )
    center = np.asarray(info["normalizationCenter"], dtype=np.float32)
    scale = max(float(info["normalizationScale"]), 1e-6)
    return ((physical - center) / scale).astype(np.float32)


def _draw_xyz_axes(image: Image.Image, matrix: np.ndarray) -> None:
    """Draw a view-linked XYZ orientation triad in the lower-left corner."""
    draw = ImageDraw.Draw(image)
    origin = np.array([82.0, image.height - 72.0], dtype=np.float32)
    vectors = matrix @ np.eye(3, dtype=np.float32)
    colors = ((255, 70, 70), (70, 235, 100), (70, 145, 255))
    names = ("X", "Y", "Z")
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill=(238, 238, 238))
    for vector, color, name in zip(vectors.T, colors, names):
        endpoint = origin + np.array((vector[0], -vector[2]), dtype=np.float32) * 48.0
        draw.line((origin[0], origin[1], endpoint[0], endpoint[1]), fill=color, width=4)
        direction = endpoint - origin
        length = max(float(np.linalg.norm(direction)), 1.0)
        unit = direction / length
        side = np.array((-unit[1], unit[0]), dtype=np.float32)
        tip1 = endpoint - unit * 9.0 + side * 4.0
        tip2 = endpoint - unit * 9.0 - side * 4.0
        draw.polygon((tuple(endpoint), tuple(tip1), tuple(tip2)), fill=color)
        draw.text((endpoint[0] + 4, endpoint[1] - 8), name, fill=color)


def prepare_point_cloud_render(
    points: np.ndarray,
    intensity: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
    point_size: int,
) -> dict:
    """Render the expensive view-dependent red/blue base once."""
    width, height = max(200, width), max(200, height)
    matrix = rotation_matrix(yaw, pitch)
    rotated, projected_u, projected_v = _project_points(points, matrix, width, height, zoom)
    u = np.rint(projected_u).astype(np.int32)
    v = np.rint(projected_v).astype(np.int32)
    visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    source_indices = np.flatnonzero(visible)
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
    return {
        "image": Image.fromarray(rgb, mode="RGB"),
        "matrix": matrix,
        "u": u,
        "v": v,
        "values": values,
        "labels": shown_labels,
        "source_indices": source_indices,
        "width": width,
        "height": height,
        "zoom": zoom,
        "point_size": int(point_size),
    }


def compose_point_cloud_render(
    base: dict,
    source_positions: np.ndarray | None = None,
    highlighted_frame: int | None = None,
    cloud_info: dict | None = None,
) -> Image.Image:
    """Add the inexpensive linked section, legend and XYZ triad to a cached base."""
    image = base["image"].copy()
    shown_labels = base["labels"]
    blue = shown_labels == 2
    if source_positions is not None and highlighted_frame is not None:
        shown_positions = source_positions[base["source_indices"]]
        tolerance = 0.18
        if cloud_info:
            tolerance = max(tolerance, float(cloud_info.get("frameStep", 1)) * 0.18)
        section = (np.abs(shown_positions - float(highlighted_frame)) <= tolerance) & ~blue
        if np.any(section):
            rgb = np.asarray(image).copy()
            selected_u = base["u"][section]
            selected_v = base["v"][section]
            radius = max(0, min(2, base["point_size"] - 1))
            for dy in range(-radius, radius + 1):
                yy = np.clip(selected_v + dy, 0, base["height"] - 1)
                for dx in range(-radius, radius + 1):
                    xx = np.clip(selected_u + dx, 0, base["width"] - 1)
                    rgb[yy, xx, 0] = 255
                    rgb[yy, xx, 1] = 220
                    rgb[yy, xx, 2] = 35
            image = Image.fromarray(rgb, mode="RGB")

    if highlighted_frame is not None and cloud_info:
        polygon = _slice_polygon(cloud_info, highlighted_frame)
        if polygon is not None:
            _, pu, pv = _project_points(
                polygon, base["matrix"], base["width"], base["height"], base["zoom"]
            )
            xy = [(float(x), float(y)) for x, y in zip(pu, pv)]
            ImageDraw.Draw(image).line(xy + [xy[0]], fill=(255, 220, 60), width=2, joint="curve")
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 14, 310, 86), fill=(5, 7, 10), outline=(70, 75, 82))
    draw.line((27, 31, 57, 31), fill=(255, 45, 45), width=4)
    draw.text((66, 22), "Red: overall marked region", fill=(230, 230, 230))
    draw.line((27, 51, 57, 51), fill=(35, 150, 255), width=5)
    draw.text((66, 42), "Blue: strong echo (highlighted)", fill=(230, 230, 230))
    draw.line((27, 71, 57, 71), fill=(255, 220, 60), width=4)
    draw.text((66, 62), "Yellow: linked DSC cross-section", fill=(230, 230, 230))
    _draw_xyz_axes(image, base["matrix"])
    return image


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
    source_positions: np.ndarray | None = None,
    highlighted_frame: int | None = None,
    cloud_info: dict | None = None,
) -> Image.Image:
    """Convenience wrapper for offline exports and validation."""
    base = prepare_point_cloud_render(
        points, intensity, labels, width, height, yaw, pitch, zoom, point_size
    )
    return compose_point_cloud_render(base, source_positions, highlighted_frame, cloud_info)


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
        self.dsc_frames: np.ndarray | None = None
        self.slice_dsc_params: ProcessParams | None = None
        self.slice_dsc_lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.metadata: dict = {}
        self.points: np.ndarray | None = None
        self.intensity: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.source_positions: np.ndarray | None = None
        self.cloud_info: dict = {}
        self.highlighted_frame = 0
        self.slice_window: tk.Toplevel | None = None
        self.slice_photos: list[ImageTk.PhotoImage] = []
        self.slice_resize_job = None
        self.slice_panel_cache: dict[tuple, tuple[Image.Image, Image.Image]] = {}
        self.yaw, self.pitch, self.zoom = -25.0, -12.0, 1.0
        self.drag_origin = None
        self.render_job = None
        self.rebuild_job = None
        self.wheel_finish_job = None
        self.is_interacting = False
        self.interaction_indices: np.ndarray | None = None
        self.render_base_cache: dict | None = None
        self.render_base_key: tuple | None = None
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
        self.point_size_var = tk.IntVar(value=3)
        self.interframe_fill_var = tk.IntVar(value=2)
        self.interpolation_var = tk.StringVar(value="平滑曲度")
        self.axis_var = tk.StringVar(value="depth")
        self._scale(quality, 0, "备用扫描总角度", self.sweep_var, 10, 180, 1)
        self._scale(quality, 1, "帧间隔", self.frame_step_var, 1, 5, 1)
        self._scale(quality, 2, "扫描线间隔", self.line_step_var, 1, 6, 1)
        self._scale(quality, 3, "深度采样间隔", self.sample_step_var, 1, 6, 1)
        self._scale(quality, 4, "连续帧间填充切片", self.interframe_fill_var, 0, 5, 1)
        self._scale(quality, 5, "显示点大小", self.point_size_var, 1, 4, 1, render_only=True)
        ttk.Label(quality, text="帧间插值").grid(row=6, column=0, sticky="w", pady=4)
        interpolation = ttk.Combobox(
            quality,
            textvariable=self.interpolation_var,
            values=("平滑曲度", "线性平均"),
            state="readonly",
            width=12,
        )
        interpolation.grid(row=6, column=1, sticky="ew", pady=4)
        interpolation.bind("<<ComboboxSelected>>", lambda _e: self._schedule_rebuild())
        ttk.Label(quality, text="旋转扫描轴").grid(row=7, column=0, sticky="w", pady=4)
        axis = ttk.Combobox(quality, textvariable=self.axis_var, values=("depth", "lateral"), state="readonly", width=12)
        axis.grid(row=7, column=1, sticky="ew", pady=4)
        axis.bind("<<ComboboxSelected>>", lambda _e: self._schedule_rebuild())
        ttk.Label(
            quality,
            text="有 angle3D 时优先逐帧读取；缺失时才按帧数和备用角度均匀分布。",
            wraplength=350,
            foreground="#3a6f8f",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.rebuild_button = ttk.Button(panel, text="生成/更新三维图像", command=self._start_rebuild)
        self.rebuild_button.pack(fill="x")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="重置视角", command=self._reset_view).pack(side="left", expand=True, fill="x")
        self.rotate_button = ttk.Button(row, text="自动旋转", command=self._toggle_auto_rotate)
        self.rotate_button.pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(row, text="保存当前视图", command=self._save_view).pack(side="left", expand=True, fill="x")
        ttk.Button(panel, text="打开联动 DSC 切片窗口", command=self._open_slice_window).pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(panel, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 3))
        self.status_var = tk.StringVar(value="等待选择目录。")
        ttk.Label(panel, textvariable=self.status_var, wraplength=360).pack(fill="x")

        ttk.Label(
            view,
            text="拖动时快速预览、松开后恢复完整质量｜左下角 XYZ 方向轴｜黄色为联动 DSC 截面",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")
        self.canvas = tk.Canvas(view, background="#050608", highlightthickness=1, highlightbackground="#333333")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(7, 0))
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
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
                    lambda x, s: self.task_queue.put(("progress", x * 0.24, s)),
                    calibration_frames=raw,
                )
                if self.source_params.dsc_enabled:
                    display_params = replace(
                        self.source_params,
                        width=meta["sampleCount"], height=meta["lineCount"], frame_count=meta["frameCount"],
                        offset_bytes=0, transpose_output=False,
                    )
                else:
                    display_params = replace(
                        params,
                        dsc_enabled=True,
                        dsc_angle_deg=float(meta["scanAngleDeg"]),
                        dsc_inner_radius=float(meta["deadRadius"]),
                        dsc_output_width=int(meta["suggestedDscWidth"]),
                        dsc_output_height=int(meta["suggestedDscHeight"]),
                    )
                # Reuse the already denoised/tone-corrected native frames.
                # This avoids running the expensive filter pipeline a second time.
                dsc_frames = dsc_scan_convert(
                    processed,
                    display_params,
                    lambda x, s: self.task_queue.put(("progress", 0.24 + x * 0.26, s)),
                )
                dsc_lookup = _build_dsc_nearest_lookup(
                    display_params, meta["lineCount"], meta["sampleCount"]
                )
                self.task_queue.put(("loaded", meta, raw, processed, dsc_frames, display_params, dsc_lookup))
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
        self.slice_panel_cache.clear()
        settings = (
            float(self.red_threshold_var.get()), float(self.blue_threshold_var.get()), int(self.diffusion_var.get()),
            float(self.sweep_var.get()), int(self.frame_step_var.get()), int(self.line_step_var.get()),
            int(self.sample_step_var.get()), int(self.interframe_fill_var.get()),
            ("smooth" if self.interpolation_var.get() == "平滑曲度" else "linear"),
            self.axis_var.get(),
        )
        self.status_var.set("正在按扫描角度和红/蓝区域重建三维图像…")

        def task():
            try:
                points, values, labels, positions, info = reconstruct_labeled_point_cloud(
                    self.processed_frames, self.metadata,
                    red_threshold=settings[0], blue_threshold=settings[1], diffusion_steps=settings[2],
                    fallback_sweep_deg=settings[3], frame_step=settings[4], line_step=settings[5],
                    sample_step=settings[6], interframe_fill=settings[7], interpolation_mode=settings[8],
                    rotation_axis=settings[9],
                    progress=lambda x, s: self.task_queue.put(("progress", 0.50 + x * 0.50, s)),
                )
                self.task_queue.put(("cloud", points, values, labels, positions, info))
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
                    self.metadata, self.raw_frames, self.processed_frames, self.dsc_frames, self.slice_dsc_params, self.slice_dsc_lookup = (
                        item[1], item[2], item[3], item[4], item[5], item[6]
                    )
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
                    self.points, self.intensity, self.labels, self.source_positions, self.cloud_info = (
                        item[1], item[2], item[3], item[4], item[5]
                    )
                    self._prepare_interaction_subset()
                    self.render_base_cache = None
                    self.render_base_key = None
                    self.rebuild_button.configure(state="normal")
                    self.progress["value"] = 0
                    self.status_var.set(
                        f'完成：{len(self.points):,} 点｜红 {np.count_nonzero(self.labels == 1):,}｜'
                        f'蓝 {np.count_nonzero(self.labels == 2):,}｜连续切片 {self.cloud_info["virtualSliceCount"]}｜'
                        f'角度来源：{self.cloud_info["angleSource"]}'
                    )
                    self._schedule_render()
                    self._update_slice_window()
                    if self.slice_window is None:
                        self._open_slice_window()
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
        self.render_job = self.after(16 if (self.is_interacting or self.auto_rotate) else 25, self._render)

    def _prepare_interaction_subset(self):
        if self.points is None:
            self.interaction_indices = None
            return
        limit = 55_000
        if len(self.points) <= limit:
            self.interaction_indices = np.arange(len(self.points), dtype=np.int64)
            return
        rng = np.random.default_rng(20260921)
        red = np.flatnonzero(self.labels == 1)
        blue = np.flatnonzero(self.labels == 2)
        blue_count = min(len(blue), limit // 3)
        red_count = min(len(red), limit - blue_count)
        blue_count = min(len(blue), limit - red_count)
        parts = []
        if red_count:
            parts.append(rng.choice(red, size=red_count, replace=False))
        if blue_count:
            parts.append(rng.choice(blue, size=blue_count, replace=False))
        self.interaction_indices = np.sort(np.concatenate(parts))

    def _invalidate_render_base(self):
        self.render_base_cache = None
        self.render_base_key = None

    def _render_image(self, width: int, height: int, force_full: bool = False) -> Image.Image:
        fast = bool((self.is_interacting or self.auto_rotate) and not force_full)
        if fast and self.interaction_indices is not None:
            indices = self.interaction_indices
            points = self.points[indices]
            intensity = self.intensity[indices]
            labels = self.labels[indices]
            positions = self.source_positions[indices]
        else:
            points, intensity, labels, positions = (
                self.points, self.intensity, self.labels, self.source_positions
            )
        render_width = max(320, int(width * 0.50)) if fast else int(width)
        render_height = max(240, int(height * 0.50)) if fast else int(height)
        render_point_size = max(1, int(round(float(self.point_size_var.get()) * 0.55))) if fast else int(self.point_size_var.get())
        key = (
            id(self.points), fast, render_width, render_height,
            round(self.yaw, 3), round(self.pitch, 3), round(self.zoom, 4),
            render_point_size,
        )
        if key != self.render_base_key or self.render_base_cache is None:
            self.render_base_cache = prepare_point_cloud_render(
                points, intensity, labels, render_width, render_height,
                self.yaw, self.pitch, self.zoom, render_point_size,
            )
            self.render_base_key = key
        image = compose_point_cloud_render(
            self.render_base_cache,
            positions,
            self.highlighted_frame,
            self.cloud_info,
        )
        if fast and image.size != (int(width), int(height)):
            image = image.resize((int(width), int(height)), Image.Resampling.NEAREST)
        return image

    def _render(self):
        self.render_job = None
        if self.points is None:
            return
        image = self._render_image(max(200, self.canvas.winfo_width()), max(200, self.canvas.winfo_height()))
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("volume")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw", tags="volume")

    def _open_slice_window(self):
        if self.dsc_frames is None:
            messagebox.showinfo(APP_TITLE, "请先读取采集目录并完成 DSC 处理。")
            return
        if self.slice_window is not None and self.slice_window.winfo_exists():
            self.slice_window.deiconify()
            self.slice_window.lift()
            self._update_slice_window()
            return
        window = tk.Toplevel(self)
        self.slice_window = window
        window.title("DSC＋调色切片联动：未标注 / 红蓝标注")
        window.geometry("1160x790")
        window.minsize(900, 620)
        window.columnconfigure((0, 1), weight=1)
        window.rowconfigure(1, weight=1)
        ttk.Label(window, text="DSC校正＋统一调色（未标注）", anchor="center").grid(row=0, column=0, sticky="ew", pady=(10, 4))
        ttk.Label(window, text="DSC校正＋统一调色＋红蓝标注", anchor="center").grid(row=0, column=1, sticky="ew", pady=(10, 4))
        self.slice_plain_label = ttk.Label(window, anchor="center")
        self.slice_plain_label.grid(row=1, column=0, sticky="nsew", padx=(10, 5))
        self.slice_marked_label = ttk.Label(window, anchor="center")
        self.slice_marked_label.grid(row=1, column=1, sticky="nsew", padx=(5, 10))

        nav = ttk.Frame(window, padding=10)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew")
        nav.columnconfigure(2, weight=1)
        ttk.Button(nav, text="上一帧", command=lambda: self._step_slice(-1)).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(nav, text="下一帧", command=lambda: self._step_slice(1)).grid(row=0, column=1, padx=(0, 8))
        self.slice_frame_var = tk.DoubleVar(value=float(self.highlighted_frame))
        self.slice_scale = ttk.Scale(
            nav,
            from_=0,
            to=max(0, len(self.dsc_frames) - 1),
            variable=self.slice_frame_var,
            command=self._on_slice_change,
        )
        self.slice_scale.grid(row=0, column=2, sticky="ew")
        self.slice_text_var = tk.StringVar()
        ttk.Label(nav, textvariable=self.slice_text_var, width=34, anchor="e").grid(row=0, column=3, padx=(8, 0))
        ttk.Label(
            nav,
            text="拖动滑块时，三维窗口中的黄色扇形截面同步移动；截面内的红色点变为黄色，蓝色强反声仍保持蓝色。",
            foreground="#826300",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        window.protocol("WM_DELETE_WINDOW", self._close_slice_window)
        window.bind("<Configure>", self._schedule_slice_update)
        self._update_slice_window()

    def _schedule_slice_update(self, _event=None):
        if self.slice_resize_job is not None:
            try:
                self.after_cancel(self.slice_resize_job)
            except tk.TclError:
                pass
        self.slice_resize_job = self.after(55, self._update_slice_window)

    def _close_slice_window(self):
        if self.slice_window is not None:
            self.slice_window.destroy()
        self.slice_window = None
        self.slice_photos = []
        self.slice_resize_job = None

    def _on_slice_change(self, value):
        if self.dsc_frames is None:
            return
        index = max(0, min(len(self.dsc_frames) - 1, int(round(float(value)))))
        if index == self.highlighted_frame and self.slice_photos:
            return
        self.highlighted_frame = index
        self.slice_frame_var.set(float(index))
        self._schedule_slice_update()
        self._schedule_render()

    def _step_slice(self, delta: int):
        if self.dsc_frames is None:
            return
        index = max(0, min(len(self.dsc_frames) - 1, self.highlighted_frame + int(delta)))
        self.highlighted_frame = index
        self.slice_frame_var.set(float(index))
        self._update_slice_window()
        self._schedule_render()

    @staticmethod
    def _fit_slice_image(frame: np.ndarray, width: int, height: int) -> Image.Image:
        image = Image.fromarray(frame) if frame.ndim == 3 else Image.fromarray(frame, mode="L").convert("RGB")
        scale = min(width / image.width, height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (width, height), (0, 0, 0))
        panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        return panel

    def _fast_marked_slice(self, index: int) -> np.ndarray:
        plain = self.dsc_frames[index]
        if self.processed_frames is None or self.slice_dsc_lookup is None:
            return add_diffused_dual_outline(
                plain,
                float(self.red_threshold_var.get()),
                float(self.blue_threshold_var.get()),
                int(self.diffusion_var.get()),
            )
        red_threshold = float(self.red_threshold_var.get())
        blue_threshold = max(float(self.blue_threshold_var.get()), red_threshold + 1.0)
        steps = int(self.diffusion_var.get())
        native = self.processed_frames[index]
        native_regions = np.stack(
            (
                diffused_region_mask(native, red_threshold, steps),
                diffused_region_mask(native, blue_threshold, steps),
            )
        ).astype(np.uint8) * 255
        line_index, sample_index, valid = self.slice_dsc_lookup
        mapped = native_regions[:, line_index, sample_index] >= 128
        mapped &= valid[None, ...]
        red_outline = _outline_from_region(mapped[0])
        blue_outline = _outline_from_region(mapped[1])
        rgb = np.repeat(plain[..., None], 3, axis=2)
        rgb[red_outline] = (255, 32, 32)
        rgb[blue_outline] = (32, 112, 255)
        return rgb

    def _update_slice_window(self):
        self.slice_resize_job = None
        if self.slice_window is None or not self.slice_window.winfo_exists() or self.dsc_frames is None:
            return
        index = max(0, min(len(self.dsc_frames) - 1, self.highlighted_frame))
        plain = self.dsc_frames[index]
        panel_width = max(380, (self.slice_window.winfo_width() - 40) // 2)
        panel_height = max(420, self.slice_window.winfo_height() - 145)
        cache_key = (
            index, int(round(float(self.red_threshold_var.get()))),
            int(round(float(self.blue_threshold_var.get()))), int(self.diffusion_var.get()),
            panel_width, panel_height,
        )
        images = self.slice_panel_cache.get(cache_key)
        if images is None:
            marked = self._fast_marked_slice(index)
            images = (
                self._fit_slice_image(plain, panel_width, panel_height),
                self._fit_slice_image(marked, panel_width, panel_height),
            )
            self.slice_panel_cache[cache_key] = images
            while len(self.slice_panel_cache) > 12:
                self.slice_panel_cache.pop(next(iter(self.slice_panel_cache)))
        self.slice_photos = [ImageTk.PhotoImage(image) for image in images]
        self.slice_plain_label.configure(image=self.slice_photos[0])
        self.slice_marked_label.configure(image=self.slice_photos[1])
        angles = self.metadata.get("angle3DValues") or []
        angle_text = f"｜angle3D {angles[index]:.3f}°" if index < len(angles) else ""
        self.slice_text_var.set(f"第 {index + 1} / {len(self.dsc_frames)} 帧{angle_text}")

    def _save_view(self):
        if self.points is None:
            messagebox.showinfo(APP_TITLE, "请先生成三维图像。")
            return
        name = filedialog.asksaveasfilename(
            title="保存当前三维视图", defaultextension=".png", filetypes=(("PNG 图像", "*.png"),)
        )
        if name:
            self._render_image(1600, 1200, force_full=True).save(name)
            self.status_var.set(f"当前三维视图已保存：{name}")

    def _drag_start(self, event):
        self.is_interacting = True
        self.drag_origin = (event.x, event.y, self.yaw, self.pitch)

    def _drag_move(self, event):
        if self.drag_origin is None:
            return
        x0, y0, yaw0, pitch0 = self.drag_origin
        self.yaw = yaw0 + (event.x - x0) * 0.45
        self.pitch = max(-89.0, min(89.0, pitch0 + (event.y - y0) * 0.45))
        self._schedule_render()

    def _drag_end(self, _event=None):
        self.is_interacting = False
        self.drag_origin = None
        self._invalidate_render_base()
        self._schedule_render()

    def _mousewheel(self, event):
        self.is_interacting = True
        self.zoom = max(0.25, min(5.0, self.zoom * (1.10 if event.delta > 0 else 1 / 1.10)))
        self._schedule_render()
        if self.wheel_finish_job is not None:
            try:
                self.after_cancel(self.wheel_finish_job)
            except tk.TclError:
                pass
        self.wheel_finish_job = self.after(140, self._finish_wheel_interaction)

    def _finish_wheel_interaction(self):
        self.wheel_finish_job = None
        if self.auto_rotate:
            return
        self.is_interacting = False
        self._invalidate_render_base()
        self._schedule_render()

    def _reset_view(self):
        self.yaw, self.pitch, self.zoom = -25.0, -12.0, 1.0
        self.is_interacting = False
        self._invalidate_render_base()
        self._schedule_render()

    def _toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate
        self.rotate_button.configure(text="停止旋转" if self.auto_rotate else "自动旋转")
        if self.auto_rotate:
            self.is_interacting = True
            self._auto_rotate_step()
        else:
            self.is_interacting = False
            self._invalidate_render_base()
            self._schedule_render()

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
