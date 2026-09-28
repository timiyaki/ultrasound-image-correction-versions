from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk


APP_TITLE = "超声二维校正工具 Image v25 · Lee + 双边滤波"
COMMON_WIDTHS = (128, 192, 256, 320, 384, 480, 512, 640, 768, 800, 1024, 1280, 1920)
CAPTURE_FILENAMES = ("raw_frames_u8.bin", "frames.json", "scan.snp", "integrity.json")


@dataclass
class ProcessParams:
    width: int = 512
    height: int = 96
    frame_count: int = 0
    offset_bytes: int = 0
    black_level: float = 0.0
    white_percentile: float = 99.8
    alpha: float = 1.10
    beta: float = 0.0
    gamma: float = 0.92
    lee_window: int = 5
    noise_percentile: float = 25.0
    original_blend: float = 0.15
    despeckle_strength: float = 0.60
    bilateral_strength: float = 0.45
    bilateral_radius: int = 2
    bilateral_sigma: float = 0.08
    edge_gain: float = 0.35
    red_outline_enabled: bool = True
    red_outline_threshold: float = 80.0
    blue_outline_threshold: float = 170.0
    red_outline_diffusion_steps: int = 5
    shadow_fill_enabled: bool = False
    shadow_fill_strength: float = 0.45
    dsc_enabled: bool = False
    dsc_angle_deg: float = 60.0
    dsc_inner_radius: float = 0.0
    dsc_output_width: int = 512
    dsc_output_height: int = 512
    transpose_output: bool = False


def find_capture_directory(selected: Path) -> Path:
    selected = selected.resolve()
    if not selected.is_dir():
        raise ValueError("请选择有效的采集目录。")
    if (selected / "raw_frames_u8.bin").is_file():
        return selected
    candidates = sorted(selected.rglob("raw_frames_u8.bin"))
    complete = [p.parent for p in candidates if (p.parent / "frames.json").is_file()]
    if len(complete) == 1:
        return complete[0]
    if len(complete) > 1:
        raise ValueError("所选目录下发现多组采集数据，请直接选择其中一个 source 文件夹。")
    if len(candidates) == 1:
        return candidates[0].parent
    raise ValueError("目录中没有找到 raw_frames_u8.bin。")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_capture_metadata(selected: Path) -> dict:
    root = find_capture_directory(selected)
    raw_path = root / "raw_frames_u8.bin"
    frames_path = root / "frames.json"
    scan_path = root / "scan.snp"
    integrity_path = root / "integrity.json"
    if not frames_path.is_file():
        raise ValueError("找到 BIN 文件，但缺少 frames.json；可使用旧版单文件尺寸推断方式。")

    manifest = json.loads(frames_path.read_text(encoding="utf-8"))
    frame_items = manifest.get("frames") or []
    if not frame_items:
        raise ValueError("frames.json 中没有帧记录。")
    first = frame_items[0]
    width = int(first["sampleCount"])
    height = int(first["lineCount"])
    frame_count = int(manifest.get("frameCount", len(frame_items)))
    data_type = str(manifest.get("dataType", "UInt8"))
    if data_type.lower() != "uint8":
        raise ValueError(f"当前版本只支持 UInt8，目录中记录的是 {data_type}。")
    expected_bytes = sum(int(x["byteLength"]) for x in frame_items)
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"BIN 大小 {raw_path.stat().st_size:,} 与 frames.json 记录的 {expected_bytes:,} 不一致。"
        )
    uniform_fields = ("lineCount", "sampleCount", "deadRadius", "sampleScale", "scanAngle", "probeType")
    for field in uniform_fields:
        if len({str(x.get(field)) for x in frame_items}) != 1:
            raise ValueError(f"各帧的 {field} 不一致，当前版本不能作为统一序列处理。")
    offsets_ok = all(
        int(item["offset"]) + int(item["byteLength"]) <= raw_path.stat().st_size
        for item in frame_items
    )
    if not offsets_ok:
        raise ValueError("frames.json 中存在越界帧偏移。")
    ordered_frames = sorted(frame_items, key=lambda x: int(x.get("ordinal", 0)))
    cursor = 0
    for item in ordered_frames:
        if int(item["offset"]) != cursor:
            raise ValueError("frames.json 中的帧不是连续顺序存储，当前版本暂不支持该布局。")
        cursor += int(item["byteLength"])

    scan_summary = {}
    if scan_path.is_file():
        outer = json.loads(scan_path.read_text(encoding="utf-8"))
        if outer:
            first_key = min(outer, key=lambda x: int(x) if str(x).isdigit() else str(x))
            raw_text = outer[first_key].get("RAW", "{}")
            snapshot = json.loads(raw_text)
            b_data = snapshot.get("raws", {}).get("b_data", {})
            scan_summary = {
                "snapshotCount": len(outer),
                "transducerKey": snapshot.get("transducerKey"),
                "imagingUnitKey": snapshot.get("imagingUnitKey"),
                "nominalDepth": b_data.get("nominalDepth"),
                "sampleRate": b_data.get("sampleRate"),
                "frequency": b_data.get("frequency"),
                "gain": b_data.get("gain"),
                "dynamicRange": b_data.get("dynamicRange"),
                "harmonic": b_data.get("harmonic"),
                "compound": b_data.get("compound"),
                "enhanceLevel": b_data.get("enhanceLevel"),
                "variableGains": b_data.get("variableGains"),
                "focuses": b_data.get("focuses"),
            }

    integrity = {"available": integrity_path.is_file(), "ok": None, "artifacts": {}}
    if integrity_path.is_file():
        integrity_doc = json.loads(integrity_path.read_text(encoding="utf-8"))
        all_ok = True
        for relative, expected in integrity_doc.get("artifacts", {}).items():
            local = root / Path(relative).name
            exists = local.is_file()
            size_ok = exists and local.stat().st_size == int(expected.get("byteLength", -1))
            hash_ok = exists and file_sha256(local) == expected.get("sha256")
            artifact_ok = bool(exists and size_ok and hash_ok)
            integrity["artifacts"][relative] = {
                "exists": exists,
                "size_ok": bool(size_ok),
                "sha256_ok": bool(hash_ok),
            }
            all_ok = all_ok and artifact_ok
        integrity["ok"] = all_ok

    scan_angle_rad = float(first.get("scanAngle", 0.0))
    scan_angle_deg = math.degrees(scan_angle_rad)
    dead_radius = float(first.get("deadRadius", 0.0))
    outer_radius = dead_radius + width - 1.0
    suggested_width = max(64, int(math.ceil(2.0 * outer_radius * math.sin(scan_angle_rad / 2.0))))
    suggested_height = max(64, int(math.ceil(outer_radius)))
    angle3d = [float(x["angle3D"]) for x in frame_items if x.get("angle3D") is not None]
    return {
        "captureDirectory": str(root),
        "rawFile": str(raw_path),
        "framesFile": str(frames_path),
        "scanFile": str(scan_path) if scan_path.is_file() else None,
        "integrityFile": str(integrity_path) if integrity_path.is_file() else None,
        "dataType": data_type,
        "layout": manifest.get("layout"),
        "frameCount": frame_count,
        "lineCount": height,
        "sampleCount": width,
        "totalBytes": raw_path.stat().st_size,
        "probeType": first.get("probeType"),
        "state": first.get("state"),
        "scanAngleRad": scan_angle_rad,
        "scanAngleDeg": scan_angle_deg,
        "deadRadius": dead_radius,
        "sampleScale": first.get("sampleScale"),
        "angle3DRange": [min(angle3d), max(angle3d)] if angle3d else None,
        "angle3DValues": angle3d,
        "suggestedDscWidth": suggested_width,
        "suggestedDscHeight": suggested_height,
        "scanSnapshot": scan_summary,
        "integrity": integrity,
    }


def validate_params(path: Path, p: ProcessParams) -> int:
    if not path.is_file():
        raise ValueError("请选择有效的 .bin 文件。")
    if p.width <= 0 or p.height <= 0:
        raise ValueError("宽度和高度必须大于 0。")
    if p.offset_bytes < 0:
        raise ValueError("文件头字节数不能小于 0。")
    if p.lee_window not in (3, 5, 7, 9):
        raise ValueError("Lee 窗口必须为 3、5、7 或 9。")
    if not 90.0 <= p.white_percentile <= 100.0:
        raise ValueError("白点百分位应在 90–100 之间。")
    if not 0.1 <= p.alpha <= 5.0:
        raise ValueError("α 对比度增益应在 0.1–5.0 之间。")
    if not -255.0 <= p.beta <= 255.0:
        raise ValueError("β 亮度偏移应在 -255–255 灰度级之间。")
    if not 0.1 <= p.gamma <= 3.0:
        raise ValueError("Gamma 应在 0.1–3.0 之间。")
    if not 0.0 <= p.original_blend <= 1.0:
        raise ValueError("原始细节混合比例应在 0–1 之间。")
    if not 0.0 <= p.despeckle_strength <= 1.0:
        raise ValueError("Lee 去散斑强度应在 0–1 之间。")
    if not 0.0 <= p.bilateral_strength <= 1.0:
        raise ValueError("双边滤波强度应在 0–1 之间。")
    if p.bilateral_radius not in (1, 2, 3):
        raise ValueError("双边滤波半径必须为 1、2 或 3。")
    if not 0.01 <= p.bilateral_sigma <= 0.25:
        raise ValueError("双边灰度相似尺度应在 0.01–0.25 之间。")
    if not 0.0 <= p.edge_gain <= 2.5:
        raise ValueError("主体边界增强应在 0–2.5 之间。")
    if not 40.0 <= p.red_outline_threshold <= 254.0:
        raise ValueError("红色整体区域阈值应在 40–254 之间。")
    if not 128.0 <= p.blue_outline_threshold <= 255.0:
        raise ValueError("蓝色强反声阈值应在 128–255 之间。")
    if not 0 <= p.red_outline_diffusion_steps <= 12:
        raise ValueError("连续扩散次数应在 0–12 之间。")
    if not 0.0 <= p.shadow_fill_strength <= 1.0:
        raise ValueError("骨骼声影补偿强度应在 0–1 之间。")
    if not 1.0 <= p.dsc_angle_deg < 180.0:
        raise ValueError("DSC 扇扫角应在 1–179 度之间。")
    if p.dsc_inner_radius < 0.0:
        raise ValueError("DSC 起始半径不能小于 0。")
    if not 64 <= p.dsc_output_width <= 2048 or not 64 <= p.dsc_output_height <= 2048:
        raise ValueError("DSC 输出宽高应在 64–2048 像素之间。")
    samples = path.stat().st_size - p.offset_bytes
    frame_size = p.width * p.height
    if samples <= 0 or samples % frame_size:
        raise ValueError(
            f"扣除文件头后有 {samples:,} 字节，不能被单帧大小 "
            f"{p.width}×{p.height}={frame_size:,} 整除。"
        )
    inferred_count = samples // frame_size
    if p.frame_count not in (0, inferred_count):
        raise ValueError(f"帧数应为 {inferred_count}；或填 0 让程序自动计算。")
    return inferred_count


def read_frames(path: Path, p: ProcessParams) -> np.ndarray:
    count = validate_params(path, p)
    data = np.fromfile(path, dtype=np.uint8, offset=p.offset_bytes)
    return data.reshape(count, p.height, p.width)


def box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    """Reflect-padded box mean for [N,H,W] float arrays."""
    k = radius * 2 + 1
    padded = np.pad(x, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")
    integ = np.pad(padded, ((0, 0), (1, 0), (1, 0)), mode="constant")
    integ = np.cumsum(np.cumsum(integ, axis=1), axis=2)
    total = (
        integ[:, k:, k:]
        - integ[:, :-k, k:]
        - integ[:, k:, :-k]
        + integ[:, :-k, :-k]
    )
    return total / float(k * k)


def estimate_noise_variance(frames: np.ndarray, p: ProcessParams) -> float:
    radius = p.lee_window // 2
    sample_ids = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    sample = frames[sample_ids].astype(np.float32)
    z = np.log1p(sample) / math.log(256.0)
    mean = box_mean(z, radius)
    var = np.maximum(box_mean(z * z, radius) - mean * mean, 0.0)
    valid = var[sample > max(8.0, p.black_level)]
    if valid.size == 0:
        return float(np.percentile(var, p.noise_percentile))
    return float(np.percentile(valid, p.noise_percentile))




def _shift_batch(frames: np.ndarray, dy: int, dx: int) -> np.ndarray:
    pad_y, pad_x = abs(dy), abs(dx)
    padded = np.pad(frames, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return padded[:, y0 : y0 + frames.shape[1], x0 : x0 + frames.shape[2]]


def bilateral_filter(
    frames: np.ndarray, strength: float, radius: int, sigma: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """One bilateral pass in normalized log intensity; no extra filter pool."""
    if strength <= 0.0:
        return frames.copy()
    spatial_sigma = max(0.8, radius / 1.5)
    center = frames.astype(np.float32, copy=False)
    active = np.ones_like(center, dtype=bool) if active_mask is None else active_mask
    pad = ((0, 0), (radius, radius), (radius, radius))
    padded = np.pad(center, pad, mode="reflect")
    mask_pad = np.pad(active, pad, mode="reflect")
    numerator = np.zeros_like(center)
    denominator = np.zeros_like(center)
    h, w = center.shape[1:]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[:, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            valid = mask_pad[:, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            spatial = math.exp(-(dx*dx + dy*dy)/(2.0*spatial_sigma**2))
            weights = spatial * np.exp(-0.5*((neighbor-center)/sigma)**2) * valid
            numerator += weights * neighbor
            denominator += weights
    smoothed = numerator / np.maximum(denominator, 1e-8)
    result = (1.0-strength)*center + strength*smoothed
    result[~active] = 0.0
    return result


def filter_chunk(frames: np.ndarray, p: ProcessParams, noise_var: float) -> np.ndarray:
    """Exactly two denoising stages: adaptive Lee, then bilateral filtering."""
    src = np.maximum(frames.astype(np.float32) - p.black_level, 0.0)
    z = np.log1p(src) / math.log(256.0)
    if p.despeckle_strength > 0.0:
        radius = p.lee_window // 2
        local_mean = box_mean(z, radius)
        local_var = np.maximum(box_mean(z*z, radius)-local_mean*local_mean, 0.0)
        weight = np.clip((local_var-noise_var)/(local_var+1e-8), 0.0, 1.0)
        lee = local_mean + weight*(z-local_mean)
        denoised = z + p.despeckle_strength*(lee-z)
    else:
        denoised = z.copy()
    filtered = bilateral_filter(
        denoised, p.bilateral_strength, p.bilateral_radius,
        p.bilateral_sigma, src > 0,
    )
    mixed = (1.0-p.original_blend)*filtered + p.original_blend*z
    corrected = np.expm1(mixed*math.log(256.0))
    corrected[src <= 0] = 0.0
    return corrected


def lateral_box_mean(frames: np.ndarray, radius: int) -> np.ndarray:
    """Mean across neighboring scan lines only for [N, lines, samples]."""
    k = radius * 2 + 1
    padded = np.pad(frames, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    integ = np.pad(padded, ((0, 0), (1, 0), (0, 0)), mode="constant")
    integ = np.cumsum(integ, axis=1)
    return (integ[:, k:, :] - integ[:, :-k, :]) / float(k)


def compensate_bone_shadow(
    frames: np.ndarray,
    active_mask: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, dict]:
    """Conservatively lift probable distal acoustic shadows.

    This is display compensation, not recovery of anatomy hidden behind bone.
    It adds a capped fraction of neighboring scan-line brightness while retaining
    the original pixel texture and never fills true zero/background pixels.
    """
    src = frames.astype(np.float32)
    corrected = src.copy()
    detected_total = 0
    lifted_total = 0.0
    active_total = int(active_mask.sum())

    for index in range(len(src)):
        active = active_mask[index]
        values = src[index][active]
        if values.size < 64:
            continue
        bone_level = max(175.0, float(np.percentile(values, 91.5)))
        bright_interface = active & (src[index] >= bone_level)

        # Propagate a decaying confidence only in the depth direction, starting
        # immediately behind a strong reflector such as bone.
        confidence = np.zeros_like(src[index], dtype=np.float32)
        state = np.zeros(src.shape[1], dtype=np.float32)
        for sample in range(1, src.shape[2]):
            state = np.maximum(state * 0.982, bright_interface[:, sample - 1].astype(np.float32))
            confidence[:, sample] = state

        neighbor_reference = lateral_box_mean(src[index : index + 1], 7)[0]
        deficit = np.maximum(neighbor_reference - src[index], 0.0)
        neighbor_gate = np.clip((neighbor_reference - 35.0) / 75.0, 0.0, 1.0)
        deficit_gate = np.clip((deficit - 5.0) / 38.0, 0.0, 1.0)
        shadow_probability = confidence * neighbor_gate * deficit_gate
        detected = active & (shadow_probability >= 0.12)

        # The additive cap is deliberately modest: at strength 1.0 no pixel is
        # lifted by more than 42 gray levels, and original texture is retained.
        lift = strength * shadow_probability * np.minimum(deficit, 42.0)
        lift[~detected] = 0.0
        corrected[index] = np.clip(src[index] + lift, 0.0, 255.0)
        corrected[index][~active] = src[index][~active]
        detected_total += int(detected.sum())
        lifted_total += float(lift.sum())

    return np.rint(corrected).astype(np.uint8), {
        "enabled": True,
        "strength": float(strength),
        "detected_fraction": float(detected_total / max(active_total, 1)),
        "mean_lift_in_active_region": float(lifted_total / max(active_total, 1)),
        "maximum_lift": float(42.0 * strength),
        "warning": "Display-only compensation; hidden anatomy is not reconstructed.",
    }


def enhance_boundaries(frames: np.ndarray, gain: float) -> np.ndarray:
    """Edge enhancement with local-range and dark-halo suppression."""
    output = np.empty_like(frames)
    pixels_per_frame = max(frames.shape[1] * frames.shape[2], 1)
    chunk_size = max(1, min(len(frames), 8_000_000 // pixels_per_frame))
    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        src = frames[start:end].astype(np.float32)
        fine = src - box_mean(src, 1)
        broad = src - box_mean(src, 3)

        # Wide unsharp masks produce a second outline beside a true boundary.
        # Keep most energy in the fine scale and use only a small broad-scale
        # contribution to improve contour continuity.
        detail = 0.86 * fine + 0.14 * broad
        detail = np.sign(detail) * np.maximum(np.abs(detail) - 1.4, 0.0)
        detail = np.clip(detail, -34.0, 34.0)

        padded = np.pad(src, ((0, 0), (1, 1), (1, 1)), mode="reflect")
        neighbors = [
            padded[:, dy : dy + src.shape[1], dx : dx + src.shape[2]]
            for dy in range(3)
            for dx in range(3)
        ]
        local_min = np.minimum.reduce(neighbors)
        local_max = np.maximum.reduce(neighbors)
        local_range = local_max - local_min

        # Reject small residual fluctuations and strongly attenuate the dark
        # lobe that visually appears as a duplicate/ghost outline.
        edge_gate = np.clip((local_range - 4.0) / 18.0, 0.0, 1.0)
        detail = np.where(detail < 0.0, detail * 0.28, detail)
        tissue_gate = np.clip((src - 10.0) / 48.0, 0.0, 1.0)
        highlight_gate = np.clip((248.0 - src) / 30.0, 0.0, 1.0)
        limited_detail = np.where(detail > 0.0, detail * highlight_gate, detail)
        candidate = src + gain * tissue_gate * edge_gate * limited_detail

        # Do not create a new extremum far outside the observed 3x3
        # neighborhood. This limits both bright rims and dark shadow bands.
        margin = np.clip(local_range * 0.10, 2.0, 7.0)
        enhanced = np.minimum(np.maximum(candidate, local_min - margin), local_max + margin)
        enhanced = np.clip(enhanced, 0.0, 255.0)
        enhanced[frames[start:end] == 0] = 0.0
        output[start:end] = np.rint(enhanced).astype(np.uint8)
    return output


def diffused_outline_mask(
    frame: np.ndarray,
    threshold: float,
    diffusion_steps: int,
) -> np.ndarray:
    """Return a one-pixel outline after edge-preserving continuous diffusion."""
    if frame.ndim != 2:
        raise ValueError("亮区描边需要二维灰度图像。")
    gray = frame.astype(np.float32) / 255.0
    threshold01 = float(threshold) / 255.0

    # A soft bright-region probability is diffused with Perona-Malik-style
    # conductance. Similar neighboring pixels connect continuously, while a
    # strong underlying gray edge stops diffusion from crossing into dark tissue.
    half_width = 20.0 / 255.0
    probability = np.clip(
        (gray - (threshold01 - half_width)) / (2.0 * half_width),
        0.0,
        1.0,
    )
    for _ in range(int(diffusion_steps)):
        padded_p = np.pad(probability, ((1, 1), (1, 1)), mode="reflect")
        padded_g = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
        update = np.zeros_like(probability)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            neighbor_p = padded_p[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
            neighbor_g = padded_g[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
            gray_delta = neighbor_g - gray
            conductance = np.exp(-((gray_delta / 0.12) ** 2))
            update += conductance * (neighbor_p - probability)
        probability = np.clip(probability + 0.18 * update, 0.0, 1.0)

    bright = (probability >= 0.5) & (frame >= max(0.0, threshold - 30.0))
    padded = np.pad(bright, ((1, 1), (1, 1)), mode="constant")
    eroded = np.ones_like(bright)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
    return bright & ~eroded


def add_diffused_dual_outline(
    frame: np.ndarray,
    red_threshold: float,
    blue_threshold: float,
    diffusion_steps: int,
) -> np.ndarray:
    """Mark the overall bright region in red and strong echoes in blue."""
    effective_blue_threshold = max(float(blue_threshold), float(red_threshold) + 1.0)
    overall_outline = diffused_outline_mask(frame, red_threshold, diffusion_steps)
    strong_echo_outline = diffused_outline_mask(
        frame, effective_blue_threshold, diffusion_steps
    )

    rgb = np.repeat(frame[..., None], 3, axis=2)
    rgb[overall_outline] = (255, 32, 32)
    # Draw blue last so a strong-echo boundary wins at overlapping pixels.
    rgb[strong_echo_outline] = (32, 112, 255)
    return rgb


def add_diffused_red_outline(
    frame: np.ndarray,
    threshold: float,
    diffusion_steps: int,
) -> np.ndarray:
    """Backward-compatible red-only overlay helper."""
    rgb = np.repeat(frame[..., None], 3, axis=2)
    rgb[diffused_outline_mask(frame, threshold, diffusion_steps)] = (255, 32, 32)
    return rgb


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="constant")
        result = np.logical_or.reduce(
            [
                padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return result


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="constant")
        result = np.logical_and.reduce(
            [
                padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return result


def scan_safe_mask(frame: np.ndarray, erosion_pixels: int = 12) -> np.ndarray:
    """Build an interior scan mask that rejects canvas and sector-edge artifacts."""
    active = frame > 0
    footprint = np.zeros_like(active)
    for row in range(active.shape[0]):
        columns = np.flatnonzero(active[row])
        if columns.size:
            footprint[row, columns[0] : columns[-1] + 1] = True
    safe = binary_erode(footprint, erosion_pixels)
    border_y = max(2, round(frame.shape[0] * 0.018))
    border_x = max(2, round(frame.shape[1] * 0.018))
    safe[:border_y] = False
    safe[-border_y:] = False
    safe[:, :border_x] = False
    safe[:, -border_x:] = False
    return safe


def automatic_subject_seed(frame: np.ndarray) -> np.ndarray:
    """Create a conservative central tissue seed without using colored outlines."""
    gray = frame.astype(np.float32)
    safe = scan_safe_mask(frame)
    if not np.any(safe):
        return safe
    smooth = box_mean(gray[None, ...], 3)[0]
    local_var = np.maximum(box_mean((gray * gray)[None, ...], 2)[0] - box_mean(gray[None, ...], 2)[0] ** 2, 0.0)
    texture = np.sqrt(local_var)
    yy, xx = np.indices(frame.shape, dtype=np.float32)
    cx, cy = (frame.shape[1] - 1) / 2.0, (frame.shape[0] - 1) * 0.54
    central = np.exp(
        -0.5
        * (
            ((xx - cx) / max(frame.shape[1] * 0.36, 1.0)) ** 2
            + ((yy - cy) / max(frame.shape[0] * 0.42, 1.0)) ** 2
        )
    )
    values = smooth[safe]
    texture_values = texture[safe]
    level = max(float(np.percentile(values, 48)), 12.0)
    texture_level = float(np.percentile(texture_values, 55))
    score = (
        0.55 * np.clip((smooth - level) / max(float(np.percentile(values, 90) - level), 1.0), 0.0, 1.0)
        + 0.25 * np.clip(texture / max(texture_level * 2.0, 1.0), 0.0, 1.0)
        + 0.20 * central
    )
    cutoff = float(np.percentile(score[safe], 70))
    return safe & (score >= cutoff) & (gray >= max(8.0, np.percentile(gray[safe], 28)))


def _neighbor_mean_4(probability: np.ndarray) -> np.ndarray:
    padded = np.pad(probability, ((1, 1), (1, 1)), mode="reflect")
    return 0.25 * (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def diffuse_subject_mask(
    frame: np.ndarray,
    seed: np.ndarray,
    intensity_center: float,
    intensity_spread: float,
    iterations: int,
) -> np.ndarray:
    """Diffuse positive annotations within tissue while respecting safe edges."""
    safe = scan_safe_mask(frame)
    seed = seed.astype(bool) & safe
    if not np.any(seed):
        return np.zeros_like(seed)
    gray = frame.astype(np.float32)
    spread = max(float(intensity_spread), 18.0)
    similarity = np.exp(-0.5 * ((gray - intensity_center) / (2.8 * spread)) ** 2)
    similarity *= np.clip((gray - 4.0) / 28.0, 0.0, 1.0)
    envelope = binary_dilate(seed, min(max(int(iterations), 1), 30)) & safe
    probability = seed.astype(np.float32)
    for _ in range(max(1, int(iterations))):
        neighbor = _neighbor_mean_4(probability)
        probability = np.clip(
            0.58 * probability + 0.30 * neighbor + 0.12 * similarity,
            0.0,
            1.0,
        )
        probability[~envelope] = 0.0
        probability[seed] = 1.0
    mask = (probability >= 0.24) & (similarity >= 0.08) & safe
    return binary_erode(binary_dilate(mask, 1), 1) & safe


def recognize_subject_sequence(
    frames: np.ndarray,
    reference_index: int,
    seed: np.ndarray | None,
    diffusion_iterations: int,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Propagate an automatic or hand-painted subject through a frame sequence."""
    if frames.ndim != 3 or not len(frames):
        raise ValueError("主体识别需要有效的灰度图像序列。")
    reference_index = max(0, min(len(frames) - 1, int(reference_index)))
    reference = frames[reference_index]
    used_auto_seed = seed is None or not np.any(seed)
    reference_seed = automatic_subject_seed(reference) if used_auto_seed else seed.astype(bool)
    reference_seed &= scan_safe_mask(reference)
    if not np.any(reference_seed):
        raise ValueError("没有获得有效主体标注；请在左下图像的主体上拖动画笔。")
    seed_values = reference[reference_seed].astype(np.float32)
    center = float(np.median(seed_values))
    spread = max(float(1.4826 * np.median(np.abs(seed_values - center))), 18.0)
    masks = np.zeros(frames.shape, dtype=bool)
    masks[reference_index] = diffuse_subject_mask(
        reference,
        reference_seed,
        center,
        spread,
        diffusion_iterations,
    )
    tracking_iterations = max(4, min(10, int(diffusion_iterations) // 3))

    completed = 1
    for direction in (1, -1):
        index = reference_index + direction
        previous = masks[reference_index]
        while 0 <= index < len(frames):
            safe = scan_safe_mask(frames[index])
            gray = frames[index].astype(np.float32)
            similarity = np.exp(-0.5 * ((gray - center) / (2.8 * spread)) ** 2)
            propagated_seed = binary_dilate(previous, 2) & safe & (similarity >= 0.08)
            if not np.any(propagated_seed):
                propagated_seed = automatic_subject_seed(frames[index])
            current = diffuse_subject_mask(
                frames[index],
                propagated_seed,
                center,
                spread,
                tracking_iterations,
            )
            masks[index] = current
            previous = current
            completed += 1
            if progress:
                progress(
                    completed / len(frames),
                    f"正在传播主体标注 {completed}/{len(frames)} 帧",
                )
            index += direction

    return masks, {
        "reference_frame": reference_index + 1,
        "seed_source": "automatic" if used_auto_seed else "manual brush",
        "seed_pixels": int(reference_seed.sum()),
        "intensity_center": center,
        "intensity_spread": spread,
        "diffusion_iterations": int(diffusion_iterations),
        "edge_artifact_exclusion_pixels": 12,
        "display_boundary_suppression": "7% of shorter image dimension (minimum 12 px)",
        "mean_subject_fraction": float(masks.mean()),
    }


def add_subject_boundary(
    frame: np.ndarray,
    subject_mask: np.ndarray,
    artifact_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay a green one-pixel subject boundary on gray or RGB imagery."""
    if frame.ndim == 2:
        rgb = np.repeat(frame[..., None], 3, axis=2)
    else:
        rgb = frame.copy()
    mask = subject_mask.astype(bool)
    boundary = mask & ~binary_erode(mask, 1)
    reference = artifact_reference
    if reference is None and frame.ndim == 2:
        reference = frame
    if reference is not None:
        suppression = max(12, round(min(reference.shape[:2]) * 0.07))
        boundary &= scan_safe_mask(reference, erosion_pixels=suppression)
    rgb[boundary] = (40, 255, 80)
    return rgb


def dsc_scan_convert(
    frames: np.ndarray,
    p: ProcessParams,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """Convert sector scan-line data [N, lines, samples] to Cartesian images."""
    frame_count, line_count, sample_count = frames.shape
    out_h, out_w = p.dsc_output_height, p.dsc_output_width
    half_angle = math.radians(p.dsc_angle_deg) / 2.0
    outer_radius = p.dsc_inner_radius + sample_count - 1.0
    lateral_limit = outer_radius * math.sin(half_angle)

    z = np.linspace(0.0, outer_radius, out_h, dtype=np.float32)[:, None]
    x = np.linspace(-lateral_limit, lateral_limit, out_w, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x + z * z)
    angle = np.arctan2(x, np.maximum(z, 1e-8))
    source_line = (angle + half_angle) * (line_count - 1) / (2.0 * half_angle)
    source_sample = radius - p.dsc_inner_radius
    valid = (
        (np.abs(angle) <= half_angle)
        & (source_sample >= 0.0)
        & (source_sample <= sample_count - 1.0)
    )

    line0 = np.clip(np.floor(source_line).astype(np.int32), 0, line_count - 1)
    line1 = np.minimum(line0 + 1, line_count - 1)
    sample0 = np.clip(np.floor(source_sample).astype(np.int32), 0, sample_count - 1)
    sample1 = np.minimum(sample0 + 1, sample_count - 1)
    wl = (source_line - line0).astype(np.float32)
    ws = (source_sample - sample0).astype(np.float32)

    converted = np.zeros((frame_count, out_h, out_w), dtype=np.uint8)
    for i, frame in enumerate(frames):
        f = frame.astype(np.float32)
        top = f[line0, sample0] * (1.0 - ws) + f[line0, sample1] * ws
        bottom = f[line1, sample0] * (1.0 - ws) + f[line1, sample1] * ws
        image = top * (1.0 - wl) + bottom * wl
        image[~valid] = 0.0
        converted[i] = np.rint(image).astype(np.uint8)
        if progress and (i % 5 == 0 or i + 1 == frame_count):
            progress((i + 1) / frame_count, f"正在进行 DSC 扫描转换 {i + 1}/{frame_count} 帧")
    return converted


def process_frames(
    frames: np.ndarray,
    p: ProcessParams,
    progress: Callable[[float, str], None] | None = None,
    calibration_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    calibration_source = frames if calibration_frames is None else calibration_frames
    noise_var = estimate_noise_variance(calibration_source, p)
    # Estimate one common white point to preserve brightness relationships between frames.
    sample_ids = np.linspace(
        0, len(calibration_source) - 1, min(8, len(calibration_source)), dtype=int
    )
    # Shared white-point calibration uses Lee only to keep preview calibration cheap.
    calibration_params = replace(p, bilateral_strength=0.0)
    filtered_sample = filter_chunk(calibration_source[sample_ids], calibration_params, noise_var)
    # Do not let zero-valued pixels outside/inside the acquisition footprint
    # pull the white point downward.  Tone statistics must represent echoes in
    # the active scan region rather than the black canvas around it.
    active_sample = calibration_source[sample_ids] > p.black_level
    white_values = filtered_sample[active_sample]
    if white_values.size == 0:
        white_values = filtered_sample.reshape(-1)
    white = float(np.percentile(white_values, p.white_percentile))
    white = max(white, 1.0)

    output = np.empty_like(frames)
    shadow_reports: list[dict] = []
    chunk_size = max(1, min(16, 24_000_000 // max(frames.shape[1] * frames.shape[2], 1)))
    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        filtered = filter_chunk(frames[start:end], p, noise_var)
        normalized = np.clip(filtered / white, 0.0, 1.0)
        # Combined contrast/brightness/gamma correction:
        # y = 255 * clip(alpha*x + beta/255, 0, 1) ** gamma
        adjusted = np.clip(p.alpha * normalized + p.beta / 255.0, 0.0, 1.0)
        encoded = np.rint(255.0 * np.power(adjusted, p.gamma)).astype(np.uint8)
        if p.shadow_fill_enabled and p.shadow_fill_strength > 0.0:
            encoded, shadow_report = compensate_bone_shadow(
                encoded,
                frames[start:end] > p.black_level,
                p.shadow_fill_strength,
            )
            shadow_reports.append(shadow_report)
        # A positive beta is useful for lifting highlights, but the acquisition
        # background and true zero echoes must remain pure black.
        encoded[frames[start:end] <= p.black_level] = 0
        output[start:end] = encoded
        if progress:
            scale = 0.8 if p.dsc_enabled else 1.0
            progress(scale * end / len(frames), f"正在处理 {end}/{len(frames)} 帧")

    if p.dsc_enabled:
        output = dsc_scan_convert(
            output,
            p,
            (lambda x, s: progress(0.8 + 0.2 * x, s)) if progress else None,
        )

    # Enhance after scan conversion so interpolation does not blur the sharpened
    # boundary. For non-DSC images this remains the final display-space step.
    if p.edge_gain > 0.0:
        output = enhance_boundaries(output, p.edge_gain)

    meta = {
        "algorithm": "log-domain adaptive Lee + bilateral filtering + alpha-beta-gamma + DSC + halo-suppressed boundary enhancement",
        "denoising_stages": {
            "despeckle": {
                "name": "adaptive Lee", "strength": p.despeckle_strength,
                "window": p.lee_window, "noise_percentile": p.noise_percentile,
            },
            "filter": {
                "name": "bilateral", "strength": p.bilateral_strength,
                "radius": p.bilateral_radius, "sigma_log": p.bilateral_sigma,
                "spatial_sigma": max(0.8, p.bilateral_radius / 1.5),
                "passes": 1,
            },
        },
        "gray_correction": {
            "name": "alpha-beta-gamma correction",
            "formula": "y = 255 * clip(alpha*x + beta/255, 0, 1)^gamma",
            "alpha_contrast_gain": p.alpha,
            "beta_brightness_gray_levels": p.beta,
            "gamma": p.gamma,
        },
        "estimated_log_noise_variance": noise_var,
        "global_white_value": white,
        "calibration_frame_count": int(len(calibration_source)),
        "bone_shadow_compensation": {
            "enabled": p.shadow_fill_enabled,
            "strength": p.shadow_fill_strength,
            "chunks": shadow_reports,
            "warning": "仅作显示补偿，不代表恢复被骨骼遮挡的真实组织。",
        },
        "parameters": asdict(p),
        "input_shape": list(frames.shape),
        "output_shape": list(output.shape),
        "dsc_scan_conversion": {
            "enabled": p.dsc_enabled,
            "geometry": "sector",
            "sector_angle_deg": p.dsc_angle_deg,
            "inner_radius_samples": p.dsc_inner_radius,
            "interpolation": "bilinear",
            "output_width": p.dsc_output_width,
            "output_height": p.dsc_output_height,
        },
        "raw_statistics": stats(frames),
        "processed_statistics": stats(output),
        "processing_mode": "one shared parameter set for all frames",
    }
    return output, meta


def auto_tune_frame(frame: np.ndarray, p: ProcessParams) -> tuple[dict, dict]:
    """Choose subject-aware tone, denoise and boundary parameters for one frame."""
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError("自动校正需要一帧有效的二维灰度图像。")

    # Tone recommendations are calibrated through the same two-stage pipeline.
    src = frame.astype(np.float32)
    local = box_mean(src[None, ...], 1)[0]
    active = src > 0
    if not np.any(active):
        raise ValueError("当前帧没有有效的非零扫描数据。")
    signal = src[active]
    dynamic = max(float(np.percentile(signal, 95) - np.percentile(signal, 10)), 1.0)
    residual = float(np.median(np.abs(src[active] - local[active]))) / dynamic
    noise_percentile = p.noise_percentile
    original_blend = p.original_blend
    despeckle_strength = p.despeckle_strength
    bilateral_strength = p.bilateral_strength
    edge_gain = float(np.clip(1.92 - residual * 3.0, 1.70, 1.90))

    tune = replace(
        p,
        black_level=0.0,
        lee_window=5,
        noise_percentile=noise_percentile,
        original_blend=original_blend,
        despeckle_strength=despeckle_strength,
        bilateral_strength=bilateral_strength,
        bilateral_radius=p.bilateral_radius,
        bilateral_sigma=p.bilateral_sigma,
        dsc_enabled=False,
    )
    noise_var = estimate_noise_variance(frame[None, ...], tune)
    filtered = filter_chunk(frame[None, ...], tune, noise_var)[0]
    values = filtered[active]
    if values.size > 120_000:
        ids = np.linspace(0, values.size - 1, 120_000, dtype=np.int64)
        values = values[ids]
    sorted_values = np.sort(values)

    # Estimate an anatomical subject region in native scan-line coordinates.
    # Dark fluid and empty background are de-emphasized; coherent medium/high
    # echoes and their nearby boundaries drive the subject-brightness targets.
    subject_smooth = box_mean(filtered[None, ...], 4)[0]
    grad_x = np.abs(np.diff(subject_smooth, axis=1, prepend=subject_smooth[:, :1]))
    grad_y = np.abs(np.diff(subject_smooth, axis=0, prepend=subject_smooth[:1, :]))
    gradient = grad_x + grad_y
    smooth_active = subject_smooth[active]
    gradient_active = gradient[active]
    tissue_level = float(np.percentile(smooth_active, 58.0))
    edge_level = float(np.percentile(gradient_active, 78.0))
    seed = active & (
        (subject_smooth >= tissue_level)
        | ((gradient >= edge_level) & (filtered >= np.percentile(values, 32.0)))
    )
    support = box_mean(seed[None, ...].astype(np.float32), 4)[0] >= 0.16
    depth = np.linspace(0.0, 1.0, frame.shape[1], dtype=np.float32)[None, :]
    subject = active & support & (depth >= 0.06) & (depth <= 0.96)
    subject_values = filtered[subject]
    if subject_values.size < max(256, int(active.sum() * 0.08)):
        subject_values = values[values >= np.percentile(values, 45.0)]
    sorted_subject = np.sort(subject_values)

    # Targets describe a diagnostic-style grayscale rendering: deep blacks,
    # readable midtones, and highlights below hard white.  The grid search is
    # intentionally constrained so it cannot produce a dramatic but destructive
    # tone curve.
    quantiles = np.array([5, 10, 25, 50, 75, 90, 95, 99, 99.8], dtype=np.float32)
    targets = np.array([0, 3, 20, 74, 162, 211, 232, 247, 252], dtype=np.float32)
    weights = np.array([2.0, 2.0, 1.4, 1.2, 1.4, 1.6, 1.6, 2.0, 2.2], dtype=np.float32)
    subject_quantiles = np.array([10, 25, 50, 75, 90, 97], dtype=np.float32)
    subject_targets = np.array([38, 78, 138, 194, 228, 247], dtype=np.float32)
    subject_weights = np.array([0.8, 1.0, 1.6, 2.0, 1.8, 1.4], dtype=np.float32)
    best: tuple[float, float, float, float, float, np.ndarray] | None = None

    for white_percentile in (99.80, 99.90, 99.95):
        white = max(float(np.percentile(sorted_values, white_percentile)), 1.0)
        source_q = np.percentile(sorted_values, quantiles)
        subject_source_q = np.percentile(sorted_subject, subject_quantiles)
        # A deterministic sample keeps interactive auto-tuning fast.
        for alpha in (1.08, 1.12, 1.16, 1.20):
            for beta in (6.0, 10.0, 14.0, 18.0):
                for gamma in (1.55, 1.65, 1.75, 1.85, 1.95):
                    q_base = np.clip(alpha * source_q / white + beta / 255.0, 0.0, 1.0)
                    q = 255.0 * np.power(q_base, gamma)
                    score = float(np.sum(weights * ((q - targets) / 18.0) ** 2))
                    subject_base = np.clip(
                        alpha * subject_source_q / white + beta / 255.0, 0.0, 1.0
                    )
                    subject_q = 255.0 * np.power(subject_base, gamma)
                    score += float(
                        np.sum(subject_weights * ((subject_q - subject_targets) / 18.0) ** 2)
                    )

                    # White must not flare, while gray detail in bright tissue
                    # should remain below clipping.  Penalize both saturation and
                    # a lifted black floor strongly.
                    saturation_base = (253.0 / 255.0) ** (1.0 / gamma)
                    saturation_x = (saturation_base - beta / 255.0) / alpha
                    saturation_value = white * saturation_x
                    saturation = float(
                        1.0 - np.searchsorted(sorted_values, saturation_value, side="left") / sorted_values.size
                    )
                    crushed_base = (1.0 / 255.0) ** (1.0 / gamma)
                    crushed_x = (crushed_base - beta / 255.0) / alpha
                    crushed_value = white * crushed_x
                    crushed = float(
                        np.searchsorted(sorted_values, crushed_value, side="right") / sorted_values.size
                    )
                    score += 2200.0 * max(0.0, saturation - 0.001)
                    score += 18.0 * max(0.0, q[1] - 9.0) ** 2 / 81.0
                    score += 5.0 * max(0.0, crushed - 0.16) ** 2
                    # Prefer the least aggressive curve when scores are close.
                    score += 0.12 * abs(alpha - 1.0) + 0.025 * abs(beta) + 0.10 * abs(gamma - 1.0)
                    if best is None or score < best[0]:
                        best = (score, white_percentile, alpha, beta, gamma, q)

    assert best is not None
    _, white_percentile, alpha, beta, gamma, output_q = best
    recommended = {
        "black_level": 0.0,
        "white_percentile": float(white_percentile),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "lee_window": 5,
        "noise_percentile": round(noise_percentile),
        "original_blend": round(original_blend, 2),
        "despeckle_strength": round(despeckle_strength, 2),
        "bilateral_strength": round(bilateral_strength, 2),
        "bilateral_radius": p.bilateral_radius,
        "bilateral_sigma": p.bilateral_sigma,
        "edge_gain": round(edge_gain, 2),
        "shadow_fill_strength": round(p.shadow_fill_strength, 2),
    }
    diagnostics = {
        "active_pixels": int(active.sum()),
        "subject_pixels": int(subject_values.size),
        "subject_fraction": float(subject_values.size / max(int(active.sum()), 1)),
        "residual_ratio": residual,
        "estimated_log_noise_variance": noise_var,
        "output_percentiles": {str(q): float(v) for q, v in zip(quantiles, output_q)},
        "design": "subject-aware fetal brightness, high contrast and boundary protection",
    }
    return recommended, diagnostics


def combine_group_recommendations(recommendations: list[dict]) -> dict:
    """Combine per-frame measurements into one robust parameter set."""
    if not recommendations:
        raise ValueError("整组自动校正没有得到有效参数。")

    def percentile(name: str, q: float, digits: int = 2) -> float:
        value = float(np.percentile([item[name] for item in recommendations], q))
        return round(value, digits)

    # Tone and edge parameters use the group median so an unusually dark or
    # bright frame cannot dominate the sequence. Denoising is biased toward
    # the noisier quartile, making the shared setting safe for most frames.
    return {
        "black_level": 0.0,
        "white_percentile": percentile("white_percentile", 50),
        "alpha": percentile("alpha", 50),
        "beta": round(percentile("beta", 50), 0),
        "gamma": percentile("gamma", 50),
        "lee_window": int(round(percentile("lee_window", 50, 0))),
        "noise_percentile": round(percentile("noise_percentile", 75), 0),
        "original_blend": percentile("original_blend", 25),
        "despeckle_strength": percentile("despeckle_strength", 75),
        "bilateral_strength": percentile("bilateral_strength", 75),
        "bilateral_radius": int(round(percentile("bilateral_radius", 50, 0))),
        "bilateral_sigma": percentile("bilateral_sigma", 50),
        "edge_gain": percentile("edge_gain", 50),
        "shadow_fill_strength": percentile("shadow_fill_strength", 50),
    }


def _gradient_magnitude(frames: np.ndarray) -> np.ndarray:
    work = frames.astype(np.float32)
    gx = 0.5 * (_shift_batch(work, 0, 1) - _shift_batch(work, 0, -1))
    gy = 0.5 * (_shift_batch(work, 1, 0) - _shift_batch(work, -1, 0))
    return np.sqrt(gx * gx + gy * gy)


def analyze_group_characteristics(
    frames: np.ndarray,
    p: ProcessParams,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Measure all frames cheaply before running any expensive candidate filter."""
    rows = []
    for index, frame in enumerate(frames):
        source = np.log1p(frame.astype(np.float32)) / math.log(256.0)
        active = frame > p.black_level
        values = source[active]
        if values.size < 64:
            continue
        local = box_mean(source[None, ...], 1)[0]
        residual = source - local
        grad = _gradient_magnitude(source[None, ...])[0]
        dynamic = max(float(np.percentile(values, 95) - np.percentile(values, 10)), 1e-4)
        active_residual = np.abs(residual[active])
        active_grad = grad[active]
        noise_ratio = float(np.median(active_residual) / dynamic)
        fine_detail = float(
            np.mean(
                (active_grad >= np.percentile(active_grad, 60))
                & (active_grad <= np.percentile(active_grad, 88))
            )
        )
        coherent_edge = float(np.percentile(active_grad, 90) / dynamic)
        rows.append(
            {
                "noise_ratio": noise_ratio,
                "fine_detail_fraction": fine_detail,
                "coherent_edge_ratio": coherent_edge,
                "dynamic_range": dynamic,
                "raw_highlight_fraction": float(np.mean(frame[active] >= 250)),
            }
        )
        if progress and (index % 8 == 0 or index + 1 == len(frames)):
            progress((index + 1) / len(frames), f"正在分析整组纹理 {index + 1}/{len(frames)} 帧")
    if not rows:
        raise ValueError("整组图像没有足够的有效像素用于自动滤波。")

    def robust(name: str, q: float) -> float:
        return float(np.percentile([row[name] for row in rows], q))

    return {
        "frame_count": len(frames),
        "valid_frame_count": len(rows),
        "noise_ratio_median": robust("noise_ratio", 50),
        "noise_ratio_p75": robust("noise_ratio", 75),
        "fine_detail_fraction_median": robust("fine_detail_fraction", 50),
        "coherent_edge_ratio_median": robust("coherent_edge_ratio", 50),
        "dynamic_range_median": robust("dynamic_range", 50),
        "raw_highlight_fraction_p90": robust("raw_highlight_fraction", 90),
    }


def _filter_candidate_score(
    raw: np.ndarray,
    filtered: np.ndarray,
    p: ProcessParams,
    target_flat_ratio: float,
    baseline_metrics: dict | None = None,
) -> tuple[float, dict]:
    raw_log = np.log1p(raw.astype(np.float32)) / math.log(256.0)
    filtered_log = np.log1p(np.maximum(filtered, 0.0)) / math.log(256.0)
    raw_grad = _gradient_magnitude(raw_log)
    filtered_grad = _gradient_magnitude(filtered_log)
    raw_residual = raw_log - box_mean(raw_log, 1)
    filtered_residual = filtered_log - box_mean(filtered_log, 1)
    metrics = []
    for index in range(len(raw)):
        active = raw[index] > p.black_level
        gradients = raw_grad[index][active]
        if gradients.size < 64:
            continue
        q35, q58, q82 = np.percentile(gradients, (35, 58, 82))
        flat = active & (raw_grad[index] <= q35)
        weak = active & (raw_grad[index] >= q58) & (raw_grad[index] < q82)
        strong = active & (raw_grad[index] >= q82)

        def ratio(mask: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> float:
            if int(mask.sum()) < 16:
                return 1.0
            return float(np.median(np.abs(numerator[mask])) / (np.median(np.abs(denominator[mask])) + 1e-6))

        flat_ratio = ratio(flat, filtered_residual[index], raw_residual[index])
        weak_ratio = ratio(weak, filtered_grad[index], raw_grad[index])
        strong_ratio = ratio(strong, filtered_grad[index], raw_grad[index])
        if int(strong.sum()) >= 16:
            a = raw_residual[index][strong]
            b = filtered_residual[index][strong]
            detail_corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        else:
            detail_corr = 1.0
        metrics.append((flat_ratio, weak_ratio, strong_ratio, detail_corr))
    if not metrics:
        return 1e9, {"error": "no valid scoring pixels"}
    values = np.asarray(metrics, dtype=np.float32)
    flat_ratio, weak_ratio, strong_ratio, detail_corr = np.median(values, axis=0)
    # Minimize residual speckle while explicitly preventing the low-gradient
    # texture and coherent boundaries from being painted into broad blobs.
    strong_floor = 0.74
    weak_floor = 0.66
    correlation_floor = 0.58
    if baseline_metrics:
        strong_floor = max(strong_floor, 0.92 * baseline_metrics["strong_edge_ratio"])
        weak_floor = max(weak_floor, 0.90 * baseline_metrics["weak_edge_ratio"])
        correlation_floor = max(correlation_floor, 0.92 * baseline_metrics["detail_correlation"])
    score = 4.0 * abs(float(flat_ratio) - target_flat_ratio)
    score += 12.0 * max(0.0, strong_floor - float(strong_ratio)) ** 2
    score += 10.0 * max(0.0, weak_floor - float(weak_ratio)) ** 2
    score += 8.0 * max(0.0, correlation_floor - float(detail_corr)) ** 2
    score += 3.0 * max(0.0, float(strong_ratio) - 1.30) ** 2
    return score, {
        "score": float(score),
        "flat_speckle_ratio": float(flat_ratio),
        "weak_edge_ratio": float(weak_ratio),
        "strong_edge_ratio": float(strong_ratio),
        "detail_correlation": float(detail_corr),
        "target_flat_speckle_ratio": float(target_flat_ratio),
        "strong_edge_floor": float(strong_floor),
        "weak_edge_floor": float(weak_floor),
        "detail_correlation_floor": float(correlation_floor),
    }


def auto_tune_group_filters(
    frames: np.ndarray, p: ProcessParams, profile: dict,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[dict, dict]:
    """Score the actual Lee + bilateral settings, then share them across all frames."""
    noise_level = float(np.clip((profile["noise_ratio_p75"]-0.018)/0.075, 0, 1))
    detail_level = float(np.clip((profile["coherent_edge_ratio_median"]-0.035)/0.11, 0, 1))
    sample = frames[_representative_indices(len(frames), maximum=4)]
    baseline_params = replace(p, despeckle_strength=0.0, bilateral_strength=0.0,
                              dsc_enabled=False, edge_gain=0.0)
    baseline = filter_chunk(sample, baseline_params, 0.0)
    _, baseline_metrics = _filter_candidate_score(
        sample, baseline, baseline_params, target_flat_ratio=0.0)
    target = float(np.clip(baseline_metrics["flat_speckle_ratio"]*(0.78-0.12*noise_level), 0.12, 0.72))
    candidates = []
    for scale in (0.55, 0.85, 1.10):
        candidate = {
            "lee_window": 5,
            "despeckle_strength": round(float(np.clip((0.65+0.18*noise_level)*scale, 0.25, 0.95)), 2),
            "bilateral_strength": round(float(np.clip((0.35+0.20*noise_level-0.08*detail_level)*scale, 0.15, 0.70)), 2),
            "bilateral_radius": 2,
            "bilateral_sigma": round(0.05+0.05*noise_level, 3),
            "original_blend": round(float(np.clip(0.15+0.08*detail_level-0.03*noise_level, 0.10, 0.25)), 2),
            "noise_percentile": round(25+10*noise_level),
            "edge_gain": round(float(np.clip(1.30+0.28*detail_level, 1.20, 1.65)), 2),
        }
        cp = replace(p, **candidate)
        filtered = filter_chunk(sample, cp, estimate_noise_variance(sample, cp))
        score, metrics = _filter_candidate_score(sample, filtered, cp, target,
                                                  baseline_metrics=baseline_metrics)
        metrics["parameters"] = dict(candidate)
        candidates.append((score, candidate, metrics))
        if progress:
            progress(len(candidates)/3, f"正在验证 Lee + 双边滤波候选 {len(candidates)}/3")
    candidates.sort(key=lambda item: item[0])
    _, best, selected = candidates[0]
    return dict(best), {
        "group_profile": profile, "noise_level_normalized": noise_level,
        "detail_level_normalized": detail_level, "selected": selected,
        "baseline": baseline_metrics, "candidates": [item[2] for item in candidates],
        "design_constraints": {
            "avoid_smearing": "strong/weak edge and residual-correlation scoring",
            "avoid_overexposure": "group tone guard",
            "shared_across_group": True, "denoising_stage_count": 2,
        },
    }


def guard_group_tone(
    frames: np.ndarray,
    p: ProcessParams,
    values: dict,
) -> tuple[dict, dict]:
    """Lower tone aggressiveness when representative frames would clip highlights."""
    sample_ids = _representative_indices(len(frames), maximum=6)
    sample = frames[sample_ids]
    test_values = dict(values)
    test_values.update({"dsc_enabled": False, "edge_gain": 0.0})
    test_params = replace(p, **test_values)
    noise_var = estimate_noise_variance(sample, test_params)
    filtered = filter_chunk(sample, test_params, noise_var)
    active = sample > test_params.black_level
    active_values = filtered[active]
    guarded = dict(values)
    iterations = []
    for _ in range(4):
        white = max(float(np.percentile(active_values, guarded["white_percentile"])), 1.0)
        adjusted = np.clip(
            guarded["alpha"] * active_values / white + guarded["beta"] / 255.0,
            0.0,
            1.0,
        )
        encoded = 255.0 * np.power(adjusted, guarded["gamma"])
        saturation = float(np.mean(encoded >= 252.0))
        highlights = float(np.percentile(encoded, 99.5))
        dark_floor = float(np.percentile(encoded, 8.0))
        iterations.append(
            {
                "saturation_fraction": saturation,
                "p99_5": highlights,
                "p08": dark_floor,
                "alpha": guarded["alpha"],
                "beta": guarded["beta"],
                "white_percentile": guarded["white_percentile"],
            }
        )
        if saturation <= 0.0035 and highlights <= 253.0 and dark_floor <= 8.0:
            break
        if saturation > 0.0035 or highlights > 253.0:
            guarded["white_percentile"] = min(99.98, guarded["white_percentile"] + 0.04)
            guarded["beta"] = max(2.0, guarded["beta"] - 2.0)
            guarded["alpha"] = max(1.04, guarded["alpha"] - 0.02)
        if dark_floor > 8.0:
            guarded["beta"] = max(0.0, guarded["beta"] - 2.0)
    # Reuse the already-filtered representative pixels to select annotation
    # thresholds.  This avoids a second expensive filtering pass.  Automatic
    # values stay inside the requested ranges; the manual sliders remain wider.
    final_white = max(float(np.percentile(active_values, guarded["white_percentile"])), 1.0)
    final_adjusted = np.clip(
        guarded["alpha"] * active_values / final_white + guarded["beta"] / 255.0,
        0.0,
        1.0,
    )
    final_encoded = 255.0 * np.power(final_adjusted, guarded["gamma"])
    raw_red = float(np.percentile(final_encoded, 55.0))
    raw_blue = float(np.percentile(final_encoded, 88.0))
    guarded["red_outline_threshold"] = float(round(np.clip(raw_red, 60.0, 90.0)))
    guarded["blue_outline_threshold"] = float(round(np.clip(raw_blue, 160.0, 180.0)))
    return guarded, {
        "iterations": iterations,
        "final": iterations[-1],
        "automatic_outline_thresholds": {
            "red": guarded["red_outline_threshold"],
            "red_allowed_range": [60, 90],
            "blue": guarded["blue_outline_threshold"],
            "blue_allowed_range": [160, 180],
            "raw_red_percentile_55": raw_red,
            "raw_blue_percentile_88": raw_blue,
        },
    }


def process_frames_individually(
    frames: np.ndarray,
    p: ProcessParams,
    frame_parameters: list[dict],
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Process a sequence with a separate auto-tuned parameter set per frame."""
    if len(frame_parameters) != len(frames):
        raise ValueError("逐帧自动参数数量与图像帧数不一致，请重新点击自动。")
    rendered: list[np.ndarray] = []
    frame_metadata: list[dict] = []
    for index, (frame, values) in enumerate(zip(frames, frame_parameters)):
        frame_params = replace(p, **values)
        corrected, meta = process_frames(frame[None, ...], frame_params)
        rendered.append(corrected[0])
        frame_metadata.append(
            {
                "frame": index + 1,
                "parameters": values,
                "global_white_value": meta["global_white_value"],
                "estimated_log_noise_variance": meta["estimated_log_noise_variance"],
                "bone_shadow_compensation": meta["bone_shadow_compensation"],
            }
        )
        if progress:
            progress(
                (index + 1) / len(frames),
                f"正在按独立参数处理第 {index + 1}/{len(frames)} 帧",
            )
    output = np.stack(rendered, axis=0)
    return output, {
        "algorithm": "per-frame auto-tuned log-domain adaptive Lee and bilateral filtering, alpha-beta-gamma correction and halo-suppressed boundary enhancement",
        "processing_mode": "independent parameters for every frame",
        "parameters": asdict(p),
        "per_frame": frame_metadata,
        "input_shape": list(frames.shape),
        "output_shape": list(output.shape),
        "raw_statistics": stats(frames),
        "processed_statistics": stats(output),
        "dsc_scan_conversion": {
            "enabled": p.dsc_enabled,
            "geometry": "sector",
            "sector_angle_deg": p.dsc_angle_deg,
            "inner_radius_samples": p.dsc_inner_radius,
            "interpolation": "bilinear",
            "output_width": p.dsc_output_width,
            "output_height": p.dsc_output_height,
        },
    }


def stats(a: np.ndarray) -> dict:
    return {
        "minimum": int(a.min()),
        "maximum": int(a.max()),
        "mean": float(a.mean()),
        "standard_deviation": float(a.std()),
        "percentiles": {
            str(q): float(v)
            for q, v in zip((1, 5, 25, 50, 75, 95, 99), np.percentile(a, (1, 5, 25, 50, 75, 95, 99)))
        },
    }


def score_layout(data: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    count = data.size // (width * height)
    if count < 2 or data.size != count * width * height:
        return (-1e9, 0.0, 0.0)
    frames = data.reshape(count, height, width).astype(np.float32)
    ids = np.linspace(0, count - 2, min(12, count - 1), dtype=int)
    correlations = []
    for i in ids:
        a, b = frames[i].ravel(), frames[i + 1].ravel()
        sa, sb = float(a.std()), float(b.std())
        if sa > 0 and sb > 0:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    corr = float(np.mean(correlations)) if correlations else 0.0
    within = float(np.mean(np.abs(np.diff(frames[: min(8, count)], axis=1))))
    boundary = float(np.mean(np.abs(frames[:-1, -1] - frames[1:, 0])))
    boundary_ratio = boundary / max(within, 1e-6)
    score = corr + 0.12 * min(boundary_ratio, 6.0)
    return score, corr, boundary_ratio


def infer_layout(path: Path, offset_bytes: int = 0) -> list[dict]:
    total = path.stat().st_size - offset_bytes
    if total <= 0:
        raise ValueError("文件中没有可读取的数据。")
    data = np.fromfile(path, dtype=np.uint8, offset=offset_bytes)
    candidates = []
    for width in COMMON_WIDTHS:
        if total % width:
            continue
        rows = total // width
        for height in range(32, min(1024, rows) + 1):
            if rows % height:
                continue
            count = rows // height
            if not 2 <= count <= 5000:
                continue
            score, corr, ratio = score_layout(data, width, height)
            candidates.append(
                {
                    "width": width,
                    "height": height,
                    "frame_count": count,
                    "score": score,
                    "adjacent_correlation": corr,
                    "boundary_ratio": ratio,
                }
            )
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:12]


def export_results(
    input_path: Path,
    output_dir: Path,
    raw: np.ndarray,
    processed: np.ndarray,
    meta: dict,
    p: ProcessParams,
    progress: Callable[[float, str], None] | None = None,
    subject_masks: np.ndarray | None = None,
    subject_report: dict | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_frames = processed.transpose(0, 2, 1) if p.transpose_output else processed
    out_subject_masks = None
    if subject_masks is not None:
        if subject_masks.shape != processed.shape:
            raise ValueError("主体识别掩膜与处理后图像尺寸不一致，请重新执行主体识别。")
        out_subject_masks = (
            subject_masks.transpose(0, 2, 1) if p.transpose_output else subject_masks
        )
    bin_path = output_dir / f"{input_path.stem}_corrected_u8.bin"
    out_frames.tofile(bin_path)

    png_dir = output_dir / f"{input_path.stem}_corrected_png"
    png_dir.mkdir(exist_ok=True)
    subject_mask_dir = None
    if out_subject_masks is not None:
        subject_mask_dir = output_dir / f"{input_path.stem}_subject_masks"
        subject_mask_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(out_frames):
        png_frame = (
            add_diffused_dual_outline(
                frame,
                p.red_outline_threshold,
                p.blue_outline_threshold,
                p.red_outline_diffusion_steps,
            )
            if p.red_outline_enabled
            else frame
        )
        if out_subject_masks is not None:
            png_frame = add_subject_boundary(png_frame, out_subject_masks[i], frame)
            Image.fromarray(
                (out_subject_masks[i].astype(np.uint8) * 255), mode="L"
            ).save(subject_mask_dir / f"mask_{i + 1:04d}.png")
        Image.fromarray(png_frame).save(png_dir / f"frame_{i + 1:04d}.png")
        if progress and (i % 10 == 0 or i + 1 == len(out_frames)):
            progress((i + 1) / len(out_frames), f"正在导出 PNG {i + 1}/{len(out_frames)}")

    # The comparison must use the same Cartesian DSC geometry on both sides.
    # Never place the native scan-line rectangle next to the processed sector.
    if p.dsc_enabled:
        comparison_before = dsc_scan_convert(raw, p)
        comparison_mode = "DSC before tone vs DSC after tone"
    else:
        comparison_before = raw.copy()
        comparison_mode = "tone before vs tone after (DSC disabled)"
    comparison_after = processed
    if p.transpose_output:
        comparison_before = comparison_before.transpose(0, 2, 1)
        comparison_after = comparison_after.transpose(0, 2, 1)
    comparison_subject_reference = comparison_after.copy()
    if p.red_outline_enabled:
        comparison_after = np.stack(
            [
                add_diffused_dual_outline(
                    frame,
                    p.red_outline_threshold,
                    p.blue_outline_threshold,
                    p.red_outline_diffusion_steps,
                )
                for frame in comparison_after
            ],
            axis=0,
        )
        comparison_mode += " with diffused red/blue region outlines"
    if out_subject_masks is not None:
        comparison_after = np.stack(
            [
                add_subject_boundary(frame, mask, reference)
                for frame, mask, reference in zip(
                    comparison_after,
                    out_subject_masks,
                    comparison_subject_reference,
                )
            ],
            axis=0,
        )
        comparison_mode += " and propagated green subject boundary"
    compare_path = output_dir / f"{input_path.stem}_dsc_tone_before_after.png"
    make_comparison(comparison_before, comparison_after, compare_path, p.dsc_enabled)
    figure_groups = make_paper_style_groups(
        input_path,
        output_dir,
        raw,
        processed,
        p,
        subject_masks=subject_masks,
    )
    meta.update(
        {
            "input_file": str(input_path),
            "binary_output": str(bin_path),
            "png_folder": str(png_dir),
            "comparison_image": str(compare_path),
            "comparison_mode": comparison_mode,
            "paper_style_2d_figure_groups": figure_groups,
            "binary_storage_shape": list(out_frames.shape),
            "red_outline_overlay": {
                "enabled": p.red_outline_enabled,
                "red_overall_region_threshold": p.red_outline_threshold,
                "blue_strong_echo_threshold": p.blue_outline_threshold,
                "continuous_diffusion_steps": p.red_outline_diffusion_steps,
                "line_width_pixels": 1,
                "red_color_rgb": [255, 32, 32],
                "blue_color_rgb": [32, 112, 255],
                "png_only": True,
                "binary_output_remains_grayscale": True,
            },
            "subject_recognition": {
                "enabled": out_subject_masks is not None,
                "boundary_color_rgb": [40, 255, 80],
                "mask_png_folder": str(subject_mask_dir) if subject_mask_dir else None,
                "png_only": True,
                "binary_output_remains_grayscale": True,
                "report": subject_report or {},
            },
        }
    )
    (output_dir / f"{input_path.stem}_processing.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bin_path, png_dir, Path(figure_groups["overview"])


def make_comparison(before: np.ndarray, after: np.ndarray, path: Path, dsc_enabled: bool = True):
    selected = np.linspace(0, len(before) - 1, min(6, len(before)), dtype=int)
    display_w = min(640, max(320, before.shape[2]))
    display_h = max(1, round(before.shape[1] * display_w / before.shape[2]))
    margin, gap, label_h = 16, 16, 20
    canvas = Image.new(
        "RGB",
        (margin * 2 + display_w * 2 + gap, 44 + len(selected) * (display_h + label_h + 8)),
        (16, 18, 21),
    )
    d = ImageDraw.Draw(canvas)
    before_label = "DSC before tone / DSC调色前" if dsc_enabled else "Before tone / 调色前"
    after_label = "DSC corrected / DSC调色后" if dsc_enabled else "Corrected / 调色后"
    d.text((margin, 12), before_label, fill=(130, 185, 255))
    d.text((margin + display_w + gap, 12), after_label, fill=(110, 230, 160))
    y = 42
    for idx in selected:
        d.text((margin, y), f"Frame {idx + 1}", fill=(220, 220, 220))
        y += label_h
        a = fit_image_to_panel(before[idx], display_w, display_h)
        b = fit_image_to_panel(after[idx], display_w, display_h)
        canvas.paste(a, (margin, y))
        canvas.paste(b, (margin + display_w + gap, y))
        y += display_h + 8
    canvas.save(path)


def fit_image_to_panel(frame: np.ndarray, panel_w: int, panel_h: int) -> Image.Image:
    image = Image.fromarray(frame) if frame.ndim == 3 else Image.fromarray(frame, mode="L")
    scale = min(panel_w / image.width, panel_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS).convert("RGB")
    panel = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
    panel.paste(image, ((panel_w - size[0]) // 2, (panel_h - size[1]) // 2))
    return panel


def _figure_font(size: int, bold: bool = False):
    """Use a CJK-capable Windows font when available, with a safe fallback."""
    names = ("msyhbd.ttc", "simhei.ttf", "arialbd.ttf") if bold else (
        "msyh.ttc",
        "simsun.ttc",
        "arial.ttf",
    )
    font_root = Path("C:/Windows/Fonts")
    for name in names:
        path = font_root / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _representative_indices(frame_count: int, maximum: int = 6) -> np.ndarray:
    if frame_count <= 0:
        return np.empty(0, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, min(maximum, frame_count), dtype=int))


def _best_detail_box(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Choose a repeatable high-information ROI while avoiding the sector border."""
    gray = frame.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    h, w = gray.shape
    box_h = max(24, min(h, round(h * 0.34)))
    box_w = max(24, min(w, round(w * 0.34)))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    score = (gx + gy) * (0.35 + gray / 255.0)
    score[gray <= 2] = 0
    # A coarse grid is faster and makes the selected crop stable across machines.
    best = (-1.0, max(0, (w - box_w) // 2), max(0, (h - box_h) // 2))
    x_min = min(max(0, round(w * 0.05)), max(0, w - box_w))
    x_max = max(x_min, max(0, w - box_w - round(w * 0.05)))
    y_min = min(max(0, round(h * 0.16)), max(0, h - box_h))
    y_max = max(y_min, max(0, h - box_h - round(h * 0.08)))
    xs = np.linspace(x_min, x_max, 9, dtype=int)
    ys = np.linspace(y_min, y_max, 9, dtype=int)
    for y in ys:
        for x in xs:
            tile = score[y : y + box_h, x : x + box_w]
            active = gray[y : y + box_h, x : x + box_w] > 2
            if tile.size == 0 or active.mean() < 0.25:
                continue
            central_x = (x + box_w / 2) / max(w, 1) - 0.5
            central_y = (y + box_h / 2) / max(h, 1) - 0.5
            central_weight = max(0.55, 1.0 - 0.35 * (abs(central_x) + abs(central_y)))
            value = float(np.mean(tile[active])) * central_weight
            if value > best[0]:
                best = (value, x, y)
    _, x0, y0 = best
    return x0, y0, min(w, x0 + box_w), min(h, y0 + box_h)


def _detail_pair(before: np.ndarray, after: np.ndarray, box: tuple[int, int, int, int], width: int, height: int) -> Image.Image:
    x0, y0, x1, y1 = box
    left = fit_image_to_panel(before[y0:y1, x0:x1], width // 2 - 2, height)
    right = fit_image_to_panel(after[y0:y1, x0:x1], width - width // 2 - 2, height)
    panel = Image.new("RGB", (width, height), (0, 0, 0))
    panel.paste(left, (0, 0))
    panel.paste(right, (width // 2 + 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.line((width // 2, 0, width // 2, height), fill=(255, 206, 64), width=2)
    small = _figure_font(18, bold=True)
    draw.text((8, 7), "Before", fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((width // 2 + 10, 7), "After", fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
    return panel


def _single_frame_six_panel(
    stages: list[np.ndarray],
    captions: list[str],
    title: str,
    path: Path,
):
    panel_w, panel_h = 470, 390
    margin, gap, title_h, caption_h = 26, 18, 64, 44
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + panel_w * 3 + gap * 2,
            title_h + (panel_h + caption_h) * 2 + gap + margin,
        ),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill=(22, 22, 22), font=_figure_font(26, bold=True))
    detail_box = _best_detail_box(stages[0])
    source_marked = fit_image_to_panel(stages[0], panel_w, panel_h)
    # Map the data ROI into the letterboxed display panel.
    data_h, data_w = stages[0].shape[:2]
    scale = min(panel_w / data_w, panel_h / data_h)
    ox = (panel_w - round(data_w * scale)) // 2
    oy = (panel_h - round(data_h * scale)) // 2
    x0, y0, x1, y1 = detail_box
    ImageDraw.Draw(source_marked).rectangle(
        (ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale),
        outline=(255, 206, 64),
        width=3,
    )
    rendered = [source_marked]
    rendered.extend(fit_image_to_panel(stage, panel_w, panel_h) for stage in stages[1:5])
    rendered.append(_detail_pair(stages[0], stages[3], detail_box, panel_w, panel_h))
    for i, (panel, caption) in enumerate(zip(rendered, captions)):
        row, col = divmod(i, 3)
        x = margin + col * (panel_w + gap)
        y = title_h + row * (panel_h + caption_h + gap)
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(80, 80, 80), width=1)
        draw.text((x + 4, y + panel_h + 7), caption, fill=(25, 25, 25), font=_figure_font(19))
    canvas.save(path, optimize=True)


def _multi_frame_three_row_group(
    original: np.ndarray,
    corrected: np.ndarray,
    outlined: np.ndarray,
    frame_indices: np.ndarray,
    path: Path,
):
    count = len(frame_indices)
    panel_w, panel_h = 270, 270
    label_w, top_h, row_caption_h, gap, margin = 190, 88, 28, 12, 18
    width = margin * 2 + label_w + count * panel_w + max(0, count - 1) * gap
    height = top_h + 3 * (panel_h + row_caption_h) + 2 * gap + margin
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = _figure_font(22, bold=True)
    label_font = _figure_font(18, bold=True)
    small_font = _figure_font(16)
    draw.text((margin, 12), "Ultrasound 2D processing sequence / 超声二维处理序列", fill=(20, 20, 20), font=title_font)
    row_names = (
        "Original DSC\nDSC原图",
        "Corrected 2D\n二维校正",
        "Echo outline\n回声描边",
    )
    row_data = (original, corrected, outlined)
    y = top_h
    for row, (name, data) in enumerate(zip(row_names, row_data)):
        draw.multiline_text((margin, y + panel_h // 2 - 22), name, fill=(35, 35, 35), font=label_font, spacing=3)
        x = margin + label_w
        for column, frame_index in enumerate(frame_indices):
            panel = fit_image_to_panel(data[column], panel_w, panel_h)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(85, 85, 85), width=1)
            if row == 0:
                draw.text((x + 4, y - 25), f"Frame {int(frame_index) + 1}", fill=(35, 35, 35), font=small_font)
            draw.text((x + 4, y + panel_h + 4), f"({chr(97 + row)}{column + 1})", fill=(35, 35, 35), font=small_font)
            x += panel_w + gap
        y += panel_h + row_caption_h + gap
    canvas.save(path, optimize=True)


def make_paper_style_groups(
    input_path: Path,
    output_dir: Path,
    raw: np.ndarray,
    processed: np.ndarray,
    p: ProcessParams,
    subject_masks: np.ndarray | None = None,
) -> dict:
    """Export non-ML, 2D figure groups using a paper-style fixed comparison layout."""
    indices = _representative_indices(len(raw), maximum=6)
    group_dir = output_dir / f"{input_path.stem}_二维图组"
    group_dir.mkdir(parents=True, exist_ok=True)
    selected_raw = raw[indices]
    original = dsc_scan_convert(selected_raw, p) if p.dsc_enabled else selected_raw.copy()

    denoise_params = replace(
        p,
        alpha=1.0,
        beta=0.0,
        gamma=1.0,
        edge_gain=0.0,
        shadow_fill_enabled=False,
        red_outline_enabled=False,
        transpose_output=False,
    )
    tone_params = replace(p, edge_gain=0.0, red_outline_enabled=False, transpose_output=False)
    denoised, _ = process_frames(selected_raw, denoise_params, calibration_frames=raw)
    toned, _ = process_frames(selected_raw, tone_params, calibration_frames=raw)
    final = processed[indices]
    selected_masks = subject_masks[indices] if subject_masks is not None else None
    if p.transpose_output:
        original = original.transpose(0, 2, 1)
        denoised = denoised.transpose(0, 2, 1)
        toned = toned.transpose(0, 2, 1)
        final = final.transpose(0, 2, 1)
        if selected_masks is not None:
            selected_masks = selected_masks.transpose(0, 2, 1)

    outlined_frames = []
    for local_index, frame in enumerate(final):
        outlined = (
            add_diffused_dual_outline(
                frame,
                p.red_outline_threshold,
                p.blue_outline_threshold,
                p.red_outline_diffusion_steps,
            )
            if p.red_outline_enabled
            else np.stack([frame, frame, frame], axis=-1)
        )
        if selected_masks is not None:
            outlined = add_subject_boundary(outlined, selected_masks[local_index], frame)
        outlined_frames.append(outlined)
    outlined_stack = np.stack(outlined_frames, axis=0)

    captions = [
        "(a) Original DSC / DSC原图",
        "(b) Despeckled / 去散斑",
        "(c) Tone corrected / 灰度校正",
        "(d) Edge enhanced / 边缘增强",
        "(e) Echo outlines / 回声描边",
        "(f) Detail: before | after / 局部放大",
    ]
    single_paths = []
    for local_index, frame_index in enumerate(indices):
        path = group_dir / f"六联图_帧{int(frame_index) + 1:04d}.png"
        _single_frame_six_panel(
            [
                original[local_index],
                denoised[local_index],
                toned[local_index],
                final[local_index],
                outlined_stack[local_index],
                final[local_index],
            ],
            captions,
            f"2D ultrasound processing stages — Frame {int(frame_index) + 1}",
            path,
        )
        single_paths.append(path)

    overview_path = group_dir / "跨帧三行总览.png"
    _multi_frame_three_row_group(original, final, outlined_stack, indices, overview_path)
    manifest = {
        "mode": "2D classical image processing only; no 3D reconstruction and no machine learning",
        "representative_frames_one_based": [int(i) + 1 for i in indices],
        "six_panel_layout": captions,
        "overview_layout": ["Original DSC", "Corrected 2D", "Echo outline"],
        "overview": str(overview_path),
        "single_frame_groups": [str(path) for path in single_paths],
    }
    manifest_path = group_dir / "图组说明.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    manifest["folder"] = str(group_dir)
    return manifest


class UltrasoundApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1420x980")
        self.minsize(1180, 820)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.task_queue: queue.Queue = queue.Queue()
        self.raw: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self.meta: dict | None = None
        self.dataset_info: dict = {}
        self.preview_images = []
        self.current_processed: np.ndarray | None = None
        self.live_preview_job = None
        self.live_preview_generation = 0
        self.frame_auto_params: dict[int, dict] = {}
        self.auto_diagnostics: dict = {}
        self._applying_frame_params = False
        self.subject_masks: np.ndarray | None = None
        self.subject_report: dict = {}
        self.manual_subject_seed: np.ndarray | None = None
        self.annotation_frame_index: int | None = None
        self.annotation_photo: ImageTk.PhotoImage | None = None
        self._brush_erasing = False
        self.is_busy = False
        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.controls_canvas = tk.Canvas(left, width=380, highlightthickness=0, borderwidth=0)
        controls_scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls = ttk.Frame(self.controls_canvas, padding=12)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls_width)
        controls.bind("<Enter>", self._enable_controls_wheel)
        controls.bind("<Leave>", self._disable_controls_wheel)
        preview = ttk.Frame(self, padding=(4, 12, 12, 12))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure((0, 1), weight=1)
        preview.rowconfigure((1, 3), weight=1)

        io = ttk.LabelFrame(controls, text="文件", padding=10)
        io.pack(fill="x", pady=(0, 10))
        self.input_var = tk.StringVar()
        self.folder_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.dataset_summary_var = tk.StringVar(value="请选择包含 BIN、frames.json、scan.snp 的采集目录。")
        self._entry_row(io, "采集目录", self.folder_var, self._browse_capture)
        self._entry_row(io, "输出目录", self.output_var, self._browse_output)
        ttk.Button(io, text="读取目录参数", command=self._load_capture).pack(fill="x", pady=(6, 4))
        ttk.Label(io, textvariable=self.dataset_summary_var, wraplength=330, foreground="#3a6f8f").pack(fill="x")

        layout = ttk.LabelFrame(controls, text="数据结构", padding=10)
        layout.pack(fill="x", pady=(0, 10))
        self.width_var = tk.StringVar(value="512")
        self.height_var = tk.StringVar(value="96")
        self.count_var = tk.StringVar(value="0")
        self.offset_var = tk.StringVar(value="0")
        self.transpose_var = tk.BooleanVar(value=False)
        self._grid_field(layout, 0, "宽度", self.width_var)
        self._grid_field(layout, 1, "高度", self.height_var)
        self._grid_field(layout, 2, "帧数（0=自动）", self.count_var)
        self._grid_field(layout, 3, "文件头字节", self.offset_var)
        ttk.Checkbutton(layout, text="导出时转置宽高", variable=self.transpose_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        params = ttk.LabelFrame(controls, text="校正参数", padding=10)
        params.pack(fill="x", pady=(0, 10))
        self.black_var = tk.StringVar(value="0")
        self.white_var = tk.StringVar(value="99.8")
        self.alpha_var = tk.DoubleVar(value=1.10)
        self.beta_var = tk.DoubleVar(value=0.0)
        self.gamma_var = tk.DoubleVar(value=0.92)
        self.window_var = tk.StringVar(value="5")
        self.noise_var = tk.DoubleVar(value=25.0)
        self.blend_var = tk.DoubleVar(value=0.15)
        self.despeckle_var = tk.DoubleVar(value=0.60)
        self.edge_var = tk.DoubleVar(value=0.35)
        self.bilateral_strength_var = tk.DoubleVar(value=0.45)
        self.bilateral_sigma_var = tk.DoubleVar(value=0.08)
        self.bilateral_radius_var = tk.StringVar(value="2")
        self._grid_field(params, 0, "黑电平", self.black_var)
        self._grid_field(params, 1, "白点百分位", self.white_var)
        self._slider_field(params, 2, "α 对比度增益", self.alpha_var, 0.50, 2.00, 0.01)
        self._slider_field(params, 3, "β 亮度偏移", self.beta_var, -50.0, 50.0, 1.0)
        self._slider_field(params, 4, "γ Gamma", self.gamma_var, 0.50, 2.00, 0.01)
        window_combo = ttk.Combobox(params, textvariable=self.window_var, values=(3, 5, 7, 9), width=12, state="readonly")
        window_combo.grid(row=5, column=1, sticky="ew", pady=3)
        window_combo.bind("<<ComboboxSelected>>", self._on_live_parameter_change)
        ttk.Label(params, text="Lee 窗口").grid(row=5, column=0, sticky="w", pady=3)
        self._slider_field(params, 6, "噪声百分位", self.noise_var, 5.0, 50.0, 1.0)
        self._slider_field(params, 7, "原始细节混合", self.blend_var, 0.0, 0.50, 0.01)
        self._slider_field(params, 8, "Lee 去散斑强度", self.despeckle_var, 0.0, 1.00, 0.05)
        self._slider_field(params, 9, "边缘清晰度增强", self.edge_var, 0.0, 2.00, 0.05)
        presets = ttk.Frame(params)
        presets.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(presets, text="保守", command=lambda: self._set_abg(1.00, 0, 1.00)).pack(side="left", expand=True, fill="x")
        ttk.Button(presets, text="推荐", command=lambda: self._set_abg(1.10, 0, 0.92)).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(presets, text="增强", command=lambda: self._set_abg(1.20, 5, 0.85)).pack(side="left", expand=True, fill="x")
        self.auto_button = ttk.Button(params, text="自动（整组分析＋防涂抹）", command=self._auto_adjust)
        self.auto_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        simple_filters = ttk.LabelFrame(controls, text="唯一滤波算法：双边滤波", padding=10)
        simple_filters.pack(fill="x", pady=(0, 10))
        self._slider_field(simple_filters, 0, "双边滤波强度", self.bilateral_strength_var, 0.0, 1.0, 0.05)
        ttk.Label(simple_filters, text="邻域半径（像素）").grid(row=1, column=0, sticky="w", pady=3)
        radius_combo = ttk.Combobox(simple_filters, textvariable=self.bilateral_radius_var,
                                   values=(1, 2, 3), width=12, state="readonly")
        radius_combo.grid(row=1, column=1, sticky="ew", pady=3)
        radius_combo.bind("<<ComboboxSelected>>", self._on_live_parameter_change)
        self._slider_field(simple_filters, 2, "灰度相似尺度（log）", self.bilateral_sigma_var, 0.01, 0.25, 0.01)
        ttk.Label(simple_filters, text="Lee 负责去散斑；双边滤波负责保边平滑。两者强度设为 0 可分别关闭。灰度相似尺度越小，越少混合不同亮度区域。",
                  wraplength=325, foreground="#3a6f8f").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        shadow = ttk.LabelFrame(controls, text="骨骼声影补偿（可选）", padding=10)
        shadow.pack(fill="x", pady=(0, 10))
        self.shadow_enabled_var = tk.BooleanVar(value=False)
        self.shadow_strength_var = tk.DoubleVar(value=0.45)
        ttk.Checkbutton(
            shadow,
            text="启用灰度补偿（仅用于显示）",
            variable=self.shadow_enabled_var,
            command=self._on_live_parameter_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._slider_field(
            shadow, 1, "补偿强度", self.shadow_strength_var, 0.0, 1.0, 0.05
        )
        ttk.Label(
            shadow,
            text="不会恢复被骨骼遮挡的真实结构，不应用于诊断或测量。",
            wraplength=325,
            foreground="#9a4f20",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        outline = ttk.LabelFrame(controls, text="亮区红/蓝细线描边（可选）", padding=10)
        outline.pack(fill="x", pady=(0, 10))
        self.red_outline_enabled_var = tk.BooleanVar(value=True)
        self.red_outline_threshold_var = tk.DoubleVar(value=80.0)
        self.blue_outline_threshold_var = tk.DoubleVar(value=170.0)
        self.red_outline_diffusion_var = tk.DoubleVar(value=5.0)
        ttk.Checkbutton(
            outline,
            text="启用红色整体区域＋蓝色强反声描边",
            variable=self.red_outline_enabled_var,
            command=self._on_live_parameter_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._slider_field(
            outline, 1, "整体区域阈值（红）", self.red_outline_threshold_var, 40.0, 240.0, 1.0
        )
        self._slider_field(
            outline, 2, "强反声阈值（蓝）", self.blue_outline_threshold_var, 128.0, 255.0, 1.0
        )
        self._slider_field(
            outline, 3, "连续扩散次数", self.red_outline_diffusion_var, 0.0, 12.0, 1.0
        )
        ttk.Label(
            outline,
            text="仅叠加到预览和PNG；灰度BIN不写入颜色。",
            wraplength=325,
            foreground="#8b2b2b",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        subject = ttk.LabelFrame(controls, text="主体识别与画笔标注（可选）", padding=10)
        subject.pack(fill="x", pady=(0, 10))
        self.subject_enabled_var = tk.BooleanVar(value=False)
        self.annotation_mode_var = tk.BooleanVar(value=False)
        self.subject_brush_size_var = tk.DoubleVar(value=14.0)
        self.subject_diffusion_var = tk.DoubleVar(value=18.0)
        ttk.Checkbutton(
            subject,
            text="显示绿色主体边界",
            variable=self.subject_enabled_var,
            command=self._on_live_parameter_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.subject_auto_button = ttk.Button(
            subject,
            text="自动识别主体（整组）",
            command=lambda: self._recognize_subject_group(use_manual_seed=False),
        )
        self.subject_auto_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 3))
        ttk.Checkbutton(
            subject,
            text="画笔标注模式（在左下图像拖动）",
            variable=self.annotation_mode_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self._slider_field(
            subject, 3, "画笔大小", self.subject_brush_size_var, 3.0, 40.0, 1.0
        )
        self._slider_field(
            subject, 4, "主体扩散次数", self.subject_diffusion_var, 6.0, 36.0, 1.0
        )
        subject_actions = ttk.Frame(subject)
        subject_actions.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(
            subject_actions,
            text="按标注识别整组",
            command=lambda: self._recognize_subject_group(use_manual_seed=True),
        ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(
            subject_actions,
            text="清除标注/结果",
            command=self._clear_subject_annotation,
        ).pack(side="left", expand=True, fill="x", padx=(3, 0))
        ttk.Label(
            subject,
            text="左键画主体，右键擦除；松开鼠标后自动扩散到其他帧。绿色边界会排除扇区边缘伪影。",
            wraplength=325,
            foreground="#287443",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))

        dsc = ttk.LabelFrame(controls, text="DSC 数字扫描转换", padding=10)
        dsc.pack(fill="x", pady=(0, 10))
        self.dsc_enabled_var = tk.BooleanVar(value=False)
        self.dsc_angle_var = tk.StringVar(value="60")
        self.dsc_inner_var = tk.StringVar(value="0")
        self.dsc_width_var = tk.StringVar(value="512")
        self.dsc_height_var = tk.StringVar(value="512")
        ttk.Checkbutton(
            dsc,
            text="启用扇扫 DSC（线阵数据不要启用）",
            variable=self.dsc_enabled_var,
            command=self._on_live_parameter_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._grid_field(dsc, 1, "扇扫角（度）", self.dsc_angle_var)
        self._grid_field(dsc, 2, "起始半径（采样）", self.dsc_inner_var)
        self._grid_field(dsc, 3, "输出宽度", self.dsc_width_var)
        self._grid_field(dsc, 4, "输出高度", self.dsc_height_var)

        actions = ttk.Frame(left, padding=(12, 8, 12, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.preview_button = ttk.Button(actions, text="重新读取并预览", command=self._preview)
        self.preview_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.export_button = ttk.Button(actions, text="处理并导出二维图组", command=self._export)
        self.export_button.pack(side="left", expand=True, fill="x")
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        self.status_var = tk.StringVar(value="请选择采集目录，程序将自动读取全部相关文件。")
        ttk.Label(left, textvariable=self.status_var, wraplength=350, padding=(12, 0, 12, 10)).grid(
            row=3, column=0, columnspan=2, sticky="ew"
        )

        ttk.Label(preview, text="左上｜原图").grid(row=0, column=0)
        ttk.Label(preview, text="右上｜DSC校正后").grid(row=0, column=1)
        self.original_label = ttk.Label(preview, anchor="center")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        self.dsc_label = ttk.Label(preview, anchor="center")
        self.dsc_label.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        ttk.Label(preview, text="左下｜DSC校正＋调色").grid(row=2, column=0)
        ttk.Label(preview, text="右下｜红/蓝亮区＋绿色主体边界").grid(row=2, column=1)
        self.tone_label = ttk.Label(preview, anchor="center")
        self.tone_label.grid(row=3, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        self.tone_label.bind("<Button-1>", self._on_brush_start)
        self.tone_label.bind("<B1-Motion>", self._on_brush_drag)
        self.tone_label.bind("<ButtonRelease-1>", self._on_brush_release)
        self.tone_label.bind("<Button-3>", self._on_brush_erase_start)
        self.tone_label.bind("<B3-Motion>", self._on_brush_erase_drag)
        self.tone_label.bind("<ButtonRelease-3>", self._on_brush_release)
        self.outline_label = ttk.Label(preview, anchor="center")
        self.outline_label.grid(row=3, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        nav = ttk.Frame(preview)
        nav.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        nav.columnconfigure(1, weight=1)
        ttk.Label(nav, text="帧").grid(row=0, column=0)
        self.frame_var = tk.IntVar(value=1)
        self.frame_scale = ttk.Scale(nav, from_=1, to=1, variable=self.frame_var, command=self._on_frame_change)
        self.frame_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.frame_text = ttk.Label(nav, text="1 / 1")
        self.frame_text.grid(row=0, column=2)

        for variable in (
            self.black_var,
            self.white_var,
            self.dsc_angle_var,
            self.dsc_inner_var,
            self.dsc_width_var,
            self.dsc_height_var,
        ):
            variable.trace_add("write", self._on_traced_parameter_change)

    @staticmethod
    def _entry_row(parent, label, variable, command):
        ttk.Label(parent, text=label).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 6))
        ttk.Entry(row, textvariable=variable, width=34).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="浏览", command=command, width=7).pack(side="left", padx=(5, 0))

    @staticmethod
    def _grid_field(parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky="ew", pady=3)

    def _slider_field(self, parent, row, label, variable, minimum, maximum, resolution):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        slider = tk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            variable=variable,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=205,
            command=self._on_live_parameter_change,
            highlightthickness=0,
        )
        slider.grid(row=row, column=1, sticky="ew", pady=1)

    def _update_controls_scrollregion(self, _event=None):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_width(self, event):
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _enable_controls_wheel(self, _event=None):
        self.bind_all("<MouseWheel>", self._on_controls_mousewheel)

    def _disable_controls_wheel(self, _event=None):
        self.unbind_all("<MouseWheel>")

    def _on_controls_mousewheel(self, event):
        units = -1 if event.delta > 0 else 1
        self.controls_canvas.yview_scroll(units * 3, "units")

    def _browse_capture(self):
        name = filedialog.askdirectory(title="选择包含超声采集文件的目录")
        if name:
            self.folder_var.set(name)
            self.output_var.set(str(Path(name) / "corrected_output"))
            self._load_capture()

    def _load_capture(self):
        preferred_index = self._current_frame_index()

        def task():
            try:
                info = read_capture_metadata(Path(self.folder_var.get()))
                self.task_queue.put(("capture_loaded", info, preferred_index))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))
        self.status_var.set("正在读取目录元数据并校验完整性…")
        self._run_task(task)

    def _browse_output(self):
        name = filedialog.askdirectory(title="选择输出目录")
        if name:
            self.output_var.set(name)

    def _params(self) -> ProcessParams:
        try:
            return ProcessParams(
                width=int(self.width_var.get()), height=int(self.height_var.get()), frame_count=int(self.count_var.get()),
                offset_bytes=int(self.offset_var.get()), black_level=float(self.black_var.get()),
                white_percentile=float(self.white_var.get()), alpha=float(self.alpha_var.get()),
                beta=float(self.beta_var.get()), gamma=float(self.gamma_var.get()),
                lee_window=int(self.window_var.get()), noise_percentile=float(self.noise_var.get()),
                original_blend=float(self.blend_var.get()),
                despeckle_strength=float(self.despeckle_var.get()),
                bilateral_strength=float(self.bilateral_strength_var.get()),
                bilateral_radius=int(self.bilateral_radius_var.get()),
                bilateral_sigma=float(self.bilateral_sigma_var.get()),
                edge_gain=float(self.edge_var.get()),
                red_outline_enabled=bool(self.red_outline_enabled_var.get()),
                red_outline_threshold=float(self.red_outline_threshold_var.get()),
                blue_outline_threshold=float(self.blue_outline_threshold_var.get()),
                red_outline_diffusion_steps=int(round(self.red_outline_diffusion_var.get())),
                shadow_fill_enabled=bool(self.shadow_enabled_var.get()),
                shadow_fill_strength=float(self.shadow_strength_var.get()),
                dsc_enabled=bool(self.dsc_enabled_var.get()),
                dsc_angle_deg=float(self.dsc_angle_var.get()),
                dsc_inner_radius=float(self.dsc_inner_var.get()),
                dsc_output_width=int(self.dsc_width_var.get()),
                dsc_output_height=int(self.dsc_height_var.get()),
                transpose_output=bool(self.transpose_var.get()),
            )
        except ValueError as exc:
            raise ValueError("参数中含有无效数字。") from exc

    def _set_abg(self, alpha: float, beta: float, gamma: float):
        self.alpha_var.set(alpha)
        self.beta_var.set(beta)
        self.gamma_var.set(gamma)
        self._remember_current_frame_params()
        self.status_var.set(f"已设置 α={alpha:.2f}、β={beta:g}、γ={gamma:.2f}；四宫格正在实时更新。")
        self._schedule_live_preview()

    def _parameter_values_from_controls(self) -> dict:
        return {
            "black_level": float(self.black_var.get()),
            "white_percentile": float(self.white_var.get()),
            "alpha": float(self.alpha_var.get()),
            "beta": float(self.beta_var.get()),
            "gamma": float(self.gamma_var.get()),
            "lee_window": int(self.window_var.get()),
            "noise_percentile": float(self.noise_var.get()),
            "original_blend": float(self.blend_var.get()),
            "despeckle_strength": float(self.despeckle_var.get()),
            "bilateral_strength": float(self.bilateral_strength_var.get()),
            "bilateral_radius": int(self.bilateral_radius_var.get()),
            "bilateral_sigma": float(self.bilateral_sigma_var.get()),
            "edge_gain": float(self.edge_var.get()),
            "shadow_fill_strength": float(self.shadow_strength_var.get()),
            "red_outline_threshold": float(self.red_outline_threshold_var.get()),
            "blue_outline_threshold": float(self.blue_outline_threshold_var.get()),
        }

    def _remember_current_frame_params(self):
        if self._applying_frame_params or self.raw is None or not self.frame_auto_params:
            return
        try:
            values = self._parameter_values_from_controls()
            self.frame_auto_params = {
                index: dict(values) for index in range(len(self.raw))
            }
        except ValueError:
            pass

    def _apply_frame_params(self, index: int):
        values = self.frame_auto_params.get(index)
        if not values:
            return
        self._applying_frame_params = True
        try:
            self.black_var.set(f'{values["black_level"]:g}')
            self.white_var.set(f'{values["white_percentile"]:.2f}')
            self.alpha_var.set(values["alpha"])
            self.beta_var.set(values["beta"])
            self.gamma_var.set(values["gamma"])
            self.window_var.set(str(values["lee_window"]))
            self.noise_var.set(values["noise_percentile"])
            self.blend_var.set(values["original_blend"])
            self.despeckle_var.set(values.get("despeckle_strength", 0.60))
            self.bilateral_strength_var.set(values.get("bilateral_strength", 0.45))
            self.bilateral_radius_var.set(str(values.get("bilateral_radius", 2)))
            self.bilateral_sigma_var.set(values.get("bilateral_sigma", 0.08))
            self.edge_var.set(values.get("edge_gain", 0.35))
            self.shadow_strength_var.set(values.get("shadow_fill_strength", 0.45))
            self.red_outline_threshold_var.set(values.get("red_outline_threshold", 80.0))
            self.blue_outline_threshold_var.set(values.get("blue_outline_threshold", 170.0))
        finally:
            self._applying_frame_params = False

    def _set_busy(self, busy: bool):
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.preview_button.configure(state=state)
        self.export_button.configure(state=state)
        self.auto_button.configure(state=state)
        self.subject_auto_button.configure(state=state)
        self.frame_scale.configure(state=state)
        if not busy:
            self.progress["value"] = 0

    def _run_task(self, func):
        self._set_busy(True)
        threading.Thread(target=func, daemon=True).start()

    def _infer(self):
        def task():
            try:
                path = Path(self.input_var.get())
                offset = int(self.offset_var.get())
                choices = infer_layout(path, offset)
                if not choices:
                    raise ValueError("没有找到可用候选，请手动填写宽度和高度。")
                self.task_queue.put(("inferred", choices))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))
        self.status_var.set("正在分析候选尺寸…")
        self._run_task(task)

    def _preview(self, preferred_index: int | None = None):
        self._invalidate_live_preview()
        if preferred_index is None:
            preferred_index = self._current_frame_index()

        def task():
            try:
                path, p = Path(self.input_var.get()), self._params()
                raw = read_frames(path, p)
                self.task_queue.put(("preview_loaded", raw, preferred_index))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))
        self.status_var.set("正在载入原始序列…")
        self._run_task(task)

    def _auto_adjust(self):
        """Analyze every frame, then apply one robust setting to the sequence."""
        self._invalidate_live_preview()
        frame_index = self._current_frame_index()
        try:
            params = self._params()
            input_path = Path(self.input_var.get())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def task():
            try:
                loaded_raw = None
                if self.raw is None:
                    loaded_raw = read_frames(input_path, params)
                    source = loaded_raw
                else:
                    source = self.raw
                profile = analyze_group_characteristics(
                    source,
                    params,
                    lambda x, s: self.task_queue.put(("progress", 0.15 * x, s)),
                )
                filter_shared, filter_report = auto_tune_group_filters(
                    source,
                    params,
                    profile,
                    lambda x, s: self.task_queue.put(("progress", 0.15 + 0.30 * x, s)),
                )
                tuning_params = replace(params, **filter_shared)
                recommendations = []
                frame_diagnostics = []
                tone_indices = _representative_indices(len(source), maximum=10)
                for position, index in enumerate(tone_indices):
                    frame = source[index]
                    recommended, detail = auto_tune_frame(frame, tuning_params)
                    recommendations.append(recommended)
                    detail["frame"] = int(index) + 1
                    frame_diagnostics.append(detail)
                    self.task_queue.put(
                        (
                            "progress",
                            0.45 + 0.38 * (position + 1) / len(tone_indices),
                            f"正在优化代表帧 {position + 1}/{len(tone_indices)}（原第 {index + 1} 帧）",
                        )
                    )
                shared = combine_group_recommendations(recommendations)
                # Candidate scoring owns the two denoising stages. Tone
                # analysis must not overwrite them with per-frame heuristics.
                shared.update(filter_shared)
                shared, tone_report = guard_group_tone(source, params, shared)
                diagnostics = {
                    "filter_optimization": filter_report,
                    "tone_guard": tone_report,
                    "representative_frame_diagnostics": frame_diagnostics,
                }
                shown_index = min(frame_index, len(source) - 1) if self.raw is not None else 0
                self.task_queue.put(
                    ("auto_group_result", loaded_raw, shown_index, shared, diagnostics)
                )
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        self.status_var.set("正在分析整组散斑、细节和边缘，并搜索防涂抹统一参数…")
        self._run_task(task)

    def _export(self):
        self._invalidate_live_preview()
        self._remember_current_frame_params()
        try:
            path = Path(self.input_var.get())
            params = self._params()
            output_text = self.output_var.get().strip()
            if not output_text:
                raise ValueError("请选择输出目录。")
            output_path = Path(output_text)
            subject_masks = (
                self.subject_masks.copy()
                if self.subject_enabled_var.get() and self.subject_masks is not None
                else None
            )
            subject_report = dict(self.subject_report)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def task():
            try:
                raw = read_frames(path, params)
                corrected, meta = process_frames(
                    raw,
                    params,
                    lambda x, s: self.task_queue.put(("progress", x * .65, s)),
                )
                meta["capture_metadata"] = self.dataset_info
                meta["automatic_group_optimization"] = self.auto_diagnostics
                files = export_results(
                    path,
                    output_path,
                    raw,
                    corrected,
                    meta,
                    params,
                    lambda x, s: self.task_queue.put(("progress", .65 + x * .35, s)),
                    subject_masks=subject_masks,
                    subject_report=subject_report,
                )
                self.task_queue.put(("exported", raw, corrected, meta, files))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))
        self.status_var.set("正在处理全部帧…")
        self._run_task(task)

    def _poll_queue(self):
        try:
            while True:
                item = self.task_queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    self.progress["value"] = item[1] * 100
                    self.status_var.set(item[2])
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("处理失败。")
                    messagebox.showerror(APP_TITLE, item[1])
                elif kind == "inferred":
                    self._set_busy(False)
                    choices = item[1]
                    best = choices[0]
                    self.width_var.set(str(best["width"]))
                    self.height_var.set(str(best["height"]))
                    self.count_var.set(str(best["frame_count"]))
                    lines = [
                        f'{i+1}. {c["frame_count"]}帧 × {c["height"]}行 × {c["width"]}列；相关={c["adjacent_correlation"]:.3f}'
                        for i, c in enumerate(choices[:5])
                    ]
                    self.status_var.set(f'已采用首选：{best["frame_count"]}×{best["height"]}×{best["width"]}')
                    messagebox.showinfo("尺寸推断结果", "首选候选已填入：\n\n" + "\n".join(lines) + "\n\n请结合图像预览确认。")
                elif kind == "capture_loaded":
                    self._set_busy(False)
                    info = item[1]
                    preferred_index = item[2]
                    self.dataset_info = info
                    self.raw = None
                    self.processed = None
                    self.current_processed = None
                    self.frame_auto_params.clear()
                    self.auto_diagnostics = {}
                    self.subject_masks = None
                    self.subject_report = {}
                    self.manual_subject_seed = None
                    self.annotation_frame_index = None
                    self.folder_var.set(info["captureDirectory"])
                    self.input_var.set(info["rawFile"])
                    self.width_var.set(str(info["sampleCount"]))
                    self.height_var.set(str(info["lineCount"]))
                    self.count_var.set(str(info["frameCount"]))
                    self.offset_var.set("0")
                    self.dsc_enabled_var.set(bool(info["scanAngleDeg"] > 0))
                    self.dsc_angle_var.set(f'{info["scanAngleDeg"]:.4f}')
                    self.dsc_inner_var.set(f'{info["deadRadius"]:g}')
                    self.dsc_width_var.set(str(info["suggestedDscWidth"]))
                    self.dsc_height_var.set(str(info["suggestedDscHeight"]))
                    integrity_text = "通过" if info["integrity"].get("ok") else "未通过或缺失"
                    scan = info.get("scanSnapshot", {})
                    self.dataset_summary_var.set(
                        f'探头 {info.get("probeType")}｜{info["frameCount"]}帧｜'
                        f'{info["lineCount"]}线×{info["sampleCount"]}采样｜'
                        f'扫描角 {info["scanAngleDeg"]:.2f}°｜死区 {info["deadRadius"]:g}｜'
                        f'深度 {scan.get("nominalDepth", "-")}｜完整性 {integrity_text}'
                    )
                    self.status_var.set("目录参数读取完成；正在自动载入图像并生成四宫格预览…")
                    self._preview(preferred_index)
                elif kind == "preview_loaded":
                    self._set_busy(False)
                    self.raw = item[1]
                    preferred_index = min(max(int(item[2]), 0), len(self.raw) - 1)
                    self.processed = None
                    self.current_processed = None
                    self.frame_auto_params.clear()
                    self.auto_diagnostics = {}
                    self.subject_masks = None
                    self.subject_report = {}
                    self.manual_subject_seed = None
                    self.annotation_frame_index = None
                    self.frame_scale.configure(to=len(self.raw))
                    self.frame_var.set(preferred_index + 1)
                    self.status_var.set(
                        f"已载入 {len(self.raw)} 帧；保留第 {preferred_index + 1} 帧并正在更新四宫格…"
                    )
                    self._on_frame_change()
                elif kind == "live_frame":
                    generation, frame_index = item[1], item[2]
                    original, dsc_only, corrected, outlined, meta = item[3], item[4], item[5], item[6], item[7]
                    current_index = self._current_frame_index()
                    if generation == self.live_preview_generation and frame_index == current_index:
                        self.current_processed = corrected
                        self.meta = meta
                        self.preview_images = [
                            self._photo(original),
                            self._photo(dsc_only),
                            self._photo(self._annotation_preview(corrected, frame_index)),
                            self._photo(outlined),
                        ]
                        self.original_label.configure(image=self.preview_images[0], text="")
                        self.dsc_label.configure(image=self.preview_images[1], text="")
                        self.tone_label.configure(image=self.preview_images[2], text="")
                        self.outline_label.configure(image=self.preview_images[3], text="")
                        self.status_var.set(
                            f"实时预览：第 {frame_index + 1}/{len(self.raw)} 帧｜"
                            f"α={self.alpha_var.get():.2f} β={self.beta_var.get():.0f} "
                            f"γ={self.gamma_var.get():.2f}｜"
                            f'{"整组自动统一参数" if self.frame_auto_params else "统一参数"}'
                        )
                elif kind == "live_error":
                    generation, message = item[1], item[2]
                    if generation == self.live_preview_generation:
                        self.status_var.set(f"当前帧预览失败：{message}")
                elif kind == "auto_group_result":
                    self._set_busy(False)
                    loaded_raw, frame_index, shared, diagnostics = item[1], item[2], item[3], item[4]
                    if loaded_raw is not None:
                        self.raw = loaded_raw
                        self.processed = None
                        self.frame_scale.configure(to=len(self.raw))
                    self.frame_auto_params = {
                        index: dict(shared) for index in range(len(self.raw))
                    }
                    self.auto_diagnostics = diagnostics
                    self.frame_var.set(frame_index + 1)
                    self._apply_frame_params(frame_index)
                    values = shared
                    self.frame_text.configure(text=f"{frame_index + 1} / {len(self.raw)}")
                    self.status_var.set(
                        f'整组 {len(self.raw)} 帧已采用同一套自动参数：'
                        f'α={values["alpha"]:.2f} β={values["beta"]:.0f} γ={values["gamma"]:.2f}｜'
                        f'Lee={values["despeckle_strength"]:.2f} 双边={values["bilateral_strength"]:.2f}｜'
                        f'红线={values["red_outline_threshold"]:.0f} '
                        f'蓝线={values["blue_outline_threshold"]:.0f}'
                    )
                    self._schedule_live_preview(delay_ms=20)
                elif kind == "subject_group_result":
                    self._set_busy(False)
                    self.subject_masks = item[1]
                    self.subject_report = item[2]
                    self.subject_enabled_var.set(True)
                    self.status_var.set(
                        f'主体识别完成：参考第 {self.subject_report["reference_frame"]} 帧，'
                        f'来源={self.subject_report["seed_source"]}，已传播到 {len(self.subject_masks)} 帧。'
                    )
                    self._schedule_live_preview(delay_ms=20)
                elif kind == "exported":
                    self._set_busy(False)
                    self.meta = item[3]
                    files = item[4]
                    self.status_var.set(f"导出完成：{files[0].parent}")
                    messagebox.showinfo(
                        APP_TITLE,
                        f"处理完成。\n\nBIN：{files[0]}\nPNG：{files[1]}\n二维图组总览：{files[2]}\n\n"
                        "图组仅包含二维经典图像处理，不包含三维成像或机器学习结果。",
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _current_frame_index(self) -> int:
        if self.raw is None:
            return 0
        return max(0, min(len(self.raw) - 1, round(float(self.frame_var.get())) - 1))

    def _annotation_preview(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if (
            self.manual_subject_seed is None
            or self.annotation_frame_index != frame_index
            or self.manual_subject_seed.shape != frame.shape
        ):
            return frame
        rgb = np.repeat(frame[..., None], 3, axis=2)
        seed = self.manual_subject_seed
        rgb[seed] = (255, 220, 32)
        return rgb

    def _paint_subject_seed(self, event, erase: bool):
        if (
            self.is_busy
            or not self.annotation_mode_var.get()
            or self.current_processed is None
            or self.raw is None
        ):
            return
        frame_index = self._current_frame_index()
        frame = self.current_processed
        if frame.ndim != 2:
            return
        if (
            self.manual_subject_seed is None
            or self.annotation_frame_index != frame_index
            or self.manual_subject_seed.shape != frame.shape
        ):
            self.manual_subject_seed = np.zeros(frame.shape, dtype=bool)
            self.annotation_frame_index = frame_index

        panel_w, panel_h = 430, 330
        widget_x = (self.tone_label.winfo_width() - panel_w) / 2.0
        widget_y = (self.tone_label.winfo_height() - panel_h) / 2.0
        panel_x, panel_y = event.x - widget_x, event.y - widget_y
        scale = min(panel_w / frame.shape[1], panel_h / frame.shape[0], 4.0)
        display_w, display_h = frame.shape[1] * scale, frame.shape[0] * scale
        image_x = (panel_w - display_w) / 2.0
        image_y = (panel_h - display_h) / 2.0
        x = int(round((panel_x - image_x) / scale))
        y = int(round((panel_y - image_y) / scale))
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return
        radius = max(1, int(round(float(self.subject_brush_size_var.get()) / scale)))
        y0, y1 = max(0, y - radius), min(frame.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(frame.shape[1], x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
        region = self.manual_subject_seed[y0:y1, x0:x1]
        region[disk] = not erase
        self.annotation_photo = self._photo(self._annotation_preview(frame, frame_index))
        self.tone_label.configure(image=self.annotation_photo, text="")

    def _on_brush_start(self, event):
        self._brush_erasing = False
        self._paint_subject_seed(event, erase=False)

    def _on_brush_drag(self, event):
        self._paint_subject_seed(event, erase=False)

    def _on_brush_erase_start(self, event):
        self._brush_erasing = True
        self._paint_subject_seed(event, erase=True)

    def _on_brush_erase_drag(self, event):
        self._paint_subject_seed(event, erase=True)

    def _on_brush_release(self, _event):
        if (
            self.annotation_mode_var.get()
            and self.manual_subject_seed is not None
            and np.any(self.manual_subject_seed)
        ):
            self._recognize_subject_group(use_manual_seed=True)

    def _clear_subject_annotation(self):
        self.manual_subject_seed = None
        self.annotation_frame_index = None
        self.subject_masks = None
        self.subject_report = {}
        self.subject_enabled_var.set(False)
        self.status_var.set("已清除画笔标注和主体识别结果。")
        self._schedule_live_preview(delay_ms=20)

    def _recognize_subject_group(self, use_manual_seed: bool):
        if self.raw is None:
            messagebox.showinfo(APP_TITLE, "请先选择采集目录；程序会自动载入图像。")
            return
        frame_index = self._current_frame_index()
        seed = None
        if use_manual_seed:
            if (
                self.manual_subject_seed is None
                or self.annotation_frame_index != frame_index
                or not np.any(self.manual_subject_seed)
            ):
                messagebox.showinfo(APP_TITLE, "请启用画笔标注模式，并在左下图像的主体区域拖动鼠标。")
                return
            seed = self.manual_subject_seed.copy()
        try:
            params = self._params()
            diffusion_iterations = int(round(self.subject_diffusion_var.get()))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        raw = self.raw

        def task():
            try:
                toned, _ = process_frames(
                    raw,
                    params,
                    lambda x, s: self.task_queue.put(("progress", x * 0.55, s)),
                )
                masks, report = recognize_subject_sequence(
                    toned,
                    frame_index,
                    seed,
                    diffusion_iterations,
                    lambda x, s: self.task_queue.put(("progress", 0.55 + x * 0.45, s)),
                )
                self.task_queue.put(("subject_group_result", masks, report))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        source = "画笔标注" if use_manual_seed else "自动种子"
        self.status_var.set(f"正在根据{source}识别整组主体并排除图像边缘伪影…")
        self._run_task(task)

    def _on_live_parameter_change(self, _value=None):
        self._remember_current_frame_params()
        self._schedule_live_preview()

    def _on_traced_parameter_change(self, *_args):
        if self._applying_frame_params:
            return
        self._remember_current_frame_params()
        self._schedule_live_preview(delay_ms=260)

    def _on_frame_change(self, _value=None):
        if self.raw is None:
            return
        i = self._current_frame_index()
        self._apply_frame_params(i)
        self.frame_text.configure(text=f"{i + 1} / {len(self.raw)}")
        self.preview_images = [self._photo(self.raw[i])]
        self.original_label.configure(image=self.preview_images[0], text="")
        self.dsc_label.configure(image="", text="正在生成DSC预览…")
        self.tone_label.configure(image="", text="正在调色…")
        self.outline_label.configure(image="", text="正在生成红/蓝双层描边…")
        self._schedule_live_preview(delay_ms=30)

    def _schedule_live_preview(self, _value=None, delay_ms: int = 120):
        if self.raw is None:
            return
        if self.live_preview_job is not None:
            try:
                self.after_cancel(self.live_preview_job)
            except tk.TclError:
                pass
        self.live_preview_generation += 1
        generation = self.live_preview_generation
        self.live_preview_job = self.after(delay_ms, lambda: self._start_live_preview(generation))

    def _invalidate_live_preview(self):
        self.live_preview_generation += 1
        if self.live_preview_job is not None:
            try:
                self.after_cancel(self.live_preview_job)
            except tk.TclError:
                pass
            self.live_preview_job = None

    def _start_live_preview(self, generation: int):
        self.live_preview_job = None
        if self.raw is None or generation != self.live_preview_generation:
            return
        frame_index = self._current_frame_index()
        try:
            params = self._params()
        except Exception as exc:
            self.status_var.set(f"参数无效：{exc}")
            return
        frame = self.raw[frame_index : frame_index + 1].copy()
        subject_enabled = bool(self.subject_enabled_var.get())
        subject_masks = self.subject_masks

        def task():
            try:
                corrected, meta = process_frames(
                    frame,
                    params,
                    calibration_frames=self.raw,
                )
                meta["capture_metadata"] = self.dataset_info
                meta["preview_scope"] = "current frame only"
                dsc_only = (
                    dsc_scan_convert(frame, params)[0]
                    if params.dsc_enabled
                    else frame[0]
                )
                outlined = (
                    add_diffused_dual_outline(
                        corrected[0],
                        params.red_outline_threshold,
                        params.blue_outline_threshold,
                        params.red_outline_diffusion_steps,
                    )
                    if params.red_outline_enabled
                    else corrected[0]
                )
                if (
                    subject_enabled
                    and subject_masks is not None
                    and frame_index < len(subject_masks)
                    and subject_masks[frame_index].shape == corrected[0].shape
                ):
                    outlined = add_subject_boundary(
                        outlined,
                        subject_masks[frame_index],
                        corrected[0],
                    )
                self.task_queue.put(
                    (
                        "live_frame",
                        generation,
                        frame_index,
                        frame[0],
                        dsc_only,
                        corrected[0],
                        outlined,
                        meta,
                    )
                )
            except Exception as exc:
                self.task_queue.put(("live_error", generation, str(exc)))

        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def _photo(frame: np.ndarray) -> ImageTk.PhotoImage:
        image = Image.fromarray(frame) if frame.ndim == 3 else Image.fromarray(frame, mode="L")
        panel_w, panel_h = 430, 330
        scale = min(panel_w / image.width, panel_h / image.height, 4.0)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        ).convert("RGB")
        panel = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
        panel.paste(image, ((panel_w - image.width) // 2, (panel_h - image.height) // 2))
        return ImageTk.PhotoImage(panel)


if __name__ == "__main__":
    UltrasoundApp().mainloop()
