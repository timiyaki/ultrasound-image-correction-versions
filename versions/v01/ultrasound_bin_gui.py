from __future__ import annotations

import json
import hashlib
import math
import queue
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk


APP_TITLE = "超声 BIN 灰度帧校正工具"
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
    sample_ids = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    sample = frames[sample_ids].astype(np.float32)
    z = np.log1p(sample) / math.log(256.0)
    mean = box_mean(z, radius)
    var = np.maximum(box_mean(z * z, radius) - mean * mean, 0.0)
    valid = var[sample > max(8.0, p.black_level)]
    if valid.size == 0:
        return float(np.percentile(var, p.noise_percentile))
    return float(np.percentile(valid, p.noise_percentile))


def filter_chunk(frames: np.ndarray, p: ProcessParams, noise_var: float) -> np.ndarray:
    radius = p.lee_window // 2
    src = frames.astype(np.float32)
    src = np.maximum(src - p.black_level, 0.0)
    z = np.log1p(src) / math.log(256.0)
    local_mean = box_mean(z, radius)
    local_var = np.maximum(box_mean(z * z, radius) - local_mean * local_mean, 0.0)
    weight = np.clip((local_var - noise_var) / (local_var + 1e-8), 0.0, 1.0)
    lee = local_mean + weight * (z - local_mean)
    mixed = (1.0 - p.original_blend) * lee + p.original_blend * z
    corrected = np.expm1(mixed * math.log(256.0))
    corrected[src <= 0] = 0.0
    return corrected


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
) -> tuple[np.ndarray, dict]:
    noise_var = estimate_noise_variance(frames, p)
    # Estimate one common white point to preserve brightness relationships between frames.
    sample_ids = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
    filtered_sample = filter_chunk(frames[sample_ids], p, noise_var)
    # Do not let zero-valued pixels outside/inside the acquisition footprint
    # pull the white point downward.  Tone statistics must represent echoes in
    # the active scan region rather than the black canvas around it.
    active_sample = frames[sample_ids] > p.black_level
    white_values = filtered_sample[active_sample]
    if white_values.size == 0:
        white_values = filtered_sample.reshape(-1)
    white = float(np.percentile(white_values, p.white_percentile))
    white = max(white, 1.0)

    output = np.empty_like(frames)
    chunk_size = max(1, min(16, 24_000_000 // max(frames.shape[1] * frames.shape[2], 1)))
    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        filtered = filter_chunk(frames[start:end], p, noise_var)
        normalized = np.clip(filtered / white, 0.0, 1.0)
        # Combined contrast/brightness/gamma correction:
        # y = 255 * clip(alpha*x + beta/255, 0, 1) ** gamma
        adjusted = np.clip(p.alpha * normalized + p.beta / 255.0, 0.0, 1.0)
        output[start:end] = np.rint(255.0 * np.power(adjusted, p.gamma)).astype(np.uint8)
        if progress:
            scale = 0.8 if p.dsc_enabled else 1.0
            progress(scale * end / len(frames), f"正在处理 {end}/{len(frames)} 帧")

    if p.dsc_enabled:
        output = dsc_scan_convert(
            output,
            p,
            (lambda x, s: progress(0.8 + 0.2 * x, s)) if progress else None,
        )

    meta = {
        "algorithm": "log-domain adaptive Lee",
        "gray_correction": {
            "name": "alpha-beta-gamma correction",
            "formula": "y = 255 * clip(alpha*x + beta/255, 0, 1)^gamma",
            "alpha_contrast_gain": p.alpha,
            "beta_brightness_gray_levels": p.beta,
            "gamma": p.gamma,
        },
        "estimated_log_noise_variance": noise_var,
        "global_white_value": white,
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
    }
    return output, meta


def auto_tune_frame(frame: np.ndarray, p: ProcessParams) -> tuple[dict, dict]:
    """Choose conservative tone and denoise parameters for one displayed frame.

    The objective deliberately protects highlight texture instead of maximizing
    global contrast: black remains black, only a very small highlight tail may
    approach white, and mid-gray samples inside bright tissue are not clipped.
    """
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError("自动校正需要一帧有效的二维灰度图像。")

    # A small Lee window and a substantial original contribution preserve thin
    # boundaries and isolated gray regions.  Noise percentile is adapted mildly
    # from the robust high-frequency residual, but kept in a conservative range.
    src = frame.astype(np.float32)
    local = box_mean(src[None, ...], 1)[0]
    active = src > 0
    if not np.any(active):
        raise ValueError("当前帧没有有效的非零扫描数据。")
    signal = src[active]
    dynamic = max(float(np.percentile(signal, 95) - np.percentile(signal, 10)), 1.0)
    residual = float(np.median(np.abs(src[active] - local[active]))) / dynamic
    noise_percentile = float(np.clip(15.0 + residual * 90.0, 15.0, 28.0))
    original_blend = float(np.clip(0.42 - residual * 0.45, 0.32, 0.42))

    tune = replace(
        p,
        black_level=0.0,
        lee_window=3,
        noise_percentile=noise_percentile,
        original_blend=original_blend,
        dsc_enabled=False,
    )
    noise_var = estimate_noise_variance(frame[None, ...], tune)
    filtered = filter_chunk(frame[None, ...], tune, noise_var)[0]
    values = filtered[active]
    if values.size > 120_000:
        ids = np.linspace(0, values.size - 1, 120_000, dtype=np.int64)
        values = values[ids]

    # Targets describe a diagnostic-style grayscale rendering: deep blacks,
    # readable midtones, and highlights below hard white.  The grid search is
    # intentionally constrained so it cannot produce a dramatic but destructive
    # tone curve.
    quantiles = np.array([5, 10, 25, 50, 75, 90, 95, 99, 99.8], dtype=np.float32)
    targets = np.array([3, 8, 30, 82, 148, 190, 212, 232, 242], dtype=np.float32)
    weights = np.array([1.5, 1.4, 1.0, 1.0, 1.1, 1.2, 1.4, 1.8, 2.0], dtype=np.float32)
    best: tuple[float, float, float, float, float, np.ndarray] | None = None

    for white_percentile in (99.80, 99.90, 99.95):
        white = max(float(np.percentile(values, white_percentile)), 1.0)
        x = np.clip(values / white, 0.0, 1.0)
        # A deterministic sample keeps interactive auto-tuning fast.
        for alpha in (0.90, 0.94, 0.96, 0.98, 1.00):
            for beta in (-8.0, -5.0, -2.0, 0.0):
                base = np.clip(alpha * x + beta / 255.0, 0.0, 1.0)
                for gamma in (0.90, 0.96, 1.02, 1.08, 1.14):
                    y = 255.0 * np.power(base, gamma)
                    q = np.percentile(y, quantiles)
                    score = float(np.sum(weights * ((q - targets) / 18.0) ** 2))

                    # White must not flare, while gray detail in bright tissue
                    # should remain below clipping.  Penalize both saturation and
                    # a lifted black floor strongly.
                    saturation = float(np.mean(y >= 248.0))
                    crushed = float(np.mean(y <= 1.0))
                    score += 1800.0 * saturation
                    score += 12.0 * max(0.0, q[1] - 12.0) ** 2 / 144.0
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
        "lee_window": 3,
        "noise_percentile": round(noise_percentile),
        "original_blend": round(original_blend, 2),
    }
    diagnostics = {
        "active_pixels": int(active.sum()),
        "residual_ratio": residual,
        "estimated_log_noise_variance": noise_var,
        "output_percentiles": {str(q): float(v) for q, v in zip(quantiles, output_q)},
        "design": "black/highlight/detail protected current-frame auto tone",
    }
    return recommended, diagnostics


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
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_frames = processed.transpose(0, 2, 1) if p.transpose_output else processed
    bin_path = output_dir / f"{input_path.stem}_corrected_u8.bin"
    out_frames.tofile(bin_path)

    png_dir = output_dir / f"{input_path.stem}_corrected_png"
    png_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(out_frames):
        Image.fromarray(frame, mode="L").save(png_dir / f"frame_{i + 1:04d}.png")
        if progress and (i % 10 == 0 or i + 1 == len(out_frames)):
            progress((i + 1) / len(out_frames), f"正在导出 PNG {i + 1}/{len(out_frames)}")

    compare_path = output_dir / f"{input_path.stem}_before_after.png"
    make_comparison(raw, processed, compare_path)
    meta.update(
        {
            "input_file": str(input_path),
            "binary_output": str(bin_path),
            "png_folder": str(png_dir),
            "comparison_image": str(compare_path),
            "binary_storage_shape": list(out_frames.shape),
        }
    )
    (output_dir / f"{input_path.stem}_processing.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bin_path, png_dir, compare_path


def make_comparison(raw: np.ndarray, processed: np.ndarray, path: Path):
    shape_changed = raw.shape[1:] != processed.shape[1:]
    selected = np.linspace(0, len(raw) - 1, min(4 if shape_changed else 6, len(raw)), dtype=int)
    display_w = 440 if shape_changed else min(640, max(320, raw.shape[2]))
    display_h = 360 if shape_changed else max(1, round(raw.shape[1] * display_w / raw.shape[2]))
    margin, gap, label_h = 16, 16, 20
    canvas = Image.new(
        "RGB",
        (margin * 2 + display_w * 2 + gap, 44 + len(selected) * (display_h + label_h + 8)),
        (16, 18, 21),
    )
    d = ImageDraw.Draw(canvas)
    d.text((margin, 12), "Raw / 原图", fill=(130, 185, 255))
    d.text((margin + display_w + gap, 12), "Corrected / 校正", fill=(110, 230, 160))
    y = 42
    for idx in selected:
        d.text((margin, y), f"Frame {idx + 1}", fill=(220, 220, 220))
        y += label_h
        a = fit_image_to_panel(raw[idx], display_w, display_h)
        b = fit_image_to_panel(processed[idx], display_w, display_h)
        canvas.paste(a, (margin, y))
        canvas.paste(b, (margin + display_w + gap, y))
        y += display_h + 8
    canvas.save(path)


def fit_image_to_panel(frame: np.ndarray, panel_w: int, panel_h: int) -> Image.Image:
    image = Image.fromarray(frame, mode="L")
    scale = min(panel_w / image.width, panel_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS).convert("RGB")
    panel = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
    panel.paste(image, ((panel_w - size[0]) // 2, (panel_h - size[1]) // 2))
    return panel


class UltrasoundApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x980")
        self.minsize(1100, 820)
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
        preview.rowconfigure(1, weight=1)

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
        presets = ttk.Frame(params)
        presets.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(presets, text="保守", command=lambda: self._set_abg(1.00, 0, 1.00)).pack(side="left", expand=True, fill="x")
        ttk.Button(presets, text="推荐", command=lambda: self._set_abg(1.10, 0, 0.92)).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(presets, text="增强", command=lambda: self._set_abg(1.20, 5, 0.85)).pack(side="left", expand=True, fill="x")
        self.auto_button = ttk.Button(params, text="自动（当前帧）", command=self._auto_adjust)
        self.auto_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(7, 0))

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
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._grid_field(dsc, 1, "扇扫角（度）", self.dsc_angle_var)
        self._grid_field(dsc, 2, "起始半径（采样）", self.dsc_inner_var)
        self._grid_field(dsc, 3, "输出宽度", self.dsc_width_var)
        self._grid_field(dsc, 4, "输出高度", self.dsc_height_var)

        actions = ttk.Frame(left, padding=(12, 8, 12, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.preview_button = ttk.Button(actions, text="载入并预览", command=self._preview)
        self.preview_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.export_button = ttk.Button(actions, text="处理并导出", command=self._export)
        self.export_button.pack(side="left", expand=True, fill="x")
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        self.status_var = tk.StringVar(value="请选择采集目录，程序将自动读取全部相关文件。")
        ttk.Label(left, textvariable=self.status_var, wraplength=350, padding=(12, 0, 12, 10)).grid(
            row=3, column=0, columnspan=2, sticky="ew"
        )

        ttk.Label(preview, text="原始帧").grid(row=0, column=0)
        ttk.Label(preview, text="校正后").grid(row=0, column=1)
        self.raw_label = ttk.Label(preview, anchor="center")
        self.raw_label.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.processed_label = ttk.Label(preview, anchor="center")
        self.processed_label.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        nav = ttk.Frame(preview)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        nav.columnconfigure(1, weight=1)
        ttk.Label(nav, text="帧").grid(row=0, column=0)
        self.frame_var = tk.IntVar(value=1)
        self.frame_scale = ttk.Scale(nav, from_=1, to=1, variable=self.frame_var, command=self._on_frame_change)
        self.frame_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.frame_text = ttk.Label(nav, text="1 / 1")
        self.frame_text.grid(row=0, column=2)

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
        def task():
            try:
                info = read_capture_metadata(Path(self.folder_var.get()))
                self.task_queue.put(("capture_loaded", info))
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
        self.status_var.set(f"已设置 α={alpha:.2f}、β={beta:g}、γ={gamma:.2f}；点击“载入并预览”查看效果。")
        self._schedule_live_preview()

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.preview_button.configure(state=state)
        self.export_button.configure(state=state)
        self.auto_button.configure(state=state)
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

    def _preview(self):
        self._invalidate_live_preview()
        def task():
            try:
                path, p = Path(self.input_var.get()), self._params()
                raw = read_frames(path, p)
                self.task_queue.put(("preview_loaded", raw))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))
        self.status_var.set("正在载入原始序列…")
        self._run_task(task)

    def _auto_adjust(self):
        """Analyze only the displayed frame and apply conservative parameters."""
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
                    index = 0
                else:
                    source = self.raw
                    index = min(frame_index, len(source) - 1)
                recommended, diagnostics = auto_tune_frame(source[index], params)
                self.task_queue.put(("auto_result", source if loaded_raw is not None else None, index, recommended, diagnostics))
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        self.status_var.set("正在分析当前帧的黑场、亮部和局部灰度细节…")
        self._run_task(task)

    def _export(self):
        self._invalidate_live_preview()
        def task():
            try:
                path, p = Path(self.input_var.get()), self._params()
                if not self.output_var.get().strip():
                    raise ValueError("请选择输出目录。")
                out = Path(self.output_var.get())
                raw = read_frames(path, p)
                corrected, meta = process_frames(raw, p, lambda x, s: self.task_queue.put(("progress", x * .65, s)))
                meta["capture_metadata"] = self.dataset_info
                files = export_results(path, out, raw, corrected, meta, p, lambda x, s: self.task_queue.put(("progress", .65 + x * .35, s)))
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
                    self.dataset_info = info
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
                    self.status_var.set("目录参数读取完成；DSC参数已自动填入，可直接预览。")
                elif kind == "preview_loaded":
                    self._set_busy(False)
                    self.raw = item[1]
                    self.processed = None
                    self.current_processed = None
                    self.frame_scale.configure(to=len(self.raw))
                    self.frame_var.set(1)
                    self.status_var.set(f"已载入 {len(self.raw)} 帧；正在处理当前帧…")
                    self._on_frame_change()
                elif kind == "live_frame":
                    generation, frame_index, corrected, meta = item[1], item[2], item[3], item[4]
                    current_index = self._current_frame_index()
                    if generation == self.live_preview_generation and frame_index == current_index:
                        self.current_processed = corrected
                        self.meta = meta
                        self.preview_images = [self._photo(self.raw[frame_index]), self._photo(corrected)]
                        self.raw_label.configure(image=self.preview_images[0])
                        self.processed_label.configure(image=self.preview_images[1], text="")
                        self.status_var.set(
                            f"实时预览：第 {frame_index + 1}/{len(self.raw)} 帧｜"
                            f"α={self.alpha_var.get():.2f} β={self.beta_var.get():.0f} "
                            f"γ={self.gamma_var.get():.2f}"
                        )
                elif kind == "live_error":
                    generation, message = item[1], item[2]
                    if generation == self.live_preview_generation:
                        self.status_var.set(f"当前帧预览失败：{message}")
                elif kind == "auto_result":
                    self._set_busy(False)
                    loaded_raw, frame_index, values, diagnostics = item[1], item[2], item[3], item[4]
                    if loaded_raw is not None:
                        self.raw = loaded_raw
                        self.processed = None
                        self.frame_scale.configure(to=len(self.raw))
                    self.frame_var.set(frame_index + 1)
                    self.black_var.set(f'{values["black_level"]:g}')
                    self.white_var.set(f'{values["white_percentile"]:.2f}')
                    self.alpha_var.set(values["alpha"])
                    self.beta_var.set(values["beta"])
                    self.gamma_var.set(values["gamma"])
                    self.window_var.set(str(values["lee_window"]))
                    self.noise_var.set(values["noise_percentile"])
                    self.blend_var.set(values["original_blend"])
                    self.frame_text.configure(text=f"{frame_index + 1} / {len(self.raw)}")
                    self.status_var.set(
                        f'自动参数（第 {frame_index + 1} 帧）：α={values["alpha"]:.2f} '
                        f'β={values["beta"]:.0f} γ={values["gamma"]:.2f}｜'
                        f'白点P{values["white_percentile"]:.2f}｜细节混合={values["original_blend"]:.2f}'
                    )
                    self._schedule_live_preview(delay_ms=20)
                elif kind == "exported":
                    self._set_busy(False)
                    self.meta = item[3]
                    files = item[4]
                    self.status_var.set(f"导出完成：{files[0].parent}")
                    messagebox.showinfo(APP_TITLE, f"处理完成。\n\nBIN：{files[0]}\nPNG：{files[1]}\n对比图：{files[2]}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _current_frame_index(self) -> int:
        if self.raw is None:
            return 0
        return max(0, min(len(self.raw) - 1, round(float(self.frame_var.get())) - 1))

    def _on_live_parameter_change(self, _value=None):
        self._schedule_live_preview()

    def _on_frame_change(self, _value=None):
        if self.raw is None:
            return
        i = self._current_frame_index()
        self.frame_text.configure(text=f"{i + 1} / {len(self.raw)}")
        self.preview_images = [self._photo(self.raw[i])]
        self.raw_label.configure(image=self.preview_images[0])
        self.processed_label.configure(image="", text="正在处理当前帧…")
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

        def task():
            try:
                corrected, meta = process_frames(frame, params)
                meta["capture_metadata"] = self.dataset_info
                meta["preview_scope"] = "current frame only"
                self.task_queue.put(("live_frame", generation, frame_index, corrected[0], meta))
            except Exception as exc:
                self.task_queue.put(("live_error", generation, str(exc)))

        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def _photo(frame: np.ndarray) -> ImageTk.PhotoImage:
        image = Image.fromarray(frame, mode="L")
        max_w, max_h = 430, 610
        scale = min(max_w / image.width, max_h / image.height, 4.0)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)


if __name__ == "__main__":
    UltrasoundApp().mainloop()
