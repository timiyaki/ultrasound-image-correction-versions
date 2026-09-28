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


APP_TITLE = "超声图像灰度校正工具 Image v10"
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
    edge_gain: float = 0.35
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
        raise ValueError("去散斑/去噪强度应在 0–1 之间。")
    if not 0.0 <= p.edge_gain <= 2.5:
        raise ValueError("主体边界增强应在 0–2.5 之间。")
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

    # A second, edge-aware pass suppresses the granular residual left by Lee.
    # Strong anatomical transitions are retained, while weak, isolated changes
    # in locally uniform tissue are treated as speckle/noise.  The operation is
    # deliberately performed in the log domain, where multiplicative ultrasound
    # speckle behaves more like additive noise.
    fine_mean = box_mean(lee, 1)
    residual = lee - fine_mean
    noise_sigma = max(math.sqrt(max(noise_var, 0.0)), 1e-4)
    edge_confidence = np.clip(np.abs(residual) / (2.2 * noise_sigma), 0.0, 1.0)
    edge_confidence = edge_confidence * edge_confidence
    edge_preserving = fine_mean + edge_confidence * residual
    despeckled = (
        (1.0 - p.despeckle_strength) * lee
        + p.despeckle_strength * edge_preserving
    )
    mixed = (1.0 - p.original_blend) * despeckled + p.original_blend * z
    corrected = np.expm1(mixed * math.log(256.0))
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
    """Multi-scale, halo-limited boundary enhancement for display images."""
    src = frames.astype(np.float32)
    fine = src - box_mean(src, 1)
    broad = src - box_mean(src, 3)
    detail = 0.60 * fine + 0.40 * broad
    # Ignore tiny fluctuations that are more likely residual speckle than anatomy.
    detail = np.sign(detail) * np.maximum(np.abs(detail) - 1.2, 0.0)
    detail = np.clip(detail, -40.0, 40.0)
    tissue_gate = np.clip((src - 10.0) / 48.0, 0.0, 1.0)
    highlight_gate = np.clip((248.0 - src) / 30.0, 0.0, 1.0)
    limited_detail = np.where(detail > 0.0, detail * highlight_gate, detail)
    enhanced = np.clip(src + gain * tissue_gate * limited_detail, 0.0, 255.0)
    enhanced[frames == 0] = 0.0
    return np.rint(enhanced).astype(np.uint8)


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
        "algorithm": "log-domain adaptive Lee + edge-aware despeckle + DSC-domain multiscale boundary enhancement",
        "gray_correction": {
            "name": "alpha-beta-gamma correction",
            "formula": "y = 255 * clip(alpha*x + beta/255, 0, 1)^gamma",
            "alpha_contrast_gain": p.alpha,
            "beta_brightness_gray_levels": p.beta,
            "gamma": p.gamma,
        },
        "estimated_log_noise_variance": noise_var,
        "global_white_value": white,
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
    }
    return output, meta


def auto_tune_frame(frame: np.ndarray, p: ProcessParams) -> tuple[dict, dict]:
    """Choose subject-aware tone, denoise and boundary parameters for one frame."""
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError("自动校正需要一帧有效的二维灰度图像。")

    # Automatic mode uses a 5x5 Lee stage followed by edge-aware residual
    # suppression.  Specklier frames receive stronger denoising and slightly
    # less original-detail mixing; later DSC-domain enhancement restores the
    # coherent anatomical outline without re-amplifying isolated noise.
    src = frame.astype(np.float32)
    local = box_mean(src[None, ...], 1)[0]
    active = src > 0
    if not np.any(active):
        raise ValueError("当前帧没有有效的非零扫描数据。")
    signal = src[active]
    dynamic = max(float(np.percentile(signal, 95) - np.percentile(signal, 10)), 1.0)
    residual = float(np.median(np.abs(src[active] - local[active]))) / dynamic
    noise_percentile = float(np.clip(25.0 + residual * 100.0, 25.0, 38.0))
    original_blend = float(np.clip(0.28 - residual * 0.55, 0.18, 0.28))
    despeckle_strength = float(np.clip(0.72 + residual * 1.5, 0.68, 0.86))
    edge_gain = float(np.clip(1.92 - residual * 3.0, 1.70, 1.90))

    tune = replace(
        p,
        black_level=0.0,
        lee_window=5,
        noise_percentile=noise_percentile,
        original_blend=original_blend,
        despeckle_strength=despeckle_strength,
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
        "algorithm": "per-frame auto-tuned log-domain adaptive Lee, edge-aware despeckle, alpha-beta-gamma correction and boundary enhancement",
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
    compare_path = output_dir / f"{input_path.stem}_dsc_tone_before_after.png"
    make_comparison(comparison_before, comparison_after, compare_path, p.dsc_enabled)
    meta.update(
        {
            "input_file": str(input_path),
            "binary_output": str(bin_path),
            "png_folder": str(png_dir),
            "comparison_image": str(compare_path),
            "comparison_mode": comparison_mode,
            "binary_storage_shape": list(out_frames.shape),
        }
    )
    (output_dir / f"{input_path.stem}_processing.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bin_path, png_dir, compare_path


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
        self.frame_auto_params: dict[int, dict] = {}
        self._applying_frame_params = False
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
        self.despeckle_var = tk.DoubleVar(value=0.60)
        self.edge_var = tk.DoubleVar(value=0.35)
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
        self._slider_field(params, 8, "去散斑/去噪强度", self.despeckle_var, 0.0, 1.00, 0.05)
        self._slider_field(params, 9, "边缘清晰度增强", self.edge_var, 0.0, 2.00, 0.05)
        presets = ttk.Frame(params)
        presets.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(presets, text="保守", command=lambda: self._set_abg(1.00, 0, 1.00)).pack(side="left", expand=True, fill="x")
        ttk.Button(presets, text="推荐", command=lambda: self._set_abg(1.10, 0, 0.92)).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(presets, text="增强", command=lambda: self._set_abg(1.20, 5, 0.85)).pack(side="left", expand=True, fill="x")
        self.auto_button = ttk.Button(params, text="自动（整组逐帧）", command=self._auto_adjust)
        self.auto_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(7, 0))

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
                despeckle_strength=float(self.despeckle_var.get()),
                edge_gain=float(self.edge_var.get()),
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
        self.status_var.set(f"已设置 α={alpha:.2f}、β={beta:g}、γ={gamma:.2f}；点击“载入并预览”查看效果。")
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
            "edge_gain": float(self.edge_var.get()),
            "shadow_fill_strength": float(self.shadow_strength_var.get()),
        }

    def _remember_current_frame_params(self):
        if self._applying_frame_params or self.raw is None or not self.frame_auto_params:
            return
        index = self._current_frame_index()
        if index not in self.frame_auto_params:
            return
        try:
            self.frame_auto_params[index] = self._parameter_values_from_controls()
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
            self.edge_var.set(values.get("edge_gain", 0.35))
            self.shadow_strength_var.set(values.get("shadow_fill_strength", 0.45))
        finally:
            self._applying_frame_params = False

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
        """Analyze the sequence and keep an independent parameter set per frame."""
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
                recommendations = []
                diagnostics = []
                for index, frame in enumerate(source):
                    recommended, detail = auto_tune_frame(frame, params)
                    recommendations.append(recommended)
                    diagnostics.append(detail)
                    self.task_queue.put(
                        ("progress", (index + 1) / len(source), f"正在自动分析第 {index + 1}/{len(source)} 帧")
                    )
                shown_index = min(frame_index, len(source) - 1) if self.raw is not None else 0
                self.task_queue.put(
                    ("auto_group_result", loaded_raw, shown_index, recommendations, diagnostics)
                )
            except Exception as exc:
                self.task_queue.put(("error", str(exc)))

        self.status_var.set("正在逐帧分析整组图像的黑场、亮部和局部灰度细节…")
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
            per_frame = {index: dict(values) for index, values in self.frame_auto_params.items()}
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def task():
            try:
                raw = read_frames(path, params)
                if len(per_frame) == len(raw) and all(i in per_frame for i in range(len(raw))):
                    ordered = [per_frame[i] for i in range(len(raw))]
                    corrected, meta = process_frames_individually(
                        raw,
                        params,
                        ordered,
                        lambda x, s: self.task_queue.put(("progress", x * .65, s)),
                    )
                else:
                    corrected, meta = process_frames(
                        raw,
                        params,
                        lambda x, s: self.task_queue.put(("progress", x * .65, s)),
                    )
                meta["capture_metadata"] = self.dataset_info
                files = export_results(
                    path,
                    output_path,
                    raw,
                    corrected,
                    meta,
                    params,
                    lambda x, s: self.task_queue.put(("progress", .65 + x * .35, s)),
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
                    self.dataset_info = info
                    self.raw = None
                    self.processed = None
                    self.current_processed = None
                    self.frame_auto_params.clear()
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
                    self.frame_auto_params.clear()
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
                            f"γ={self.gamma_var.get():.2f}｜"
                            f'{"逐帧自动参数" if frame_index in self.frame_auto_params else "统一参数"}'
                        )
                elif kind == "live_error":
                    generation, message = item[1], item[2]
                    if generation == self.live_preview_generation:
                        self.status_var.set(f"当前帧预览失败：{message}")
                elif kind == "auto_group_result":
                    self._set_busy(False)
                    loaded_raw, frame_index, recommendations, diagnostics = item[1], item[2], item[3], item[4]
                    if loaded_raw is not None:
                        self.raw = loaded_raw
                        self.processed = None
                        self.frame_scale.configure(to=len(self.raw))
                    self.frame_auto_params = {
                        index: dict(values) for index, values in enumerate(recommendations)
                    }
                    self.frame_var.set(frame_index + 1)
                    self._apply_frame_params(frame_index)
                    values = self.frame_auto_params[frame_index]
                    self.frame_text.configure(text=f"{frame_index + 1} / {len(self.raw)}")
                    self.status_var.set(
                        f'整组 {len(recommendations)} 帧已分别自动调色；当前第 {frame_index + 1} 帧：'
                        f'α={values["alpha"]:.2f} β={values["beta"]:.0f} γ={values["gamma"]:.2f}'
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
        self._remember_current_frame_params()
        self._schedule_live_preview()

    def _on_frame_change(self, _value=None):
        if self.raw is None:
            return
        i = self._current_frame_index()
        self._apply_frame_params(i)
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
