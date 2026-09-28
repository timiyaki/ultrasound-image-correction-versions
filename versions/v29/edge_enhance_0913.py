"""NumPy port of the enhancement path in edgeEnhance-0913/extract_main_boundary.m.

This is the image-enhancement path only. Its separate contour-extraction and
optional post-enhancement tone-curve paths are intentionally not used: the
host application already has its own ROI annotation and alpha/beta/gamma tone.
"""

from functools import lru_cache

import numpy as np


# edgeEnhance-0913/edgeEnhanceApp.m default preset (not the lower-level
# extract_main_boundary.m function defaults).
PRE_SIGMA = 1.0
GRAD_HW = 3
GRAD_SIGMA = 2.0
EDGE_C = 2.5
COH_MIN = 0.30
ALPHA_LO = 0.15
ALPHA_HI = 0.50
PAR_HW = 2
PAR_SIGMA = 2.0
PAR_ITER = 2
PERP_HW = 2
PERP_SIGMA = 3.0
PERP_ITER = 1
EEF_PERP = 2.0
N_DIR = 16


def _gaussian_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return weights / weights.sum()


def _filter_1d(image: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    radius = len(weights) // 2
    pad = ((radius, radius), (0, 0)) if axis == 0 else ((0, 0), (radius, radius))
    padded = np.pad(image, pad, mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    for offset, weight in enumerate(weights):
        if axis == 0:
            result += weight * padded[offset : offset + height, :]
        else:
            result += weight * padded[:, offset : offset + width]
    return result


def _gaussian(image: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    weights = _gaussian_1d(sigma, radius)
    return _filter_1d(_filter_1d(image, weights, 1), weights, 0)


def _line_coefficients(hw: int, variance: float, iterations: int, eef: float | None) -> np.ndarray:
    base = _gaussian_1d(np.sqrt(variance), hw)
    if eef is None:
        one_pass = base
    else:
        # Match the 0913 MATLAB source: negative side lobes and a centre
        # coefficient of EefPerp. Do not silently substitute a DoG kernel.
        one_pass = (1.0 - eef) * base
        one_pass[hw] = eef
    weights = one_pass
    for _ in range(1, iterations):
        weights = np.convolve(weights, one_pass).astype(np.float32)
    return weights


@lru_cache(maxsize=2)
def _directional_offsets(kind: str) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    if kind == "parallel":
        coeff = _line_coefficients(PAR_HW, PAR_SIGMA, PAR_ITER, None)
    elif kind == "perpendicular":
        coeff = _line_coefficients(PERP_HW, PERP_SIGMA, PERP_ITER, EEF_PERP)
    else:
        raise ValueError("Unknown directional kernel")
    half = len(coeff) // 2
    kernels = []
    for direction in range(N_DIR):
        angle = direction * np.pi / N_DIR
        step = 1.0 / max(abs(np.cos(angle)), abs(np.sin(angle)))
        offsets: dict[tuple[int, int], float] = {}
        for t, weight in zip(range(-half, half + 1), coeff):
            # The 0913 source uses +sin for the image row coordinate. This
            # sign is deliberately preserved; later edgeEnhance.zip flips it.
            dx = int(np.sign(t * step * np.cos(angle)) * np.floor(abs(t * step * np.cos(angle)) + 0.5))
            dy = int(np.sign(t * step * np.sin(angle)) * np.floor(abs(t * step * np.sin(angle)) + 0.5))
            offsets[(dy, dx)] = offsets.get((dy, dx), 0.0) + float(weight)
        kernels.append(tuple((dy, dx, value) for (dy, dx), value in offsets.items()))
    return tuple(kernels)


def _sparse_filter(image: np.ndarray, kernel: tuple[tuple[int, int, float], ...]) -> np.ndarray:
    radius = max(max(abs(dy), abs(dx)) for dy, dx, _ in kernel)
    padded = np.pad(image, radius, mode="edge")
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    for dy, dx, weight in kernel:
        result += weight * padded[radius + dy : radius + dy + height,
                                  radius + dx : radius + dx + width]
    return result


def _directional_filter(image: np.ndarray, angle_deg: np.ndarray, kind: str) -> np.ndarray:
    # This is the 0913 triangular interpolation between neighbouring direction
    # bins; angles are 180-degree periodic because a line has no arrowhead.
    coordinate = np.mod(angle_deg, 180.0) * (N_DIR / 180.0)
    lower = np.floor(coordinate).astype(np.uint8)
    fraction = coordinate - lower
    result = np.zeros_like(image, dtype=np.float32)
    for direction, kernel in enumerate(_directional_offsets(kind)):
        mask_lo = lower == direction
        mask_hi = (lower + 1) % N_DIR == direction
        if not (np.any(mask_lo) or np.any(mask_hi)):
            continue
        weight = (1.0 - fraction) * mask_lo + fraction * mask_hi
        result += weight * _sparse_filter(image, kernel)
    return result


def prepare_edge_response(frame: np.ndarray) -> np.ndarray:
    """Compute the gain-independent 0913 response for one grayscale frame."""
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError("Expected one uint8 grayscale frame")
    source = frame.astype(np.float32)
    smooth = _gaussian(source, PRE_SIGMA, 3)
    padded = np.pad(smooth, 1, mode="edge")
    gx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    tensor_sigma = np.sqrt(GRAD_SIGMA)
    jxx = _gaussian(gx * gx, tensor_sigma, GRAD_HW)
    jxy = _gaussian(gx * gy, tensor_sigma, GRAD_HW)
    jyy = _gaussian(gy * gy, tensor_sigma, GRAD_HW)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-7)
    energy = np.sqrt(jxx + jyy)
    reference = max(float(np.percentile(energy, 90)), 1e-7)
    confidence = np.minimum(1.0, energy / (EDGE_C * reference)) * np.clip(coherence, 0.0, 1.0)
    confidence[coherence < COH_MIN] = 0.0

    normal = np.rad2deg(theta)
    along = _directional_filter(smooth, normal + 90.0, "parallel")
    across = _directional_filter(along, normal, "perpendicular")
    t = np.clip((confidence - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0)
    active_weight = confidence * (t * t * (3.0 - 2.0 * t))
    response = (active_weight * (across - source)).astype(np.float32)
    response[frame == 0] = 0.0
    return response


def apply_edge_response(frame: np.ndarray, response: np.ndarray, gain: float) -> np.ndarray:
    """Apply a different Gain without recalculating any directional filters."""
    if frame.ndim != 2 or frame.dtype != np.uint8 or response.shape != frame.shape:
        raise ValueError("Frame and prepared edge response must have the same 2D shape")
    if gain <= 0:
        return frame.copy()
    result = frame.astype(np.float32) + gain * response
    result[frame == 0] = 0.0
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def enhance_boundaries(frames: np.ndarray, gain: float) -> np.ndarray:
    """Return the 0913 direction-adaptive enhancement for 8-bit B-mode frames."""
    if frames.ndim != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 frames with shape [frame, row, column]")
    if gain <= 0:
        return frames.copy()
    output = np.empty_like(frames)
    for index, frame in enumerate(frames):
        output[index] = apply_edge_response(frame, prepare_edge_response(frame), gain)
    return output
