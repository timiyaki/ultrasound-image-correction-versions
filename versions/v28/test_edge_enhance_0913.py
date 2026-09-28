"""Dependency-light smoke tests for the transplanted 0913 enhancement."""

import unittest
import time
from dataclasses import replace

import numpy as np

from edge_enhance_0913 import _directional_offsets, enhance_boundaries
from ultrasound_bin_gui import PreviewRenderer, ProcessParams, UltrasoundApp, process_frames


class EdgeEnhance0913Tests(unittest.TestCase):
    def test_zero_gain_is_identical(self):
        image = np.arange(64 * 64, dtype=np.uint8).reshape(1, 64, 64)
        np.testing.assert_array_equal(enhance_boundaries(image, 0), image)

    def test_constant_area_and_black_canvas_are_preserved(self):
        image = np.zeros((1, 96, 96), dtype=np.uint8)
        image[:, 16:80, 16:80] = 80
        result = enhance_boundaries(image, 1.0)
        self.assertTrue(np.all(result[image == 0] == 0))
        self.assertEqual(int(result[0, 48, 48]), 80)
        self.assertFalse(np.array_equal(result, image))

    def test_0913_diagonal_kernel_uses_positive_row_sign(self):
        kernel = _directional_offsets("parallel")[4]  # 45 degrees of 16 bins
        self.assertTrue(any(dy > 0 and dx > 0 for dy, dx, _ in kernel))

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            enhance_boundaries(np.zeros((32, 32), dtype=np.uint8), 1.0)

    def test_cached_preview_matches_full_processing_after_parameter_changes(self):
        rng = np.random.default_rng(5)
        raw = rng.integers(15, 210, (3, 64, 64), dtype=np.uint8)
        raw[:, :7, :] = 0
        p = ProcessParams(width=64, height=64, edge_gain=1.0)
        renderer = PreviewRenderer()
        for settings in (
            p,
            replace(p, edge_gain=0.45),
            replace(p, alpha=1.25, beta=5),
            replace(p, despeckle_strength=0.4, bilateral_strength=0.2),
        ):
            _, _, preview, _ = renderer.render(raw, 1, settings)
            full, _ = process_frames(raw[1:2], settings, calibration_frames=raw)
            np.testing.assert_array_equal(preview, full[0])

    def test_slider_refreshes_visible_preview_without_load_button(self):
        app = UltrasoundApp()
        app.withdraw()
        try:
            rng = np.random.default_rng(18)
            app.raw = rng.integers(15, 210, (2, 64, 64), dtype=np.uint8)
            app.width_var.set("64")
            app.height_var.set("64")
            app.frame_scale.configure(to=2)
            app._on_frame_change()

            deadline = time.monotonic() + 8
            while app.current_processed is None and time.monotonic() < deadline:
                app.update()
                time.sleep(0.01)
            self.assertIsNotNone(app.current_processed)
            before = app.current_processed.copy()

            app.edge_var.set(2.0)
            app._on_live_parameter_change()
            deadline = time.monotonic() + 8
            while (app.meta is None or app.meta["edge_enhancement"]["gain"] != 2.0) and time.monotonic() < deadline:
                app.update()
                time.sleep(0.01)
            self.assertEqual(app.meta["edge_enhancement"]["gain"], 2.0)
            self.assertFalse(np.array_equal(before, app.current_processed))
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
