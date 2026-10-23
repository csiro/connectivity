import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rust_conn import validate_device

from connectivity import beri, connectedness


class DeviceSelectionTests(unittest.TestCase):
    def test_invalid_devices_before_raster_io(self):
        for function in (connectedness, beri):
            arguments = {"condition_file": "missing-condition.tif"}
            if function is beri:
                arguments["current_file"] = "missing-current.tif"
            for device in (None, 42, "", "gpu", "cuda:bogus"):
                with self.subTest(function=function.__name__, device=device):
                    with self.assertRaisesRegex(ValueError, "device"):
                        function(**arguments, device=device)

    def test_normalized_cpu_before_raster_io(self):
        for function in (connectedness, beri):
            arguments = {"condition_file": "missing-condition.tif"}
            if function is beri:
                arguments["current_file"] = "missing-current.tif"
            with self.subTest(function=function.__name__):
                with self.assertRaises(RasterioIOError):
                    function(**arguments, device=" CPU ")

    def test_gpu_devices_before_raster_io(self):
        for device in ("metal", "cuda"):
            try:
                validate_device(device)
            except RuntimeError as error:
                message = str(error)
            else:
                message = None

            for function in (connectedness, beri):
                arguments = {"condition_file": "missing-condition.tif"}
                if function is beri:
                    arguments["current_file"] = "missing-current.tif"
                with self.subTest(function=function.__name__, device=device):
                    if message is None:
                        with self.assertRaises(RasterioIOError):
                            function(**arguments, device=f" {device.upper()} ")
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            function(**arguments, device=f" {device.upper()} ")
                        self.assertEqual(str(caught.exception), message)


class DeviceRasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.paths = {}
        rng = np.random.default_rng(21)
        condition = rng.uniform(0.1, 1.0, (8, 8)).astype(np.float32)
        condition[0, 0] = np.nan
        rasters = {
            "condition": condition,
            "resistance": rng.uniform(0.0, 1.0, (8, 8)).astype(np.float32),
            "pa": (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float32),
            "current": rng.uniform(0.0, 1.0, (2, 8, 8)).astype(np.float32),
            "future": rng.uniform(0.0, 1.0, (2, 8, 8)).astype(np.float32),
        }
        for name, values in rasters.items():
            path = Path(cls.directory.name) / f"{name}.tif"
            bands = values if values.ndim == 3 else values[np.newaxis, :, :]
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=bands.shape[0],
                dtype="float32",
                nodata=np.nan,
                crs="EPSG:3577",
                transform=from_origin(300000, 6000000, 1000, 1000),
            ) as dataset:
                dataset.write(bands)
            cls.paths[name] = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_cpu_default_and_normalized_device(self):
        for function in (connectedness, beri):
            arguments = {
                "condition_file": self.paths["condition"],
                "levels": [2],
                "closed_border": True,
                "lambdas": [2, 20],
                "n_threads": 1,
            }
            if function is beri:
                arguments["current_file"] = self.paths["current"]
                arguments["future_files"] = [self.paths["future"]]
            with self.subTest(function=function.__name__):
                expected = function(**arguments)
                actual = function(**arguments, device=" CPU ")
                self.assertEqual(actual.shape, (8, 8))
                self.assertTrue(np.isfinite(actual).any())
                np.testing.assert_array_equal(actual, expected)

    def test_gpu_matches_cpu(self):
        devices = []
        errors = []
        for device in ("metal", "cuda"):
            try:
                validate_device(device)
            except RuntimeError as error:
                errors.append(str(error))
            else:
                devices.append(device)
        if not devices:
            self.skipTest("; ".join(errors))

        cases = (
            (connectedness, {}),
            (connectedness, {"pa_file": self.paths["pa"], "pa_to_pa": True}),
            (beri, {"current_file": self.paths["current"], "future_files": [self.paths["future"]]}),
        )
        for device in devices:
            for function, options in cases:
                for window_mode in ("circular", "square"):
                    arguments = {
                        "condition_file": self.paths["condition"],
                        "resistance_file": self.paths["resistance"],
                        "levels": [2],
                        "closed_border": True,
                        "lambdas": [2, 20],
                        "n_threads": 1,
                        "window_mode": window_mode,
                        **options,
                    }
                    with self.subTest(device=device, function=function.__name__, window_mode=window_mode, options=options):
                        expected = function(**arguments, device="cpu")
                        actual = function(**arguments, device=f" {device.upper()} ")
                        self.assertTrue(np.isfinite(actual).any())
                        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
