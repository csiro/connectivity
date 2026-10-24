"""Regression tests for the rust_conn bindings on small synthetic arrays.

The expected values are the frozen samples of the Rust ``multilevel_values`` test
(crates/connectivity-core/src/core.rs), reached here through the Python binding, so they
also check how arguments are passed to the core.
"""

import numpy as np
import pytest

import rust_conn

CELLS = [(0, 1), (3, 5), (6, 8), (12, 13)]
EXPECTED = {
    ("circular", False, False): [0.035036724, 0.060928639, 0.057472181, 0.023413222],
    ("circular", False, True): [0.031013189, 0.051940750, 0.050136760, 0.017623197],
    ("circular", True, False): [0.287769228, 0.330879152, 0.336351663, 0.270322204],
    ("circular", True, True): [0.287769228, 0.330879152, 0.336351663, 0.270322204],
    ("square", False, False): [0.036365397, 0.062699854, 0.057887629, 0.023465034],
    ("square", False, True): [0.032288730, 0.053552385, 0.050358951, 0.017685590],
    ("square", True, False): [0.288273066, 0.334522188, 0.339534730, 0.269480884],
    ("square", True, True): [0.288273066, 0.334522188, 0.339534730, 0.269480884],
}
F32 = np.float32


def multilevel(mode="circular", beri=False, pa_to_pa=False):
    """Arguments of the Rust ``Fixture::multilevel`` fixture."""
    i, j = np.indices((13, 14))
    cond = ((i * 7 + j * 13) % 10 + 1).astype(F32) / F32(10)
    cond[(i * 7 + j * 11) % 29 == 0] = np.nan
    mask = ~np.isfinite(cond) | ((i * 3 + j * 5) % 7 == 0)
    traversal = ((i * 5 + j * 3) % 11).astype(F32) / F32(10)
    traversal[(i + j * 3) % 13 == 0] = np.nan
    grids = None
    if beri:
        i3, j3, k3 = np.indices((13, 14, 2))
        grids = []
        for s in range(2):
            grid = ((i3 * 7 + j3 * 3 + s * 11 + k3 * 2) % 23).astype(F32) / F32(23)
            grid[(i3 * 3 + j3 * 5 + s + k3) % 31 == 0] = np.nan
            grids.append(grid)
    levels, offsets = [1, 2, 4], (3, 5)
    transforms = {}
    for level in levels:
        scale = 1000.0 * level
        row0, col0 = offsets[0] // level * level, offsets[1] // level * level
        transforms[level] = (scale, 0.0, 500_000.0 + col0 * 1000.0, 0.0, -scale, 6_000_000.0 - row0 * 1000.0)
    return dict(condition=cond, mask=mask, transgrid_list=grids, transforms=transforms, levels=levels,
                lambdas=[2.0, 20.0], is_geo=False, max_cost=20.0, window_size=3, outer_window=5,
                offsets=offsets, traversal=traversal, window_mode=mode, pa_to_pa=pa_to_pa)


def run(**changes):
    args = multilevel()
    args.update(changes)
    return rust_conn.connectivity(**args)


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_multilevel_values(key):
    args = multilevel(*key)
    out = rust_conn.connectivity(**args)
    assert out.dtype == np.float32 and out.shape == (13, 14)
    np.testing.assert_allclose([out[c] for c in CELLS], EXPECTED[key], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.isnan(out), args["mask"])


def test_band_major():
    args = multilevel(beri=True)
    expected = rust_conn.connectivity(**args)
    # np.moveaxis views, as read_raster returns multi-band rasters
    args["transgrid_list"] = [np.moveaxis(np.ascontiguousarray(np.moveaxis(g, -1, 0)), 0, -1)
                              for g in args["transgrid_list"]]
    assert not args["transgrid_list"][0].flags.c_contiguous
    np.testing.assert_array_equal(bits(rust_conn.connectivity(**args)), bits(expected))


@pytest.mark.parametrize("beri", [False, True])
def test_threads_and_repeats(beri):
    args = multilevel(beri=beri, pa_to_pa=True)
    expected = rust_conn.connectivity(**args, n_threads=1)
    for n_threads in (None, 0, 2, 4, 1):
        np.testing.assert_array_equal(bits(rust_conn.connectivity(**{**args, "n_threads": n_threads})),
                                      bits(expected))


def test_optional_inputs():
    plain = run(traversal=None)
    np.testing.assert_array_equal(bits(run(traversal=None, transgrid_list=[])), bits(plain))
    assert not np.array_equal(bits(run()), bits(plain))
    beri = multilevel(beri=True)
    grids = beri["transgrid_list"]
    expected = rust_conn.connectivity(**beri)
    beri["transgrid_list"] = [None, grids[0], None, grids[1]]
    np.testing.assert_array_equal(bits(rust_conn.connectivity(**beri)), bits(expected))


def test_mode_parsing():
    np.testing.assert_array_equal(bits(run(window_mode=" SqUaRe ")), bits(run(window_mode="square")))
    with pytest.raises(ValueError, match=r"^Invalid window_mode 'hexagon'. Expected 'square' or 'circular'.$"):
        run(window_mode="hexagon")


def test_masks_and_nan():
    args = multilevel()
    assert np.isnan(run(mask=np.ones((13, 14), dtype=bool))).all()
    # Unmasked cells without condition reach nothing and score 0
    out = run(mask=np.zeros((13, 14), dtype=bool))
    assert (out[np.isnan(args["condition"])] == 0).all()
    assert np.isfinite(out).all()


def test_no_lambdas():
    out = run(lambdas=[])
    mask = multilevel()["mask"]
    assert (out[~mask] == 0).all() and np.isnan(out[mask]).all()


@pytest.mark.parametrize("changes, error, message", [
    (dict(condition=None), RuntimeError, "Connectivity failed: No base level (key=1) in condition array"),
    (dict(mask=None), RuntimeError, "Reading mask failed: Mask must be provided!"),
    (dict(condition=np.zeros((13, 14))), RuntimeError, "Reading condition failed: Failed to extract 2D array"),
    (dict(mask=np.zeros((13, 14), dtype=np.int8)), RuntimeError, "Reading mask failed: Failed to extract mask array"),
    (dict(traversal=np.zeros((13, 14))), RuntimeError, "Reading traversal failed: Failed to extract 2D array"),
    (dict(traversal=np.zeros((12, 14), dtype=F32)), RuntimeError,
     "Connectivity failed: Dimension mismatch for key"),
    (dict(transforms=[(1, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0))]), RuntimeError,
     "Faild getting transform information: AttributeError"),
    (dict(transgrid_list=()), TypeError, "'tuple' object is not an instance of 'list'"),
])
def test_input_errors(changes, error, message):
    with pytest.raises(error) as info:
        run(**changes)
    assert type(info.value) is error
    assert str(info.value).startswith(message)


@pytest.mark.parametrize("changes, message", [
    (dict(window_size=4), "Window size must be odd-numbered"),
    (dict(window_size=3, outer_window=6), "Outer window size must be odd-numbered"),
    (dict(levels=[1, 3]), "Failed average resampling.: Level 3 is not a power of 2"),
    (dict(levels=[2, 4]), "Base condition level not found!"),
    (dict(levels=[1, 4]), "levels must form a continuous power-of-two sequence"),
    (dict(transforms={1: (1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0)}), "Missing level in Affine set."),
    (dict(transforms={2: (2000.0, 0.0, 0.0, 0.0, -2000.0, 0.0)}), "Missing base-level transform (key=1)."),
    (dict(max_cost=3.0e7), "Path cost overflow; check max_cost and resistance scaling!"),
    (dict(transgrid_list=[np.zeros((13, 14, 2))]),
     "called `Result::unwrap()` on an `Err` value: Failed to extract 3D array"),
])
def test_setting_panics(changes, message):
    # pyo3_runtime.PanicException derives from BaseException, not Exception
    with pytest.raises(BaseException) as info:
        run(**changes)
    assert type(info.value).__name__ == "PanicException"
    assert not isinstance(info.value, Exception)
    assert str(info.value).startswith(message)


def test_small_outer_window():
    np.testing.assert_array_equal(bits(run(window_size=5, outer_window=3)),
                                  bits(run(window_size=5, outer_window=5)))


TRANSFORM = (10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)


def ring(*xy):
    return np.asarray(xy, dtype=np.float64)


def test_coverage_fractions():
    # Pixel (row 1, col 2) spans x 1020..1030 and y 1980..1990
    full = ring((1020, 1990), (1030, 1990), (1030, 1980), (1020, 1980))
    half = ring((1040, 1990), (1045, 1990), (1045, 1980), (1040, 1980))
    outer = ring((1000, 1960), (1040, 1960), (1040, 1940), (1000, 1940), (1000, 1960))
    hole = ring((1010, 1955), (1030, 1955), (1030, 1945), (1010, 1945))
    out = rust_conn.pixel_coverage([(full, []), (half, []), (outer, [hole])], TRANSFORM, (6, 6))
    assert out.dtype == np.float32 and out.shape == (6, 6)
    assert out[1, 2] == 1 and out[1, 4] == 0.5
    np.testing.assert_array_equal(out[4:], [[1.0, 0.5, 0.5, 1.0, 0.0, 0.0]] * 2)
    assert out.sum() == pytest.approx(1 + 0.5 + 8 - 2)
    for n_threads in (1, 3):
        np.testing.assert_array_equal(
            rust_conn.pixel_coverage([(full, []), (half, []), (outer, [hole])], TRANSFORM, (6, 6), n_threads),
            out)
    np.testing.assert_array_equal(rust_conn.pixel_coverage([], TRANSFORM, (2, 3)), np.zeros((2, 3), F32))


@pytest.mark.parametrize("polygons, message", [
    ((), "Reading polygons failed: Expected a list of polygons"),
    ([(np.zeros((4, 3)), [])], "Reading polygons failed: Ring array must have shape (N, 2), got [4, 3]"),
    ([(np.zeros((4, 2)), ())], "Reading polygons failed: Holes must be a list of arrays"),
])
def test_coverage_errors(polygons, message):
    with pytest.raises(RuntimeError) as info:
        rust_conn.pixel_coverage(polygons, TRANSFORM, (6, 6))
    assert str(info.value).startswith(message)
