"""Regression tests for the Python workflows on the bundled Tasmania example data.

Exact checks (NaN placement, bit-identical repeats, seamless tiles, Connected Habitat
post-processing) are platform independent; the frozen means come from the 2.4.0 baseline
and allow for floating-point differences between platforms.
"""

import numpy as np
import pytest
import rasterio

import connectivity as cn
from connectivity import example_data_path
from connectivity.rastio import read_raster
from connectivity.utils import _crop_array

COND = example_data_path("site_condition")
PA = example_data_path("pa_proportion")
CURRENT = example_data_path("transgrids/1990")
FUTURES = [example_data_path(f"transgrids/{n}") for n in ("ACC50_85", "GFD50_85", "IPS50_45")]
THREADS = 4
SHAPE = (516, 588)
RTOL = 1e-5

# Frozen from the 2.4.0 baseline build (default parameters unless noted)
FROZEN = {
    "connectedness": 0.3594061735617171,
    "parc": 0.6910263746707314,
    "parc_pa_to_pa": 0.5578077798671234,
    "beri": 0.40609480758632494,
    "beri_current": 0.46590158319867997,
    "resistance": 0.3100888245434447,
    "square_short": 0.4162536757680946,  # square windows, levels [2, 4, 8], outer_window 5
}


def bits(a):
    return np.ascontiguousarray(np.ma.getdata(a)).view(np.uint32)


def nanmean(a):
    return float(np.nanmean(np.asarray(np.ma.getdata(a), dtype=np.float64)))


@pytest.fixture(scope="module")
def condition():
    return read_raster(COND, levels=[1])[0]


@pytest.fixture(scope="module")
def connectedness():
    return cn.connectedness(condition_file=COND, option=1, n_threads=THREADS)


def test_habitat_options(condition, connectedness):
    assert connectedness.dtype == np.float32 and connectedness.shape == SHAPE
    np.testing.assert_array_equal(np.isnan(connectedness), np.isnan(condition))
    finite = connectedness[np.isfinite(connectedness)]
    assert finite.min() >= 0 and finite.max() <= 1
    assert nanmean(connectedness) == pytest.approx(FROZEN["connectedness"], rel=RTOL)
    # Connected Habitat stays in Python: option 2 = condition x connectedness, 3 = its sqrt
    option2 = cn.connectedness(condition_file=COND, option=2, n_threads=THREADS)
    option3 = cn.connectedness(condition_file=COND, n_threads=THREADS)
    np.testing.assert_array_equal(bits(option2), bits(condition * connectedness))
    np.testing.assert_array_equal(bits(option3), bits(np.sqrt(np.multiply(condition, connectedness, dtype=np.float32))))
    with pytest.raises(ValueError, match="option must be one of 1, 2, or 3"):
        cn.connectedness(condition_file=COND, option=4, levels=[2, 4, 8], outer_window=5, n_threads=THREADS)


def test_thread_counts():
    kwargs = dict(condition_file=COND, levels=[2, 4, 8], outer_window=5, option=1)
    expected = cn.connectedness(**kwargs, n_threads=1)
    for n_threads in (2, THREADS, None):
        np.testing.assert_array_equal(bits(cn.connectedness(**kwargs, n_threads=n_threads)), bits(expected))


def test_window_modes(connectedness):
    square = cn.connectedness(condition_file=COND, option=1, window_mode="square", levels=[2, 4, 8],
                              outer_window=5, n_threads=THREADS)
    np.testing.assert_array_equal(np.isnan(square), np.isnan(connectedness))
    assert nanmean(square) == pytest.approx(FROZEN["square_short"], rel=RTOL)
    with pytest.raises(ValueError, match="window_mode must be one of"):
        cn.connectedness(condition_file=COND, window_mode="block")


def test_parc_and_pa_to_pa(condition):
    pa = read_raster(PA, levels=[1])[0]
    parc = cn.connectedness(condition_file=COND, pa_file=PA, n_threads=THREADS)
    gated = cn.connectedness(condition_file=COND, pa_file=PA, pa_to_pa=True, n_threads=THREADS)
    # PARC evaluates protected cells only; the output is connectedness (no Connected Habitat)
    protected = (pa > 0) & ~np.isnan(np.maximum(condition, pa))
    np.testing.assert_array_equal(np.isfinite(parc), protected)
    np.testing.assert_array_equal(np.isfinite(gated), protected)
    assert nanmean(parc) == pytest.approx(FROZEN["parc"], rel=RTOL)
    assert nanmean(gated) == pytest.approx(FROZEN["parc_pa_to_pa"], rel=RTOL)
    # Gating only removes unprotected destinations from the numerator
    assert (gated[protected] <= parc[protected]).all()
    assert (gated[protected] < parc[protected]).any()


def test_beri(connectedness):
    out = cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES, n_threads=THREADS)
    current = cn.beri(condition_file=COND, current_file=CURRENT, n_threads=THREADS)
    for result in (out, current):
        assert result.dtype == np.float32 and result.shape == SHAPE
        np.testing.assert_array_equal(np.isnan(result), np.isnan(connectedness))
        assert (result[np.isfinite(result)] >= 0).all()
    assert nanmean(out) == pytest.approx(FROZEN["beri"], rel=RTOL)
    assert nanmean(current) == pytest.approx(FROZEN["beri_current"], rel=RTOL)
    assert nanmean(out) < nanmean(current)


def write_resistance(path):
    """Resistance rising from west to east, missing in a few rows of valid condition."""
    with rasterio.open(COND) as src:
        profile = src.profile | {"nodata": np.nan, "dtype": "float32"}
        valid = np.isfinite(src.read(1, masked=True).filled(np.nan))
    cols = np.broadcast_to(np.arange(SHAPE[1], dtype=np.float32), SHAPE)
    resistance = np.where(valid, np.float32(0.2) + np.float32(0.6) * cols / np.float32(SHAPE[1]), np.nan)
    resistance[300:303] = np.nan
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(resistance.astype(np.float32), 1)
    return int((valid[300:303]).sum())


def test_resistance_surface(tmp_path, connectedness):
    res_file = str(tmp_path / "resistance.tif")
    gaps = write_resistance(res_file)
    with pytest.warns(UserWarning, match=f"^{gaps} cell\\(s\\) have valid condition but missing resistance"):
        out = cn.connectedness(condition_file=COND, resistance_file=res_file, option=1, n_threads=THREADS)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(connectedness))
    assert not np.array_equal(bits(out), bits(connectedness))
    assert nanmean(out) == pytest.approx(FROZEN["resistance"], rel=RTOL)


def test_seamless_tiles():
    whole = cn.connectedness(condition_file=COND, n_threads=THREADS)
    with rasterio.open(COND) as src:
        transform = src.transform
    tiles = cn.make_tiles(COND, nrows=2, ncols=2, align_to=32)
    assert list(tiles["row_start"]) == [0, 0, 256, 256] and list(tiles["col_start"]) == [0, 288, 0, 288]
    for k in range(len(tiles)):
        tile = tiles.iloc[[k]]
        out = cn.connectedness(condition_file=COND, polygon_mask=tile, n_threads=THREADS)
        expected, _ = _crop_array(whole, transform, tile)
        assert np.ma.isMaskedArray(out) and out.shape == expected.shape
        np.testing.assert_array_equal(np.ma.getmaskarray(out), np.ma.getmaskarray(expected))
        # Open-border tiles see the same overview cells and windows as the whole raster
        inside = ~np.ma.getmaskarray(out)
        np.testing.assert_array_equal(bits(out)[inside], bits(expected)[inside])
        closed = cn.connectedness(condition_file=COND, polygon_mask=tile, closed_border=True, n_threads=THREADS)
        assert np.isfinite(closed).any()


def test_pixel_coverage(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.ops import unary_union

    shapes = [Point(146.5, -42.0).buffer(0.8).difference(Point(146.4, -42.1).buffer(0.3)),
              box(144.0, -41.0, 145.0, -40.5)]
    polygons = gpd.GeoDataFrame(geometry=shapes, crs="EPSG:4326")
    out_file = str(tmp_path / "coverage.tif")
    coverage = cn.pixel_coverage(polygons, COND, n_threads=THREADS, filename=out_file)
    assert coverage.dtype == np.float32 and coverage.shape == SHAPE
    assert coverage.min() == 0 and coverage.max() == 1
    with rasterio.open(COND) as src:
        pixel_area = abs(src.transform.a * src.transform.e)
    assert coverage.sum(dtype=np.float64) * pixel_area == pytest.approx(unary_union(shapes).area, rel=1e-5)
    for n_threads in (1, None):
        np.testing.assert_array_equal(cn.pixel_coverage(polygons, COND, n_threads=n_threads), coverage)
    with rasterio.open(out_file) as written:
        np.testing.assert_array_equal(written.read(1), coverage)
