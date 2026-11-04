"""Regression tests for the Python workflows on the bundled Tasmania example data.

Exact checks (NaN placement, bit-identical repeats, seamless tiles, Connected Habitat
post-processing) are platform independent; the frozen means come from the 2.5.0 build
and allow for floating-point differences between platforms.
"""

import warnings

import numpy as np
import pytest
import rasterio
from affine import Affine

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
SHORT = dict(levels=[2, 4], outer_window=3, window_size=3, n_threads=THREADS)

# Frozen from the 2.5.0 build (default parameters unless noted, window_size 5); the R
# package 0.1.0 gives the same means to within 1e-7 relative
FROZEN = {
    "connectedness": 0.36226272030321477,
    "parc": 0.6971164168245471,
    "parc_pa_to_pa": 0.5655157079463478,
    "beri": 0.4077840752430562,
    "beri_current": 0.46763685714287495,
    "resistance": 0.31148093179243047,
    "square_short": 0.4180251384944627,  # square windows, levels [2, 4, 8], outer_window 5
}


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def nanmean(a):
    return float(np.nanmean(np.asarray(a, dtype=np.float64)))


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


def write_grid(path, crs, transform):
    """The condition values with another CRS (None for none) and transform."""
    with rasterio.open(COND) as src:
        profile = src.profile | {"crs": crs, "transform": transform}
        values = src.read(1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return str(path)


def moved(path, source, cols=0, rows=0, bands=None, pad=0):
    """`source` moved by (fractional) pixels, with only `bands` and `pad` nodata cells around it."""
    with rasterio.open(source) as src:
        values, profile = src.read(bands), src.profile
    nodata = np.nan if profile["nodata"] is None else profile["nodata"]
    values = np.pad(values, ((0, 0), (pad, pad), (pad, pad)), constant_values=nodata)
    profile |= {"transform": profile["transform"] * Affine.translation(cols - pad, rows - pad),
                "count": values.shape[0], "height": values.shape[1], "width": values.shape[2]}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values)
    return str(path)


def test_input_grids(tmp_path):
    # Every input must share the condition grid and cover the analysis window, as in R
    cover = "must cover the analysis window on the same grid as condition_file$"
    with pytest.raises(ValueError, match=f"^pa_file {cover}"):
        cn.connectedness(condition_file=COND, pa_file=moved(tmp_path / "pa.tif", PA, cols=50), **SHORT)
    future = moved(tmp_path / "future.tif", FUTURES[1], rows=50)
    with pytest.raises(ValueError, match=f"^future_files\\[1\\] {cover}"):
        cn.beri(condition_file=COND, current_file=CURRENT, future_files=[FUTURES[0], future], **SHORT)
    with pytest.raises(ValueError, match=f"^resistance_file {cover}"):
        cn.connectedness(condition_file=COND, resistance_file=moved(tmp_path / "half.tif", COND, cols=0.5), **SHORT)
    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    same = "must have the same CRS, resolution and grid alignment as condition_file$"
    with pytest.raises(ValueError, match=f"^resistance_file {same}"):
        cn.connectedness(condition_file=COND, resistance_file=write_grid(tmp_path / "3577.tif", "EPSG:3577", transform),
                         **SHORT)
    with pytest.raises(ValueError, match=f"^current_file {same}"):
        cn.beri(condition_file=COND, current_file=write_grid(tmp_path / "coarse.tif", crs, transform * Affine.scale(2)),
                **SHORT)
    with pytest.raises(ValueError, match="^pa_file must have exactly one layer$"):
        cn.connectedness(condition_file=COND, pa_file=CURRENT, **SHORT)
    fewer = moved(tmp_path / "bands.tif", FUTURES[0], bands=list(range(1, 19)))
    with pytest.raises(ValueError, match="^Every climate scenario must have the same band count as current_file$"):
        cn.beri(condition_file=COND, current_file=CURRENT, future_files=[fewer], **SHORT)


def test_larger_inputs(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    # Larger rasters on the condition grid are read on the condition's window
    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    x0, y0, r = transform.c, transform.f, transform.a
    polygon = gpd.GeoDataFrame(geometry=[box(x0 + 100.3 * r, y0 - 300.7 * r, x0 + 220.6 * r, y0 - 180.2 * r)], crs=crs)
    pa = moved(tmp_path / "pa.tif", PA, pad=7)
    resistance = str(tmp_path / "resistance.tif")
    write_resistance(resistance)
    larger = moved(tmp_path / "larger.tif", resistance, pad=5)
    grids = [moved(tmp_path / f"{k}.tif", f, pad=3 + k) for k, f in enumerate([CURRENT, *FUTURES[:2]])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # missing resistance cells
        for border in ({}, dict(polygon_mask=polygon), dict(polygon_mask=polygon, closed_border=True)):
            args = SHORT | border
            np.testing.assert_array_equal(
                bits(cn.connectedness(condition_file=COND, pa_file=pa, resistance_file=larger, **args)),
                bits(cn.connectedness(condition_file=COND, pa_file=PA, resistance_file=resistance, **args)))
            np.testing.assert_array_equal(
                bits(cn.beri(condition_file=COND, current_file=grids[0], future_files=grids[1:], **args)),
                bits(cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES[:2], **args)))


def test_single_band_beri(tmp_path):
    # Climate rasters with one band keep their band axis, as in R
    current, future = (moved(tmp_path / f"{k}.tif", f, bands=[1]) for k, f in enumerate([CURRENT, FUTURES[0]]))
    out = cn.beri(condition_file=COND, current_file=current, future_files=[future], **SHORT)
    reference = cn.connectedness(condition_file=COND, **SHORT)
    assert out.dtype == np.float32 and out.shape == SHAPE
    np.testing.assert_array_equal(np.isnan(out), np.isnan(reference))
    assert (out[np.isfinite(out)] >= 0).all()
    assert not np.array_equal(bits(out), bits(cn.beri(condition_file=COND, current_file=CURRENT,
                                                       future_files=FUTURES[:1], **SHORT)))


def test_crs_units(tmp_path):
    # Projected units are converted to metres for the engine, as in R, so one grid in metres,
    # kilometres or US survey feet gives the same result
    aea = "+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +ellps=GRS80 +units={} +no_defs"
    out = {}
    for units, metres in (("m", 1.0), ("km", 1000.0), ("us-ft", 1200 / 3937)):
        transform = Affine(1000 / metres, 0, 1e6 / metres, 0, -1000 / metres, -4e6 / metres)
        grid = write_grid(tmp_path / f"{units}.tif", aea.format(units), transform)
        out[units] = cn.connectedness(condition_file=grid, **SHORT)
    np.testing.assert_array_equal(bits(out["km"]), bits(out["m"]))
    np.testing.assert_allclose(out["us-ft"], out["m"], rtol=1e-6)  # feet are inexact in binary


def test_missing_crs(tmp_path):
    with rasterio.open(COND) as src:
        transform = src.transform
    # Cells smaller than one unit are taken as degrees, larger cells as metres
    with pytest.warns(UserWarning, match="^Raster CRS is missing; assuming longitude/latitude degrees$"):
        geo = cn.connectedness(condition_file=write_grid(tmp_path / "geo.tif", None, transform), **SHORT)
    np.testing.assert_array_equal(bits(geo), bits(cn.connectedness(condition_file=COND, **SHORT)))
    metric = Affine(1000, 0, 1e6, 0, -1000, -4e6)
    with pytest.warns(UserWarning, match="^Raster CRS is missing; assuming projected metres$"):
        proj = cn.connectedness(condition_file=write_grid(tmp_path / "proj.tif", None, metric), **SHORT)
    albers = cn.connectedness(condition_file=write_grid(tmp_path / "m.tif", "EPSG:3577", metric), **SHORT)
    np.testing.assert_array_equal(bits(proj), bits(albers))


def test_polygon_crs(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    x0, y0, r = transform.c, transform.f, transform.a
    naive = gpd.GeoDataFrame(geometry=[box(x0 + 100.3 * r, y0 - 300.7 * r, x0 + 220.6 * r, y0 - 180.2 * r)])
    # The polygon and the raster must both have a CRS, or both omit one, as in R
    rule = "and the raster must both have a CRS, or both omit one$"
    with pytest.raises(ValueError, match=f"^polygon_mask {rule}"):
        cn.connectedness(condition_file=COND, polygon_mask=naive, **SHORT)
    with pytest.raises(ValueError, match=f"^polygon {rule}"):
        cn.pixel_coverage(naive, COND)
    no_crs = write_grid(tmp_path / "no_crs.tif", None, transform)
    with pytest.warns(UserWarning, match="Raster CRS is missing"), pytest.raises(ValueError, match=rule):
        cn.beri(condition_file=no_crs, current_file=CURRENT, polygon_mask=naive.set_crs(crs), **SHORT)
    # Without either CRS, the polygon is in the raster's coordinates
    with pytest.warns(UserWarning, match="Raster CRS is missing"):
        out = cn.connectedness(condition_file=no_crs, polygon_mask=naive, **SHORT)
    expected = cn.connectedness(condition_file=COND, polygon_mask=naive.set_crs(crs), **SHORT)
    np.testing.assert_array_equal(bits(out), bits(expected))


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
        assert type(out) is np.ndarray and out.dtype == np.float32 and out.shape == expected.shape
        # Open-border tiles see the same overview cells and windows as the whole raster
        inside = np.isfinite(expected)
        np.testing.assert_array_equal(np.isfinite(out), inside)
        np.testing.assert_array_equal(bits(out)[inside], bits(expected)[inside])
        closed = cn.connectedness(condition_file=COND, polygon_mask=tile, closed_border=True, n_threads=THREADS)
        assert np.isfinite(closed).any()


def test_polygon_cells(connectedness):
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from shapely.geometry import Point, box

    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    x0, y0, r = transform.c, transform.f, transform.a
    shapes = [box(x0 + 100.3 * r, y0 - 300.7 * r, x0 + 220.6 * r, y0 - 180.2 * r),  # off pixel edges
              box(x0 + 40 * r, y0 - 160 * r, x0 + 130 * r, y0 - 70 * r),  # on pixel edges
              Point(x0 + 300.45 * r, y0 - 250.55 * r).buffer(60.3 * r)]
    for shape in shapes:
        polygon = gpd.GeoDataFrame(geometry=[shape], crs=crs)
        out = cn.connectedness(condition_file=COND, polygon_mask=polygon, option=1, n_threads=THREADS)
        assert type(out) is np.ndarray and out.dtype == np.float32
        # Open borders return every valid cell the polygon touches, as the whole raster scores
        # it; R's connectedness() returns the same cells and values
        touched = ~geometry_mask([shape], SHAPE, transform, all_touched=True)
        assert np.isfinite(out).sum() == (touched & np.isfinite(connectedness)).sum()
        expected, _ = _crop_array(connectedness, transform, polygon)
        np.testing.assert_array_equal(bits(out), bits(expected))
    # The crop reprojects a polygon in another CRS, as the read does
    polygon = gpd.GeoDataFrame(geometry=shapes[:1], crs=crs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # no margin is used internally
        out = cn.connectedness(condition_file=COND, polygon_mask=polygon, option=1, n_threads=THREADS)
    projected = cn.connectedness(condition_file=COND, polygon_mask=polygon.to_crs(3577), option=1,
                                 n_threads=THREADS)
    np.testing.assert_array_equal(bits(projected), bits(out))
    # Deprecated margins: margin_px is ignored, read_raster's margin still applies until 3.0
    with pytest.warns(DeprecationWarning, match="'margin_px' is deprecated and ignored"):
        margin = cn.connectedness(condition_file=COND, polygon_mask=polygon, margin_px=32, option=1,
                                  n_threads=THREADS)
    np.testing.assert_array_equal(bits(margin), bits(out))
    with pytest.warns(DeprecationWarning, match="'valid_margin_px' is deprecated"):
        read_raster(COND, polygon=polygon, levels=[1, 2, 4, 8], expand_px=3)


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
