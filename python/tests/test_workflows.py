"""Regression tests for the Python workflows on the bundled Tasmania example data.

Exact checks (NaN placement, bit-identical repeats, seamless tiles, Connected Habitat
post-processing) are platform independent; the frozen means come from the 2.5.0 build, run with
the legacy NoData handling (`nodata_crossing=None`) that 2.5.0 had, and allow for floating-point
differences between platforms.
"""

import warnings

import numpy as np
import pytest
import rasterio
from affine import Affine

import connectivity as cn
import rust_conn
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
LEGACY = dict(nodata_crossing=None)  # the NoData handling of 2.x, which FROZEN comes from

# Frozen from the 2.5.0 build (default parameters unless noted, window_size 5), reproduced with
# LEGACY; the R package 0.1.0 gives the same means to within 1e-7 relative
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
    return cn.connectedness(condition_file=COND, option=1, n_threads=THREADS, **LEGACY)


def test_habitat_options(condition, connectedness):
    assert connectedness.dtype == np.float32 and connectedness.shape == SHAPE
    np.testing.assert_array_equal(np.isnan(connectedness), np.isnan(condition))
    finite = connectedness[np.isfinite(connectedness)]
    assert finite.min() >= 0 and finite.max() <= 1
    assert nanmean(connectedness) == pytest.approx(FROZEN["connectedness"], rel=RTOL)
    # Connected Habitat stays in Python: option 2 = condition x connectedness, 3 = its sqrt
    option2 = cn.connectedness(condition_file=COND, option=2, n_threads=THREADS, **LEGACY)
    option3 = cn.connectedness(condition_file=COND, n_threads=THREADS, **LEGACY)
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
                              outer_window=5, n_threads=THREADS, **LEGACY)
    np.testing.assert_array_equal(np.isnan(square), np.isnan(connectedness))
    assert nanmean(square) == pytest.approx(FROZEN["square_short"], rel=RTOL)
    with pytest.raises(ValueError, match="window_mode must be one of"):
        cn.connectedness(condition_file=COND, window_mode="block")


def test_parc_and_pa_to_pa(condition):
    pa = read_raster(PA, levels=[1])[0]
    parc = cn.connectedness(condition_file=COND, pa_file=PA, n_threads=THREADS, **LEGACY)
    gated = cn.connectedness(condition_file=COND, pa_file=PA, pa_to_pa=True, n_threads=THREADS, **LEGACY)
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
    out = cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES, n_threads=THREADS, **LEGACY)
    current = cn.beri(condition_file=COND, current_file=CURRENT, n_threads=THREADS, **LEGACY)
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
    with pytest.raises(ValueError, match="^pa_file must have exactly one layer, but it has 19; band selects a layer"):
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


def write_series(path, source, count=3):
    """`source` as a `count`-band series whose band k holds source ** (k / 2)."""
    with rasterio.open(source) as src:
        values, profile = src.read(1, masked=True), src.profile
    series = np.ma.stack([values ** np.float32(k / 2) for k in range(1, count + 1)])
    nodata = np.nan if profile["nodata"] is None else profile["nodata"]
    with rasterio.open(path, "w", **(profile | {"count": count})) as dst:
        dst.write(series.astype(np.float32).filled(nodata))
    return str(path)


def test_band(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    resistance = str(tmp_path / "resistance.tif")
    write_resistance(resistance)
    cond, pa, res = (write_series(tmp_path / f"{n}_series.tif", f)
                     for n, f in (("cond", COND), ("pa", PA), ("res", resistance)))
    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    x0, y0, r = transform.c, transform.f, transform.a
    polygon = gpd.GeoDataFrame(geometry=[box(x0 + 100.3 * r, y0 - 300.7 * r, x0 + 220.6 * r, y0 - 180.2 * r)], crs=crs)
    climate = dict(current_file=CURRENT, future_files=FUTURES[:1])
    # Band k of a condition series gives the result of a single-band copy of band k, bit for bit.
    # The band selects condition only: single-band PA and resistance rasters are used as they
    # are with every band, and the climate rasters keep all their bands
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # missing resistance cells
        for k in (1, 3):
            one = moved(tmp_path / f"cond{k}.tif", cond, bands=[k])
            borders = ({}, dict(polygon_mask=polygon), dict(polygon_mask=polygon, closed_border=True)) if k == 3 else ({},)
            for border in borders:
                args = SHORT | border
                out_file = str(tmp_path / "out.tif")
                out = cn.connectedness(condition_file=cond, band=k, filename=out_file, **args)
                np.testing.assert_array_equal(bits(out), bits(cn.connectedness(condition_file=one, **args)))
                with rasterio.open(out_file) as written:
                    np.testing.assert_array_equal(bits(written.read(1)), bits(out))
                np.testing.assert_array_equal(
                    bits(cn.connectedness(condition_file=cond, pa_file=PA, resistance_file=resistance, band=k, **args)),
                    bits(cn.connectedness(condition_file=one, pa_file=PA, resistance_file=resistance, **args)))
                np.testing.assert_array_equal(
                    bits(cn.beri(condition_file=cond, resistance_file=resistance, band=k, **climate, **args)),
                    bits(cn.beri(condition_file=one, resistance_file=resistance, **climate, **args)))
    # A single-band condition raster is band 1
    np.testing.assert_array_equal(bits(cn.connectedness(condition_file=COND, band=1, **SHORT)),
                                  bits(cn.connectedness(condition_file=COND, **SHORT)))
    # GDAL's vrt:// strings read one band too, and the output is still a GeoTIFF
    k = 3
    out_file = str(tmp_path / "vrt.tif")
    out = cn.connectedness(condition_file=f"vrt://{cond}?bands={k}", filename=out_file, **SHORT)
    np.testing.assert_array_equal(bits(out), bits(cn.connectedness(condition_file=cond, band=k, **SHORT)))
    with rasterio.open(out_file) as written:
        assert written.driver == "GTiff"
        np.testing.assert_array_equal(bits(written.read(1)), bits(out))
    # Errors: a multi-band condition needs band; multi-band PA and resistance rasters are refused
    # with or without band, which never picks one of their bands
    with pytest.raises(ValueError, match="^condition_file must have exactly one layer; choose one with band$"):
        cn.connectedness(condition_file=cond, **SHORT)
    single = ("must have exactly one layer, but it has 3; band selects a layer of condition_file only, so "
              "write the layer to use to a single-band raster first$")
    for band in (None, 1, 3):
        with pytest.raises(ValueError, match=f"^pa_file {single}"):
            cn.connectedness(condition_file=cond if band else COND, pa_file=pa, band=band, **SHORT)
        with pytest.raises(ValueError, match=f"^resistance_file {single}"):
            cn.connectedness(condition_file=cond if band else COND, resistance_file=res, band=band, **SHORT)
        with pytest.raises(ValueError, match=f"^resistance_file {single}"):
            cn.beri(condition_file=cond if band else COND, resistance_file=res, band=band, **climate, **SHORT)
    for count, args in ((3, dict(condition_file=cond, band=4)), (1, dict(condition_file=COND, band=2))):
        with pytest.raises(ValueError, match=f"^band must be between 1 and {count}, the number of layers in condition_file$"):
            cn.connectedness(**args, **SHORT)
    for band in (0, -1, 1.0, True, "1"):
        with pytest.raises(ValueError, match="^band must be a positive integer$"):
            cn.beri(condition_file=cond, current_file=CURRENT, band=band, **SHORT)
    with pytest.raises(ValueError, match="^band must be between 1 and 3, the number of layers in file_path$"):
        read_raster(cond, levels=[1], band=4)


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
        out = cn.connectedness(condition_file=COND, resistance_file=res_file, option=1, n_threads=THREADS, **LEGACY)
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
        out = cn.connectedness(condition_file=COND, polygon_mask=polygon, option=1, n_threads=THREADS, **LEGACY)
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
        warnings.simplefilter("error", DeprecationWarning)  # no deprecated path is used internally
        out = cn.connectedness(condition_file=COND, polygon_mask=polygon, option=1, n_threads=THREADS, **LEGACY)
    projected = cn.connectedness(condition_file=COND, polygon_mask=polygon.to_crs(3577), option=1,
                                 n_threads=THREADS, **LEGACY)
    np.testing.assert_array_equal(bits(projected), bits(out))
    # The margin arguments deprecated in 2.5 are gone, so passing them fails
    with pytest.raises(TypeError, match="margin_px"):
        cn.connectedness(condition_file=COND, polygon_mask=polygon, margin_px=32, **SHORT)
    with pytest.raises(TypeError, match="margin_px"):
        cn.beri(condition_file=COND, current_file=CURRENT, polygon_mask=polygon, margin_px=0, **SHORT)
    with pytest.raises(TypeError, match="valid_margin_px"):
        read_raster(COND, polygon=polygon, levels=[1, 2, 4, 8], expand_px=3, valid_margin_px=0)
    # A sixth positional argument, once valid_margin_px, is refused rather than taken as band
    with pytest.raises(TypeError):
        read_raster(COND, polygon, [1, 2], None, 3, 1)
    # read_raster marks only the cells the polygon touches as valid, with open and closed borders;
    # an open border reads the cells around them as context
    for expand in (0, 3):
        data, mask, transforms, _, _, _ = read_raster(COND, polygon=polygon, levels=[1, 2, 4, 8], expand_px=expand)
        touched = ~geometry_mask(shapes[:1], data.shape, Affine(*transforms[1]), all_touched=True)
        np.testing.assert_array_equal(~mask, touched & np.isfinite(data))
        assert (np.isfinite(data) & ~touched).any() == (expand > 0)


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


CROSS = dict(levels=[2, 4, 8], window_size=3, outer_window=5, n_threads=THREADS)


def pieces(land):
    """8-connected pieces of `land`, numbered from 1, and their sizes."""
    from collections import deque

    labels = np.zeros(land.shape, dtype=np.int32)
    sizes = [0]
    for start in zip(*np.nonzero(land)):
        if labels[start]:
            continue
        sizes.append(0)
        labels[start] = len(sizes) - 1
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            sizes[-1] += 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    r, c = i + di, j + dj
                    if 0 <= r < land.shape[0] and 0 <= c < land.shape[1] and land[r, c] and not labels[r, c]:
                        labels[r, c] = len(sizes) - 1
                        queue.append((r, c))
    return labels, np.array(sizes)


# Without a permitted crossing, small pieces of land score the same, bit for bit, alone and with
# all the other land, in both modes; with the legacy handling some do not
def test_crossing_islands(tmp_path, condition):
    labels, sizes = pieces(np.isfinite(condition))
    picks = [p for p in np.argsort(sizes)[::-1][1:] if 100 <= sizes[p] <= 700][:3]
    assert len(picks) == 3
    with rasterio.open(COND) as src:
        profile = src.profile
    alone = {}
    for p in picks:
        path = tmp_path / f"piece_{p}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.where(labels == p, condition, profile["nodata"]).astype(np.float32), 1)
        alone[p] = str(path)
    for mode in (None, "landmass", "strict"):
        settings = dict(nodata_crossing=mode) if mode is None else dict(nodata_crossing=mode, crossing_width=0)
        together = cn.connectedness(condition_file=COND, option=1, **settings, **CROSS)
        same = []
        for p in picks:
            out = cn.connectedness(condition_file=alone[p], option=1, **settings, **CROSS)
            inside = labels == p
            same.append(np.array_equal(bits(out)[inside], bits(together)[inside]))
        if mode is None:
            assert not all(same)
        else:
            assert all(same), mode


# Open-border tiles give the whole raster's output in both modes, with a permitted crossing
def test_crossing_tiles():
    with rasterio.open(COND) as src:
        transform = src.transform
    tiles = cn.make_tiles(COND, nrows=2, ncols=2, align_to=32)
    for mode in ("landmass", "strict"):
        settings = dict(nodata_crossing=mode, crossing_width=2.0, crossing_cost=3.0, **CROSS)
        whole = cn.connectedness(condition_file=COND, **settings)
        for k in range(len(tiles)):
            tile = tiles.iloc[[k]]
            out = cn.connectedness(condition_file=COND, polygon_mask=tile, **settings)
            expected, _ = _crop_array(whole, transform, tile)
            inside = np.isfinite(expected)
            np.testing.assert_array_equal(np.isfinite(out), inside)
            np.testing.assert_array_equal(bits(out)[inside], bits(expected)[inside], err_msg=f"{mode} tile {k}")


# PARC and BERI run in both modes; a crossing changes some cells and leaves NaN where it was
def test_crossing_indicators():
    plain = cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES[:1], **LEGACY, **CROSS)
    parc = cn.connectedness(condition_file=COND, pa_file=PA, pa_to_pa=True, **LEGACY, **CROSS)
    for mode in ("landmass", "strict"):
        settings = dict(nodata_crossing=mode, crossing_width=2.0, **CROSS)
        beri = cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES[:1], **settings)
        np.testing.assert_array_equal(np.isfinite(beri), np.isfinite(plain))
        assert not np.array_equal(bits(beri), bits(plain)) and np.nanmax(beri) <= 1
        gated = cn.connectedness(condition_file=COND, pa_file=PA, pa_to_pa=True, **settings)
        np.testing.assert_array_equal(np.isfinite(gated), np.isfinite(parc))
        assert np.nanmin(gated) >= 0 and np.nanmax(gated) <= 1


def test_crossing_settings(capsys):
    for bad, error, message in [
        (dict(nodata_crossing="river"), ValueError, "nodata_crossing must be 'strict', 'landmass' or None"),
        (dict(nodata_crossing="none"), ValueError, "nodata_crossing must be 'strict', 'landmass' or None"),
        (dict(nodata_crossing=1), TypeError, "nodata_crossing must be 'strict', 'landmass' or None"),
        (dict(crossing_width=-1), ValueError, "crossing_width must be a finite number of kilometres >= 0, got -1"),
        (dict(crossing_width=float("inf")), ValueError, "crossing_width must be a finite number"),
        (dict(crossing_width=float("nan")), ValueError, "crossing_width must be a finite number"),
        (dict(crossing_width="2"), ValueError, "crossing_width must be a finite number"),
        (dict(crossing_cost=0.5), ValueError, "crossing_cost must be a finite number >= 1, got 0.5"),
        (dict(crossing_cost=0), ValueError, "crossing_cost must be a finite number >= 1, got 0"),
        (dict(crossing_cost=-4), ValueError, "crossing_cost must be a finite number >= 1, got -4"),
        (dict(crossing_cost=float("inf")), ValueError, "crossing_cost must be a finite number >= 1, got inf"),
        (dict(crossing_cost=float("nan")), ValueError, "crossing_cost must be a finite number >= 1, got nan"),
        (dict(crossing_cost=True), ValueError, "crossing_cost must be a finite number >= 1, got True"),
        # Invalid values fail also where they would have no effect
        (dict(nodata_crossing=None, crossing_cost=0.5), ValueError, "crossing_cost must be"),
        (dict(nodata_crossing="landmass", crossing_width=0, crossing_cost=0.5), ValueError, "crossing_cost must be"),
    ]:
        for function, extra in ((cn.connectedness, {}), (cn.beri, dict(current_file=CURRENT))):
            with pytest.raises(error, match=f"^{message}"):
                function(condition_file=COND, **bad, **extra, **SHORT)
    # Under the legacy handling a width and cost have no effect: a notice when they are changed
    legacy = cn.connectedness(condition_file=COND, **LEGACY, **SHORT)
    assert "ignored" not in capsys.readouterr().out
    for changed in (dict(crossing_width=2.0), dict(crossing_cost=1.0), dict(crossing_width=0, crossing_cost=30)):
        out = cn.connectedness(condition_file=COND, **LEGACY, **changed, **SHORT)
        np.testing.assert_array_equal(bits(out), bits(legacy))
        assert "'crossing_width' and 'crossing_cost' are ignored" in capsys.readouterr().out
    # Integer and numpy values are accepted, and the mode ignores case and spaces
    np.testing.assert_array_equal(
        bits(cn.connectedness(condition_file=COND, nodata_crossing=" Strict ", crossing_width=np.float32(0.5),
                              crossing_cost=np.int64(4), **SHORT)),
        bits(cn.connectedness(condition_file=COND, **SHORT)))


CONNECTEDNESS_ORDER = [
    "condition_file", "band", "scale", "pa_file", "pa_to_pa", "option", "resistance_file", "resistance_scale",
    "lambdas", "max_cost", "levels", "window_size", "outer_window", "window_mode", "polygon_mask", "closed_border",
    "nodata_crossing", "crossing_width", "crossing_cost", "n_threads", "filename", "save_log",
]
BERI_ORDER = [
    "condition_file", "band", "scale", "current_file", "future_files", "resistance_file", "resistance_scale",
    "lambdas", "max_cost", "levels", "window_size", "outer_window", "window_mode", "polygon_mask", "closed_border",
    "nodata_crossing", "crossing_width", "crossing_cost", "n_threads", "filename", "save_log",
]
# Defaults kept from 2.x; the crossing defaults are the core's
DEFAULTS = dict(band=None, scale=None, pa_file=None, pa_to_pa=False, option=3, resistance_file=None,
                resistance_scale=None, lambdas=[2, 20, 200], max_cost=2.0, levels=[2, 4, 8, 16, 32], window_size=5,
                outer_window=9, window_mode="circular", polygon_mask=None, closed_border=False, future_files=None,
                n_threads=None, filename="", save_log=None)


# Every argument but condition_file is keyword-only, in the documented order, so a positional
# call of 2.x fails instead of binding to another argument
def test_signatures():
    import inspect

    for function, order in ((cn.connectedness, CONNECTEDNESS_ORDER), (cn.beri, BERI_ORDER)):
        params = inspect.signature(function).parameters
        assert list(params) == order
        assert params["condition_file"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(params[k].kind is inspect.Parameter.KEYWORD_ONLY for k in order[1:])
        crossing = tuple(params[k].default for k in ("nodata_crossing", "crossing_width", "crossing_cost"))
        assert crossing == rust_conn.DEFAULT_CROSSING == ("strict", 0.5, 4.0)
        for name in order[1:]:
            if name in DEFAULTS:
                assert params[name].default == DEFAULTS[name], (function.__name__, name)
        # The docstring documents every argument, in this order
        if function.__doc__:
            documented = [line.split(" : ")[0].strip() for line in function.__doc__.splitlines() if " : " in line]
            assert [name for name in documented if name in order] == order, function.__name__
    assert inspect.signature(cn.beri).parameters["current_file"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="positional argument"):
        cn.connectedness(COND, PA)
    with pytest.raises(TypeError, match="positional argument"):
        cn.connectedness(COND, None, False, None, False, None, [2, 20])
    with pytest.raises(TypeError, match="positional argument"):
        cn.beri(COND, CURRENT)
    with pytest.raises(TypeError, match="current_file"):
        cn.beri(condition_file=COND, **SHORT)
    for removed in ("margin_px", "valid_margin_px"):
        with pytest.raises(TypeError, match=removed):
            cn.connectedness(condition_file=COND, **{removed: 0}, **SHORT)
    np.testing.assert_array_equal(bits(cn.connectedness(COND, **SHORT)),
                                  bits(cn.connectedness(condition_file=COND, **SHORT)))
    np.testing.assert_array_equal(bits(cn.beri(COND, current_file=CURRENT, **SHORT)),
                                  bits(cn.beri(condition_file=COND, current_file=CURRENT, **SHORT)))


# Frozen from the 3.0.0 development build with the default strict crossing (regression values,
# not independent ones); the legacy values are FROZEN
FROZEN_STRICT = {
    "connectedness": 0.3652536108175584,
    "beri": 0.4091173515425484,
}


# The default is strict crossing at 0.5 km and cost 4, and differs from the legacy handling. The
# bundled cells, about 0.69 x 0.93 km, leave no gap of 0.5 km or less, so it crosses nothing here
# and equals strict mode at width 0
def test_default_crossing(connectedness):
    out = cn.connectedness(condition_file=COND, option=1, n_threads=THREADS)
    explicit = cn.connectedness(condition_file=COND, option=1, nodata_crossing="strict", crossing_width=0.5,
                                crossing_cost=4.0, n_threads=THREADS)
    np.testing.assert_array_equal(bits(out), bits(explicit))
    zero = cn.connectedness(condition_file=COND, option=1, crossing_width=0, n_threads=THREADS)
    np.testing.assert_array_equal(bits(out), bits(zero))
    np.testing.assert_array_equal(np.isnan(out), np.isnan(connectedness))
    assert not np.array_equal(bits(out), bits(connectedness))
    assert nanmean(out) == pytest.approx(FROZEN_STRICT["connectedness"], rel=RTOL)
    beri = cn.beri(condition_file=COND, current_file=CURRENT, future_files=FUTURES, n_threads=THREADS)
    assert nanmean(beri) == pytest.approx(FROZEN_STRICT["beri"], rel=RTOL)


def write_river(path, river=(20, 21), condition=1.0, climate=None):
    """A 30 x 40 raster of 1 km cells (EPSG:3577): land of `condition`, and NoData in the
    columns `river`, a river as wide as their count. With `climate`, a (bands, 30, 40) array,
    it is written instead, with NoData where the condition is."""
    transform = Affine(1000.0, 0, 1_000_000.0, 0, -1000.0, -4_000_000.0)
    values = np.full((30, 40), np.float32(condition), dtype=np.float32)
    values[:, river[0]:river[1]] = np.nan
    if climate is not None:
        values = climate.astype(np.float32)
    values = values.reshape((-1, 30, 40))
    profile = dict(driver="GTiff", width=40, height=30, count=values.shape[0], dtype="float32", crs="EPSG:3577",
                   transform=transform, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values)
    return str(path)


RIVER = dict(levels=[2, 4, 8], window_size=3, outer_window=5, option=1, n_threads=THREADS)


# The crossing cost is a movement cost per kilometre relative to intact land, independent of
# max_cost. On intact land (condition 1) every land cell costs 1 whatever max_cost is, so results
# do not depend on max_cost at any crossing cost; a crossing costing 1 is as cheap as the land, so
# adjusted paths equal intact ones up to routing's 10 m rounding of intact lengths and scores
# stay near 1, while a dearer crossing lowers cells whose destinations lie across it
def test_crossing_cost(tmp_path):
    river = write_river(tmp_path / "river.tif")
    cross = dict(nodata_crossing="strict", crossing_width=1.5)
    runs = {(m, c): cn.connectedness(condition_file=river, max_cost=m, crossing_cost=c, **cross, **RIVER)
            for m in (1.0, 2.0, 20.0) for c in (1, 4)}
    for (m, c), out in runs.items():
        np.testing.assert_array_equal(bits(out), bits(runs[1.0, c]), err_msg=f"max_cost {m} cost {c}")
    one, four = runs[1.0, 1], runs[1.0, 4]
    assert np.nanmin(one) > 0.999 and np.nanmax(one) <= 1
    assert (four[np.isfinite(four)] <= one[np.isfinite(four)]).all() and np.nanmin(four) < 0.99
    # A cost above max_cost lowers scores further
    dear = cn.connectedness(condition_file=river, max_cost=2.0, crossing_cost=30, **cross, **RIVER)
    assert (dear[np.isfinite(dear)] <= four[np.isfinite(dear)]).all() and np.nanmin(dear) < np.nanmin(four)
    # Widths just below, at and above the river's 1 km: crossed at 1 km and wider only
    zero = cn.connectedness(condition_file=river, nodata_crossing="strict", crossing_width=0, **RIVER)
    for width, crossed in ((0.99, False), (1.0, True), (1.01, True)):
        out = cn.connectedness(condition_file=river, nodata_crossing="strict", crossing_width=width, **RIVER)
        assert np.array_equal(bits(out), bits(zero)) != crossed, width
    # Not crossed: strict mode at width 0 ignores the cost; the legacy handling differs from it
    for cost in (1, 30):
        np.testing.assert_array_equal(
            bits(cn.connectedness(condition_file=river, nodata_crossing="strict", crossing_width=0, crossing_cost=cost,
                                  **RIVER)), bits(zero))
    assert not np.array_equal(bits(cn.connectedness(condition_file=river, **LEGACY, **RIVER)), bits(zero))


# Crossed NoData adds no climate: under the crossing modes, climate values on the river (NaN or
# not) leave BERI unchanged bit for bit, with multi-band climate rasters; the legacy handling
# averages them into coarse cells
def test_crossing_climate(tmp_path):
    i, j = np.indices((30, 40))
    cond_values = np.where((j >= 20) & (j < 21), np.nan, 0.3 + 0.6 * ((i * 7 + j * 3) % 10) / 10)
    cond = write_river(tmp_path / "cond.tif", climate=cond_values)
    grids = {}
    for name, river in (("nan", np.nan), ("hot", 5.0)):
        for scenario, shift in (("now", 0.0), ("then", 0.3)):
            bands = np.stack([0.1 * k + 0.02 * i + 0.01 * j + shift for k in range(3)])
            bands[:, :, 20:21] = river
            grids[name, scenario] = write_river(tmp_path / f"{name}_{scenario}.tif", climate=bands)
    settings = dict(levels=[2, 4, 8], window_size=3, outer_window=5, n_threads=THREADS)
    run = lambda name, **kw: cn.beri(condition_file=cond, current_file=grids[name, "now"],
                                     future_files=[grids[name, "then"]], **kw, **settings)
    for mode in ("strict", "landmass"):
        np.testing.assert_array_equal(bits(run("nan", nodata_crossing=mode, crossing_width=1.5)),
                                      bits(run("hot", nodata_crossing=mode, crossing_width=1.5)), err_msg=mode)
    assert not np.array_equal(bits(run("nan", **LEGACY)), bits(run("hot", **LEGACY)))
