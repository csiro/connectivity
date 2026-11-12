"""Run logs, in the format shared with the R package and Eco-Connectivity Studio."""

import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio

import connectivity as cn
from connectivity import example_data_path
from connectivity.runlog import (
    _absolute,
    _format_log,
    _quoted,
    _real,
    _scale,
    log_path,
)
from rust_conn import CORE_VERSION

COND = example_data_path("site_condition")
PA = example_data_path("pa_proportion")
CURRENT = example_data_path("transgrids/1990")
FUTURE = example_data_path("transgrids/ACC50_85")
SHORT = dict(levels=[2, 4], outer_window=3, window_size=3, n_threads=4)
CREATED = datetime(2026, 11, 11, 3, 14, 15, tzinfo=timezone.utc)


def read_log(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def lines(text):
    return text.splitlines()


@pytest.mark.skipif(os.name == "nt", reason="Windows makes the example's paths C:\\...")
def test_documented_example():
    # The documented example of the format, as R and Eco-Connectivity Studio write it from
    # [parameters] to the end of [output]
    text = _format_log(
        created=CREATED, elapsed=42.48, software="eco-connectivity Python package 2.6.0",
        indicator="connected_habitat", option=3, lambdas=[2, 20, 200], levels=[1, 2, 4, 8, 16, 32],
        max_cost=2.0, window_size=5, outer_window=9, window_mode="circular",
        resistance_scale=100, condition="/data/tasmania/site_condition.tif",
        resistance="/data/tasmania/resistance.tif", output="/data/results/connected_habitat.tif",
        threads=8,
    )
    assert text == f"""# Connectivity run log
format = 1
software = "eco-connectivity Python package 2.6.0"
engine = "connectivity-core {CORE_VERSION}"
created = "2026-11-11T03:14:15Z"
elapsed_seconds = 42.5

[parameters]
indicator = "connected_habitat"
habitat_output = "connected_habitat"
lambdas_km = [2, 20, 200]
levels = [1, 2, 4, 8, 16, 32]
max_cost = 2
window_size = 5
outer_window = 9
window_mode = "circular"
condition_scale = 1
resistance_scale = 100

[inputs]
condition = "/data/tasmania/site_condition.tif"
resistance = "/data/tasmania/resistance.tif"

[output]
file = "/data/results/connected_habitat.tif"

[execution]
threads = 8
"""


def test_values():
    assert _quoted("C:\\maps\\\"x\"\t\u0001é\x7f") == '"C:\\\\maps\\\\\\"x\\"\\t\\u0001é\\u007F"'
    # Shortest round-trip text, without an exponent or a decimal point for whole numbers
    assert [_real(v) for v in (2.0, 2, 0.5, 0.1 + 0.2, 1 / 3, 1e-7, 1e20, np.float32(0.1))] == [
        "2", "2", "0.5", "0.30000000000000004", "0.3333333333333333", "0.0000001",
        "100000000000000000000", "0.1",
    ]
    assert [_scale(v) for v in (None, 0, 1, 255, 0.5)] == ["1", "1", "1", "255", "0.5"]


def test_paths(monkeypatch, tmp_path):
    assert log_path("maps/connected_habitat.tif") == "maps/connected_habitat.toml"
    assert log_path("beri.v2.TIFF") == "beri.v2.toml"
    assert log_path("run.v2/result") == "run.v2/result.toml"
    # GDAL connection strings are logged as given
    for name in ("vrt:///data/x.tif?bands=2", "/vsis3/bucket/x.tif", 'NETCDF:"x.nc":tas'):
        assert _absolute(name) == name
    if os.name != "nt":
        monkeypatch.chdir(tmp_path)
        assert _absolute("a/./b//../c.tif") == os.getcwd() + "/a/b/../c.tif"
        assert _absolute("/data/./x.tif") == "/data/x.tif"


def test_connectedness_log(tmp_path):
    out = str(tmp_path / "habitat.tif")
    cn.connectedness(condition_file=COND, filename=out, option=2, lambdas=[0.5, 20], **SHORT)
    text = read_log(tmp_path / "habitat.toml")
    assert "\r" not in text and text.endswith("\n") and not text.startswith("\ufeff")
    assert lines(text)[:3] == ["# Connectivity run log", "format = 1", f'software = "eco-connectivity Python package {cn_version()}"']
    assert f'engine = "connectivity-core {CORE_VERSION}"' in text
    assert lines(text)[7:] == [
        "[parameters]",
        'indicator = "connected_habitat"',
        'habitat_output = "condition_times_connectedness"',
        "lambdas_km = [0.5, 20]",
        "levels = [1, 2, 4]",
        "max_cost = 2",
        "window_size = 3",
        "outer_window = 3",
        'window_mode = "circular"',
        "condition_scale = 1",
        "",
        "[inputs]",
        f"condition = {_quoted(os.path.abspath(COND))}",
        "",
        "[output]",
        f"file = {_quoted(_absolute(out))}",
        "",
        "[execution]",
        "threads = 4",
    ]
    check_toml(text)


def test_parc_log(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    with rasterio.open(COND) as src:
        transform, crs = src.transform, src.crs
    x0, y0, r = transform.c, transform.f, transform.a
    polygon = gpd.GeoDataFrame(geometry=[box(x0 + 100 * r, y0 - 300 * r, x0 + 220 * r, y0 - 180 * r)], crs=crs)
    out = str(tmp_path / "parc.tif")
    cn.connectedness(condition_file=COND, pa_file=PA, pa_to_pa=True, scale=(None, 100),
                     polygon_mask=polygon, closed_border=True, filename=out, **SHORT)
    text = read_log(tmp_path / "parc.toml")
    for line in ('indicator = "parc"', "pa_to_pa = true", "protected_areas_scale = 100",
                 "closed_border = true", f"protected_areas = {_quoted(os.path.abspath(PA))}",
                 'polygon_mask = "<in memory>"'):
        assert line in lines(text)
    assert "habitat_output" not in text and "resistance" not in text
    check_toml(text)


def test_beri_log(tmp_path):
    out = str(tmp_path / "beri.tif")
    cn.beri(condition_file=COND, current_file=CURRENT, future_files=[FUTURE], filename=out,
            resistance_file=COND, resistance_scale=0, band=1, **SHORT)
    text = read_log(tmp_path / "beri.toml")
    parameters = text[text.index("[parameters]"):text.index("[inputs]")]
    assert lines(parameters)[-4:] == ["condition_scale = 1", "resistance_scale = 1", "band = 1", ""]
    for line in ('indicator = "beri"', f"current_climate = {_quoted(os.path.abspath(CURRENT))}",
                 f"future_climates = [{_quoted(os.path.abspath(FUTURE))}]",
                 f"resistance = {_quoted(os.path.abspath(COND))}"):
        assert line in lines(text)
    assert "pa_to_pa" not in text and "habitat_output" not in text
    check_toml(text)


def test_turning_logs_off(tmp_path, monkeypatch):
    out = str(tmp_path / "x.tif")
    log = tmp_path / "x.toml"
    cn.connectedness(condition_file=COND, filename=out, save_log=False, **SHORT)
    assert not log.exists()
    monkeypatch.setenv("CONNECTIVITY_SAVE_LOG", " Off ")
    cn.connectedness(condition_file=COND, filename=out, **SHORT)
    assert not log.exists()
    # An explicit argument wins over the variable
    cn.connectedness(condition_file=COND, filename=out, save_log=True, **SHORT)
    assert log.exists()
    log.unlink()
    monkeypatch.setenv("CONNECTIVITY_SAVE_LOG", "1")
    cn.connectedness(condition_file=COND, filename=out, **SHORT)
    assert log.exists()


def test_existing_files(tmp_path):
    out = str(tmp_path / "x.tif")
    log = tmp_path / "x.toml"
    # A file that is not a run log is kept
    log.write_text("title = 'mine'\n")
    with pytest.warns(UserWarning, match="Run log not saved: .*x.toml exists and is not a run log"):
        cn.connectedness(condition_file=COND, filename=out, **SHORT)
    assert log.read_text() == "title = 'mine'\n"
    # An older run log is replaced
    log.write_bytes(b"# Connectivity run log\r\nformat = 1\r\n")
    with warnings.catch_warnings():
        # Only run-log warnings: dependencies may warn of their own deprecations
        warnings.filterwarnings("error", message="Run log")
        cn.connectedness(condition_file=COND, filename=out, **SHORT)
    assert "levels = [1, 2, 4]" in read_log(log)
    # An unwritable log only warns
    log.unlink()
    log.mkdir()
    with pytest.warns(UserWarning, match=r"Run log not saved: .*x\.toml is a directory"):
        result = cn.connectedness(condition_file=COND, filename=out, **SHORT)
    assert result.shape == (516, 588)


def cn_version():
    from importlib.metadata import version

    return version("eco-connectivity")


def check_toml(text):
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(text)
    assert parsed["format"] == 1 and "parameters" in parsed and "output" in parsed
