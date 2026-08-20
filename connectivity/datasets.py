"""Example datasets bundled with the package.

The rasters here back the notebooks in ``examples/``. They cover Tasmania,
Australia at roughly 1 km resolution (588 x 516 cells, EPSG:4326, 30 arc-second
grid) and are small enough to ship with the library so the examples run
straight after ``pip install connectivity``. Every layer shares the same grid,
so they can be combined without resampling.

See ``connectivity/data/README.md`` for the source and licence of each layer.

Examples
--------
>>> from connectivity import connectedness, example_data_path
>>> conns = connectedness(condition_file=example_data_path("site_condition"))
"""

from importlib.resources import files

__all__ = ["example_data_path"]

_DATASETS = {
    "site_condition": "site_condition.tif",
    "pa_proportion": "pa_proportion.tif",
    "transgrids/1990": "transgrids/1990.tif",
    "transgrids/ACC50_85": "transgrids/ACC50_85.tif",
    "transgrids/GFD50_85": "transgrids/GFD50_85.tif",
    "transgrids/IPS50_26": "transgrids/IPS50_26.tif",
    "transgrids/IPS50_45": "transgrids/IPS50_45.tif",
    "transgrids/IPS50_60": "transgrids/IPS50_60.tif",
    "transgrids/IPS50_85": "transgrids/IPS50_85.tif",
}


def example_data_path(name: str) -> str:
    """Return a filesystem path to a bundled example raster.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"site_condition"`` or ``"transgrids/IPS50_45"``.

    Returns
    -------
    str
        Path to the raster, ready to pass to any function in this package.
    """
    try:
        relative = _DATASETS[name]
    except KeyError:
        available = ", ".join(sorted(_DATASETS))
        raise ValueError(
            f"unknown example dataset {name!r}; choose from: {available}"
        ) from None

    return str(files("connectivity.data").joinpath(relative))
