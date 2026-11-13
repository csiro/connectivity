"""Run logs: a text record, beside each output, of the settings that produced it.

The format is shared with the R package and Eco-Connectivity Studio, which restores the
settings of a log dropped on its window.
"""

import os
import re
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from rust_conn import CORE_VERSION

LOG_FORMAT = 1
IN_MEMORY = "<in memory>"
_HEADER = "# Connectivity run log"
_OFF = {"0", "false", "no", "off"}
_HABITAT_OUTPUTS = {1: "connectedness", 2: "condition_times_connectedness", 3: "connected_habitat"}
# GDAL connection strings and subdatasets, such as vrt://..., /vsis3/... and NETCDF:"x.nc":v,
# are not file paths, so they are logged as given. A Windows drive letter is one character.
_NOT_A_FILE = re.compile(r"^(/vsi|[A-Za-z][A-Za-z0-9+._-]+:)")


def log_path(filename: str) -> str:
    """The run log of an output: its path with a `.toml` extension."""
    return os.path.splitext(filename)[0] + ".toml"


def _log_wanted(save_log) -> bool:
    """`save_log`, or when it is None, whether CONNECTIVITY_SAVE_LOG leaves logs on."""
    if save_log is None:
        return os.environ.get("CONNECTIVITY_SAVE_LOG", "").strip().lower() not in _OFF
    return bool(save_log)


def _is_run_log(path: str) -> bool:
    """Whether the file at `path` starts as a run log does."""
    with open(path, "rb") as f:
        first = f.readline().removeprefix(b"\xef\xbb\xbf").rstrip(b"\r\n")
    return first == _HEADER.encode()


def _save_log(filename: str, save_log, started: float, **settings) -> None:
    """Writes the log of a run that wrote `filename` and started at `started`, a
    `time.perf_counter()` reading, unless logs are off. Failing to write it only warns."""
    if len(filename) <= 3 or not _log_wanted(save_log) or _NOT_A_FILE.match(filename):
        return
    path = log_path(filename)
    try:
        if os.path.isdir(path):
            warnings.warn(f"Run log not saved: {path} is a directory", stacklevel=3)
            return
        # A file of the same name that is not a run log is the user's, so it is kept
        if os.path.exists(path) and not _is_run_log(path):
            warnings.warn(f"Run log not saved: {path} exists and is not a run log", stacklevel=3)
            return
        text = _format_log(
            created=datetime.now(timezone.utc),
            elapsed=time.perf_counter() - started,
            output=filename,
            **settings,
        )
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except Exception as error:  # the analysis succeeded, so a log never fails the run
        warnings.warn(f"Run log not saved: {error}", stacklevel=3)


def _format_log(
        *,
        created: datetime,
        elapsed: float,
        indicator: str,
        lambdas,
        levels,
        max_cost,
        window_size,
        outer_window,
        window_mode: str,
        condition,
        output: str,
        option: int = 3,
        pa_to_pa: bool = False,
        condition_scale=None,
        protected_areas=None,
        protected_areas_scale=None,
        resistance=None,
        resistance_scale=None,
        current_climate=None,
        future_climates=(),
        band=None,
        polygon_mask=None,
        closed_border: bool = False,
        threads=None,
        nodata_crossing=None,
        crossing_width=0.0,
        crossing_traversal=0.0,
        software: str | None = None,
    ) -> str:
    """The run log text. Keys are written in the order, and under the conditions, of the
    format; `indicator` is "connected_habitat", "parc" or "beri"."""
    if software is None:
        software = f"eco-connectivity Python package {_package_version()}"
    log = _Writer(_HEADER)
    log.entry("format", LOG_FORMAT)
    log.entry("software", _quoted(software))
    log.entry("engine", _quoted(f"connectivity-core {CORE_VERSION}"))
    log.entry("created", _quoted(created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    log.entry("elapsed_seconds", f"{elapsed:.1f}")

    log.table("parameters")
    log.entry("indicator", _quoted(indicator))
    if indicator == "connected_habitat":
        log.entry("habitat_output", _quoted(_HABITAT_OUTPUTS[option]))
    elif indicator == "parc":
        log.entry("pa_to_pa", _boolean(pa_to_pa))
    log.entry("lambdas_km", _array([_real(v) for v in lambdas]))
    log.entry("levels", _array([_integer(v) for v in levels]))
    log.entry("max_cost", _real(max_cost))
    log.entry("window_size", _integer(window_size))
    log.entry("outer_window", _integer(outer_window))
    log.entry("window_mode", _quoted(window_mode))
    if nodata_crossing is not None:
        log.entry("nodata_crossing", _quoted(nodata_crossing))
        log.entry("crossing_width_km", _real(crossing_width))
        log.entry("crossing_traversal", _real(crossing_traversal))
    log.entry("condition_scale", _scale(condition_scale))
    if indicator == "parc":
        log.entry("protected_areas_scale", _scale(protected_areas_scale))
    if resistance is not None:
        log.entry("resistance_scale", _scale(resistance_scale))
    if band is not None:
        log.entry("band", _integer(band))
    if polygon_mask is not None:
        log.entry("closed_border", _boolean(closed_border))

    log.table("inputs")
    log.entry("condition", _path_text(condition))
    if resistance is not None:
        log.entry("resistance", _path_text(resistance))
    if indicator == "parc":
        log.entry("protected_areas", _path_text(protected_areas))
    if indicator == "beri":
        log.entry("current_climate", _path_text(current_climate))
        log.entry("future_climates", _array([_path_text(p) for p in future_climates]))
    if polygon_mask is not None:
        log.entry("polygon_mask", _path_text(polygon_mask))

    log.table("output")
    log.entry("file", _path_text(output))

    if threads is not None:
        log.table("execution")
        log.entry("threads", _integer(threads))
    return log.text


class _Writer:
    """Builds the log text, one `key = value` line at a time."""

    def __init__(self, first_line: str):
        self.lines = [first_line]

    def entry(self, key: str, value) -> None:
        self.lines.append(f"{key} = {value}")

    def table(self, name: str) -> None:
        self.lines += ["", f"[{name}]"]

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _package_version() -> str:
    try:
        return version("eco-connectivity")
    except PackageNotFoundError:
        return "unknown"


def _quoted(text: str) -> str:
    """A TOML basic string, escaped as the format requires."""
    out = ['"']
    for c in text:
        code = ord(c)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\t":
            out.append("\\t")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        elif 0xD800 <= code <= 0xDFFF:
            # Undecodable bytes of a file name, which UTF-8 cannot hold
            out.append("\ufffd")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def _real(value) -> str:
    """Plain decimal text, the shortest that reads back to `value`; `2`, not `2.0`."""
    return np.format_float_positional(value, trim="-")


def _integer(value) -> str:
    return str(int(value))


def _boolean(value) -> str:
    return "true" if value else "false"


def _scale(value) -> str:
    """A divisor; None and 0 leave a raster unscaled, as 1 does."""
    return "1" if value is None or value == 0 else _real(value)


def _array(items) -> str:
    return "[" + ", ".join(items) + "]"


def _path_text(path) -> str:
    """An input or output as the log names it: an absolute path, or `<in memory>`."""
    if not isinstance(path, (str, os.PathLike)):
        return _quoted(IN_MEMORY)
    return _quoted(_absolute(os.fspath(path)))


def _absolute(path: str) -> str:
    """`path` made absolute without resolving symbolic links, as Eco-Connectivity Studio
    does: `.` components and repeated separators go, `..` stays."""
    if _NOT_A_FILE.match(path):
        return path
    if os.name == "nt":
        return os.path.abspath(path)
    if not path.startswith("/"):
        path = os.path.join(os.getcwd(), path)
    return "/" + "/".join(part for part in path.split("/") if part not in ("", "."))
