# Run logs: a text record, beside each output, of the settings that produced
# it. The format is shared with the Python package and Eco-Connectivity Studio,
# which restores the settings of a log dropped on its window. Format 2, as in
# the Python package 3.0, always records the NoData crossing: "strict",
# "landmass" or "none" (the legacy handling, nodata_crossing = NULL), with
# crossing_width_km and crossing_cost under a mode.

.LOG_FORMAT <- 2L

.LOG_HEADER <- "# Connectivity run log"
.IN_MEMORY <- "<in memory>"
# GDAL connection strings and subdatasets, such as vrt://..., /vsis3/... and
# NETCDF:"x.nc":v, are not file paths, so they are logged as given. A Windows
# drive letter is one character.
.NOT_A_FILE <- "^(/vsi|[A-Za-z][A-Za-z0-9+._-]+:)"

# The default of save_log when the connectivity.save_log option is unset:
# whether the CONNECTIVITY_SAVE_LOG environment variable leaves logs on.
.log_by_env <- function() {
    value <- tolower(trimws(Sys.getenv("CONNECTIVITY_SAVE_LOG")))
    return(!value %in% c("0", "false", "no", "off"))
}

# The run log of an output: its path with a .toml extension.
.log_path <- function(filename) {
    return(paste0(sub("([^/\\\\])\\.[^./\\\\]*$", "\\1", filename), ".toml"))
}

# Whether the file at `path` starts as a run log does.
.is_run_log <- function(path) {
    con <- file(path, "r", encoding = "UTF-8-BOM")
    on.exit(close(con))
    return(identical(readLines(con, n = 1L, warn = FALSE), .LOG_HEADER))
}

# Writes the log of a run that wrote `filename` and started at `started`, a
# proc.time() elapsed reading, unless logs are off. Settings in `...` go to
# .format_log(). Failing to write the log only warns: the run succeeded.
.save_log <- function(filename, save_log, started, ...) {
    if (!nzchar(filename) || !save_log || grepl(.NOT_A_FILE, filename)) {
        return(invisible(NULL))
    }
    path <- .log_path(filename)
    # The first warning or error stops the log and becomes the reason given
    problem <- tryCatch({
        if (dir.exists(path)) {
            paste(path, "is a directory")
        } else if (file.exists(path) && !.is_run_log(path)) {
            # A file of the same name that is not a run log is the user's
            paste(path, "exists and is not a run log")
        } else {
            text <- .format_log(
                created = Sys.time(), elapsed = proc.time()[["elapsed"]] - started,
                output = filename, ...
            )
            # Binary, so that Windows does not write CRLF line endings
            writeBin(charToRaw(enc2utf8(text)), path)
            NULL
        }
    }, warning = conditionMessage, error = conditionMessage)
    if (!is.null(problem)) warning("Run log not saved: ", problem, call. = FALSE)
    return(invisible(NULL))
}

# The run log text. Keys are written in the order, and under the conditions, of
# the format. `settings` is from .normalise_settings(), `indicator` is
# "connected_habitat", "parc" or "beri", `condition` is from .log_condition(),
# other rasters are texts from .log_input(), and `polygon_mask` is the argument.
# Callers pass inputs unevaluated through .save_log(), so that reading their
# sources happens only for a log, and an error there only warns.
.format_log <- function(created, elapsed, indicator, settings, condition, output,
                        option = 3L, pa_to_pa = FALSE, condition_scale = NULL,
                        protected_areas = NULL, protected_areas_scale = NULL,
                        resistance = NULL, resistance_scale = NULL,
                        current_climate = NULL, future_climates = list(),
                        polygon_mask = NULL, closed_border = FALSE,
                        software = NULL) {
    if (is.null(software)) {
        software <- paste("connectivity R package", utils::packageVersion("connectivity"))
    }
    habitat_outputs <- c("connectedness", "condition_times_connectedness", "connected_habitat")
    entry <- function(key, value) return(paste(key, "=", value))
    table <- function(name) return(c("", paste0("[", name, "]")))

    lines <- c(
        .LOG_HEADER,
        entry("format", .LOG_FORMAT),
        entry("software", .log_quote(software)),
        entry("engine", .log_quote(paste("connectivity-core", core_version_impl()))),
        entry("created", .log_quote(format(created, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"))),
        entry("elapsed_seconds", sprintf("%.1f", elapsed)),
        table("parameters"),
        entry("indicator", .log_quote(indicator)),
        if (indicator == "connected_habitat") {
            entry("habitat_output", .log_quote(habitat_outputs[[option]]))
        },
        if (indicator == "parc") entry("pa_to_pa", .log_boolean(pa_to_pa)),
        entry("lambdas_km", .log_array(.log_real(settings$lambdas))),
        entry("levels", .log_array(.log_integer(settings$levels))),
        entry("max_cost", .log_real(settings$max_cost)),
        entry("window_size", .log_integer(settings$window_size)),
        entry("outer_window", .log_integer(settings$outer_window)),
        entry("window_mode", .log_quote(settings$window_mode)),
        # Always written, so a log of a legacy run is never restored with a newer default
        entry("nodata_crossing", .log_quote(
            if (is.null(settings$nodata_crossing)) "none" else settings$nodata_crossing
        )),
        if (!is.null(settings$nodata_crossing)) {
            c(entry("crossing_width_km", .log_real(settings$crossing_width)),
              entry("crossing_cost", .log_real(settings$crossing_cost)))
        },
        entry("condition_scale", .log_scale(condition_scale)),
        if (indicator == "parc") entry("protected_areas_scale", .log_scale(protected_areas_scale)),
        if (!is.null(resistance)) entry("resistance_scale", .log_scale(resistance_scale)),
        if (!is.null(condition$band)) entry("band", .log_integer(condition$band)),
        if (!is.null(polygon_mask)) entry("closed_border", .log_boolean(closed_border)),
        table("inputs"),
        entry("condition", .log_quote(condition$path)),
        if (!is.null(resistance)) entry("resistance", .log_quote(resistance)),
        if (indicator == "parc") entry("protected_areas", .log_quote(protected_areas)),
        if (indicator == "beri") {
            c(entry("current_climate", .log_quote(current_climate)),
              entry("future_climates", .log_array(vapply(future_climates, .log_quote, ""))))
        },
        if (!is.null(polygon_mask)) {
            entry("polygon_mask", .log_quote(
                if (is.character(polygon_mask)) .log_absolute(polygon_mask) else .IN_MEMORY
            ))
        },
        table("output"),
        entry("file", .log_quote(.log_absolute(output))),
        if (!is.null(settings$n_threads)) {
            c(table("execution"), entry("threads", .log_integer(settings$n_threads)))
        }
    )
    return(paste0(paste(lines, collapse = "\n"), "\n"))
}

# A raster input as the log names it: an absolute path when it is a filename or
# holds every layer of one file in order, and "<in memory>" otherwise.
.log_input <- function(x) {
    if (is.character(x)) return(.log_absolute(x))
    if (inherits(x, "SpatRaster")) {
        sources <- terra::sources(x, bands = TRUE)
        file <- sources$source[[1L]]
        if (nzchar(file) && all(sources$source == file) &&
            identical(as.integer(sources$bands), seq_len(.log_layers(file)))) {
            return(.log_absolute(file))
        }
    }
    return(.IN_MEMORY)
}

# The condition input and band as the log names them. `used` is the one layer
# the run used. A SpatRaster layer of a file is logged as that file and its
# band, which a multi-layer file needs to be restored.
.log_condition <- function(x, used, band) {
    if (is.character(x)) return(list(path = .log_absolute(x), band = band))
    sources <- terra::sources(used, bands = TRUE)
    if (nrow(sources) != 1L || !nzchar(sources$source)) {
        return(list(path = .IN_MEMORY, band = band))
    }
    file_band <- if (!is.null(band) || .log_layers(sources$source) != 1L) sources$bands
    return(list(path = .log_absolute(sources$source), band = file_band))
}

.log_layers <- function(file) {
    return(tryCatch(terra::nlyr(terra::rast(file)), error = function(e) return(0L),
                    warning = function(w) return(0L)))
}

# `path` made absolute without resolving symbolic links, as Eco-Connectivity
# Studio does: "." components and repeated separators go, ".." stays.
.log_absolute <- function(path) {
    path <- path.expand(path)
    if (grepl(.NOT_A_FILE, path)) return(path)
    if (.Platform$OS.type == "windows") {
        return(normalizePath(path, winslash = "\\", mustWork = FALSE))
    }
    if (!startsWith(path, "/")) path <- file.path(getwd(), path)
    parts <- strsplit(path, "/", fixed = TRUE)[[1L]]
    return(paste0("/", paste(parts[nzchar(parts) & parts != "."], collapse = "/")))
}

# A TOML basic string, escaped as the format requires.
.log_quote <- function(text) {
    text <- enc2utf8(text)
    codes <- utf8ToInt(text)
    if (anyNA(codes)) codes <- utf8ToInt(iconv(text, "UTF-8", "UTF-8", sub = "\uFFFD"))
    chars <- vapply(codes, function(code) {
        return(switch(as.character(code),
            "92" = "\\\\", "34" = "\\\"", "9" = "\\t", "10" = "\\n", "13" = "\\r",
            if (code < 32L || code == 127L) sprintf("\\u%04X", code) else intToUtf8(code)
        ))
    }, "")
    return(paste0("\"", paste(chars, collapse = ""), "\""))
}

# Plain decimal text, the shortest that reads back to each value; whole numbers
# without a decimal point. `format()` pads the elements of a vector to a common
# width, so they are formatted one at a time.
.log_real <- function(x) {
    return(vapply(as.numeric(x), function(value) {
        for (digits in 15:17) {
            text <- format(value, digits = digits, scientific = FALSE,
                           decimal.mark = ".", big.mark = "")
            if (as.numeric(text) == value) break
        }
        return(text)
    }, ""))
}

.log_integer <- function(x) return(as.character(as.integer(x)))

.log_boolean <- function(x) return(if (isTRUE(x)) "true" else "false")

# A divisor; NULL and 0 leave a raster unscaled, as 1 does.
.log_scale <- function(x) return(if (is.null(x) || x == 0) "1" else .log_real(x))

.log_array <- function(items) return(paste0("[", paste(items, collapse = ", "), "]"))
