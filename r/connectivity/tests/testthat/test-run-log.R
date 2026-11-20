# Run logs, in the format shared with the Python package and Eco-Connectivity
# Studio.

read_log <- function(path) return(rawToChar(readBin(path, "raw", file.size(path))))

log_lines <- function(text) return(strsplit(text, "\n", fixed = TRUE)[[1L]])

write_tif <- function(raster) {
    path <- tempfile(fileext = ".tif")
    terra::writeRaster(raster, path)
    return(path)
}

run_quietly <- function(fun, ...) {
    # The fixture's resistance is missing at some valid condition cells.
    return(suppressWarnings(do.call(fun, c(list(..., overwrite = TRUE), workflow_settings))))
}

test_that("the documented example is written as Python and Studio write it", {
    skip_on_os("windows")
    settings <- list(lambdas = c(2, 20, 200), levels = c(1L, 2L, 4L, 8L, 16L, 32L),
                     max_cost = 2, window_size = 5L, outer_window = 9L,
                     window_mode = "circular", n_threads = 8L,
                     nodata_crossing = "strict", crossing_width = 1, crossing_cost = 4)
    text <- .format_log(
        created = as.POSIXct("2026-11-11 03:14:15", tz = "UTC"), elapsed = 42.48,
        indicator = "connected_habitat", settings = settings,
        condition = list(path = "/data/tasmania/site_condition.tif"),
        resistance = "/data/tasmania/resistance.tif", resistance_scale = 100,
        output = "/data/results/connected_habitat.tif",
        software = "connectivity R package 0.1.0.9000"
    )
    expect_identical(text, paste0(paste(c(
        "# Connectivity run log",
        "format = 2",
        "software = \"connectivity R package 0.1.0.9000\"",
        paste0("engine = \"connectivity-core ", core_version_impl(), "\""),
        "created = \"2026-11-11T03:14:15Z\"",
        "elapsed_seconds = 42.5",
        "",
        "[parameters]",
        "indicator = \"connected_habitat\"",
        "habitat_output = \"connected_habitat\"",
        "lambdas_km = [2, 20, 200]",
        "levels = [1, 2, 4, 8, 16, 32]",
        "max_cost = 2",
        "window_size = 5",
        "outer_window = 9",
        "window_mode = \"circular\"",
        "nodata_crossing = \"strict\"",
        "crossing_width_km = 1",
        "crossing_cost = 4",
        "condition_scale = 1",
        "resistance_scale = 100",
        "",
        "[inputs]",
        "condition = \"/data/tasmania/site_condition.tif\"",
        "resistance = \"/data/tasmania/resistance.tif\"",
        "",
        "[output]",
        "file = \"/data/results/connected_habitat.tif\"",
        "",
        "[execution]",
        "threads = 8"
    ), collapse = "\n"), "\n"))
    expect_match(core_version_impl(), "^[0-9]+\\.[0-9]+\\.[0-9]+")
})

test_that("values are written as the format requires", {
    expect_identical(.log_quote("C:\\maps\\\"x\"\t\u0001\u00e9\u007f"),
                     "\"C:\\\\maps\\\\\\\"x\\\"\\t\\u0001\u00e9\\u007F\"")
    old <- options(OutDec = ",", scipen = -9, digits = 3)
    on.exit(options(old))
    # Shortest round-trip text, as Python and Rust write it
    expect_identical(.log_real(c(2, 0.5, 0.1 + 0.2, 1 / 3, 1e-7, 1e20)),
                     c("2", "0.5", "0.30000000000000004", "0.3333333333333333",
                       "0.0000001", "100000000000000000000"))
    expect_identical(vapply(list(NULL, 0, 1, 255, 0.5), .log_scale, ""),
                     c("1", "1", "1", "255", "0.5"))
    expect_identical(.log_array(character()), "[]")
})

test_that("logs sit beside their output and name absolute paths", {
    expect_identical(.log_path("maps/connected_habitat.tif"), "maps/connected_habitat.toml")
    expect_identical(.log_path("beri.v2.TIFF"), "beri.v2.toml")
    expect_identical(.log_path("run.v2/result"), "run.v2/result.toml")
    for (name in c("vrt:///data/x.tif?bands=2", "/vsis3/bucket/x.tif", "NETCDF:\"x.nc\":tas")) {
        expect_identical(.log_absolute(name), name)
    }
    skip_on_os("windows")
    old <- setwd(tempdir())
    on.exit(setwd(old))
    expect_identical(.log_absolute("a/./b//../c.tif"), paste0(getwd(), "/a/b/../c.tif"))
    expect_identical(.log_absolute("/data/./x.tif"), "/data/x.tif")
})

test_that("Connected Habitat runs write a log of the settings they used", {
    fixture <- raster_fixture()
    condition <- write_tif(fixture$condition)
    out <- tempfile(fileext = ".tif")
    run_quietly(connectedness, condition_file = condition, option = 2, filename = out)
    log <- sub("\\.tif$", ".toml", out)
    text <- read_log(log)
    expect_false(grepl("\r", text, fixed = TRUE))
    expect_true(endsWith(text, "\n"))
    lines <- log_lines(text)
    expect_identical(lines[[1L]], "# Connectivity run log")
    expect_identical(lines[[3L]], paste0("software = \"connectivity R package ",
                                         utils::packageVersion("connectivity"), "\""))
    expect_identical(lines[-(1:7)], c(
        "[parameters]",
        "indicator = \"connected_habitat\"",
        "habitat_output = \"condition_times_connectedness\"",
        "lambdas_km = [2, 20]",
        "levels = [1, 2, 4]",
        "max_cost = 20",
        "window_size = 3",
        "outer_window = 5",
        "window_mode = \"circular\"",
        "nodata_crossing = \"none\"",
        "condition_scale = 1",
        "",
        "[inputs]",
        paste("condition =", .log_quote(.log_absolute(condition))),
        "",
        "[output]",
        paste("file =", .log_quote(.log_absolute(out))),
        "",
        "[execution]",
        "threads = 1"
    ))
})

test_that("PARC logs name protected areas, scales and polygon masks", {
    fixture <- raster_fixture(93L, 101L)
    pa <- write_tif(fixture$pa)
    out <- tempfile(fileext = ".tif")
    run_quietly(connectedness, condition_file = fixture$condition, pa_file = pa,
                pa_to_pa = TRUE, scale = c(0, 2), polygon_mask = fixture_polygon(),
                closed_border = TRUE, filename = out)
    lines <- log_lines(read_log(sub("\\.tif$", ".toml", out)))
    expected <- c("indicator = \"parc\"", "pa_to_pa = true", "condition_scale = 1",
                  "protected_areas_scale = 2", "closed_border = true",
                  "condition = \"<in memory>\"",
                  paste("protected_areas =", .log_quote(.log_absolute(pa))),
                  "polygon_mask = \"<in memory>\"")
    expect_true(all(expected %in% lines))
    expect_false(any(grepl("^(habitat_output|resistance|band) ", lines)))
    # A polygon file is named by its path
    polygon <- tempfile(fileext = ".gpkg")
    terra::writeVector(fixture_polygon(), polygon)
    run_quietly(connectedness, condition_file = fixture$condition, pa_file = fixture$pa,
                polygon_mask = polygon, filename = out)
    lines <- log_lines(read_log(sub("\\.tif$", ".toml", out)))
    expect_true(all(c("closed_border = false", "protected_areas = \"<in memory>\"",
                      paste("polygon_mask =", .log_quote(.log_absolute(polygon)))) %in% lines))
})

test_that("a layer of a multi-layer file is logged as the file and its band", {
    fixture <- raster_fixture()
    series <- write_tif(c(fixture$condition^0.5, fixture$condition))
    single <- write_tif(fixture$condition)
    resistance <- write_tif(fixture$resistance)
    resistances <- write_tif(c(fixture$resistance, fixture$resistance))
    out <- tempfile(fileext = ".tif")
    logged <- function(...) {
        run_quietly(connectedness, ..., filename = out)
        lines <- log_lines(read_log(sub("\\.tif$", ".toml", out)))
        return(lines[grepl("^(band|condition|resistance|resistance_scale) ", lines)])
    }
    file <- function(key, path) return(paste(key, "=", .log_quote(.log_absolute(path))))
    expect_identical(logged(condition_file = series, band = 2L), c("band = 2", file("condition", series)))
    expect_identical(logged(condition_file = terra::rast(series), band = 2L),
                     c("band = 2", file("condition", terra::sources(terra::rast(series)))))
    # Without band, the layer's band of its file is logged, which restoring needs
    expect_identical(logged(condition_file = terra::rast(series)[[2L]]),
                     c("band = 2", file("condition", terra::sources(terra::rast(series)))))
    expect_identical(logged(condition_file = terra::rast(single)),
                     file("condition", terra::sources(terra::rast(single))))
    # A raster holding every layer of its file is named by it; some layers are not
    expect_identical(logged(condition_file = series, band = 1L, resistance_file = terra::rast(resistance)),
                     c("resistance_scale = 1", "band = 1", file("condition", series),
                       file("resistance", terra::sources(terra::rast(resistance)))))
    expect_identical(logged(condition_file = single, resistance_file = terra::rast(resistances)[[1L]]),
                     c("resistance_scale = 1", file("condition", single), "resistance = \"<in memory>\""))
})

test_that("BERI logs name the climate scenarios in order", {
    fixture <- raster_fixture()
    current <- write_tif(fixture$current)
    out <- tempfile(fileext = ".tif")
    run_quietly(beri, condition_file = fixture$condition, current_file = current,
                future_files = list(fixture$future, current), filename = out)
    lines <- log_lines(read_log(sub("\\.tif$", ".toml", out)))
    expect_true(all(c("indicator = \"beri\"",
                      paste("current_climate =", .log_quote(.log_absolute(current))),
                      paste0("future_climates = [\"<in memory>\", ", .log_quote(.log_absolute(current)), "]"))
                    %in% lines))
    expect_false(any(grepl("^(habitat_output|pa_to_pa|protected_areas_scale) ", lines)))
    run_quietly(beri, condition_file = fixture$condition, current_file = current, filename = out)
    expect_true("future_climates = []" %in% log_lines(read_log(sub("\\.tif$", ".toml", out))))
})

test_that("logs can be turned off by argument, option or environment variable", {
    fixture <- raster_fixture()
    out <- tempfile(fileext = ".tif")
    log <- sub("\\.tif$", ".toml", out)
    run <- function(...) {
        unlink(c(out, log))
        run_quietly(connectedness, condition_file = fixture$condition, filename = out, ...)
        return(file.exists(log))
    }
    old <- options(connectivity.save_log = NULL)
    old_env <- Sys.getenv("CONNECTIVITY_SAVE_LOG", unset = NA)
    on.exit({
        options(old)
        if (is.na(old_env)) Sys.unsetenv("CONNECTIVITY_SAVE_LOG") else Sys.setenv(CONNECTIVITY_SAVE_LOG = old_env)
    })
    Sys.unsetenv("CONNECTIVITY_SAVE_LOG")
    expect_true(run())
    expect_false(run(save_log = FALSE))
    Sys.setenv(CONNECTIVITY_SAVE_LOG = " Off ")
    expect_false(run())
    expect_true(run(save_log = TRUE))
    # The option comes before the variable, and the argument before both
    options(connectivity.save_log = TRUE)
    expect_true(run())
    Sys.setenv(CONNECTIVITY_SAVE_LOG = "yes")
    options(connectivity.save_log = FALSE)
    expect_false(run())
    expect_true(run(save_log = TRUE))
    # In-memory results never write one
    options(connectivity.save_log = TRUE)
    expect_no_warning(run_quietly(connectedness, condition_file = fixture$condition))
    expect_error(run(save_log = "yes"), "save_log must be TRUE or FALSE")
})

test_that("only run logs are replaced, and failing to write one only warns", {
    fixture <- raster_fixture()
    out <- tempfile(fileext = ".tif")
    log <- sub("\\.tif$", ".toml", out)
    run <- function() {
        unlink(out)
        return(do.call(connectedness, c(list(condition_file = fixture$condition, filename = out),
                                        workflow_settings)))
    }
    writeLines("title = 'mine'", log)
    expect_warning(run(), "Run log not saved: .*\\.toml exists and is not a run log")
    expect_identical(readLines(log), "title = 'mine'")
    writeBin(charToRaw("# Connectivity run log\r\nformat = 1\r\n"), log)
    expect_no_warning(run())
    expect_true("levels = [1, 2, 4]" %in% log_lines(read_log(log)))
    unlink(log)
    dir.create(log)
    on.exit(unlink(log, recursive = TRUE))
    expect_warning(result <- run(), "Run log not saved: .*\\.toml is a directory")
    expect_s4_class(result, "SpatRaster")
})
