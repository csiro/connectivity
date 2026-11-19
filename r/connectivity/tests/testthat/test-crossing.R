# NoData crossing and the named arguments of connectedness() and beri(), as in
# the Python package 3.0.

crossing_settings <- workflow_settings[names(workflow_settings) != "nodata_crossing"]

run_crossing <- function(fun = connectedness, ...) {
    # The fixture's resistance is missing at some valid condition cells.
    return(suppressWarnings(do.call(fun, c(list(...), crossing_settings))))
}

values_of <- function(raster) return(terra::values(raster, mat = FALSE))

test_that("arguments after condition_file must be named in full", {
    fixture <- raster_fixture()
    expect_error(connectedness(fixture$condition, fixture$pa),
                 "^Arguments after condition_file must be named; unknown: <unnamed>$")
    expect_error(connectedness(fixture$condition, pa_fil = fixture$pa, NULL),
                 "^Arguments after condition_file must be named; unknown: pa_fil, <unnamed>$")
    # No partial matching after ...
    expect_error(connectedness(fixture$condition, pa = fixture$pa), "unknown: pa$")
    expect_error(beri(fixture$condition, fixture$current), "unknown: <unnamed>$")
    expect_error(beri(fixture$condition, future_files = fixture$future),
                 "^current_file is required; pass it by name")
})

test_that("crossing settings are checked with Python's messages", {
    fixture <- raster_fixture()
    run <- function(...) return(run_crossing(connectedness, condition_file = fixture$condition, ...))
    for (mode in list("none", "", NA_character_, 1, TRUE, c("strict", "landmass"))) {
        expect_error(run(nodata_crossing = mode),
                     "^nodata_crossing must be 'strict', 'landmass' or NULL$")
    }
    for (width in list(-1, Inf, NaN, NA_real_, "1", c(1, 2), TRUE)) {
        expect_error(run(crossing_width = width),
                     "^crossing_width must be a finite number of kilometres >= 0, got ")
    }
    expect_error(run(crossing_cost = 0.5), "^crossing_cost must be a finite number >= 1, got 0.5$")
    for (cost in list(Inf, NA_real_, "4", c(2, 4))) {
        expect_error(run(crossing_cost = cost), "^crossing_cost must be a finite number >= 1, got ")
    }
    # Modes are matched as in the core: any case, surrounding spaces ignored
    expect_identical(values_of(run(nodata_crossing = " LandMass ")),
                     values_of(run(nodata_crossing = "landmass")))
})

test_that("custom widths and costs are ignored under NULL, with a message", {
    fixture <- raster_fixture()
    run <- function(...) {
        return(run_crossing(connectedness, condition_file = fixture$condition,
                            nodata_crossing = NULL, ...))
    }
    expect_no_message(legacy <- run())
    expect_message(changed <- run(crossing_width = 3),
                   "crossing_width and crossing_cost are ignored because nodata_crossing is NULL")
    expect_message(run(crossing_cost = 1), "ignored because nodata_crossing is NULL")
    expect_identical(values_of(changed), values_of(legacy))
    expect_no_message(run_crossing(connectedness, condition_file = fixture$condition))
})

test_that("crossing modes change results only near NoData", {
    fixture <- raster_fixture(31L, 33L)
    condition <- fixture$condition
    run <- function(raster, ...) {
        return(values_of(run_crossing(connectedness, condition_file = raster, option = 1, ...)))
    }
    legacy <- run(condition, nodata_crossing = NULL)
    strict <- run(condition)
    # The defaults are strict, 0.5 km and 4
    expect_identical(run(condition, nodata_crossing = "strict", crossing_width = 0.5,
                         crossing_cost = 4), strict)
    expect_identical(is.na(strict), is.na(legacy))
    expect_false(identical(strict, legacy))
    expect_false(identical(run(condition, nodata_crossing = "strict", crossing_width = 3),
                           strict))
    expect_false(identical(run(condition, nodata_crossing = "landmass", crossing_width = 3),
                           run(condition, nodata_crossing = "landmass", crossing_width = 3,
                               crossing_cost = 1)))
    # Without NoData every setting gives the legacy result
    full <- terra::ifel(is.na(condition), 0.5, condition)
    expected <- run(full, nodata_crossing = NULL)
    for (mode in c("strict", "landmass")) {
        for (width in c(0, 3)) {
            expect_identical(run(full, nodata_crossing = mode, crossing_width = width,
                                 crossing_cost = 2), expected)
        }
    }
})

test_that("BERI and PARC accept crossing settings", {
    fixture <- raster_fixture()
    for (args in list(
        list(beri, condition_file = fixture$condition, current_file = fixture$current,
             future_files = fixture$future),
        list(connectedness, condition_file = fixture$condition, pa_file = fixture$pa,
             pa_to_pa = TRUE)
    )) {
        legacy <- values_of(do.call(run_crossing, c(args, list(nodata_crossing = NULL))))
        strict <- values_of(do.call(run_crossing, args))
        expect_identical(is.na(strict), is.na(legacy))
        expect_false(identical(strict, legacy))
        expect_true(all(strict >= 0, na.rm = TRUE))
    }
})

test_that("crossing confines traversal to land in the matrix backend", {
    args <- multilevel()
    args$nodata_crossing <- "strict"
    args$crossing_width <- 3
    expected <- do.call(connectivity_matrix, args)
    # Traversal on NoData is ignored: crossed NoData costs crossing_cost
    args$traversal[is.na(args$condition)] <- 1
    expect_identical(do.call(connectivity_matrix, args), expected)
    args$traversal[is.na(args$condition)] <- 0
    expect_identical(do.call(connectivity_matrix, args), expected)
})

test_that("open-border tiles with NoData crossing equal the whole raster", {
    fixture <- raster_fixture(93L, 101L)
    # A river two cells wide just inside the right-hand tiles, crossable at 3 km
    condition <- fixture$condition
    condition[, 54:55] <- NA
    resistance <- terra::ifel(is.na(fixture$resistance), 0.5, fixture$resistance)
    tiles <- make_tiles(condition, 2, 2, align_to = 4)
    for (mode in c("strict", "landmass")) {
        cases <- list(
            habitat = list(connectedness, list()),
            beri = list(beri, list(current_file = fixture$current, resistance_file = resistance))
        )
        for (name in names(cases)) {
            run <- function(polygon = NULL) {
                return(do.call(cases[[name]][[1L]], c(
                    list(condition_file = condition, polygon_mask = polygon,
                         nodata_crossing = mode, crossing_width = 3),
                    cases[[name]][[2L]], crossing_settings)))
            }
            whole <- run()
            mosaic <- terra::merge(terra::sprc(lapply(seq_len(nrow(tiles)), function(i) {
                return(run(tiles[i]))
            })))
            actual <- terra::as.matrix(mosaic, wide = TRUE)
            expected <- terra::as.matrix(whole, wide = TRUE)
            info <- paste(mode, name)
            expect_identical(is.na(actual), is.na(expected), info = info)
            expect_lte(max(abs(actual - expected), na.rm = TRUE), 1e-6)
        }
    }
})

test_that("the read window grows by the crossing width", {
    projected <- raster_fixture()$condition
    # 1 km cells: 3 km is 3 cells, plus one, in coarsest cells of 4
    expect_identical(.crossing_pad(projected, 3, .affine(projected), FALSE, 4L), 1)
    expect_identical(.crossing_pad(projected, 7.5, .affine(projected), FALSE, 4L), 3)
    expect_identical(.crossing_pad(projected, 0, .affine(projected), FALSE, 4L), 0)
    # Geographic cells are narrowest nearest a pole
    geographic <- geographic_fixture(raster_fixture())$condition
    cell <- 6371 * pi / 180 * terra::res(geographic)[1] * cos(-(40 + 13 / 120) * pi / 180)
    expect_identical(.crossing_pad(geographic, 3, .affine(geographic), TRUE, 2L),
                     ceiling((ceiling(3 / cell) + 1) / 2))
})

test_that("run logs record the NoData crossing used", {
    fixture <- raster_fixture()
    out <- tempfile(fileext = ".tif")
    crossing_lines <- function(...) {
        run_crossing(connectedness, condition_file = fixture$condition, filename = out,
                     overwrite = TRUE, ...)
        lines <- strsplit(rawToChar(readBin(sub("\\.tif$", ".toml", out), "raw", 1e5)), "\n")[[1L]]
        expect_identical(lines[[2L]], "format = 2")
        return(lines[grepl("^(nodata_crossing|crossing_width_km|crossing_cost) ", lines)])
    }
    expect_identical(crossing_lines(),
                     c("nodata_crossing = \"strict\"", "crossing_width_km = 0.5", "crossing_cost = 4"))
    expect_identical(crossing_lines(nodata_crossing = "landmass", crossing_width = 2.5,
                                    crossing_cost = 1.5),
                     c("nodata_crossing = \"landmass\"", "crossing_width_km = 2.5",
                       "crossing_cost = 1.5"))
    expect_identical(crossing_lines(nodata_crossing = NULL), "nodata_crossing = \"none\"")
})
