test_that("resolution planning matches Python dimensions, spans and thresholds", {
    grid <- terra::rast(nrows = 31, ncols = 65, xmin = 0, xmax = 6500,
                        ymin = 0, ymax = 6200, crs = "EPSG:3857")
    output <- capture.output(result <- withVisible(resolution_info(
        grid, levels = c(4, 1, 3, 2, 4), min_dimension = 16, outer_window = 7
    )))
    expect_false(result$visible)
    plan <- result$value
    expect_equal(plan$level, c(1, 2, 3, 4))
    expect_equal(plan$width, c(65, 33, 22, 17))
    expect_equal(plan$height, c(31, 16, 11, 8))
    expect_equal(plan$cells, c(2015, 528, 242, 136))
    expect_equal(plan$percent_base, 100 * c(2015, 528, 242, 136) / 2015)
    expect_equal(plan$x_resolution, c(100, 200, 300, 400))
    expect_equal(plan$y_resolution, c(200, 400, 600, 800))
    expect_equal(plan$reach, c(1400, 2800, 4200, 5600))
    expect_equal(plan$reach_km, c(1.4, 2.8, 4.2, 5.6))
    expect_equal(plan$note, c("", "", "small grid; review", "small grid; review"))
    expect_equal(attr(plan, "unit"), "metre")
    expect_match(paste(output, collapse = "\n"), "100 x 200 metre", fixed = TRUE)
    expect_match(paste(output, collapse = "\n"), "Reach km", fixed = TRUE)
})

test_that("geographic reach uses the raster centre latitude", {
    grid <- terra::rast(nrows = 20, ncols = 20, xmin = 140, xmax = 144,
                        ymin = -61, ymax = -59, crs = "EPSG:4326")
    output <- capture.output(plan <- resolution_info(grid, levels = c(1, 4)))
    # East-west cells are twice as wide in degrees but equal in km at 60 S.
    expect_equal(plan$reach, c(1, 4))
    expect_equal(plan$reach_km, c(55.66, 222.64))
    expect_equal(attr(plan, "unit"), "degrees")
    expect_match(paste(output, collapse = "\n"), "centre latitude -60.000", fixed = TRUE)
})

test_that("projected non-metre units and missing CRS are handled", {
    grid <- terra::rast(nrows = 10, ncols = 10, xmin = 0, xmax = 1000,
                        ymin = 0, ymax = 1000, crs = "EPSG:2263")
    capture.output(plan <- resolution_info(grid, levels = 1))
    expect_equal(plan$reach_km, 500 * (1200 / 3937) / 1000)
    expect_match(attr(plan, "unit"), "foot", ignore.case = TRUE)
    terra::crs(grid) <- ""
    expect_warning(output <- capture.output(plan <- resolution_info(grid, levels = 1)), NA)
    expect_equal(attr(plan, "unit"), "map units")
    expect_true(is.na(plan$reach_km))
    expect_match(paste(output, collapse = "\n"), "CRS: unknown", fixed = TRUE)
})

test_that("filenames, multiple layers and metadata-only rasters are supported", {
    grid <- terra::rast(nrows = 3, ncols = 5, nlyrs = 2, xmin = 0, xmax = 500,
                        ymin = 0, ymax = 300, crs = "EPSG:3857")
    capture.output(empty <- resolution_info(grid, levels = c(1, 2)))
    terra::values(grid) <- seq_len(30)
    path <- tempfile(fileext = ".tif")
    on.exit(unlink(path))
    terra::writeRaster(grid, path, datatype = "INT2S", NAflag = -9999)
    output <- capture.output(stored <- resolution_info(path, levels = c(1, 2)))
    expect_equal(stored, empty)
    expect_match(paste(output, collapse = "\n"), path, fixed = TRUE)
    expect_match(paste(output, collapse = "\n"), "Bands: 2", fixed = TRUE)
    expect_match(paste(output, collapse = "\n"), "Data type(s): INT2S", fixed = TRUE)
    expect_match(paste(output, collapse = "\n"), "Nodata: -9999", fixed = TRUE)
    stored_raster <- terra::rast(path)
    terra::NAflag(stored_raster) <- -99
    expect_match(paste(capture.output(resolution_info(stored_raster, levels = 1)), collapse = "\n"),
                 "Nodata: -99", fixed = TRUE)
})

test_that("default levels and clamped thresholds match the Python helper", {
    grid <- terra::rast(nrows = 1, ncols = 1)
    capture.output(plan <- resolution_info(grid, min_dimension = -2, outer_window = 0))
    expect_equal(plan$level, 2^(0:9))
    expect_equal(plan$width, rep(1, 10))
    expect_equal(plan$height, rep(1, 10))
    expect_equal(plan$percent_base, rep(100, 10))
    expect_equal(plan$note, rep("", 10))
    expect_equal(plan$reach, 360 * 2^(0:9))
})

test_that("invalid planning arguments fail with useful errors", {
    grid <- terra::rast()
    for (levels in list(numeric(), 0, -1, 1.5, NA_real_, Inf, "2", 2^31)) {
        expect_error(resolution_info(grid, levels = levels), "levels")
    }
    for (value in list(numeric(), c(1, 2), NA_real_, Inf, 1.5, "2")) {
        expect_error(resolution_info(grid, min_dimension = value), "min_dimension")
        expect_error(resolution_info(grid, outer_window = value), "outer_window")
    }
    expect_error(resolution_info(matrix(1)), "SpatRaster or a raster filename")
})
