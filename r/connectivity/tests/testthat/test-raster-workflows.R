test_that("habitat options and window modes match frozen Python workflows", {
    fixture <- raster_fixture()
    for (option in 1:3) {
        actual <- do.call(connectedness, c(list(condition_file = fixture$condition, option = option), workflow_settings))
        expect_python_result(actual, paste0("option", option))
        expect_true(terra::compareGeom(actual, fixture$condition, stopOnError = FALSE))
    }
    square <- do.call(connectedness, c(list(condition_file = fixture$condition, window_mode = " SQUARE "), workflow_settings))
    expect_python_result(square, "square")
})

test_that("PARC preparation, protected targets and scales match Python", {
    fixture <- raster_fixture()
    for (gated in c(FALSE, TRUE)) {
        actual <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                                pa_file = fixture$pa, pa_to_pa = gated, option = 2L), workflow_settings))
        expect_python_result(actual, if (gated) "parc_gated" else "parc")
    }
    scaled <- do.call(connectedness, c(list(condition_file = fixture$condition * 100,
                                            pa_file = fixture$pa * 100, scale = c(100, 100)), workflow_settings))
    expect_python_result(scaled, "parc")
    pa_only <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                             pa_file = fixture$pa * 100, scale = list(NULL, 100)), workflow_settings))
    expect_python_result(pa_only, "parc")
    scaled_habitat <- do.call(connectedness, c(list(condition_file = fixture$condition * 100,
                                                    scale = 100, option = 1), workflow_settings))
    expect_python_result(scaled_habitat, "option1")
})

test_that("resistance fallback preserves condition nodata and warns", {
    fixture <- raster_fixture()
    expect_warning(actual <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                                           resistance_file = fixture$resistance * 100, resistance_scale = 100, option = 1),
                                                      workflow_settings)), "valid condition but missing resistance")
    expect_python_result(actual, "resistance")
    blank <- fixture$resistance
    terra::values(blank) <- NA_real_
    expect_warning(fallback <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                                             resistance_file = blank, option = 1), workflow_settings)), "missing resistance")
    expect_python_result(fallback, "option1")
})

test_that("BERI preserves scenario and band order and current-only results", {
    fixture <- raster_fixture()
    futures <- list(fixture$future)
    actual <- do.call(beri, c(list(condition_file = fixture$condition,
                                   current_file = fixture$current, future_files = futures), workflow_settings))
    expect_python_result(actual, "beri")
    expect_length(futures, 1L)
    current <- do.call(beri, c(list(condition_file = fixture$condition,
                                    current_file = fixture$current), workflow_settings))
    expect_python_result(current, "beri_current")
    expect_warning(resistance <- do.call(beri, c(list(condition_file = fixture$condition,
                                                      current_file = fixture$current, future_files = futures,
                                                      resistance_file = fixture$resistance), workflow_settings)), "missing resistance")
    expect_python_result(resistance, "beri_resistance")
    one_future <- do.call(beri, c(list(condition_file = fixture$condition,
                                       current_file = fixture$current, future_files = fixture$future), workflow_settings))
    expect_equal(terra::values(one_future), terra::values(actual))
    expect_error(do.call(beri, c(list(condition_file = fixture$condition,
                                      current_file = fixture$current, future_files = fixture$future[[1L]]), workflow_settings)), "band count")
})

test_that("geographic distances and planar pixel margins match Python", {
    fixture <- geographic_fixture(raster_fixture())
    actual <- do.call(connectedness, c(list(condition_file = fixture$condition, option = 1), workflow_settings))
    expect_python_result(actual, "geographic")
    large <- geographic_fixture(raster_fixture(93L, 101L))
    actual <- do.call(connectedness, c(list(condition_file = large$condition,
                                            polygon_mask = geographic_polygon(), option = 1), workflow_settings))
    expect_python_result(actual, "geographic_polygon")
})

test_that("projected units are converted and an absent CRS is diagnosed", {
    fixture <- raster_fixture()
    feet <- terra::deepcopy(fixture$condition)
    terra::ext(feet) <- terra::ext(as.vector(terra::ext(feet)) / (1200 / 3937))
    terra::crs(feet) <- "+proj=utm +zone=55 +datum=WGS84 +units=us-ft +no_defs"
    result <- do.call(connectedness, c(list(condition_file = feet, option = 1), workflow_settings))
    expect_python_result(result, "option1")
    expect_true(terra::same.crs(result, feet))
    unlabelled <- terra::deepcopy(fixture$condition)
    terra::crs(unlabelled) <- ""
    expect_warning(result <- do.call(connectedness, c(list(condition_file = unlabelled,
                                                           option = 1), workflow_settings)), "CRS is missing")
    expect_python_result(result, "option1")
})

test_that("single-band climate remains a 3D input to the matrix backend", {
    fixture <- raster_fixture()
    climate <- fixture$current[[1L]]
    actual <- do.call(beri, c(list(condition_file = fixture$condition,
                                   current_file = climate), workflow_settings))
    expected <- do.call(connectivity_matrix, c(list(
        condition = terra::as.array(fixture$condition)[, , 1L],
        transform = c(1000, 0, 500000, 0, -1000, 6000000),
        transgrids = list(terra::as.array(climate))), workflow_settings))
    expect_equal(terra::as.array(actual)[, , 1L], expected)
})

test_that("open and closed polygon analyses match non-zero-offset Python windows", {
    fixture <- raster_fixture(93L, 101L)
    polygon <- fixture_polygon()
    for (closed in c(FALSE, TRUE)) {
        actual <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                                polygon_mask = polygon, closed_border = closed, margin_px = 0, option = 1), workflow_settings))
        expect_python_result(actual, if (closed) "polygon_closed" else "polygon_open")
        expect_equal(unname(as.vector(terra::ext(actual))), c(544000, 551000, 5955000, 5962000))
    }
    parc <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                          pa_file = fixture$pa, pa_to_pa = TRUE, polygon_mask = polygon, margin_px = 0), workflow_settings))
    expect_python_result(parc, "polygon_parc")
    climate <- do.call(beri, c(list(condition_file = fixture$condition,
                                    current_file = fixture$current, future_files = list(fixture$future),
                                    polygon_mask = polygon, margin_px = 0), workflow_settings))
    expect_python_result(climate, "polygon_beri")
})

test_that("terra file IO preserves values, metadata and nodata", {
    fixture <- raster_fixture()
    paths <- vapply(fixture[c("condition", "current", "future")], function(raster) {
        path <- tempfile(fileext = ".tif")
        terra::writeRaster(raster, path, datatype = "FLT4S", NAflag = -9999)
        return(path)
    }, character(1))
    filename <- tempfile(fileext = ".tif")
    result <- do.call(connectedness, c(list(condition_file = paths[["condition"]],
                                            option = 1, filename = filename), workflow_settings))
    expect_true(file.exists(filename))
    expect_python_result(result, "option1")
    expect_true(terra::compareGeom(result, fixture$condition, stopOnError = FALSE))
    expect_equal(terra::datatype(result), "FLT4S")
    expect_error(do.call(connectedness, c(list(condition_file = paths[["condition"]],
                                               filename = filename), workflow_settings)), "exist")
    climate <- do.call(beri, c(list(condition_file = paths[["condition"]],
                                    current_file = paths[["current"]], future_files = paths[["future"]]), workflow_settings))
    expect_python_result(climate, "beri")
})

test_that("empty domains and input validation have predictable results", {
    fixture <- raster_fixture()
    blank <- fixture$condition
    terra::values(blank) <- NA_real_
    actual <- do.call(connectedness, c(list(condition_file = blank), workflow_settings))
    expect_true(all(is.na(terra::values(actual))))
    none <- fixture$pa
    terra::values(none) <- 0
    actual <- do.call(connectedness, c(list(condition_file = fixture$condition,
                                            pa_file = none), workflow_settings))
    expect_true(all(is.na(terra::values(actual))))
    shifted <- terra::shift(fixture$pa, dx = 500)
    expect_error(do.call(connectedness, c(list(condition_file = fixture$condition,
                                               pa_file = shifted), workflow_settings)), "same grid")
    for (change in list(list(option = 4), list(scale = -1), list(scale = c(1, 2, 3)),
                        list(margin_px = -1), list(window_size = 4), list(levels = c(2, 8)))) {
        args <- utils::modifyList(c(list(condition_file = fixture$condition), workflow_settings), change)
        expect_error(do.call(connectedness, args))
    }
    adjusted <- utils::modifyList(workflow_settings, list(outer_window = 1))
    expect_message(do.call(connectedness, c(list(condition_file = fixture$condition), adjusted)), "increased")
})
