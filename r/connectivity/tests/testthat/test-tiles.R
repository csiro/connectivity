test_that("tile layouts match Python's planner", {
    # Python make_tiles() 2.4.0 results; R starts are one larger.
    cases <- list(
        list(tile_fixture(), 2, 3, 8, TRUE, 0.1,
             c(1, 56, 1, 48, 1420.8), c(57, 96, 73, 104, 1408)),
        list(tile_fixture(93L, 101L), 3, 2, 4, TRUE, 0.1,
             c(1, 36, 1, 64, 1382.4), c(69, 93, 53, 101, 1347.5)),
        list(tile_fixture(), 2, 3, 8, TRUE, 0,
             c(1, 56, 1, 56, 1600), c(57, 96, 73, 104, 1280))
    )
    for (case in cases) {
        tiles <- make_tiles(case[[1]], case[[2]], case[[3]], align_to = case[[4]],
                            balanced = case[[5]], io_weight = case[[6]])
        table <- terra::as.data.frame(tiles)
        columns <- c("row_start", "row_stop", "col_start", "col_stop", "cost")
        expect_equal(unlist(table[1L, columns]), setNames(case[[7]], columns), tolerance = 1e-6)
        expect_equal(unlist(table[nrow(table), columns]), setNames(case[[8]], columns), tolerance = 1e-6)
    }
    tiles <- make_tiles(tile_fixture(93L, 101L), 3, 2, align_to = 4)
    expect_identical(tiles$id, 1:6)
    expect_identical(tiles$row_stop, rep(c(32L, 64L, 93L), each = 2L))
    expect_identical(tiles$col_stop, rep(c(52L, 101L), 3L))
    expect_false("cost" %in% names(tiles))
    expect_equal(as.vector(terra::ext(tiles[2])), c(xmin = 552000, xmax = 601000,
                                                       ymin = 5968000, ymax = 6000000))
    expect_true(terra::same.crs(tiles, tile_fixture()))
    expect_equal(sum(terra::expanse(tiles, transform = FALSE)), 93 * 101 * 1e6)
})

test_that("merged 2x2 tiles equal the whole raster, including tile edges", {
    fixture <- raster_fixture(93L, 101L)
    resistance <- terra::ifel(is.na(fixture$resistance), 0.5, fixture$resistance)
    cases <- list(
        habitat = list(connectedness, fixture$condition, list()),
        geographic = list(connectedness, geographic_fixture(fixture)$condition, list()),
        parc = list(connectedness, fixture$condition, list(pa_file = fixture$pa, pa_to_pa = TRUE)),
        beri = list(beri, fixture$condition, list(
            current_file = fixture$current, future_files = list(fixture$future),
            resistance_file = resistance))
    )
    edges <- list(rows = c(48L, 49L), cols = c(52L, 53L))
    for (name in names(cases)) {
        case <- cases[[name]]
        run <- function(polygon = NULL) {
            return(do.call(case[[1L]], c(list(condition_file = case[[2L]], polygon_mask = polygon),
                                         case[[3L]], workflow_settings)))
        }
        tiles <- make_tiles(case[[2L]], 2, 2, align_to = 4)
        expect_identical(tiles$row_stop, c(48L, 48L, 93L, 93L))
        expect_identical(tiles$col_stop, c(52L, 101L, 52L, 101L))
        whole <- run()
        mosaic <- terra::merge(terra::sprc(lapply(seq_len(nrow(tiles)), function(i) {
            return(run(tiles[i]))
        })))
        expect_true(terra::compareGeom(mosaic, whole, stopOnError = FALSE), info = name)
        actual <- terra::as.matrix(mosaic, wide = TRUE)
        expected <- terra::as.matrix(whole, wide = TRUE)
        expect_identical(is.na(actual), is.na(expected), info = name)
        expect_lte(max(abs(actual - expected), na.rm = TRUE), 1e-6)
        boundary <- c(actual[edges$rows, ] - expected[edges$rows, ],
                      actual[, edges$cols] - expected[, edges$cols])
        expect_gt(sum(!is.na(boundary)), 100L)
        expect_lte(max(abs(boundary), na.rm = TRUE), 1e-6)
    }
})

test_that("make_tile selects one tile and arguments are validated", {
    raster <- tile_fixture(93L, 101L)
    tiles <- make_tiles(raster, 3, 2, align_to = 4, balanced = TRUE)
    one <- make_tile(raster, 3, 2, tile_id = 5, align_to = 4, balanced = TRUE)
    expect_identical(terra::as.data.frame(one), terra::as.data.frame(tiles[5]))
    expect_equal(terra::geom(one)[, c("x", "y")], terra::geom(tiles[5])[, c("x", "y")])
    expect_error(make_tile(raster, 3, 2, tile_id = 7), "between 1 and 6")
    expect_error(make_tiles(raster, 3, 1, align_to = 64), "Cannot split")
    expect_error(make_tiles(raster, 30, 1, align_to = 4, balanced = TRUE), "Cannot split")
    for (change in list(list(nrows = 0), list(ncols = 1.5), list(align_to = NA),
                        list(balanced = "yes"), list(io_weight = -1))) {
        args <- utils::modifyList(list(raster_file = raster, nrows = 2, ncols = 2), change)
        expect_error(do.call(make_tiles, args))
    }
    empty <- terra::rast(raster)
    expect_s4_class(make_tiles(empty, 2, 2), "SpatVector")
    expect_error(make_tiles(empty, 2, 2, balanced = TRUE), "no raster values")
})
