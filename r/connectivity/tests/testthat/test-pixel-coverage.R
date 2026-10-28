test_that("Rust polygon coverage matches Python, including overlap and holes", {
    fixture <- raster_fixture()
    polygons <- coverage_polygons()
    result <- pixel_coverage(polygons, fixture$condition, n_threads = 1)
    expect_python_result(result, "coverage")
    parallel <- pixel_coverage(polygons, fixture$condition, n_threads = 3)
    expect_identical(terra::values(parallel), terra::values(result))
})

test_that("union is performed before clipping and reference nodata is ignored", {
    reference <- terra::rast(nrows = 1, ncols = 1, xmin = 0, xmax = 10,
                             ymin = 0, ymax = 10, crs = "EPSG:3857")
    polygons <- terra::vect(c("POLYGON ((0 0,6 0,6 10,0 10,0 0))",
                              "POLYGON ((4 0,8 0,8 10,4 10,4 0))"), crs = terra::crs(reference))
    result <- pixel_coverage(polygons, reference, n_threads = 1)
    expect_equal(as.numeric(terra::values(result)), 0.8, tolerance = 1e-7)
    terra::values(reference) <- NA_real_
    expect_equal(terra::values(pixel_coverage(polygons, reference, n_threads = 1)), terra::values(result))
    empty <- pixel_coverage(polygons[FALSE, ], reference, n_threads = 1)
    expect_equal(as.numeric(terra::values(empty)), 0)
    unlabelled_empty <- pixel_coverage(terra::vect(), reference, n_threads = 1)
    expect_equal(as.numeric(terra::values(unlabelled_empty)), 0)
})

test_that("polygon file IO, multipart geometries and CRS conversion work", {
    fixture <- raster_fixture()
    polygons <- coverage_polygons()
    vector_path <- tempfile(fileext = ".gpkg")
    raster_path <- tempfile(fileext = ".tif")
    output_path <- tempfile(fileext = ".tif")
    terra::writeVector(polygons, vector_path)
    terra::writeRaster(fixture$condition, raster_path)
    result <- pixel_coverage(vector_path, raster_path, filename = output_path, n_threads = 1)
    expect_python_result(result, "coverage")
    expect_true(terra::compareGeom(result, fixture$condition, stopOnError = FALSE))
    combined <- terra::aggregate(polygons)
    expect_python_result(pixel_coverage(combined, fixture$condition, n_threads = 1), "coverage")
    geographic <- terra::project(polygons, "EPSG:4326")
    expect_python_result(pixel_coverage(geographic, fixture$condition, n_threads = 1), "coverage")
    expect_error(pixel_coverage(terra::centroids(polygons), fixture$condition), "polygon geometries")
    no_crs <- terra::deepcopy(polygons)
    terra::crs(no_crs) <- ""
    expect_error(pixel_coverage(no_crs, fixture$condition), "both have a CRS")
})
