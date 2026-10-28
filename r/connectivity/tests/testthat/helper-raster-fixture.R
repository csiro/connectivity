fixture_raster <- function(values) {
  dims <- dim(values)
  layers <- if (length(dims) == 2L) 1L else dims[3L]
  raster <- terra::rast(nrows = dims[1L], ncols = dims[2L], nlyrs = layers,
                         xmin = 500000, xmax = 500000 + dims[2L] * 1000,
                         ymin = 6000000 - dims[1L] * 1000, ymax = 6000000,
                         crs = "EPSG:3857")
  if (layers == 1L) {
    terra::values(raster) <- as.vector(t(matrix(values, dims[1L], dims[2L])))
  } else {
    terra::values(raster) <- vapply(seq_len(layers), function(band) {
      as.vector(t(values[, , band]))
    }, numeric(dims[1L] * dims[2L]))
  }
  raster
}

raster_fixture <- function(rows = 13L, cols = 14L) {
  i <- row(matrix(0, rows, cols)) - 1L
  j <- col(i) - 1L
  condition <- ((i * 7L + j * 13L) %% 10L + 1L) / 10
  condition[(i * 7L + j * 11L) %% 29L == 0L] <- NA_real_
  pa <- ((i * 3L + j * 7L) %% 5L) / 5
  pa[(i + j) %% 19L == 0L] <- NA_real_
  resistance <- ((i * 5L + j * 3L) %% 11L) / 10
  resistance[(i + j * 3L) %% 13L == 0L] <- NA_real_
  grids <- lapply(0:1, function(scenario) {
    grid <- array(0, c(rows, cols, 2L))
    for (band in 0:1) {
      layer <- ((i * 7L + j * 3L + scenario * 11L + band * 2L) %% 23L) / 23
      layer[(i * 3L + j * 5L + scenario + band) %% 31L == 0L] <- NA_real_
      grid[, , band + 1L] <- layer
    }
    fixture_raster(grid)
  })
  list(condition = fixture_raster(condition), pa = fixture_raster(pa),
       resistance = fixture_raster(resistance), current = grids[[1L]], future = grids[[2L]])
}

workflow_settings <- list(levels = c(2, 4), outer_window = 5,
                          lambdas = c(2, 20), max_cost = 20, n_threads = 1)

fixture_polygon <- function() {
  terra::vect(paste0(
    "POLYGON ((544200 5961800,550800 5961800,550800 5955200,",
    "548400 5955200,548400 5957800,544200 5957800,544200 5961800),",
    "(545100 5960900,546900 5960900,546900 5959100,545100 5959100,545100 5960900))"
  ), crs = "EPSG:3857")
}

coverage_polygons <- function() {
  terra::vect(c(
    "POLYGON ((500100 5987200,502600 5987200,502600 5991200,500100 5991200,500100 5987200))",
    "POLYGON ((501300 5988000,504800 5988000,504800 5990400,501300 5990400,501300 5988000))",
    paste0("POLYGON ((505000 5999000,508600 5997300,507100 5994100,505000 5999000),",
           "(506200 5997300,507100 5997000,506800 5996200,506200 5997300))")
  ), crs = "EPSG:3857")
}

geographic_fixture <- function(fixture) {
  lapply(fixture, function(raster) {
    out <- terra::deepcopy(raster)
    terra::ext(out) <- terra::ext(144, 144 + ncol(out) / 120,
                                 -40 - nrow(out) / 120, -40)
    terra::crs(out) <- "EPSG:4326"
    out
  })
}

geographic_polygon <- function() {
  geometry <- terra::geom(fixture_polygon())
  geometry[, "x"] <- 144 + (geometry[, "x"] - 500000) / 120000
  geometry[, "y"] <- -40 + (geometry[, "y"] - 6000000) / 120000
  terra::vect(geometry, type = "polygons", crs = "EPSG:4326")
}

reference_matrix <- function(case) {
  reference <- utils::read.csv(test_path("fixtures", "python-workflows.csv"))
  selected <- reference[reference$case == case, ]
  stopifnot(nrow(selected) > 0L)
  out <- matrix(NA_real_, max(selected$row), max(selected$col))
  out[cbind(selected$row, selected$col)] <- selected$value
  out
}

expect_python_result <- function(raster, case, tolerance = 1e-6) {
  expect_s4_class(raster, "SpatRaster")
  actual <- matrix(terra::as.array(raster)[, , 1L], nrow(raster), ncol(raster))
  expected <- reference_matrix(case)
  expect_identical(dim(actual), dim(expected))
  expect_identical(is.na(actual), is.na(expected))
  expect_true(all(abs(actual - expected) <= tolerance, na.rm = TRUE), info = case)
}
