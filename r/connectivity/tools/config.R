# Note: Any variables prefixed with `.` are used for text
# replacement in the Makevars.in and Makevars.win.in

# check the packages MSRV first
source("tools/msrv.R")

# check DEBUG and NOT_CRAN environment variables
env_debug <- Sys.getenv("DEBUG")
env_not_cran <- Sys.getenv("NOT_CRAN")

# Vendored crates from tools/prepare-source.R are used only while Cargo.lock
# matches vendor.lock, the lockfile they were made from. readLines() ignores
# line endings.
vendor_files <- file.path("src/rust", c("vendor.tar.xz", "vendor-config.toml", "vendor.lock"))
vendor_exists <- file.exists(vendor_files[1L])
vendor_current <- all(file.exists(vendor_files)) &&
    identical(readLines(vendor_files[3L], warn = FALSE),
              readLines("src/rust/Cargo.lock", warn = FALSE))

is_not_cran <- env_not_cran != ""
is_debug <- env_debug != ""

if (is_debug) {
    # if we have DEBUG then we set not cran to true
    # CRAN is always release build
    is_not_cran <- TRUE
    message("Creating DEBUG build.")
}

if (!is_not_cran) {
    message("Building for CRAN.")
}

# A stale tarball stops CRAN-style builds, which must compile offline;
# development builds ignore it and download the locked crates instead.
if (vendor_exists && !vendor_current) {
    stale <- "src/rust/vendor.tar.xz was not made from the current src/rust/Cargo.lock"
    if (!is_not_cran) {
        stop(stale, "; delete it, or refresh it by running ",
             "Rscript -e 'source(\"tools/prepare-source.R\"); prepare_source(\".\")' ",
             "in r/connectivity", call. = FALSE)
    }
    message(stale, "; ignoring it and downloading the locked crates.")
}

# used to replace @VENDOR@
.vendor <- ifelse(vendor_current, "yes", "no")

# we set cran flags only if NOT_CRAN is empty and if
# the vendored crates are current.
.cran_flags <- ifelse(
    !is_not_cran && vendor_current,
    "-j 2 --offline",
    ""
)

# when DEBUG env var is present we use `--debug` build
.profile <- ifelse(is_debug, "", "--release")
.clean_targets <- ifelse(is_debug, "", "$(TARGET_DIR)")

# We specify this target when building for webR
webr_target <- "wasm32-unknown-emscripten"

# here we check if the platform we are building for is webr
is_wasm <- identical(R.version$platform, webr_target)

# print to terminal to inform we are building for webr
if (is_wasm) {
    message("Building for WebR")
}

# we check if we are making a debug build or not
# if so, the LIBDIR environment variable becomes:
# LIBDIR = $(TARGET_DIR)/{wasm32-unknown-emscripten}/debug
# this will be used to fill out the LIBDIR env var for Makevars.in
target_libpath <- if (is_wasm) "wasm32-unknown-emscripten" else NULL
cfg <- if (is_debug) "debug" else "release"

# used to replace @LIBDIR@
.libdir <- paste(c(target_libpath, cfg), collapse = "/")

# use this to replace @TARGET@
# we specify the target _only_ on webR
# there may be use cases later where this can be adapted or expanded
.target <- ifelse(is_wasm, paste0("--target=", webr_target), "")

# add panic exports only for WASM builds
.panic_exports <- ifelse(
    is_wasm,
    "CARGO_PROFILE_DEV_PANIC=\"abort\" CARGO_PROFILE_RELEASE_PANIC=\"abort\" ",
    ""
)

# read in the Makevars.in file checking
is_windows <- .Platform[["OS.type"]] == "windows"

# if windows we replace in the Makevars.win.in
mv_fp <- ifelse(
    is_windows,
    "src/Makevars.win.in",
    "src/Makevars.in"
)

# set the output file
mv_ofp <- ifelse(
    is_windows,
    "src/Makevars.win",
    "src/Makevars"
)

# delete the existing Makevars{.win/.wasm}
if (file.exists(mv_ofp)) {
    message("Cleaning previous `", mv_ofp, "`.")
    invisible(file.remove(mv_ofp))
}

# read as a single string
mv_txt <- readLines(mv_fp)

# replace placeholder values
new_txt <- gsub("@CRAN_FLAGS@", .cran_flags, mv_txt) |>
  gsub("@VENDOR@", .vendor, x = _) |>
  gsub("@PROFILE@", .profile, x = _) |>
  gsub("@CLEAN_TARGET@", .clean_targets, x = _) |>
  gsub("@LIBDIR@", .libdir, x = _) |>
  gsub("@TARGET@", .target, x = _) |>
  gsub("@PANIC_EXPORTS@", .panic_exports, x = _)

message("Writing `", mv_ofp, "`.")
con <- file(mv_ofp, open = "wb")
writeLines(new_txt, con, sep = "\n")
close(con)

message("`tools/config.R` has finished.")
