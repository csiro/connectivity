# Development tool: vendor published Rust dependencies consumed by R CMD build.
# Only writes generated files inside the R package; never edits the Python/core
# workspace or resolves dependencies from its source tree.
prepare_source <- function(package = "r/connectivity") {
    package <- normalizePath(package, mustWork = TRUE)
    repository <- normalizePath(file.path(package, "..", ".."), mustWork = TRUE)
    rust <- file.path(package, "src", "rust")
    if (!requireNamespace("rextendr", quietly = TRUE)) {
        stop("Install the development dependency rextendr before preparing sources")
    }
    license <- file.path(repository, "LICENSE")
    if (file.exists(license)) {
        stopifnot(file.copy(license, file.path(package, "LICENSE.md"), overwrite = TRUE))
    }

    # Remove snapshots left by earlier development builds.
    unlink(file.path(rust, "connectivity-core"), recursive = TRUE)

    # cargo vendor --locked uses this binding's lockfile, including the published
    # connectivity-core release, and fails if dependency changes need a lock update.
    rextendr::vendor_crates(package, clean = TRUE)
    message("Vendored published Rust dependencies for standalone R installation.")
    return(invisible(package))
}
