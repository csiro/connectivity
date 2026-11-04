# Run from any directory. Optional argument: output directory for the tarball.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
package <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
source(file.path(package, "tools", "prepare-source.R"))
output <- commandArgs(trailingOnly = TRUE)
if (length(output) > 1L) stop("Expected at most one output directory")
# The vendored crates are needed only inside the built archive. Remove them
# afterwards so a later Cargo.lock change cannot leave a stale copy here.
vendored <- file.path(package, "src", "rust", c("vendor.tar.xz", "vendor-config.toml", "vendor.lock"))
status <- tryCatch({
    prepare_source(package)
    if (length(output)) {
        dir.create(output, recursive = TRUE, showWarnings = FALSE)
        setwd(normalizePath(output, mustWork = TRUE))
    }
    system2(file.path(R.home("bin"), "R"),
            c("CMD", "build", "--no-manual", shQuote(package)))
}, finally = unlink(vendored))
if (status != 0L) stop("R source package build failed")
