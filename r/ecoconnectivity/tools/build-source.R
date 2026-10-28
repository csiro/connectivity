# Run from any directory. Optional argument: output directory for the tarball.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
package <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
source(file.path(package, "tools", "prepare-source.R"))
prepare_source(package)
output <- commandArgs(trailingOnly = TRUE)
if (length(output) > 1L) stop("Expected at most one output directory")
if (length(output)) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  setwd(normalizePath(output, mustWork = TRUE))
}
status <- system2(file.path(R.home("bin"), "R"),
                  c("CMD", "build", "--no-manual", shQuote(package)))
if (status != 0L) stop("R source package build failed")
