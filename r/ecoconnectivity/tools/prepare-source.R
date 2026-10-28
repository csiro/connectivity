# Development tool: create the standalone sources consumed by R CMD build.
# Only writes generated files inside the R package; never edits the core or
# either manifest/lock file of the existing Python/core workspace.
prepare_source <- function(package = "r/ecoconnectivity") {
  package <- normalizePath(package, mustWork = TRUE)
  repository <- normalizePath(file.path(package, "..", ".."), mustWork = TRUE)
  core <- file.path(repository, "crates", "connectivity-core")
  rust <- file.path(package, "src", "rust")
  if (!file.exists(file.path(core, "Cargo.toml"))) {
    stop("Run preparation from a checkout containing crates/connectivity-core")
  }
  if (!requireNamespace("rextendr", quietly = TRUE)) {
    stop("Install the development dependency rextendr before preparing sources")
  }

  # Resolve the core's inherited dependency declarations into an ordinary,
  # standalone Cargo manifest. Fail rather than silently mishandle a new form.
  workspace <- readLines(file.path(repository, "Cargo.toml"))
  first <- match("[workspace.dependencies]", trimws(workspace))
  if (is.na(first)) stop("Cannot find the core's workspace dependency table")
  dependencies <- workspace[seq.int(first + 1L, length(workspace))]
  next_table <- which(grepl("^\\s*\\[", dependencies))
  if (length(next_table)) dependencies <- head(dependencies, next_table[1] - 1L)
  manifest <- readLines(file.path(core, "Cargo.toml"))
  inherited <- grepl("workspace\\s*=\\s*true", manifest)
  for (line in which(inherited)) {
    name <- sub("\\s*=.*$", "", trimws(manifest[line]))
    expected <- paste0("^", name, "\\s*=\\s*\\{\\s*workspace\\s*=\\s*true\\s*\\}\\s*$")
    replacement <- dependencies[grepl(paste0("^", name, "\\s*="), dependencies)]
    if (!grepl(expected, manifest[line]) || length(replacement) != 1L) {
      stop("Unsupported inherited dependency declaration: ", manifest[line])
    }
    manifest[line] <- replacement
  }
  # Development-only tests of the core stay in its original workspace.
  dev <- match("[dev-dependencies]", trimws(manifest))
  if (!is.na(dev)) {
    after <- which(seq_along(manifest) > dev & grepl("^\\s*\\[", manifest))
    last <- if (length(after)) after[1] - 1L else length(manifest)
    manifest <- manifest[-seq.int(dev, last)]
  }
  destination <- file.path(rust, "connectivity-core")
  unlink(destination, recursive = TRUE)
  dir.create(destination)
  stopifnot(file.copy(file.path(core, c("src", "README.md")), destination, recursive = TRUE))
  stopifnot(file.copy(file.path(repository, "LICENSE"), destination))
  writeLines(manifest, file.path(destination, "Cargo.toml"))

  # Start with the current engine dependency versions, then retain this binding's
  # independent lock on subsequent preparations. Registry dependencies are not
  # upgraded by --workspace; changed constraints must still be reviewed.
  lock <- file.path(rust, "Cargo.lock")
  if (!file.exists(lock)) stopifnot(file.copy(file.path(repository, "Cargo.lock"), lock))
  status <- system2("cargo", c("update", "--workspace", "--manifest-path",
                              shQuote(file.path(rust, "Cargo.toml"))))
  if (status != 0L) stop("Could not resolve the R binding's Cargo dependencies")
  rextendr::vendor_crates(package, clean = TRUE)
  message("Prepared a core snapshot and vendored dependencies for standalone R installation.")
  invisible(package)
}
