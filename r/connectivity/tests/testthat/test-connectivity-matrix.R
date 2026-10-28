test_that("R reaches the frozen Rust and Python results without transposing cells", {
    cases <- list(
        list("circular", FALSE, FALSE, c(0.035036724, 0.060928639, 0.057472181, 0.023413222)),
        list("circular", FALSE, TRUE, c(0.031013189, 0.051940750, 0.050136760, 0.017623197)),
        list("circular", TRUE, FALSE, c(0.287769228, 0.330879152, 0.336351663, 0.270322204)),
        list("square", FALSE, FALSE, c(0.036365397, 0.062699854, 0.057887629, 0.023465034)),
        list("square", FALSE, TRUE, c(0.032288730, 0.053552385, 0.050358951, 0.017685590)),
        list("square", TRUE, FALSE, c(0.288273066, 0.334522188, 0.339534730, 0.269480884))
    )
    for (case in cases) {
        args <- multilevel(case[[1]], case[[2]], case[[3]])
        score <- do.call(connectivity_matrix, args)
        expect_identical(dim(score), c(13L, 14L))
        expect_true(all(abs(score[fixture_cells] - case[[4]]) <= 1e-6))
        expect_identical(is.na(score), args$mask)
    }
})

test_that("thread counts preserve complete results, including BERI", {
    for (beri in c(FALSE, TRUE)) {
        args <- multilevel(beri = beri)
        one <- do.call(connectivity_matrix, args)
        args$n_threads <- 3L
        expect_identical(do.call(connectivity_matrix, args), one)
    }
})

test_that("R missing values, dimnames and integer matrices are handled", {
    args <- multilevel()
    dimnames(args$condition) <- list(paste0("r", 1:13), paste0("c", 1:14))
    args$condition[2, 3] <- NaN
    score <- do.call(connectivity_matrix, args)
    expect_identical(dimnames(score), dimnames(args$condition))
    expect_true(is.na(score[2, 3]))
    score <- connectivity_matrix(matrix(1L, 4, 5), c(1000, 0, 0, 0, -1000, 0),
                                 levels = 1, n_threads = 1)
    expect_true(all(is.finite(score)))
    args <- multilevel()
    args$condition[,] <- NA_real_
    expect_true(all(is.na(do.call(connectivity_matrix, args))))
})

test_that("level 1 is inserted and optional traversal can be omitted", {
    args <- multilevel()
    args$traversal <- NULL
    expected <- do.call(connectivity_matrix, args)
    args$levels <- c(4, 2, 2)
    expect_identical(do.call(connectivity_matrix, args), expected)
})

test_that("invalid arguments fail before entering the core", {
    cases <- list(
        list(levels = c(1, 4)), list(levels = c(1, 3)),
        list(window_size = 4), list(outer_window = 1),
        list(lambdas = 0), list(lambdas = Inf), list(lambdas = numeric()),
        list(max_cost = 0), list(max_cost = NA_real_),
        list(n_threads = 0), list(n_threads = 1.5), list(offsets = c(-1, 0)),
        list(transform = rep(0, 6)), list(transform = 1:5),
        list(is_geo = NA), list(pa_to_pa = 1), list(window_mode = "hexagon"),
        list(condition = matrix(2, 13, 14)), list(condition = matrix(Inf, 13, 14)),
        list(mask = matrix(NA, 13, 14)), list(mask = matrix(FALSE, 2, 3)),
        list(traversal = matrix(0.5, 2, 3)), list(transgrids = list()),
        list(transgrids = list(array(0, c(13, 14, 2)), array(0, c(13, 14, 3)))),
        list(transgrids = list(array(0, c(2, 3, 2))))
    )
    for (change in cases) {
        args <- multilevel()
        args[names(change)] <- change
        expect_error(do.call(connectivity_matrix, args))
    }
    expect_error(do.call(connectivity_matrix, multilevel(beri = TRUE, pa_to_pa = TRUE)),
                 "pa_to_pa is only supported")
    args <- multilevel()
    args$lambdas <- 1e100
    expect_error(do.call(connectivity_matrix, args), "float32")
})
