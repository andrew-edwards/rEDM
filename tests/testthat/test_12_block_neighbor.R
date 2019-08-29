context("Check nearest neighbors")

# Use test_05_simplex_calculations.R as a template. ts is from a simple
#  population model, and showed the problems. Using Deyle et al. (2013, PNAS,
#  110:6430-6435) Supporting Information notation. Time series has values X(t),
#  and the lagged space consists of vectors (for E=2) x(t) = (X(t), X(t-1)).
#  For target time t^*, we are trying to predict the X(t^* + 1) value.
#  By definition, X(t^* + 1) is included in x(t^* + 2) = (X(t^* + 2), X(t^* +
#  1)), and so we should not be allowed to use x(t^* + 2). This test will test
#  if x(t^* + 2) is correctly not allowed. For this time series, my (Andrew
#  Edwards) own independent code found that for t^* = 75 and t^* = 94, that
#  x(t^* + 2) was (incorrectly) allowed.
#  Answers from my manual code in 2017 are:
#   t^* = 75:
#    Data: X(76) = -0.4154818     = ts[76]
#    \hat{X}(76) = 0.8378772 from AME manual code
#    \hat{X}(76) = 1.367744  from rEDM (in 2017)
#   t^* = 94:
#    Data: X(95) = -0.7987376
#    \hat{X}(95) = 0.4123431 from AME manual code
#    \hat{X}(95) = 0.1773531 from rEDM (in 2017)
#    \hat{X}(95} = 0.1773531 from rEDM in 2019 (model_est) in this code with
#                    ts[95] known, from commenting out ts[tstar+1] <- NA
#    \hat{X}(95} = 0.4123431 from rEDM in 2019 (model_est) in this code with
#                    ts[95] <- NA, to ensure that it's not used at all
#    \hat{X}(95} = 0.1773531 from manual calculations here (est) with ts[95] known
#    \hat{X}(95} = 0.1800465 from manual calculations here with ts[95] not known
# So it looks like either (i) my concern from 2017 still exists
#                         (ii) my manual calculations are wrong.
# Adapt the manual calculations here to replicate mine.

testthat::test_that("Simplex (block_LNLP) does not use x(t^* + 2) = (X(t^* + 2, X(t^*  + 1)) as a nearest neighbor for E=2", {
    ts <- c(-0.056531409251883, 0.059223778257432, 5.24124928046977, -4.85399581474521,
            -0.46134818068973, 0.273317575696793, 0.801806230470337, -0.888891901824982,
            -0.202777622745051, 0.497565422757662, 5.10219324323769, -5.36826459373178,
            -0.17467165498718, 1.06545333399298, 1.97419279178678, -2.91448405223082,
            -0.179969867858605, 0.237962494942611, 1.47828327622468, -1.54267507064286,
            -0.180342027136338, 0.238919610831881, 1.06140368490958, -1.06522901782019,
            -0.214923527940395, 0.452847221462308, 2.13053391555372, -2.55145224744286,
            -0.0307653352327702, 1.1448014288826, -0.0675575239486375, -1.04711881585576,
            -0.00910890042051652, 0.726257323277433, 0.732271192186161, -1.35460378982395,
            -0.0322955446760023, 0.507606440290776, 3.73396587274012, -4.19686615950143,
            -0.0997201857962038, 0.753392632401029, 2.41347231553437, -3.03677401452137,
            -0.141112562089696, 0.446002103079665, 0.223768504955365, -0.615452831633047,
            -0.0216659723974975, 0.292246351104258, 0.20006105300258, -0.469596514211075,
            0.0422676544887819, 0.474264989176278, -0.0416811459395667, -0.53555712696719,
            0.118860281628173, 0.176335117268894, -0.10364820567334, -0.153572235117542,
            0.180339482186409, 0.0566876206447625, -0.140537892644139, 0.0252441742388871,
            0.340689505466622, 0.852833653689839, -1.07051231019616, -0.0937704380137284,
            0.460677118593916, 0.37444382348273, -0.83783628206217, -0.0154896108244113,
            1.34259279914848, -0.495978821807168, -0.472464634960208, -0.415481769949074,
            1.36767605087962, -0.891896943918948, -0.279228283931612, -0.148703043863421,
            2.04524590138255, -1.98431486665908, 0.0602356391036573, -0.0902730939678147,
            0.243344379963862, -0.074421904114315, -0.309150440565139, 0.43675531763949,
            0.178787692802827, 0.0799271040758849, -0.657946157906476, 1.14668210755046,
            -0.791665479471326, 0.482533897248175, -0.798737571552661, 0.439024256063545,
            0.177114631209318, 2.19942374686687, -2.9488856529422)

    tstar <- 94
    # ts[tstar+1] <- NA    # to ensure no use of the value we are trying to predict
    # construct lagged block
    lag_block <- cbind(c(ts[2:length(ts)], NA), ts, c(NA, ts[1:(length(ts) - 1)]))

    t <- c(2:(tstar-1), (tstar+1):99)

    # lib and pred portions
    lib_block <- cbind(t + 1, lag_block[t, ])
    pred_block <- cbind(tstar+1, lag_block[tstar, , drop = FALSE])

    block <- rbind(lib_block, pred_block)

    # make EDM forecast
    out <- block_lnlp(block,
                      lib = c(1, NROW(lib_block)),
                      pred = c(NROW(lib_block) + 1, NROW(lib_block) + 1),
                      first_column_time = TRUE,
                      tp = 0,
                      columns = c(2, 3), target_column = 1,
                      stats_only = FALSE, silent = TRUE)
    model_est <- out$model_output[[1]]$pred

    # manually calculate distances and neighbors (Hao's code)
    dist_mat <- as.matrix(dist(block[, 3:4]))
    dist_vec <- dist_mat[NROW(dist_mat), ]
    dist_vec[length(dist_vec)] <- NA
    nn <- order(dist_vec)[1:3] # 3 closest neighbors
    weights <- exp(-dist_vec[nn] / dist_vec[nn[1]])
    est <- sum(weights * block[nn, 2]) / sum(weights) # weighted average

    # adapt the above calculations to ignore x(t^*+2) as a nearest neighbour
    # block[(tstar-3):(tstar+2),]   # this gives (for t^*=94):
    #                            ts
    # [1,] 93 -0.7916655  1.1466821 -0.6579462
    # [2,] 94  0.4825339 -0.7916655  1.1466821
    # [3,] 96  0.4390243 -0.7987376  0.4825339
    # [4,] 97  0.1771146  0.4390243 -0.7987376
    # [5,] 98  2.1994237  0.1771146  0.4390243
    # [6,] 99 -2.9488857  2.1994237  0.1771146
    #  confirming that ts[95] = -0.7987376 appears in block due to the lagging,
    #   but that is the value we are trying to predict. So need to create
    #   block_adapt without those.
    block_adapt <- as.data.frame(block)
    names(block_adapt) <- c("t", "Xt", "Xt.min.1", "Xt.min.2")
    rows_to_exclude <- which(block_adapt$t %in% c(tstar+2, tstar+3))  # since contain
                                        # X(tstar+1), but keep X(tstar+1) as final line
                                        # since predicting it
    block_adapt <- block_adapt[-rows_to_exclude, ]

    dist_mat_adapt <- as.matrix(dist(block_adapt[, 3:4]))
    dist_vec_adapt <- dist_mat_adapt[NROW(dist_mat_adapt), ]  # distances from X(tstar+1)
    dist_vec_adapt[length(dist_vec_adapt)] <- NA     # equals 0 by definition
                                        # (distance from itself)
    nn_adapt <- order(dist_vec_adapt)[1:3] # 3 closest neighbors, since E=2
    weights_adapt <- exp(-dist_vec_adapt[nn_adapt] / dist_vec_adapt[nn_adapt[1]])
    est_adapt <- sum(weights_adapt * block_adapt[nn_adapt, 2]) / sum(weights_adapt) # weighted average

    testthat::expect_equal(model_est, est)
})
