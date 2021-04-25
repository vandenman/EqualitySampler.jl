
@testset "compare with BayesFactor package" begin

	#=

		getRversion()
		# [1] ‘4.0.5’
		packageVersion("BayesFactor")
		# [1] ‘0.9.12.4.2’
		ncomparisons <- 50

		f_values      <-       seq(from = 0.2, to = 5,   length.out = ncomparisons)
		n_values      <- floor(seq(from = 5,   to = 500, length.out = ncomparisons))
		j_values      <- floor(seq(from = 5,   to = 12,  length.out = ncomparisons))
		rscale_values <-       seq(from = 0.2, to = 5,   length.out = ncomparisons)

		set.seed(42)
		bayes_factors <- unlist(Map(function(...) BayesFactor:::oneWayAOV.Fstat(...)$bf,
			F      = f_values,
			N      = n_values,
			J      = j_values,
			rscale = rscale_values
		))
		julia_expr <- capture.output(dput(bayes_factors))
		julia_expr[1] <- substring(julia_expr[1], 2)
		substr(julia_expr[1], 1, 1) <- "["
		len <- length(julia_expr)
		substring(julia_expr[len], nchar(julia_expr[len])) <- "]"
		print(paste(julia_expr, collapse = ""))

	=#

	ncomparisons  = 50
	f_values      =        range(0.2, stop = 5,   length = ncomparisons)
	n_values      = floor.(range(5,   stop = 500, length = ncomparisons))
	j_values      = floor.(range(5,   stop = 12,  length = ncomparisons))
	rscale_values =        range(0.2, stop = 5,   length = ncomparisons)

	replication = [
		bayes_factor_one_way_anova(f_stat, n_obs_per_group, n_groups, rscale)[:log_bf]
		for (f_stat, n_obs_per_group, n_groups, rscale) in zip(f_values, n_values, j_values, rscale_values)
	]

	# obtained from R code above
	expected = [-0.706030671019979, -1.62959342780057, -2.4799680965703, -3.25221442063222, -3.94080612321899, -4.54866081369211, -5.08282121362426, -6.52056092523155, -7.02271776214815, -7.46589743766432, -7.87926743256166, -8.22361953930493, -8.52816339966176, -8.79751191668703, -10.3788760944123, -10.6285659492446, -10.8487262038146, -11.0422505452206, -11.2116528699508, -11.359125615427, -11.5007823855227, -13.0682469925063, -13.1743708851499, -13.2625730645949, -13.3341999638837, -13.3904559907163, -13.4324221600269, -13.4610718936853, -14.9213385900277, -14.9261471031495, -14.9313077627387, -14.9116029313324, -14.8809210223282, -14.839886081655, -14.7890711544711, -16.0857630543688, -16.0080980681435, -15.9210972501016, -15.8252264739729, -15.7209179668227, -15.6194036629214, -15.4991375964277, -16.5931088259111, -16.4434782298928, -16.2864442888214, -16.1223221456341, -15.9514073971233, -15.7739776633015, -15.5902940044255, -16.4635769424055]

	@test isapprox(replication, expected, atol = 1e-7)
	# maximum(abs, replication - expected) is about 1.5e-8

end