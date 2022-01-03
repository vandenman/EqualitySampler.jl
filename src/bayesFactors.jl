
# function marginal_g_one_way(g, f_stat, n_obs_per_group, n_groups, rscale, return_log = false, log_const = 0.0)
# 	# adapted from https://github.com/richarddmorey/BayesFactor/blob/452624486c111436910061770bd1fa7ea6a69d62/pkg/BayesFactor/R/oneWayAOV-utility.R

# 	dfs = (n_groups - 1) / (n_obs_per_group * n_groups - n_groups)
# 	omega = (1 + (n_obs_per_group * g / (dfs * f_stat + 1))) / (n_obs_per_group * g + 1)
# 	m = log(rscale) - 0.5*log(2*pi) - 1.5*log(g) - rscale^2 / (2*g) - (n_groups - 1) / 2 * log(n_obs_per_group * g + 1) - (n_obs_per_group * n_groups - 1) / 2 * log(omega) - log_const

# 	return_log && return m
# 	return exp(m)

# end

# """
# adapted from:
# 	https://github.com/richarddmorey/BayesFactor/blob/452624486c111436910061770bd1fa7ea6a69d62/pkg/BayesFactor/R/oneWayAOV_Fstat.R
# 	https://github.com/richarddmorey/BayesFactor/blob/452624486c111436910061770bd1fa7ea6a69d62/pkg/BayesFactor/R/oneWayAOV-utility.R
# 	https://github.com/richarddmorey/BayesFactor/blob/452624486c111436910061770bd1fa7ea6a69d62/pkg/BayesFactor/R/common.R
# """
# function bayes_factor_one_way_anova(f_stat, n_obs_per_group, n_groups, rscale = 0.5)

# 	log_const = marginal_g_one_way(1.0, f_stat, n_obs_per_group, n_groups, rscale, true)
# 	integral, err = QuadGK.quadgk(g -> marginal_g_one_way(g, f_stat, n_obs_per_group, n_groups, rscale, false, log_const), 0, Inf, rtol=1e-8)
# 	return (log_bf = log(integral) + log_const, integral = integral, log_const = log_const, error = err)
# end
