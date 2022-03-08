
"""
dpp_find_α(k::Integer)

Computes α such that the probability of a model where all variables are the same is equal to that of a model where all variables are distinct.
This does not consider duplicte models, i.e., it minimizes (EqualitySampler.logpdf_model_distinct(D, null_model) - EqualitySampler.logpdf_model_distinct(D, full_model))^2 as a function of α.
"""
function dpp_find_α(k::Integer)

	return exp(-(1 / (1 - k)) * logabsgamma(k))

	# brute force version
	# null_model = fill(1, k)
	# full_model = collect(1:k)

	# function target(α)
	# 	dpp = RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(exp(first(α))))
	# 	(
	# 		EqualitySampler.logpdf_model_distinct(dpp, full_model) -
	# 		EqualitySampler.logpdf_model_distinct(dpp, null_model)
	# 	)^2
	# end

	# result = Optim.optimize(target, [log(1.817)], Optim.BFGS())

	# return exp(first(result.minimizer))#, result

end
