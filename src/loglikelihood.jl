import LinearAlgebra

function myloglikelihood(n, b, ρ, τ)

	prec = ρ .* (τ * length(n))
	out =
		-logpdf(InverseGamma(1, 1), τ) +
		-log(τ) +
		sum(n .* log.(prec)) +
		-0.5 * sum(prec .* b)
	return out
end

function multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
	pop_prec = 1 ./ (pop_sds .^2)
	return - n / 2 * (
		2 * sum(log, pop_sds) +
		length(pop_sds) * log(2 * float(pi)) +
		LinearAlgebra.dot(obs_var,  pop_prec) -
		2 * LinearAlgebra.dot(obs_mean, pop_prec .* pop_mean) +
		LinearAlgebra.dot(pop_mean, pop_prec .* pop_mean)
	)
end
