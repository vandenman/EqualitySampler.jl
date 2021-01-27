function myloglikelihood(n, b, ρ, τ)

	prec = ρ .* (τ * length(n))
	out =
		-logpdf(Distributions.InverseGamma(1, 1), τ) +
		-log(τ) +
		sum(n .* log.(prec)) +
		-0.5 * sum(prec .* b)
	return out
end

# function __multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
# 	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
# 	pop_prec = 1 ./ (pop_sds .^2)
# 	return - n / 2 * (
# 		2 * sum(log, pop_sds) +
# 		length(pop_sds) * log(2 * float(pi)) +
# 		LinearAlgebra.dot(obs_var,  pop_prec) -
# 		2 * LinearAlgebra.dot(obs_mean, pop_prec .* pop_mean) +
# 		LinearAlgebra.dot(pop_mean, pop_prec .* pop_mean)
# 	)
# end

function _multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
	# unlike the commented version above, this one doesn't depend on LinearAlgebra
	result = length(obs_mean) * log(2 * float(pi))
	for i in eachindex(obs_mean)
		pop_prec_i = 1.0 / pop_sds[i] / pop_sds[i]
		result +=
			2 * log(pop_sds[i]) +
			obs_var[i] * pop_prec_i +
			(pop_mean[i] - 2 * obs_mean[i]) * pop_mean[i] * pop_prec_i

	end
	return - n / 2 * result
end

function _univariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)

	pop_prec = 1.0 / (pop_sds * pop_sds)
	result = 
		log(2 * float(pi)) +
		2 * log(pop_sds) +
		obs_var * pop_prec +
		(pop_mean - 2 * obs_mean) * pop_mean * pop_prec

	return - n / 2 * result
end

# MvNormal Distribution parametrized with sufficient statistics for a diagonal covariance matrix
struct MvNormalSuffStat <: Distributions.AbstractMvNormal
	obs_var::AbstractVector{<:Real}
	pop_mean::AbstractVector{<:Real}
	pop_var::AbstractVector{<:Real}
	n::Int
end
Distributions.logpdf(D::MvNormalSuffStat, obs_mean::AbstractVector) = _multivariate_normal_likelihood(obs_mean, D.obs_var, D.pop_mean, D.pop_var, D.n)
# this method isn't necessary for observe statements in Turing
# Distributions.rand(rng::Random.AbstractRNG, D::MvNormalSuffStat) = rand(rng, MvNormal(D.pop_mean, D.pop_var ./ D.n))

struct NormalSuffStat <: Distributions.ContinuousUnivariateDistribution
	obs_var::Real
	pop_mean::Real
	pop_var::Real
	n::Int
end
Distributions.logpdf(D::NormalSuffStat, obs_mean::T) where T<:Real = _univariate_normal_likelihood(obs_mean, D.obs_var, D.pop_mean, D.pop_var, D.n)


