function myloglikelihood(n, b, ρ, τ)

	prec = ρ .* (τ * length(n))
	out =
		-Distributions.logpdf(Distributions.InverseGamma(1, 1), τ) +
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
	# efficient evaluation of log likelihood multivariate normal with a diagonal covariance matrix correlations given sufficient statistics
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

function _univariate_normal_likelihood(obs_mean, obs_var, obs_n, pop_mean, pop_var)

	# TODO: use obs_n and don't use corrected variance!
	return -obs_n / 2.0 * (log(2pi) + log(pop_var)) - 1 / (2.0pop_var) * ((obs_n - 1) * obs_var + obs_n * (obs_mean - pop_mean)^2)
	# pop_prec = 1.0 / (pop_sds * pop_sds)
	# result =
	# 	log(2 * float(pi)) +
	# 	2 * log(pop_sds) +
	# 	obs_var * pop_prec +
	# 	(pop_mean - 2 * obs_mean) * pop_mean * pop_prec

	# return - n / 2 * result
end

# maybe this is faster for AD than explicitly looping over _univariate_normal_likelihood
# function _multivariate_normal_diagonal_covariance_likelihood(obs_mean, obs_var, obs_n, pop_mean, pop_var)
# 	# untested!
# 	# pop_var should be a number, the others arrays of number
# 	(-obs_n+length(obs_mean)) / 2.0 * (log(2pi) + log(pop_var)) +
# 		- 1 / (2.0pop_var) * (
# 			LA.dot((obs_n .- 1), obs_var) + LA.dot(obs_n, (obs_mean .- pop_mean)^2)
# 		)
# end

# MvNormal Distribution parametrized with sufficient statistics for a diagonal covariance matrix
struct MvNormalSuffStat{T<:AbstractVector{<:Real}, U<:AbstractVector{<:Real}} <: Distributions.AbstractMvNormal
# struct MvNormalSuffStat{T<:AbstractVector{<:Real}, U<:AbstractVector{<:Real}} <: Distributions.Distribution{F, S}
	obs_var::T
	pop_mean::U
	pop_var::U
	obs_n::Int
end
Distributions.logpdf(D::MvNormalSuffStat, obs_mean::AbstractVector) = _multivariate_normal_likelihood(obs_mean, D.obs_var, D.obs_n, D.pop_mean, D.pop_var)
# this method isn't necessary for observe statements in Turing
# Distributions.rand(rng::Random.AbstractRNG, D::MvNormalSuffStat) = rand(rng, MvNormal(D.pop_mean, D.pop_var ./ D.n))

struct NormalSuffStat{T<:Real, U<:Real, V<:Real} <: Distributions.ContinuousUnivariateDistribution
# struct NormalSuffStat{T<:Real, U<:Real} <: Distributions.Distribution
	obs_var::T
	pop_mean::U
	pop_var::V
	obs_n::Int
end
Distributions.logpdf(D::NormalSuffStat, obs_mean::T) where T<:Real = _univariate_normal_likelihood(obs_mean, D.obs_var, D.obs_n, D.pop_mean, D.pop_var)

struct MvNormalDenseSuffStat{T<:AbstractMatrix{<:Real}, F<:Real, U<:AbstractVector{F}, W<:AbstractMatrix{F}} <: Distributions.AbstractMvNormal
	obs_cov::T
	obs_n::Int
	pop_mean::U
	pop_cov::W
end

Distributions.logpdf(D::MvNormalDenseSuffStat, obs_mean::AbstractVector) = begin
	return logpdf_mv_normal_suffstat(obs_mean, D.obs_cov, D.obs_n, D.pop_mean, D.pop_cov)
end

function logpdf_mv_normal_suffstat(x̄, S, n, μ, Σ)
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			LinearAlgebra.logdet(Σ) +
			(x̄ .- μ)' / Σ * (x̄ .- μ) +
			LinearAlgebra.tr(Σ \ S)
		)
	)
end

struct MvNormalCholDenseSuffStat{T<:Real, U<:Real, W<:AbstractVector{U}} <: Distributions.AbstractMvNormal
	obs_cov_chol::LinearAlgebra.UpperTriangular{T, Matrix{T}}
	obs_n::Int
	pop_mean::W
	pop_cov_chol::LinearAlgebra.UpperTriangular{U, Matrix{U}}
end

Distributions.logpdf(D::MvNormalCholDenseSuffStat, obs_mean::AbstractVector) = begin
	return logpdf_mv_normal_chol_suffstat(obs_mean, D.obs_cov_chol, D.obs_n, D.pop_mean, D.pop_cov_chol)
end


function logpdf_mv_normal_chol_suffstat(x̄, S_chol::LinearAlgebra.UpperTriangular, n, μ, Σ_chol::LinearAlgebra.UpperTriangular)
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			2 * sum(i->log(@inbounds Σ_chol[i, i]), axes(Σ_chol, 1)) +
			# 2LinearAlgebra.logdet(Σ_chol) +
			sum(x->x^2, (x̄ .- μ)' / Σ_chol) +
			sum(x->x^2, S_chol / Σ_chol)
		)
	)
end


function logpdf_mv_normal_precision_chol_suffstat(x̄, S_chol::LinearAlgebra.UpperTriangular, n, μ, Ω_chol::LinearAlgebra.UpperTriangular)
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			-2 * sum(i->log(Ω_chol[i, i]), axes(Ω_chol, 1)) +
			sum(x->x^2, (x̄ .- μ)' * Ω_chol) +
			sum(x->x^2, S_chol * Ω_chol)
		)
	)
end

"""
get_normal_dense_suff_stats(x::AbstractMatrix{<:Real})

Returns the sufficients statistics for a multivariate normal with dense covariance matrix.
Note that the sample covariance is the "biased" version (divided by n, instead of n - 1).
"""
function get_normal_dense_suff_stats(x::AbstractMatrix{<:Real})
	n = size(x, 2)
	obs_mean = vec(Statistics.mean(x, dims = 2))
	# we want the "biased" sample covariance
	obs_cov = Statistics.cov(x') .* ((n - 1) / n)
	return obs_mean, obs_cov, n
end

"""
get_normal_dense_chol_suff_stats(x::AbstractMatrix{<:Real})

Returns the sufficients statistics for a multivariate normal with dense covariance matrix.
Note that the cholesky of the sample covariance is the "biased" version (divided by n, instead of n - 1).
"""
function get_normal_dense_chol_suff_stats(x::AbstractMatrix{<:Real})
	obs_mean, obs_cov, n = get_normal_dense_suff_stats(x)
	obs_cov_chol = LinearAlgebra.cholesky(obs_cov).U
	return obs_mean, obs_cov_chol, n
end
