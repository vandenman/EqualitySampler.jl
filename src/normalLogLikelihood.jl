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

function _univariate_normal_likelihood(obs_mean, obs_var, obs_n, pop_mean, pop_var)

	return -obs_n / 2.0 * (log(2pi) + log(pop_var)) - 1 / (2.0pop_var) * ((obs_n - 1) * obs_var + obs_n * (obs_mean - pop_mean)^2)
	# pop_prec = 1.0 / (pop_sds * pop_sds)
	# result =
	# 	log(2 * float(pi)) +
	# 	2 * log(pop_sds) +
	# 	obs_var * pop_prec +
	# 	(pop_mean - 2 * obs_mean) * pop_mean * pop_prec

	# return - n / 2 * result
end

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

struct NormalSuffStat{T<:Real, U<:Real} <: Distributions.ContinuousUnivariateDistribution
# struct NormalSuffStat{T<:Real, U<:Real} <: Distributions.Distribution
	obs_var::T
	pop_mean::U
	pop_var::U
	obs_n::Int
end
Distributions.logpdf(D::NormalSuffStat, obs_mean::T) where T<:Real = _univariate_normal_likelihood(obs_mean, D.obs_var, D.obs_n, D.pop_mean, D.pop_var)

# NormalSuffStat(::Float64,
# ::ForwardDiff.Dual{ForwardDiff.Tag{Turing.Core.var"#f#1"{TypedVarInfo{NamedTuple{(:σ², :μ_grand, :θ_r), Tuple{DynamicPPL.Metadata{Dict{VarName{:σ², Tuple{}}, Int64}, Vector{InverseGamma{Float64}}, Vector{VarName{:σ², Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{VarName{:μ_grand, Tuple{}}, Int64}, Vector{Normal{Float64}}, Vector{VarName{:μ_grand, Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{VarName{:θ_r, Tuple{}}, Int64}, Vector{DistributionsAD.TuringScalMvNormal{Vector{Float64}, Float64}}, Vector{VarName{:θ_r, Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}}}, Float64}, Model{var"#24#25", (:obs_mean, :obs_var, :obs_n, :Q, :T), (:T,), (), Tuple{Vector{Float64}, Vector{Float64}, Vector{Int64}, Matrix{Float64}, Type{Float64}}, Tuple{Type{Float64}}}, Sampler{NUTS{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.DiagEuclideanMetric}}, DefaultContext}, Float64}, Float64, 7},
# ::ForwardDiff.Dual{ForwardDiff.Tag{Turing.Core.var"#f#1"{TypedVarInfo{NamedTuple{(:σ², :μ_grand, :θ_r), Tuple{DynamicPPL.Metadata{Dict{VarName{:σ², Tuple{}}, Int64}, Vector{InverseGamma{Float64}}, Vector{VarName{:σ², Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{VarName{:μ_grand, Tuple{}}, Int64}, Vector{Normal{Float64}}, Vector{VarName{:μ_grand, Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{VarName{:θ_r, Tuple{}}, Int64}, Vector{DistributionsAD.TuringScalMvNormal{Vector{Float64}, Float64}}, Vector{VarName{:θ_r, Tuple{}}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}}}, Float64}, Model{var"#24#25", (:obs_mean, :obs_var, :obs_n, :Q, :T), (:T,), (), Tuple{Vector{Float64}, Vector{Float64}, Vector{Int64}, Matrix{Float64}, Type{Float64}}, Tuple{Type{Float64}}}, Sampler{NUTS{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.DiagEuclideanMetric}}, DefaultContext}, Float64}, Float64, 7},
# ::Int64)