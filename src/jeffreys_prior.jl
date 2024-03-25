abstract type AbstractJeffreysPrior <: Distributions.ContinuousUnivariateDistribution end

Distributions.rand(::Random.AbstractRNG, d::AbstractJeffreysPrior) 	= throw(error("rand is not implemented"))
Distributions.sampler(::AbstractJeffreysPrior) 						= throw(error("sampler is not implemented"))
Distributions.cdf(::AbstractJeffreysPrior, x::Real) 				= throw(error("cdf is not implemented"))
Distributions.quantile(::AbstractJeffreysPrior, x::Real) 			= throw(error("quantile is not implemented"))
Distributions.minimum(::AbstractJeffreysPrior) = 0.0
Distributions.maximum(::AbstractJeffreysPrior) = Inf

Distributions.insupport(::AbstractJeffreysPrior, x::Real)			= x > zero(x)

Distributions.pdf(d::AbstractJeffreysPrior, x::Real) = exp(Distributions.logpdf(d, x))

# required for compatability with Turing
# Bijectors.bijector(::AbstractJeffreysPrior) = Bijectors.Log{0}()
Bijectors.bijector(::AbstractJeffreysPrior) = Bijectors.elementwise(log)

struct JeffreysPriorStandardDeviation <: AbstractJeffreysPrior end
Distributions.logpdf(d::JeffreysPriorStandardDeviation, x::Real) = Distributions.insupport(d, x) ? -log(x) : -Inf

struct JeffreysPriorVariance <: AbstractJeffreysPrior end
Distributions.logpdf(d::JeffreysPriorVariance, x::Real) = Distributions.insupport(d, x) ? -2log(x) : -Inf

"""
	Following, https://en.wikipedia.org/wiki/Jeffreys_prior#Gaussian_distribution_with_standard_deviation_parameter
	we have that:

	Equivalently, the Jeffreys prior for log(σ) = ∫ dσ / σ is the unnormalized uniform distribution
	on the real line, and thus this distribution is also known as the logarithmic prior.

	Therefore, `rand(d::AbstractJeffreysPrior)` is implemented as a transformation of sampling from the real line.
	Sampling uniformly from the real line is done via `exp(rand(Turing.Flat()))`.
	Note that uniformly sampling from the entire real line is impossible and the validity of `rand(::JeffreysPriorXXX)` hinges on at the validity of `rand(Turing.Flat())`.
"""
Distributions.rand(rng::Random.AbstractRNG, ::JeffreysPriorStandardDeviation) = exp(rand(rng, Turing.Flat()))
Distributions.rand(rng::Random.AbstractRNG, ::JeffreysPriorVariance)          = exp(rand(rng, Turing.Flat()))^2