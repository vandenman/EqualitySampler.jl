abstract type AbstractJeffreysPrior <: Distributions.ContinuousUnivariateDistribution end

Distributions.rand(::Random.AbstractRNG, d::AbstractJeffreysPrior) 	= throw(error("rand is not implemented"))
Distributions.sampler(::AbstractJeffreysPrior) 						= throw(error("sampler is not implemented"))
Distributions.cdf(::AbstractJeffreysPrior, x::Real) 				= throw(error("cdf is not implemented"))
Distributions.quantile(::AbstractJeffreysPrior, x::Real) 			= throw(error("quantile is not implemented"))
Distributions.minimum(::AbstractJeffreysPrior) = 0.0
Distributions.maximum(::AbstractJeffreysPrior) = Inf

Distributions.insupport(::AbstractJeffreysPrior, x::Real)			= x > zero(x)
pdf(d::AbstractJeffreysPrior, x::Real) = exp(logpdf(d, x))

struct JeffreysPriorStandardDeviation <: AbstractJeffreysPrior end
logpdf(d::JeffreysPriorStandardDeviation, x::Real) = Distributions.insupport(d, x) ? -log(x) : -Inf

struct JeffreysPriorVariance <: AbstractJeffreysPrior end
logpdf(d::JeffreysPriorStandardDeviation, x::Real) = Distributions.insupport(d, x) ? -2log(x) : -Inf