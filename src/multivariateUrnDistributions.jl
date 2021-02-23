#=

#region AbstractMvUrnDistribution
abstract type AbstractMvUrnDistribution{T} <: Distributions.DiscreteMultivariateDistribution where T <: Integer end

Distributions.minimum(::AbstractMvUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractMvUrnDistribution{T}) where T = T(length(d))

Distributions.logpdf(::AbstractMvUrnDistribution, x::AbstractVector{T}) where T <: Real = -Inf
Distributions.pdf(d::AbstractMvUrnDistribution, x::AbstractVector{T}) where T <: Real = zero(T)

#endregion

#region UniformMvUrnDistribution
struct UniformMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
end

pdf(d::UniformMvUrnDistribution) = 1.0 / bellnumr(d.k, 0)
logpdf(d::UniformMvUrnDistribution) = - log(bellnumr(d.k, 0))

function _rand!(rng::AbstractRNG, s::Spl, x::AbstractVector{T}) where T<:Real



end

#endregion

#region BetaBinomialMvUrnDistribution
struct BetaBinomialMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
	α::Float64
	β::Float64
end

# pdf(d::BetaBinomialMvUrnDistribution) = 1.0 / bellnumr(d.k, 0)
# logpdf(d::BetaBinomialMvUrnDistribution) = - log(bellnumr(d.k, 0))

# function _rand!(rng::AbstractRNG, s::Spl, x::AbstractVector{T}) where T<:Real
# end

#endregion

=#