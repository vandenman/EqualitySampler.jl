"""
```
BetaBinomialPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}
```

Beta binomial distribution over partitions.
If ``\\rho \\sim \\text{BetaBinomialPartitionDistribution}(k, \\alpha, \\beta)`` then ``\\text{count\\_parameters}(\\rho) \\sim \\text{BetaBinomial}(k - 1, \\alpha, \\beta)``.

"""
struct BetaBinomialPartitionDistribution{T <: Integer} <: AbstractSizePartitionDistribution{T}
	k::T
	α::Float64
	β::Float64
	function BetaBinomialPartitionDistribution(k::T, α::Float64 = 1.0, β::Float64 = 1.0) where T<:Integer
		new{T}(k, α, β)
	end
end

function BetaBinomialPartitionDistribution(k::Integer, α::Number, β::Number = 1.0; check_args::Bool = true)
	Distributions.@check_args BetaBinomialPartitionDistribution (k, k > zero(k)) (α, α > zero(α)) (β, β > zero(β))
	BetaBinomialPartitionDistribution(k, convert(Float64, α), convert(Float64, β))
end

function logpdf_incl_no_check(d::BetaBinomialPartitionDistribution{T}, no_parameters::Integer) where T
    Distributions.logpdf(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), no_parameters - one(T))
end







# logpdf_model_distinct(d::BetaBinomialPartitionDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))


# function logpdf_incl(d::BetaBinomialPartitionDistribution, no_parameters::Integer)
# 	in_eqsupport(d, no_parameters) || return -Inf
#     Distributions.logpdf(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), no_parameters - one(T))
# end

