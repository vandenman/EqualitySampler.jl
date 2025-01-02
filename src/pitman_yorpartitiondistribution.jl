"""
```
DirichletProcessPartitionDistribution(k::Integer, α::Float64)
DirichletProcessPartitionDistribution(k::Integer, s::Symbol)
DirichletProcessPartitionDistribution(k::Integer, α::Real)
```

Dirichlet process distribution over partitions.
Either set α directly by passing a float, or pass a symbol for a prespecified option.
`:Gopalan_Berry` uses `EqualitySampler.dpp_find_α` to specify α, which uses the heuristic by
Gopalan & Berry (1998) so that P(everything equal) == P(everything unequal).
`harmonic` sets `α` to the inverse of the kth harmonic number, which implies that the prior over partitions is decreasing.
"""
struct DirichletProcessPartitionDistribution{T<:Integer, U<:Number} <: AbstractProcessPartitionDistribution{T}
	k::T
	α::U
	DirichletProcessPartitionDistribution{T, U}(k::T, α::U) where {T<:Integer, U<:Number} = new{T, U}(k, α)
end
function DirichletProcessPartitionDistribution(k::T, s::Symbol) where T<:Integer
	U = float(T)
	if s === :Gopalan_Berry
		α = U(dpp_find_α(k))
	elseif s === :harmonic
		α = isone(k) ? U(one(k)) : U(inv(sum(inv, 1:k)))
	else
		throw(DomainError(s, "Symbol should be either Gopalan_Berry or :harmonic."))
	end
	DirichletProcessPartitionDistribution(k, α)
end

function DirichletProcessPartitionDistribution(k::T, α::U; check_args::Bool = true) where {T<:Integer, U<:Number}
	Distributions.@check_args DirichletProcessPartitionDistribution (k, k > zero(k)) (α, α > zero(α))
	DirichletProcessPartitionDistribution{T, U}(k, α)
end

struct PitmanYorProcessPartitionDistribution{T<:Integer, U<:Number} <: AbstractProcessPartitionDistribution{T}
	k::T
	d::U
	θ::U
	PitmanYorProcessPartitionDistribution{T, U}(k::T, d::U, θ::U) where {T<:Integer, U<:Number} = new{T, U}(k, d, θ)
end

function PitmanYorProcessPartitionDistribution(k::T, d::U, θ::U; check_args::Bool = true) where {T<:Integer, U<:Number}
	Distributions.@check_args PitmanYorProcessPartitionDistribution (k, k > zero(k)) (d, zero(d) <= d < one(d)) (θ, θ > -d)
	PitmanYorProcessPartitionDistribution{T, U}(k, d, θ)
end
PitmanYorProcessPartitionDistribution(k::Integer, d::Number, θ::Number) = PitmanYorProcessPartitionDistribution(k, promote(d, θ)...)

_pdf_helper!(probvec, d::PitmanYorProcessPartitionDistribution, i, x, partition_sizes) = _pdf_helper_py_dp!(probvec, d.d,       d.θ, i, x, partition_sizes)
_pdf_helper!(probvec, d::DirichletProcessPartitionDistribution, i, x, partition_sizes) = _pdf_helper_py_dp!(probvec, zero(d.α), d.α, i, x, partition_sizes)

function _pdf_helper_py_dp!(probvec, d, θ, index, x, partition_sizes)

	isone(index) && return fill_uniform!(probvec)
	k = length(probvec)

	v_known_urns = view(x, 1:index-1)

	tb = fast_countmap_partition_incl_zero!(partition_sizes, v_known_urns)
	mi = sum(!iszero, tb)
	no_zeros = k - mi

	den = (θ + index - 1)
	prob_new_label = (θ + d * mi) / den
	# @show tb, prob_new_label, den, k, v_known_urns
	for i in eachindex(probvec)
		if iszero(tb[i])
			probvec[i] = prob_new_label / no_zeros
		else
			probvec[i] = (tb[i] - d) / den
		end
	end

	return probvec

end

function pdf_model_distinct(d::DirichletProcessPartitionDistribution, partition::AbstractVector)

	Distributions.insupport(d, partition) || return zero(d.α)

	α = d.α
	partition_sizes = fast_countmap_partition_incl_zero(partition)
	t = sum(!iszero, partition_sizes)
	n = length(partition)
	α^t * SpecialFunctions.gamma(α) / SpecialFunctions.gamma(α + n) * prod(SpecialFunctions.gamma(s) for s in partition_sizes if !iszero(s))
end

function logpdf_model_distinct(d::DirichletProcessPartitionDistribution{T}, partition::AbstractVector) where T
	U = float(T)
	Distributions.insupport(d, partition) || return U(-Inf)

	α = d.α
	partition_sizes = fast_countmap_partition_incl_zero(partition)
	t = sum(!iszero, partition_sizes)
	n = length(partition)
	t * log(α) + SpecialFunctions.loggamma(α) - SpecialFunctions.loggamma(α + n) + sum(SpecialFunctions.loggamma(s) for s in partition_sizes if !iszero(s))
end

function logpdf_incl(d::DirichletProcessPartitionDistribution, no_parameters::Integer)
	n = length(d)
	α = d.α
	k = no_parameters # number of unique values

	logunsignedstirlings1(n, k) +
		k * log(α) +
		SpecialFunctions.loggamma(α) -
		SpecialFunctions.loggamma(α + n)
end

expected_inclusion_probabilities(d::AbstractProcessPartitionDistribution) = pdf_incl.(Ref(d), 1:length(d))

"""
Computes (x)_{n↑δ}
"""
py_helper(x, n, δ) = prod(i-> x + i * δ, 0:n-1; init = one(x))
log_py_helper(x, n, δ) = sum(i-> log(x + i * δ), 0:n-1; init = zero(x))
function foo_py(d, θ, partition)
	partition_sizes = fast_countmap_partition_incl_zero(partition)
	t = sum(!iszero, partition_sizes)
	n = length(partition)
	py_helper(d + θ, t - 1, d) / (py_helper(1 + θ, n - 1, 1)) * prod(
		py_helper(1 - d, s - 1, 1) for s in partition_sizes if !iszero(s)
	)
end

function logpdf_model_distinct(d::PitmanYorProcessPartitionDistribution, partition::AbstractVector)
	log(foo_py(d.d, d.θ, partition))
end

function logpdf_incl(d::PitmanYorProcessPartitionDistribution, no_parameters::Integer)
	n = length(d)
	k = no_parameters # number of unique values
	d, θ = d.d, d.θ

	# TODO: it's possible to implement this on a log scale, but it's not clear if it's worth it right now
    U = promote_type(typeof(d), typeof(θ))
	log(
		py_helper(d + θ, k - 1, d) / (py_helper(1 + θ, n - 1, 1)) *
        U(generalized_stirling_number(big(-one(d)), big(-d), big(n), big(k)))
		# generalized_stirling_number(-one(d), -d, n, k)
	)
end
