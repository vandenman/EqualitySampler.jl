"""
	reduce a model to a unique representation. For example, [2, 2, 2] -> [1, 1, 1]
"""
function reduce_model(x::AbstractVector{T}) where T <: Integer

	y = copy(x)
	for i in eachindex(x)
		if !any(==(x[i]), x[1:i - 1])
			if x[i] != i
				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)
				y[idx] .= i
			end
		end
	end
	return y
end

function get_conditional_counts(n::Int, known::AbstractVector{T} = [1], include_new::Bool = true) where T <: Integer

	# TODO: just pass d.partitions and d.index to this function! (or just d itself)
	# then we can get rid of the include_new and use dispatch, although it might duplicate some code
	# or maybe not?
	known = reduce_model(known) # TODO: I don't think this is necessary!
	refinement = length(unique(known))
	n_known = length(known)

	res = zeros(Int, n_known + (include_new ? 1 : 0))
	idx = findall(i->known[i] == i, 1:n_known)
	n_idx = length(idx)
	res[idx] .= bellnumr(n - n_known - 1, n_idx)
	if include_new
		res[n_known + 1] = bellnumr(n - n_known - 1, n_idx + 1)
	end
	res
end

count_equalities(urns::AbstractVector{T}) where T <: Integer = length(urns) - length(Set(urns))
count_equalities(urns::AbstractString) = length(urns) - length(Set(urns))

#region AbstractConditionalUrnDistribution
abstract type AbstractConditionalUrnDistribution{T} <: Distributions.DiscreteUnivariateDistribution where T <: Integer end

Distributions.minimum(::AbstractConditionalUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractConditionalUrnDistribution{T}) where T = T(length(d))
Base.length(d::AbstractConditionalUrnDistribution) = length(d.urns)

Distributions.logpdf(::AbstractConditionalUrnDistribution, x::Real) = -Inf
Distributions.pdf(d::AbstractConditionalUrnDistribution{T}, x::Real) where T = exp(logpdf(d, x))
function Distributions.logpdf(d::AbstractConditionalUrnDistribution{T}, x::T) where T <: Integer
	Distributions.insupport(d, x) || return -Inf
	return log(_pdf(d)[x])
end

Distributions.cdf(d::AbstractConditionalUrnDistribution, x::Real) = 0.0
function Distributions.cdf(d::AbstractConditionalUrnDistribution, x::T) where T <: Integer
	cdf = cumsum(_pdf(d))
	return cumsum[x]
end

# TODO: this could be done more efficiently
Distributions.rand(::Random.AbstractRNG, d::AbstractConditionalUrnDistribution) = rand(Distributions.Categorical(_pdf(d)), 1)[1]

count_distinct_models(d::AbstractConditionalUrnDistribution) = count_distinct_models(length(d))

outcomes(d::AbstractConditionalUrnDistribution) = eachindex(d.urns)
Distributions.mean(d::AbstractConditionalUrnDistribution) = sum(_pdf(d) .* outcomes(d))
Distributions.var(d::AbstractConditionalUrnDistribution) = sum(_pdf(d) .* outcomes(d) .^2) - Distributions.mean(d)^2

#endregion 

#region UniformConditionalUrnDistribution
struct UniformConditionalUrnDistribution{T<:Integer} <: AbstractConditionalUrnDistribution{T}
	urns::AbstractVector{T}
	index::T
	function UniformConditionalUrnDistribution(urns::AbstractVector{T}, index::T = 1) where T <: Integer
		n = length(urns)
		all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
		one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
		return new{T}(urns, index)
	end
end


function _pdf(d::UniformConditionalUrnDistribution)

	index, k = d.index, length(d.urns)
	isone(index) && return fill(1 / k, k)
	urns = view(d.urns, 1:index - 1)

	count = get_conditional_counts(k, urns)
	result = zeros(Float64, k)
	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
	other = setdiff(1:k, urns)
	result[other] .= count[length(urns) + 1] ./ length(other)
	return result ./ sum(result)

end

#endregion
#region BetaBinomialConditionalUrnDistribution
struct BetaBinomialConditionalUrnDistribution{T<:Integer} <: AbstractConditionalUrnDistribution{T}
	urns::AbstractVector{T}
	index::T
	α::Float64
	β::Float64
	# logpdf::Vector{Float64}
	function BetaBinomialConditionalUrnDistribution(urns::AbstractVector{T}, index::T = 1, α::Float64 = 1.0, β::Float64 = 1.0) where T <: Integer
		n = length(urns)
		all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
		one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
		0.0 <= α || throw(DomainError(α, "condition: 0 <= α is violated"))
		0.0 <= β || throw(DomainError(β, "condition: 0 <= β is violated"))
		new{T}(urns, index, α, β)
	end
end

function BetaBinomialConditionalUrnDistribution(urns::AbstractVector{T}, index::T, α::Real, β::Real) where T <: Integer
	BetaBinomialConditionalUrnDistribution(urns, index, convert(Float64, α), convert(Float64, β))
end

function BetaBinomialConditionalUrnDistribution(k::T, α::Real = 1.0, β::Real = 1.0) where T <: Integer
	BetaBinomialConditionalUrnDistribution(T[one(T)], k, convert(Float64, α), convert(Float64, β))
end

function _pdf(d::BetaBinomialConditionalUrnDistribution)

	index, k = d.index, length(d.urns)
	isone(index) && return fill(1 / k, k)

	urns = d.urns
	index_already_sampled = 1:index - 1
	n0 = k
	
	# no_duplicated = count_equalities(view(urns, index_already_sampled))
	v_known_urns = view(urns, index_already_sampled)
	r = length(Set(v_known_urns))
	n = n0 - (index - r - 1)

	# TODO: this could also be stored inside d...
	model_probs_by_incl = exp.(Distributions.logpdf.(Distributions.BetaBinomial(n0 - 1, d.α, d.β), 0:n0 - 1) .- log.(expected_inclusion_counts(n0)))

	num = sum(model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    ) for k in 1:n0)
	den = sum(model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1) for k in 1:n0)
	probEquality = r*num / (r*num + den)

	probs = Vector{Float64}(undef, n0)

	# probability of an equality
	known = reduce_model(v_known_urns)
	counts = get_conditional_counts(k, known, false)
	idx_nonzero = findall(!iszero, counts)
	probs[v_known_urns[idx_nonzero]] .= probEquality .* (counts[idx_nonzero] ./ sum(counts[idx_nonzero]))

	# probability of an inequality
	inequality_options = setdiff(1:n0, v_known_urns)
	probs[inequality_options] .= (1 - probEquality) ./ length(inequality_options)

	if !Distributions.isprobvec(probs)
		@show d, probs
	end
	return probs

end
#endregion

#region expected model + inclusion probabilities
function expected_model_probabilities(k::Int)
	x = count_distinct_models(k)
	return fill(1 / x, x)
end
expected_inclusion_counts(k::Integer) = stirlings2.(k, k:-1:1)
function expected_inclusion_probabilities(k::Integer)
	counts = expected_inclusion_counts(k)
	return counts ./ sum(counts)
end

expected_model_probabilities(d::UniformConditionalUrnDistribution) = expected_model_probabilities(length(d))
expected_inclusion_counts(d::UniformConditionalUrnDistribution) = expected_inclusion_counts(length(d))
expected_inclusion_probabilities(d::UniformConditionalUrnDistribution) = expected_inclusion_probabilities(length(d))

function expected_inclusion_probabilities(d::BetaBinomialConditionalUrnDistribution)
	k = length(d) - 1
	return Distributions.pdf.(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
end

function expected_model_probabilities(d::BetaBinomialConditionalUrnDistribution)
	incl_probs  = expected_inclusion_probabilities(d)
	no_models_with_incl = expected_inclusion_counts(length(d))
	
	# probability of j equalities for j in 1...k
	probs = incl_probs ./ no_models_with_incl
	result = Vector{Float64}(undef, sum(no_models_with_incl))
	index = 1
	for i in eachindex(probs)
		result[index:index + no_models_with_incl[i] - 1] .= probs[i]
		index += no_models_with_incl[i]
	end
	return result
end
#endregion