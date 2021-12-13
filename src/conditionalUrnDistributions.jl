# TODO: this entire file needs to be refactored!

"""
	reduce a model to a unique representation. For example, [2, 2, 2] -> [1, 1, 1]
"""
function reduce_model(x::AbstractVector{T}) where T <: Integer

	#= TODO:

		It would be nice if the result is identical to something that the DirichletProcesses expect

	=#

	y = copy(x)
	# reduce_model!(y)
	# return y
	for i in eachindex(x)
		if x[i] != i
			if !any(==(x[i]), x[1:i - 1])
				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)
				y[idx] .= i
			end
		end
	end
	return y
end

function reduce_model_dpp(x::AbstractVector{<:Integer})

	y = similar(x)
	currentMax = 0
	visited = Set{Int}()
	for i in eachindex(y)
		if x[i] ∉ visited
			currentMax += 1
			y[i] = currentMax
			for j in i+1:length(x)
				if x[i] == x[j]
					y[j] = currentMax
				end
			end
			push!(visited, x[i])
		end
	end
	return y
end

# function reduce_model!(x::AbstractVector{T}) where T <: Integer
# 	# TODO: this is less efficient than just copying x first
# 	for i in eachindex(x)
# 		if x[i] != i
# 			if !any(==(x[i]), x[1:i - 1])
# 				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)

# 				otherIdx = findall(==(i), x[i:end]) .+ (i - 1)
# 				v = x[i]
# 				x[idx] .= i
# 				x[otherIdx] .= v

# 			end
# 		end
# 	end
# 	return x
# end


function get_conditional_counts(n::Int, known::AbstractVector{T} = [1], include_new::Bool = true) where T <: Integer

	# TODO: just pass d.partitions and d.index to this function! (or just d itself)
	# then we can get rid of the include_new and use dispatch, although it might duplicate some code
	# or maybe not?
	# NOTE: profiling shows that reduce_model is by far the most expensive function here

	# known = reduce_model(known) # TODO: double check that this is unnecessary!
	refinement = length(unique(known))
	n_known = length(known)

	res = zeros(Int, n_known + (include_new ? 1 : 0))
	# idx = findall(i->known[i] == i, 1:n_known) # This step fails when not doing reduce_model
	idx = get_idx_for_conditional_counts(known)

	n_idx = length(idx)
	res[idx] .= bellnumr(n - n_known - 1, n_idx)
	if include_new
		res[n_known + 1] = bellnumr(n - n_known - 1, n_idx + 1)
	end
	res
end

function get_idx_for_conditional_counts(known)

	idx = Vector{Int}(undef, length(Set(known)))
	s = Set{Int}()
	count = 1
	for i in eachindex(known)
		if known[i] ∉ s
			idx[count] = i
			count += 1
			push!(s, known[i])
		end
	end
	idx

end

count_equalities(urns::AbstractVector{T}) where T <: Integer = length(urns) - length(Set(urns))
count_equalities(urns::AbstractString) = length(urns) - length(Set(urns))

count_parameters(urns::U) where U <: Union{AbstractVector{<:Integer}, AbstractString} = length(Set(urns))

#region AbstractConditionalUrnDistribution
abstract type AbstractConditionalUrnDistribution{T} <: Distributions.DiscreteUnivariateDistribution where T <: Integer end

Distributions.minimum(::AbstractConditionalUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractConditionalUrnDistribution{T}) where T = T(length(d))
Base.length(d::AbstractConditionalUrnDistribution) = length(d.urns)

Distributions.logpdf(::AbstractConditionalUrnDistribution, x::Real) = -Inf
Distributions.pdf(d::AbstractConditionalUrnDistribution{T}, x::Real) where T = exp(Distributions.logpdf(d, x))
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
struct UniformConditionalUrnDistribution{T, U<:AbstractVector{T}} <: AbstractConditionalUrnDistribution{T}
	urns::U
	index::T
	function UniformConditionalUrnDistribution(urns::U, index::T = 1) where {T <: Integer, U <: AbstractVector{T}}
		n = length(urns)
		all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
		one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
		return new{T, U}(urns, index)
	end
end
UniformConditionalUrnDistribution(k::T) where T <: Integer = UniformConditionalUrnDistribution(ones(T, k), k)


# function _pdf(d::UniformConditionalUrnDistribution)

# 	index, k = d.index, length(d.urns)
# 	isone(index) && return fill(1 / k, k)
# 	urns = view(d.urns, 1:index - 1)

# 	count = get_conditional_counts(k, urns)
# 	result = zeros(Float64, k)
# 	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
# 	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
# 	other = setdiff(1:k, urns)
# 	result[other] .= count[length(urns) + 1] ./ length(other)
# 	return result ./ sum(result)

# end
_pdf(d::UniformConditionalUrnDistribution) = _pdf_helper(d, d.index, d.urns)

# _pdf_uniform_helper exists so that it can also be reused by the multivariate distribution without explicitly creating a UniformConditionalUrnDistribution
function _pdf_helper(d::Union{AbstractConditionalUrnDistribution, AbstractMvUrnDistribution}, index, complete_urns)

	k = length(complete_urns)
	result = zeros(Float64, length(complete_urns))
	_pdf_helper!(result, d, index, complete_urns)
	return result

end

function _pdf_helper!(result, ::Union{UniformConditionalUrnDistribution, UniformMvUrnDistribution}, index, complete_urns)

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	urns = view(complete_urns, 1:index - 1)

	count = get_conditional_counts(k, urns)

	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
	other = setdiff(1:k, urns)
	result[other] .= count[length(urns) + 1] ./ length(other)
	result ./= sum(result)
	return
end

# function _pdf_helper!(result, d::Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}, index, complete_urns)

# 	k = length(result)
# 	if isone(index)
# 		fill!(result, 1 / k)
# 		return nothing
# 	end

# 	urns = view(complete_urns, 1:index - 1)

# 	count = get_conditional_counts(k, urns)

# 	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
# 	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
# 	other = setdiff(1:k, urns)
# 	result[other] .= count[length(urns) + 1] ./ length(other)
# 	result ./= sum(result)
# 	return nothing
# end

#endregion
#region BetaBinomialConditionalUrnDistribution
struct BetaBinomialConditionalUrnDistribution{T, U<:AbstractVector{T}} <: AbstractConditionalUrnDistribution{T}
	urns::U
	index::T
	α::Float64
	β::Float64
	_log_model_probs_by_incl::Vector{Float64}
	function BetaBinomialConditionalUrnDistribution(urns::U, index::T = 1, α::Float64 = 1.0, β::Float64 = 1.0) where {T <: Integer, U <: AbstractVector{T}}
		n = length(urns)
		all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
		one(T) <= index <= n || throw(DomainError(urns, "condition: 0 <= index <= length(urns) is violated"))
		0.0 <= α || throw(DomainError(α, "condition: 0 <= α is violated"))
		0.0 <= β || throw(DomainError(β, "condition: 0 <= β is violated"))
		log_model_probs_by_incl = Distributions.logpdf.(Distributions.BetaBinomial(n - 1, α, β), 0:n - 1) .- log_expected_equality_counts(n)
		new{T, U}(urns, index, α, β, log_model_probs_by_incl)
	end
end

function BetaBinomialConditionalUrnDistribution(urns::AbstractVector{T}, index::T, α::Real, β::Real) where T <: Integer
	BetaBinomialConditionalUrnDistribution(urns, index, convert(Float64, α), convert(Float64, β))
end

function BetaBinomialConditionalUrnDistribution(k::T, α::Real = 1.0, β::Real = 1.0) where T <: Integer
	BetaBinomialConditionalUrnDistribution(ones(T, k), k, convert(Float64, α), convert(Float64, β))
	# BetaBinomialConditionalUrnDistribution(T[one(T)], k, convert(Float64, α), convert(Float64, β))
end

_pdf(d::BetaBinomialConditionalUrnDistribution) = _pdf_helper(d, d.index, d.urns)

log_model_probs_by_incl(d::BetaBinomialConditionalUrnDistribution) = d._log_model_probs_by_incl
function _pdf_helper!(result, d::T, index, complete_urns) where T<:Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	index_already_sampled = 1:index - 1
	n0 = k

	# no_duplicated = count_equalities(view(urns, index_already_sampled))
	v_known_urns = view(complete_urns, index_already_sampled)
	r = length(Set(v_known_urns))
	n = n0 - (index - r - 1)

	model_probs_by_incl = exp.(log_model_probs_by_incl(d))

	# no. parameters
	num = r * sum(model_probs_by_incl[k] * stirlings2r(n - 1, k, r    ) for k in 1:n0)
	den =     sum(model_probs_by_incl[k] * stirlings2r(n    , k, r + 1) for k in 1:n0)
	probEquality = num / (num + den)

	# no. equalities
	# num = r * sum(model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    ) for k in 1:n0)
	# den =     sum(model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1) for k in 1:n0)
	# probEquality = num / (num + den)

	# probability of an equality
	known = reduce_model(v_known_urns)
	counts = get_conditional_counts(k, known, false)
	idx_nonzero = findall(!iszero, counts)
	result[v_known_urns[idx_nonzero]] .= probEquality .* (counts[idx_nonzero] ./ sum(counts[idx_nonzero]))

	# probability of an inequality
	inequality_options = setdiff(1:n0, v_known_urns)
	result[inequality_options] .= (1 - probEquality) ./ length(inequality_options)

	return

end
#endregion

#region expected model + inclusion probabilities
function expected_model_probabilities(k::Integer)
	x = count_distinct_models(k)
	return fill(one(k) / x, x)
end
expected_inclusion_counts(k::Integer) = stirlings2.(k, 1:k)
function expected_inclusion_probabilities(k::Integer)
	counts = expected_inclusion_counts(k)
	return counts ./ sum(counts)
end
log_expected_equality_counts(k::Integer) = logstirlings2.(k, 1:k)


const UniformUrnDists = Union{UniformConditionalUrnDistribution, UniformMvUrnDistribution}
expected_model_probabilities(d::UniformUrnDists) = expected_model_probabilities(length(d))
expected_inclusion_counts(d::UniformUrnDists) = expected_inclusion_counts(length(d))
expected_inclusion_probabilities(d::UniformUrnDists) = expected_inclusion_probabilities(length(d))
log_expected_equality_counts(d::UniformUrnDists) = log_expected_equality_counts(length(d))
function log_expected_inclusion_probabilities(d::UniformUrnDists)
	vals = log_expected_equality_counts(length(d))
	z = logsumexp_batch(vals)
	return vals .- z
end

function expected_inclusion_probabilities(d::T) where T<:Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}
	# return exp.(d._log_model_probs_by_incl)
	k = length(d) - 1
	return Distributions.pdf.(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
end

function log_expected_inclusion_probabilities(d::T) where T<:Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}
	k = length(d) - 1
	return Distributions.logpdf.(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
end


function expected_model_probabilities(d::T, compact::Bool = false) where T<:Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}
	incl_probs  = expected_inclusion_probabilities(d)
	no_models_with_incl = expected_inclusion_counts(length(d))
	probs = incl_probs ./ no_models_with_incl

	# TODO: this compact creates type instabilities!
	if compact

		result = hcat(0:length(d)-1, no_models_with_incl, probs)

	else

		# probability of j equalities for j in 1...k
		result = Vector{Float64}(undef, sum(no_models_with_incl))
		index = 1
		for i in eachindex(probs)
			result[index:index + no_models_with_incl[i] - 1] .= probs[i]
			index += no_models_with_incl[i]
		end
	end
	return result
end
#endregion