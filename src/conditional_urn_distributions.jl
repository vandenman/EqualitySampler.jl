"""
$(TYPEDSIGNATURES)

Reduce a partition to a unique representation. For example, [2, 2, 2] -> [1, 1, 1]
# Examples
```julia-repl
julia> reduce_model([2, 2, 2])
[1, 1, 1]
julia> reduce_model([2, 1, 1])
[1, 2, 2]
```
"""
function reduce_model(x::AbstractVector{T}) where T <: Integer

	#= TODO:

		It would be nice if the result is identical to something that the DirichletProcesses expect

	=#

	y = copy(x)
	# reduce_model!(y)
	# return y
	@inbounds for i in eachindex(x)
		if x[i] != i
			if !any(==(x[i]), x[1:i - 1])
				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)
				y[idx] .= i
			end
		end
	end
	return y
end

function get_conditional_counts(n::Int, known::AbstractVector{T} = [1], include_new::Bool = true) where T <: Integer

	n_known = length(known)

	res = zeros(Int, n_known + (include_new ? 1 : 0))
	idx = get_idx_for_conditional_counts(known)

	n_idx = length(idx)
	@inbounds res[idx] .= bellnumr(n - n_known - 1, n_idx)
	if include_new
		@inbounds res[n_known + 1] = bellnumr(n - n_known - 1, n_idx + 1)
	end
	res
end

function get_idx_for_conditional_counts(known)

	idx = Vector{Int}(undef, no_distinct_groups_in_partition(known))
	s = Set{Int}()
	count = 1
	@inbounds for i in eachindex(known)
		if known[i] ∉ s
			idx[count] = i
			count += 1
			push!(s, known[i])
		end
	end
	idx

end

"""
$(TYPEDSIGNATURES)

Count the number of equality constraints implied by a partition.
This assumes some elegant ordering of the constraints.
For example the partition ``\\{\\{\\theta_1, \\theta_3\\}, \\{\\theta_2\\}\\}`` can be written as ``\\theta_1 = \\theta_3 \\neq \\theta_2\`` and therefore
`count_equalities([1, 2, 1]) == 1`.
"""
count_equalities(urns::AbstractVector{T}) where T <: Integer = length(urns) - no_distinct_groups_in_partition(urns)
count_equalities(urns::AbstractString) = length(urns) - length(Set(urns))

"""
$(TYPEDSIGNATURES)

Count the number of free parameters implied by a partition.
"""
count_parameters(urns::AbstractString) = length(Set(urns))
count_parameters(urns::AbstractVector{<:Integer}) = no_distinct_groups_in_partition(urns)

# _pdf_uniform_helper exists so that it can also be used by the multivariate distribution
# function _pdf_helper(d::AbstractPartitionDistribution{T}, index::T, complete_urns::AbstractVector{T}) where T<:Integer

# 	k = length(complete_urns)
# 	result = zeros(Float64, length(complete_urns))
# 	_pdf_helper!(result, d, index, complete_urns)
# 	return result

# end


#endregion

#region expected model + inclusion probabilities

# TODO: these names are terrible, they only hold for the uniform case!
function expected_model_probabilities(k::Integer)
	x = bellnum(k)
	return fill(inv(x), x)
end

expected_inclusion_counts(k::Integer) = stirlings2.(k, 1:k)
expected_inclusion_counts(k::Integer, j::Integer) = stirlings2(k, j)

log_expected_equality_counts(k::Integer) = logstirlings2.(k, 1:k)
log_expected_equality_counts(k::Integer, j::Integer) = logstirlings2(k, j)

# function expected_inclusion_probabilities(k::Integer)
# 	counts = expected_inclusion_counts(k)
# 	return counts ./ sum(counts)
# end


# expected_model_probabilities(d::UniformPartitionDistribution) = expected_model_probabilities(length(d))
# expected_inclusion_counts(d::UniformPartitionDistribution) = expected_inclusion_counts(length(d))
# expected_inclusion_probabilities(d::UniformPartitionDistribution) = expected_inclusion_probabilities(length(d))
# log_expected_equality_counts(d::UniformPartitionDistribution) = log_expected_equality_counts(length(d))

# function log_expected_inclusion_probabilities(d::UniformPartitionDistribution)
# 	vals = log_expected_equality_counts(length(d))
# 	z = LogExpFunctions.logsumexp(vals)
# 	return vals .- z
# end


# function expected_inclusion_probabilities(d::BetaBinomialPartitionDistribution)
# 	# return exp.(d._log_model_probs_by_incl)
# 	k = length(d) - 1
# 	return Distributions.pdf.(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
# end

# function log_expected_inclusion_probabilities(d::BetaBinomialPartitionDistribution)
# 	k = length(d) - 1
# 	return Distributions.logpdf.(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
# end


# function expected_model_probabilities(d::BetaBinomialPartitionDistribution, compact::Bool = false)
# 	incl_probs  = expected_inclusion_probabilities(d)
# 	no_models_with_incl = expected_inclusion_counts(length(d))
# 	probs = incl_probs ./ no_models_with_incl

# 	# TODO: this compact creates type instabilities!
# 	if compact

# 		result = hcat(0:length(d)-1, no_models_with_incl, probs)

# 	else

# 		# probability of j equalities for j in 1...k
# 		result = Vector{Float64}(undef, sum(no_models_with_incl))
# 		index = 1
# 		for i in eachindex(probs)
# 			result[index:index + no_models_with_incl[i] - 1] .= probs[i]
# 			index += no_models_with_incl[i]
# 		end
# 	end
# 	return result
# end
#endregion