#=

    Note: the only required method for a subtype of `AbstractSizePartitionDistribution` is
    `logpdf_incl_no_check(d::AbstractSizePartitionDistribution, no_parameters::Integer)`.

=#

"""
```
AbstractSizePartitionDistribution{<:Integer} <: Distributions.DiscreteMultivariateDistribution
```

Supertype for distributions over partitions that are fully determined by the number of free parameters they imply.
For example [1, 1, 1] implies one parameter and [1, 1, 2, 3, 4] implies four parameters.
"""
abstract type AbstractSizePartitionDistribution{T} <: AbstractPartitionDistribution{T} end

function logpdf_incl(d::AbstractSizePartitionDistribution, no_parameters::Integer)
    Distributions.insupport(d, no_parameters) || return -Inf
    logpdf_incl_no_check(d, no_parameters)
end

function logpdf_incl_no_check(d::AbstractSizePartitionDistribution)
    logpdf_incl_no_check.(Ref(d), 1:length(d))
end

function log_model_probs_by_incl(d::AbstractSizePartitionDistribution)
    logpdf_incl_no_check(d) .- log_expected_equality_counts(d)
end

function log_model_probs_by_incl(d::AbstractSizePartitionDistribution, no_parameters::Integer)
    logpdf_incl(d, no_parameters) - log_expected_equality_counts(d, no_parameters)
end

function logpdf_model_distinct(d::AbstractSizePartitionDistribution, x::AbstractVector{<:Integer})
    Distributions.insupport(d, x) || return -Inf
    logpdf_model_distinct(d, count_parameters(x))
end

function logpdf_model_distinct(d::AbstractSizePartitionDistribution, no_parameters::Integer)
    Distributions.insupport(d, no_parameters) || return -Inf
    log_model_probs_by_incl(d, no_parameters)
end

function _pdf_helper!(probvec::AbstractVector{<:AbstractFloat},
    d::AbstractSizePartitionDistribution{T}, index::T,
    complete_urns::AbstractVector{T},
    partition_sizes::AbstractVector{T}) where T<:Integer

    isone(index) && return fill_uniform!(probvec)
    k = length(probvec)

    index_already_sampled = 1:index - 1

    # no_duplicated = count_equalities(view(urns, index_already_sampled))
    v_known_urns = view(complete_urns, index_already_sampled)

    tb = fast_countmap_partition_incl_zero!(partition_sizes, v_known_urns)

    # v_known_set = Set(v_known_urns)
    # r = length(v_known_set)
    r = sum(!iszero, tb)
    n = k - (index - r - one(r))

    r = oftype(n, r)
    k = oftype(n, k)

    log_incl_probs = log_model_probs_by_incl(d)
    log_num = LogExpFunctions.logsumexp(
        log_incl_probs[i] + logstirlings2r(n - one(n), i, r    )
        for i in 1:k
    )
    log_den = LogExpFunctions.logsumexp(
        log_incl_probs[i] + logstirlings2r(n    , i, r + one(r))
        for i in 1:k
    )

    num = r * exp(log_num)
    den = 	  exp(log_den)

    prob_new_label = den / (den + num)
    @inbounds for i in eachindex(probvec)
        # if i in v_known_set
        if !iszero(tb[i])
            probvec[i] = (1 - prob_new_label) / r
        else
            probvec[i] = prob_new_label / (k - r)
        end
    end
    return probvec
end

log_expected_equality_counts(d::AbstractSizePartitionDistribution) = log_expected_equality_counts(length(d))
log_expected_equality_counts(d::AbstractSizePartitionDistribution, j::Integer) = log_expected_equality_counts(length(d), j)

log_expected_inclusion_probabilities(d::AbstractSizePartitionDistribution) = logpdf_incl_no_check(d)
expected_inclusion_probabilities(d::AbstractSizePartitionDistribution) = exp.(log_expected_inclusion_probabilities(d))

# function expected_model_probabilities(d::AbstractSizePartitionDistribution, compact::Bool = false)
#     incl_probs  = expected_inclusion_probabilities(d)
#     no_models_with_incl = expected_inclusion_counts(d)
#     probs = incl_probs ./ no_models_with_incl

#     # TODO: this compact creates type instabilities!
#     if compact

#         result = hcat(0:length(d)-1, no_models_with_incl, probs)

#     else

#         # probability of j equalities for j in 1...k
#         result = Vector{Float64}(undef, sum(no_models_with_incl))
#         index = 1
#         for i in eachindex(probs)
#             result[index:index + no_models_with_incl[i] - 1] .= probs[i]
#             index += no_models_with_incl[i]
#         end
#     end
#     return result
# end

function expected_model_probabilities(d::AbstractSizePartitionDistribution)

	incl_probs  = expected_inclusion_probabilities(d)
	no_models_with_incl = expected_inclusion_counts(length(d))
	probs = incl_probs ./ no_models_with_incl

	# probability of j equalities for j in 1...k
	result = similar(incl_probs, sum(no_models_with_incl))
	index = 1
	for i in eachindex(probs)
		result[index:index + no_models_with_incl[i] - 1] .= probs[i]
		index += no_models_with_incl[i]
	end

	return result
end
