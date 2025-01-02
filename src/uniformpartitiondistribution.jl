"""
```
UniformPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}
```

Uniform distribution over partitions.
"""
struct UniformPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}
	k::T
	UniformPartitionDistribution{T}(k::T) where {T<:Integer} = new{T}(k)
end
function UniformPartitionDistribution(k::T; check_args::Bool = true) where T<:Integer
	Distributions.@check_args UniformPartitionDistribution (k, k > zero(k))
	UniformPartitionDistribution{T}(k)
end

function _pdf_helper!(probvec::AbstractVector{<:AbstractFloat}, ::UniformPartitionDistribution{T}, index::T, complete_urns::AbstractVector{T}, partition_sizes::AbstractVector{T}) where T<:Integer

	isone(index) && return fill_uniform!(probvec)

	k = length(probvec)

	v_known_urns = view(complete_urns, 1:index - 1)

	tb = fast_countmap_partition_incl_zero!(partition_sizes, v_known_urns)
	r = sum(!iszero, tb)

	lognum = logbellnumr(k - index, r + one(r))
	logden = log(r) + logbellnumr(k - index, r)
	prob_new_label = exp(lognum - LogExpFunctions.logsumexp([lognum, logden]))

	@inbounds for i in eachindex(probvec)
		# if i in urns_set
		if !iszero(tb[i])
			probvec[i] = (1 - prob_new_label) / r
		else
			probvec[i] = prob_new_label / (k - r)
		end
	end

	return probvec

end


"""
```
logpdf_model_distinct(d::AbstractPartitionDistribution, x)
```

Computes the log probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).
"""
function logpdf_model_distinct(d::UniformPartitionDistribution, ::AbstractVector{T}) where T <: Integer
	# return -logbellnumr(convert(T, length(d)), zero(T))
	return -logbellnumr(length(d), zero(length(d)))
end
logpdf_model_distinct(d::UniformPartitionDistribution, ::T) where T <: Integer = -logbellnumr(convert(T, length(d)), zero(T))

logpdf_incl_no_check(d::UniformPartitionDistribution, no_parameters::Integer) = logpdf_model_distinct(d, no_parameters) + log_expected_equality_counts(d, j)

function logpdf_incl(d::UniformPartitionDistribution, no_parameters::T) where T<:Integer
	Distributions.insupport(d, no_parameters) || return convert(float(T), -Inf)
	logpdf_model_distinct(d, no_parameters) + logstirlings2(length(d), no_parameters)
end

expected_model_probabilities(d::UniformPartitionDistribution) = expected_model_probabilities(length(d))
expected_inclusion_counts(d::UniformPartitionDistribution) = expected_inclusion_counts(length(d))
log_expected_equality_counts(d::UniformPartitionDistribution) = log_expected_equality_counts(length(d))

