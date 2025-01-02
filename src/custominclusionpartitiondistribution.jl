"""
```
CustomInclusionPartitionDistribution(k::T, logpdf::NTuple{N, Float64})
```

CustomInclusionPartitionDistribution is similar to the BetaBinomialPartitionDistribution in that the model probabilities are completely determined by the size of the partition.
Whereas the BetaBinomialPartitionDistribution uses a BetaBinomial distribution to obtain the probabilities, the CustomInclusionPartitionDistribution can be used to specify any vector of probabilities.
This distribution is particularly useful to sample uniformly from partitions of a given size.
For example:
```julia
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==1), Val(4)))) # always all equal (1 parameter)
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==3), Val(4)))) # always 3 parameters
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==4), Val(4)))) # always completely distinct (4 parameters)
```
The function does not check if sum(exp, logpdf) ≈ 1.0, that is the callers responsibility.
"""
struct CustomInclusionPartitionDistribution{T<:Integer, U<:AbstractFloat} <: AbstractSizePartitionDistribution{T}
	k::T
	logpdf::Vector{U}
	# function CustomInclusionPartitionDistribution(k::T, logpdf::AbstractVector{U}) where {T<:Integer, U<:AbstractFloat}
	# 	k != length(logpdf) && throw(DomainError(logpdf, "Length musth match k"))
	# 	new{T, U}(k, logpdf)
	# end
end
CustomInclusionPartitionDistribution(logpdf::AbstractVector) = CustomInclusionPartitionDistribution(length(logpdf), logpdf)

function CustomInclusionPartitionDistribution(k::T, logpdf::AbstractVector{U}; check_args::Bool = true) where {T<:Integer, U<:AbstractFloat}
	Distributions.@check_args CustomInclusionPartitionDistribution (k, k > zero(k)) (logpdf, length(logpdf) == k)
	CustomInclusionPartitionDistribution{T, U}(k, logpdf)
end

function logpdf_incl_no_check(d::CustomInclusionPartitionDistribution, no_parameters::Integer)
    d.logpdf[no_parameters]
end

struct PrecomputedCustomInclusionPartitionDistribution{T <: Integer, U<:Real} <: AbstractSizePartitionDistribution{T}
	k::T
	logpdf::Vector{U}
	log_expected_equality_counts::Vector{U}
	# function PrecomputedCustomInclusionPartitionDistribution(k::T, logpdf::AbstractVector{U}, log_expected_equality_counts::AbstractVector{U}) where {T<:Integer, U<:Real}
	# 	k != length(logpdf) 					  && throw(DomainError(logpdf, 						 "length(logpdf) must match k"))
	# 	k != length(log_expected_equality_counts) && throw(DomainError(log_expected_equality_counts, "length(log_expected_equality_counts) must match k"))
	# 	new{T, U}(k, logpdf, log_expected_equality_counts)
	# end
end
PrecomputedCustomInclusionPartitionDistribution(logpdf::AbstractVector, log_expected_equality_counts::AbstractVector) = PrecomputedCustomInclusionPartitionDistribution(length(logpdf), logpdf, log_expected_equality_counts)
PrecomputedCustomInclusionPartitionDistribution(logpdf::AbstractVector) = PrecomputedCustomInclusionPartitionDistribution(logpdf, log_expected_equality_counts(length(logpdf)))

PrecomputedCustomInclusionPartitionDistribution(d::PrecomputedCustomInclusionPartitionDistribution) = d
PrecomputedCustomInclusionPartitionDistribution(d::AbstractSizePartitionDistribution) = PrecomputedCustomInclusionPartitionDistribution(log_expected_inclusion_probabilities(d))

function PrecomputedCustomInclusionPartitionDistribution(k::T, logpdf::AbstractVector{U}, log_expected_equality_counts::AbstractVector{U}; check_args::Bool = true) where {T<:Integer, U<:AbstractFloat}
	Distributions.@check_args CustomInclusionPartitionDistribution (k, k > zero(k)) (logpdf, length(logpdf) == k) (log_expected_equality_counts, length(log_expected_equality_counts) == k)
	PrecomputedCustomInclusionPartitionDistribution{T, U}(k, logpdf, log_expected_equality_counts)
end

function logpdf_incl_no_check(d::PrecomputedCustomInclusionPartitionDistribution, no_parameters::Integer)
    d.logpdf[no_parameters]
end

function log_expected_equality_counts(d::PrecomputedCustomInclusionPartitionDistribution, no_parameters::Integer)
    d.log_expected_equality_counts[no_parameters]
end

# log_model_probs_by_incl(d::PrecomputedCustomInclusionPartitionDistribution) = d.logpdf .- d.log_expected_equality_counts
# log_model_probs_by_incl(d::PrecomputedCustomInclusionPartitionDistribution, no_parameters::Integer) = d.logpdf[no_parameters] - d.log_expected_equality_counts[no_parameters]
# logpdf_model_distinct(d::PrecomputedCustomInclusionPartitionDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
# function logpdf_model_distinct(d::PrecomputedCustomInclusionPartitionDistribution, no_parameters::Integer)
# 	in_eqsupport(d, no_parameters) || return -Inf
# 	log_model_probs_by_incl(d, no_parameters)
# end

# function logpdf_incl(d::PrecomputedCustomInclusionPartitionDistribution, no_parameters::Integer)
# 	in_eqsupport(d, no_parameters) || return -Inf
# 	@inbounds d.logpdf[no_parameters]
# end


