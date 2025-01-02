struct DuplicatedPartitionDistribution{T<:Integer, D<:AbstractPartitionDistribution{T}} <: AbstractPartitionDistribution{T}
    dist::D
end

Base.length(d::DuplicatedPartitionDistribution) = length(d.dist)

log_count_combinations(::DuplicatedPartitionDistribution, x::AbstractVector) = log_count_combinations(x)
log_count_combinations(d::DuplicatedPartitionDistribution, x::Integer) = log_count_combinations(length(d), x)

maybe_reduce_model!(x::AbstractVector{<:Integer}, ::Union{DuplicatedPartitionDistribution, PartitionSampler{<:DuplicatedPartitionDistribution}}) = x
