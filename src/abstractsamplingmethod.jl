abstract type AbstractSamplingMethod{T, U} end

function _validate_sampling_method_args(iter, initial_partition, split_merge_prob = 0.5)
    iter > 0 || throw(DomainError(iter, "iter must be positive"))
    zero(split_merge_prob) <= split_merge_prob <= one(split_merge_prob) || throw(DomainError(split_merge_prob, "split_merge_prob must be between 0 and 1"))
    k = length(initial_partition)
    V = eltype(initial_partition)
    if !iszero(k)
        all(x -> zero(V) < x <= V(k), initial_partition) || throw(DomainError(initial_partition, "initial_partition is invalid, some value is outside the range of 1 to $k."))
    end
end

function _interpret_max_cache_size(max_cache_size::Integer, initial_partition::AbstractVector)

    max_cache_size != -1 && return max_cache_size
    isempty(initial_partition) && return typemax(Int)

    # Get the size in bytes of each element in initial_partition
    element_size = sizeof(initial_partition)

    # Estimate 60% of the available system memory (in bytes)
    available_ram = Sys.free_memory() * 60 // 100

    # Calculate the number of elements that can fit into 60% of the available RAM
    num_keys = available_ram ÷ element_size

    return num_keys
end


struct Enumerate{T, U} <: AbstractSamplingMethod{T, U}
    integral_type::Type{T}
    initial_partition::U # TODO: this could also just store a type?

    function Enumerate(integral_type::Type{T}, initial_partition::U) where {T <: Number, U <: AbstractVector{<:Integer}}
        new{T, U}(integral_type, initial_partition)
    end
end
function Enumerate(; integral_type::Type{T} = Float64, initial_partition::U = Int8[]) where {T <: Number, U <: AbstractVector{<:Integer}}
    Enumerate(integral_type, initial_partition)
end


struct EnumerateThenSample{T, U} <: AbstractSamplingMethod{T, U}
    integral_type::Type{T}
    initial_partition::U # TODO: this could also just store a type?
    iter::Int

    function EnumerateThenSample(integral_type::Type{T}, initial_partition::U, iter::Integer) where {T <: Number, U <: AbstractVector{<:Integer}}

        _validate_sampling_method_args(iter, initial_partition)
        new{T, U}(integral_type, initial_partition, iter)
    end
end
function EnumerateThenSample(; integral_type::Type{T} = Float64, initial_partition::U = Int8[], iter::Integer = 10_000) where {T <: Number, U <: AbstractVector{<:Integer}}
    EnumerateThenSample(integral_type, initial_partition, iter)
end


struct SampleIntegrated{T, U} <: AbstractSamplingMethod{T, U}
    integral_type::Type{T}
    initial_partition::U
    iter::Int
    max_cache_size::Int
    split_merge_prob::Float64

    function SampleIntegrated(integral_type::Type{T}, initial_partition::U, iter::Integer, max_cache_size::Integer, split_merge_prob::Float64) where {T <: Number, U <: AbstractVector{<:Integer}}

        _validate_sampling_method_args(iter, initial_partition, split_merge_prob)
        max_cache_size = _interpret_max_cache_size(max_cache_size, initial_partition)
        new{T, U}(integral_type, initial_partition, iter, max_cache_size, split_merge_prob)
    end
end
function SampleIntegrated(; integral_type::Type{T} = Float64, initial_partition::U = Int8[], iter::Integer = 10_000, max_cache_size::Integer = -1, split_merge_prob::Float64 = 0.5) where {T <: Number, U <: AbstractVector{<:Integer}}
    SampleIntegrated(integral_type, initial_partition, iter, max_cache_size, split_merge_prob)
end


struct SampleRJMCMC{T, U} <: AbstractSamplingMethod{T, U}
    integral_type::Type{T}
    initial_partition::U
    iter::Int
    max_cache_size::Int
    fullmodel_only::Bool
    split_merge_prob::Float64

    function SampleRJMCMC(integral_type::Type{T}, initial_partition::U, iter::Integer, max_cache_size::Integer, fullmodel_only::Bool, split_merge_prob::Float64) where {T <: Number, U <: AbstractVector{<:Integer}}

        _validate_sampling_method_args(iter, initial_partition, split_merge_prob)
        max_cache_size = _interpret_max_cache_size(max_cache_size, initial_partition)
        new{T, U}(integral_type, initial_partition, iter, max_cache_size, fullmodel_only, split_merge_prob)
    end
end
function SampleRJMCMC(; integral_type::Type{T} = Float64, initial_partition::U = Int8[], iter::Integer = 10_000, max_cache_size::Integer = -1, fullmodel_only::Bool = false, split_merge_prob = 0.5) where {T <: Number, U <: AbstractVector{<:Integer}}
    SampleRJMCMC(integral_type, initial_partition, iter, max_cache_size, fullmodel_only, split_merge_prob)
end

mutable struct SamplingStats
    moved::Int
    no_local_moves::Int

    slit_merge_accepted::Int
    no_split_merge_moves::Int

    no_cache_hits::Int
    no_cache_checks::Int
end
function SamplingStats(; moved::Int = 0, no_local_moves::Int = 0, slit_merge_accepted::Int = 0, no_split_merge_moves::Int = 0, no_cache_hits::Int = 0, no_cache_checks::Int = 0)
    SamplingStats(moved, no_local_moves, slit_merge_accepted, no_split_merge_moves, no_cache_hits, no_cache_checks)
end

abstract type AbstractParameterSamples end
struct AnovaParameterSamples <: AbstractParameterSamples
    μ   ::Vector{Float64}
    σ²  ::Vector{Float64}
    g   ::Vector{Float64}
    θ_u ::Matrix{Float64}
    θ_s ::Matrix{Float64}
    θ_cp::Matrix{Float64}
end

struct ProportionParameterSamples <: AbstractParameterSamples
    θ_p_samples::Matrix{Float64}
end

abstract type AbstractMCMCResult end

struct EnumerateResult{T<:Number} <: AbstractMCMCResult
    k                    ::Int
    logml                ::Vector{T}
    bfs                  ::Vector{T}
    logbfs               ::Vector{T}
    err                  ::Vector{T}
    log_prior_model_probs::Vector{T}
    log_posterior_probs  ::Vector{T}
    posterior_probs      ::Vector{T}
end

struct IntegratedResult{U<:Integer} <: AbstractMCMCResult
    partition_samples::Matrix{U}
end

struct RJMCMCResult{T<:AbstractParameterSamples, U<:Integer} <: AbstractMCMCResult
    partition_samples::Matrix{U}
    parameter_samples::T
end

struct EnumerateThenSampleResult{T<:AbstractParameterSamples, U<:Integer, V<:Number} <: AbstractMCMCResult
    enumerate_result::EnumerateResult{V}
    partition_samples::Matrix{U}
    parameter_samples::T
end

no_groups(x::EnumerateResult) = x.k
function no_groups(x::Union{EnumerateResult, IntegratedResult, RJMCMCResult, EnumerateThenSampleResult})
    return size(x.partition_samples, 1)
end