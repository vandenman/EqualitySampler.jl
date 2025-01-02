# TODO: needs a better name
struct TrueValues{T<:Real, U<:AbstractVector{T}, W<:AbstractVector{<:Integer}}
    μ::T
    σ::T
    θ::U
    partition::W
end

struct AnovaSimulatedData{T, U}
    data::SimpleDataSet
    distribution::T
    true_values::U
end

function normalize_θ(offset::AbstractFloat, true_model::Vector{T}) where T<:Integer

    copy_model = copy(true_model)
    current_max = copy_model[1]

    for i in eachindex(copy_model)
        if copy_model[i] > current_max
            copy_model[copy_model .== i] .= current_max
            current_max += 1
        elseif copy_model[i] == i
            current_max += 1
        end
    end

    θ = copy_model .* offset
    return θ .- StatsBase.mean(θ)
end

function simulate_data_one_way_anova(
    n_groups::Integer,
    n_obs_per_group::Union{Integer, AbstractVector{<:Integer}},
    θ::AbstractVector{<:AbstractFloat} = Float64[],
    partition::AbstractVector{<:Integer} = 1:n_groups,
    μ::AbstractFloat = 0.0,
    σ::AbstractFloat = 1.0,
    rng::Random.AbstractRNG = Random.default_rng()
)

    if isempty(θ)
        θ = 0.2 .* partition
    end

    length(θ) != n_groups && throw(error("length(θ) != n_groups"))

    n_obs_per_group2 = n_obs_per_group isa Integer ? FillArrays.Fill(n_obs_per_group, n_groups) : n_obs_per_group

    n_obs = sum(n_obs_per_group2)
    θc = θ .- StatsBase.mean(θ)

    g = Vector{UnitRange{Int}}(undef, n_groups)
    g[1] = 1:n_obs_per_group2[1]
    for i in 2:length(g)
        g[i] = 1 + last(g[i-1]) : last(g[i-1]) + n_obs_per_group2[i]
    end

    group_idx = Vector{Int}(undef, n_obs)
    for (i, idx) in enumerate(g)
        group_idx[idx] .= i
    end

    # we have an offset because we assume a centered design matrix, but the construction above is not centered
    # this avoids creating a dense centered matrix
    # offset = LinearAlgebra.dot((n_obs_per_group ./ n_obs), θc)

    mean_vec = (μ#= - offset=#) .+ view(θc, group_idx)
    # @show StatsBase.mean(mean_vec)

    D = Distributions.MvNormal(mean_vec, abs2(σ) * LinearAlgebra.I)
    y = rand(rng, D)

    dat = SimpleDataSet(y, g)

    true_values = TrueValues(μ, σ, θc, partition)

    return AnovaSimulatedData(dat, D, true_values)

end

function simulate_data_one_way_anova(;
    n_groups::Integer,
    n_obs_per_group::Union{Integer, AbstractVector{<:Integer}},
    θ::AbstractVector{<:AbstractFloat} = Float64[],
    partition::AbstractVector{<:Integer} = 1:n_groups,
    μ::AbstractFloat = 0.0,
    σ::AbstractFloat = 1.0,
    rng::Random.AbstractRNG = Random.default_rng()
)

    simulate_data_one_way_anova(n_groups, n_obs_per_group, θ, partition, μ, σ, rng)
end



# these two need to go elsewhere
# perhaps also reduce_partition?
# add a method to pass a dict

"""
Convert a partition to its canonical form.
The canonical form implies that, from left to right, each value equals the lowest possible value.
For example,
```julia
reduce_model_dct([2, 2, 2, 2]) = [1, 1, 1, 1]
reduce_model_dct([2, 1, 1, 1]) = [1, 2, 2, 2]
```

NOTE: if it can be assumed that `all(1 <= p <= length(partition) for p in partition)` then [`reduce_model_2`](@ref) is faster.
"""
reduce_model_dct(partition::AbstractVector{<:Integer}) = reduce_model_dct!(copy(partition))
reduce_model_dct!(partition::AbstractVector{T}) where T<:Integer = reduce_model_dct!(partition, Dict{T, T}())

function reduce_model_dct!(partition::AbstractVector{T}, lookup::AbstractDict{T, T}) where T<:Integer
    current = one(T)
    for i in eachindex(partition)

        # @show current, lookup
        # partition[i] = get!(lookup, partition[i]) do
        #     current0 = current
        #     current += one(T) # <- causes boxing...
        #     current0
        # end
        # @show current, lookup

        if haskey(lookup, partition[i])
            partition[i] = lookup[partition[i]]
        else
            lookup[partition[i]] = current
            partition[i]         = current
            current += one(T)
        end
    end
    partition
end



"""
Convert a partition to its canonical form.
The canonical form implies that, from left to right, each value equals the lowest possible value.
For example,
```julia
reduce_model_2([2, 2, 2, 2]) = [1, 1, 1, 1]
reduce_model_2([2, 1, 1, 1]) = [1, 2, 2, 2]
```

NOTE: this function is assumes and does not check that `all(1 <= p <= length(partition) for p in partition)`.
if this is not guaranteed, use the slower [`reduce_model_dct`](@ref) instead.
"""
reduce_model_2(partition::AbstractVector{T}) where T<:Integer = reduce_model_2!(copy(partition))

function reduce_model_2!(partition::AbstractVector{T}) where T<:Integer
    lookup = zeros(T, length(partition))
    reduce_model_2!(partition, lookup)
end


function reduce_model_2!(partition::AbstractVector{T}, lookup::AbstractVector{T}) where T<:Integer
    current = one(T)
    @inbounds for i in eachindex(partition)
    # @inbounds for i in start_at:length(partition)

        if iszero(lookup[partition[i]])
            lookup[partition[i]] = current
            partition[i]         = current
            current += one(T)
        else
            partition[i] = lookup[partition[i]]
        end
    end
    partition
end

function reduce_model_2_and_clear_lookup!(partition::AbstractVector{T}, lookup::AbstractVector{T}) where T<:Integer
    reduce_model_2!(partition, lookup)
    fill!(lookup, zero(eltype(lookup)))
end



# TODO: not sure if this type should be exposed to users at all?
abstract type AbstractPriors end
Base.@kwdef struct AnovaPriors{T<:AbstractPartitionDistribution, U<:Number} <: AbstractPriors
    partition::T
    rscale::U    = 1.0
    function AnovaPriors(partition::T, rscale::U = 1.0) where {T<:AbstractPartitionDistribution, U<:Number}
        rscale > zero(rscale) || throw(DomainError(rscale, "rscale must be positive"))
        new{T, U}(partition, rscale)
    end
end

function _get_initial_partition(method::AbstractSamplingMethod, priors::AbstractPriors, k::Integer)
    ip = method.initial_partition
    isempty(ip) && return convert(typeof(ip), 1:k)

    # TODO: impossible? checked in the constructor of method?
    Distributions.insupport(priors.partition, ip) || throw(ArgumentError("Initial partition is not valid"))
    return reduce_model_2(ip)
end

abstract type AbstractSuffstats end
# TODO: SuffstatsANOVA or ANOVASuffstats?
struct SuffstatsANOVA <: AbstractSuffstats
    y_mean::Float64
    y_var::Float64
    n::Int
    y_mean_by_group::Vector{Float64}
    y_vars_by_group::Vector{Float64}
    n_by_group::Vector{Int}
end

_get_k(obj::SuffstatsANOVA) = length(obj.n_by_group)

_get_means(obj::SuffstatsANOVA) = obj.y_mean_by_group
_get_vars( obj::SuffstatsANOVA) = obj.y_vars_by_group

function _validate_partition_and_suffstats(obj::AbstractSuffstats, partition_prior::AbstractPartitionDistribution)
    len_obj   = _get_k(obj)
    len_prior = length(partition_prior)
    len_obj == len_prior || throw(ArgumentError("Length of partition_prior ($len_prior) does not match number of groups ($len_obj)"))
end


function extract_suffstats_one_way_anova(y_mean_by_group::AbstractVector, y_var_by_group::AbstractVector, ns::AbstractVector{<:Integer})

    n = sum(ns)
    y_mean = LinearAlgebra.dot(y_mean_by_group, ns) / n
    y_var = sum(i-> ns[i] * (y_var_by_group[i] + abs2(y_mean_by_group[i])), eachindex(ns)) / n - abs2(y_mean)

    return SuffstatsANOVA(y_mean, y_var, n, y_mean_by_group, y_var_by_group, ns)
end

function extract_suffstats_one_way_anova(y::AbstractVector, g::AbstractVector)

    ns     = [count(==(i), g) for i in unique(g)]
    y_mean_by_group = [StatsBase.mean(y[g .== i]) for i in unique(g)]
    y_var_by_group  = [StatsBase.var(y[g .== i], corrected = false) for i in unique(g)]

    return extract_suffstats_one_way_anova(y_mean_by_group, y_var_by_group, ns)

end

function extract_suffstats_one_way_anova(y::AbstractVector, g::AbstractVector{<:UnitRange})

    ns     = length.(g)
    y_mean_by_group = [StatsBase.mean(y[gᵢ]) for gᵢ in g]
    y_var_by_group  = [StatsBase.var(y[gᵢ], corrected = false) for gᵢ in g]

    return extract_suffstats_one_way_anova(y_mean_by_group, y_var_by_group, ns)

end

function apply_partition_to_suffstats(obj::SuffstatsANOVA, partition0::AbstractVector, reduce::Bool = true)

    partition = reduce ? reduce_model_2(partition0) : partition0
    new_length = no_distinct_groups_in_partition(partition)
    ns_new = zeros(eltype(obj.n_by_group), new_length)
    y_mean_by_group_new = zeros(eltype(obj.y_mean_by_group), new_length)
    y_var_by_group_new  = zeros(eltype(obj.y_vars_by_group), new_length)
    for i in eachindex(partition)
        # @show length(ns_new), i, partition[i], length(partition), length(obj.n_by_group)
        ns_new[partition[i]] += obj.n_by_group[i]
        y_mean_by_group_new[partition[i]] += obj.y_mean_by_group[i] * obj.n_by_group[i]
        y_var_by_group_new[partition[i]]  += (obj.y_vars_by_group[i] + abs2(obj.y_mean_by_group[i])) * obj.n_by_group[i]
    end
    for i in eachindex(ns_new)
        y_mean_by_group_new[i] /= ns_new[i]
        y_var_by_group_new[i]  = y_var_by_group_new[i] / ns_new[i] - abs2(y_mean_by_group_new[i])
    end

    SuffstatsANOVA(obj.y_mean, obj.y_var, obj.n, y_mean_by_group_new, y_var_by_group_new, ns_new)
end

get_XTX(obj::SuffstatsANOVA) = LinearAlgebra.Diagonal(obj.n_by_group)
get_XTX_over_n(obj::SuffstatsANOVA) = LinearAlgebra.Diagonal(obj.n_by_group ./ obj.n)

get_XTy(obj::SuffstatsANOVA) = obj.y_mean_by_group .* obj.n_by_group
get_XTy_over_n(obj::SuffstatsANOVA) = obj.y_mean_by_group .* (obj.n_by_group ./ obj.n)

# assumes centered design matrix
get_XTX_c(obj::SuffstatsANOVA) = LinearAlgebra.Diagonal(obj.n_by_group) .- obj.n_by_group * obj.n_by_group' ./ obj.n
get_yTX_c(obj::SuffstatsANOVA) = (obj.y_mean_by_group .- obj.y_mean) .* obj.n_by_group

get_XTy_c_over_n(obj) = (obj.y_mean_by_group .- obj.y_mean) .* (obj.n_by_group ./ obj.n)
function get_XTX_c_over_n(obj)
    temp = obj.n_by_group ./ obj.n
    LinearAlgebra.Diagonal(temp) .- temp * temp'
end


# module EqualitySamplerStatsModelsExt

# using EqualitySampler
# import StatsModels, DataFrames

"""
```julia
anova_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame, args..., ; kwargs...)
anova_test(y::AbstractVector{<:Number}, g::AbstractVector{<:Integer}, args..., ; kwargs...)
anova_test(y::AbstractVector{<:Number}, g::AbstractVector{<:Integer}, args..., ; kwargs...)
anova_test(y_means::AbstractVector{<:Number}, y_vars::AbstractVector{<:Number}, ns::AbstractVector{<:Integer}, args..., ; kwargs...)
anova_test(obj::SuffstatsANOVA, method::AbstractSamplingMethod, prior::AbstractPartitionDistribution; rscale::Number = 1.0, verbose::Bool = true, threaded::Bool = false)
```

Conduct a one way anova test.

Data can be input in four ways.
1. using a formula and dataframe (`@formula(outcome ~ 1 + grouping), df`).
2. vector of rawdata and vector of group membership (`y = randn(6), g = [1, 1, 2, 2, 3, 3]`).
3. rawdata and group ranges (`y = randn(6), g = [1:2, 3:4, 5:6]`)
4. means, variances, and group sizes (`y_means = randn(3), y_vars = randexp(3), ns  = rand(2:2, 3)`).

In each case, the data is reduced to an object of type `SuffstatsANOVA` which contains the sufficient statistics for the test.

The method should be one of `Enumerate`, `EnumerateThenSample`, `SampleIntegrated`, or `SampleRJMCMC`.
The return type depends on the method used. Specifically, `Enumerate` returns an `EnumerateResult`,
`EnumerateThenSample` returns an `EnumerateThenSampleResult`, `SampleIntegrated` returns an `IntegratedResult`, and
`SampleRJMCMC` returns an `RJMCMCResult`.
These different types do not really matter and are mostly used to make various extractor functions work.

For `EnumerateThenSample` and `SampleRJMCMC`, the raw posterior samples can be found inside the field `parameter_samples`, see also
`AnovaParameterSamples`.

"""
function anova_test(
    f::StatsModels.FormulaTerm, df::DataFrames.DataFrame,
    method::AbstractSamplingMethod, prior::AbstractPartitionDistribution;
    kwargs...)

    # f = StatsModels.@formula(outcome ~ 1 + grouping)
    ts = StatsModels.apply_schema(f, StatsModels.schema(df))

    if StatsModels.hasintercept(ts)
        ts = StatsModels.drop_term(ts, StatsModels.term(1))
    end

    mc = StatsModels.modelcols(ts, df)

    y = mc[1]
    g = Integer.(vec(mc[2]))

    ts = StatsModels.apply_schema(f, StatsModels.schema(df))
    ts = StatsModels.drop_term(ts, StatsModels.term(1))

    mc = StatsModels.modelcols(ts, df)

    length(mc) != 2        && throw(ArgumentError("No predictors in formula."))
    !isone(size(mc[2], 2)) && throw(ArgumentError("Expected `$f` to only specify one predictor, for example `y ~ g`"))

    y = vec(mc[1])
    g = vec(Integer.(mc[2]))

    # @show extract_suffstats_one_way_anova(y, g)

    anova_test(y, g, method, prior; kwargs...)
end

# end

function anova_test(
    y::AbstractVector{<:Number}, group_idx::AbstractVector{<:Integer},
    method::AbstractSamplingMethod, prior::AbstractPartitionDistribution;
    kwargs...)
    anova_test(extract_suffstats_one_way_anova(y, group_idx), method, prior; kwargs...)
end

function anova_test(
    y::AbstractVector{<:Number}, group_idx::AbstractVector{<:UnitRange},
    method::AbstractSamplingMethod, prior::AbstractPartitionDistribution;
    kwargs...)
    anova_test(extract_suffstats_one_way_anova(y, group_idx), method, prior; kwargs...)
end

function anova_test(
    y_means::AbstractVector{<:Number}, y_vars::AbstractVector{<:Number}, ns::AbstractVector{<:Integer},
    method::AbstractSamplingMethod, prior::AbstractPartitionDistribution;
    kwargs...)
    anova_test(extract_suffstats_one_way_anova(y_means, y_vars, ns), method, prior; kwargs...)
end

function anova_test(
    obj::SuffstatsANOVA,
    method::Enumerate,
    partition_prior::AbstractPartitionDistribution;
    # these two fit all methods
    rscale::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = true)

    _enumerate(obj, method, AnovaPriors(partition_prior, rscale), verbose, threaded)

end

function _enumerate(
    obj::AbstractSuffstats,
    method::Enumerate,
    priors::AbstractPriors,
    verbose::Bool = true,
    threaded::Bool = true)

    k = _get_k(obj)

    modelspace = PartitionSpace(k, DistinctPartitionSpace)
    nmodels = length(modelspace)

    result_object = instantiate_results(obj, method, nmodels, k)

    logml_h0 = logml_H0(obj)
    result_object.logml[1] = logml_h0

    if threaded
        _enumerate_internal_threaded!(result_object, obj, logml_h0, modelspace, verbose, method, priors)
    else
        _enumerate_internal!(         result_object, obj, logml_h0, modelspace, verbose, method, priors)
    end

    # could also just pass the object instead of the elements?
    logbf_to_logposteriorprob!(result_object.log_posterior_probs, result_object.logbfs, result_object.log_prior_model_probs)
    result_object.posterior_probs .= exp.(result_object.log_posterior_probs)
    return result_object
end

function anova_test(
    obj::SuffstatsANOVA,
    method::SampleIntegrated,
    partition_prior::AbstractPartitionDistribution;
    # these two fit all methods
    rscale::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = false)

    _sample_integrated(obj, method, AnovaPriors(partition_prior, rscale), verbose, threaded)

end

function _sample_integrated(
        obj::AbstractSuffstats,
        method::SampleIntegrated,
        priors::AbstractPriors,
        verbose::Bool = true,
        threaded::Bool = false
    )

    iter = method.iter

    k = _get_k(obj)

    _validate_partition_and_suffstats(obj, priors.partition)

    partition = _get_initial_partition(method, priors, k)

    logbf10_dict = Dict{typeof(partition), Float64}()
    max_keys_logbf10_dict = method.max_cache_size

    partition_samples = similar(partition, k, iter)
    sampling_stats = SamplingStats()#Ref{Int}(0)
    ProgressMeter.@showprogress enabled=verbose for it in 1:iter

        sample_partition_rjmcmc_integrated!(partition, obj, logbf10_dict, method, sampling_stats, priors, threaded, max_keys_logbf10_dict)
        partition_samples[:, it]  = partition

    end

    return IntegratedResult(partition_samples)
    # return (; partition_samples)

end

function anova_test(
    obj_full::SuffstatsANOVA,
    method::M,
    partition_prior::AbstractPartitionDistribution;
    # these two fit all methods
    rscale::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = false
    ) where M <: Union{SampleRJMCMC, EnumerateThenSample}

    priors = AnovaPriors(partition_prior, rscale)
    iter = method.iter
    fullmodel_only = !(M <: EnumerateThenSample) && method.fullmodel_only

    !fullmodel_only && _validate_partition_and_suffstats(obj_full, partition_prior)

    # TODO: rscale should also be used during sampling!

    k = _get_k(obj_full)

    # intial values
    μ  = obj_full.y_mean
    σ² = obj_full.y_var
    n  = obj_full.n

    y_mean = obj_full.y_mean
    # ss2_y_over_n = StatsBase.mean(abs2, y)
    ss2_y_over_n = obj_full.y_var + abs2(y_mean)

    # g  = 1.0
    # θ_u = randn(k - 1)
    # θ_s = similar(θ_u, k)
    # θ_cs = similar(θ_u, k)

    partition = _get_initial_partition(method, priors, k)

    obj = apply_partition_to_suffstats(obj_full, partition, false)

    # TODO: these are pretty meaningless except for g, the first full conditional is θ_u
    # Q = getQ_Rouder(length(obj_full.n_by_group))
    # θ_s = obj_full.y_mean_by_group .- StatsBase.mean(obj_full.y_mean_by_group)

    Q = getQ_Rouder(length(obj.y_mean_by_group))
    θ_c = obj.y_mean_by_group .- StatsBase.mean(obj.y_mean_by_group)
    θ_u = Q' * θ_c
    g  = isone(length(θ_c)) ? 1.0 : StatsBase.var(θ_c)
    θ_cp = similar(θ_c, k)

    μ_samples         = Vector{Float64}(undef, iter)
    σ²_samples        = Vector{Float64}(undef, iter)
    g_samples         = Vector{Float64}(undef, iter)
    if fullmodel_only
        θ_u_samples       = Matrix{Float64}(undef, length(θ_u)    , iter)
        θ_s_samples       = Matrix{Float64}(undef, length(θ_u) + 1, iter)
        partition_samples = similar(partition, 0, 0)
    else
        θ_u_samples       = Matrix{Float64}(undef, k - 1, iter)
        θ_s_samples       = Matrix{Float64}(undef, k    , iter)
        partition_samples = similar(partition, k, iter)

        resize!(θ_u, k - 1)
        resize!(θ_c, k)
    end
    θ_cp_samples      = Matrix{Float64}(undef, k,     iter)

    # this can be improved upon!
    # design_mat = zeros(Int, n, k)
    # for i in 1:k
    #     design_mat[group_idx .== i, i] .= 1
    # end

    # tb_g = StatsBase.countmap(group_idx)
    # XtX_over_n_full = LinearAlgebra.Diagonal([tb_g[i] for i in axes(design_mat, 2)])
    # Xty_over_n_full = design_mat' * y ./ n

    # XtX_over_n_full = LinearAlgebra.Diagonal(obj_full.n_by_group)
    # Xty_over_n_full = (obj_full.y_mean_by_group .* obj_full.n_by_group) ./ obj_full.n

    # TODO: _full is a misnomer, this need not be the full model anymore
    # XtX_over_n_full = LinearAlgebra.Diagonal(obj.n_by_group)
    # Xty_over_n_full = (obj.y_mean_by_group .* obj.n_by_group) ./ obj.n

    # XtX_over_n_full = get_XTX_c(obj)
    # Xty_over_n_full = get_yTX_c(obj) ./ obj.n

    if !fullmodel_only
        # type of XtX_over_n could also be Int?
        # partition_to_suffstats = Dict{Vector{Int}, Tuple{LinearAlgebra.Diagonal{Float64, Vector{Float64}}, Vector{Float64}}}()

        # partition_to_suffstats = Dict{Vector{Int}, Tuple{Matrix{Float64}, Vector{Float64}}}()
        # partition_to_suffstats[partition] = (XtX_over_n_full, Xty_over_n_full)

        # could also use eltype(partition) instead of Int?
        partition_size_to_Q = Dict{Int, Matrix{Float64}}(length(obj.y_mean_by_group) => Q) # <- could also be Vector{Union{Nothing, ...}} ? probably no benefit

        seen = Set{Int}()
        sizehint!(seen, k)
    else
        # XtX_over_n, Xty_over_n = XtX_over_n_full, Xty_over_n_full
        partition_size = no_distinct_groups_in_partition(partition)
    end

    if M <: EnumerateThenSample
        enumerate_results = _enumerate(obj_full, Enumerate(), priors, verbose, threaded)
        enumerate_results_for_resampling = Distributions.sampler(Distributions.Categorical(enumerate_results.posterior_probs))
        partition_space = Matrix(PartitionSpace(k, DistinctPartitionSpace))
        sampling_stats = SamplingStats()
    else
        logbf10_dict = Dict{typeof(partition), Float64}()
        max_size_logbf10_dict = method.max_cache_size
        sampling_stats = SamplingStats()
    end

    same_count = 0
    stats_before_after = Vector{SamplingStats}(undef, 2)


    prog = ProgressMeter.Progress(iter; enabled = verbose, showspeed = true)
    for it in 1:iter

        if !fullmodel_only

            if M <: EnumerateThenSample
                partition_idx = rand(enumerate_results_for_resampling)
                partition .= @view partition_space[:, partition_idx]
            else

                sample_partition_rjmcmc_integrated!(partition, obj_full, logbf10_dict, method, sampling_stats, priors, threaded, max_size_logbf10_dict)

            end

            partition_size = no_distinct_groups_in_partition(partition)
            Q = getQ_rouder_fromObject!(partition_size_to_Q, partition_size)

            # could be cached, but is that really worth it?
            obj = apply_partition_to_suffstats(obj_full, partition, false)

        end

        θ_u_v = view(θ_u, range(; stop = partition_size - 1))
        θ_c_v = view(θ_c, range(; stop = partition_size))

        sample_conditional_θ_u_v!(θ_u_v, obj, Q, μ, σ², g)

        LinearAlgebra.mul!(θ_c_v, Q, θ_u_v)

        μ  = sample_conditional_μ(obj, σ², θ_c_v)
        σ² = sample_conditional_σ(obj, ss2_y_over_n, μ, θ_c_v, g)
        g  = sample_conditional_g(σ², θ_u_v)


        if !fullmodel_only

            for l in eachindex(partition)
                pl = partition[l]
                θ_cp[l] = θ_c[pl]
            end

            partition_samples[:, it] = partition
            # if it > 1
            #     if view(partition_samples, :, it - 1) == partition
            #         same_count += 1
            #         if same_count == 1
            #             stats_before_after[1] = deepcopy(sampling_stats)
            #         end
            #     else
            #         same_count = 0
            #     end
            #     max_same_count = 500
            #     if same_count > max_same_count
            #         stats_before_after[2] = deepcopy(sampling_stats)
            #         @show partition
            #         @show stats_before_after
            #         @show length(unique(eachcol(view(partition_samples, :, it - max_same_count:it))))
            #         error("same partition for over $max_same_count iterations!")
            #     end
            # end

            θ_cp_samples[:, it] = θ_cp

        end

        μ_samples[it]       = μ
        σ²_samples[it]      = σ²
        g_samples[it]       = g
        θ_u_samples[:, it]  = θ_u
        θ_s_samples[:, it]  = θ_c

        showvalues = if M <: EnumerateThenSample
            [
                (:it, it),
                (:acc_local,       sampling_stats.moved / sampling_stats.no_local_moves),
                (:acc_split_merge, sampling_stats.slit_merge_accepted / sampling_stats.no_split_merge_moves),
                (:cache_hits,      sampling_stats.no_cache_hits),
                (:no_cache_checks, sampling_stats.no_cache_checks),
                # (:cache_available, 1.0 - length(logbf10_dict) / max_size_logbf10_dict),
                (:same_count,      same_count)
            ]
        else
            [
                (:it, it),
                (:acc_local,       sampling_stats.moved / sampling_stats.no_local_moves),
                (:acc_split_merge, sampling_stats.slit_merge_accepted / sampling_stats.no_split_merge_moves),
                (:cache_hits,      sampling_stats.no_cache_hits),
                (:no_cache_checks, sampling_stats.no_cache_checks),
                (:cache_available, 1.0 - length(logbf10_dict) / max_size_logbf10_dict),
                (:same_count,      same_count)
            ]
        end
        ProgressMeter.next!(prog, showvalues = showvalues)

    end

    if fullmodel_only
        # TODO: why is this needed again?
        # double check what it does!]

        @views for l in eachindex(partition)
            θ_cp_samples[l, :] = θ_s_samples[partition[l], :]
        end

        if partition != eachindex(partition)

            # arbitrary rescaling so that sum(θ_cp) = 0
            @views for i in axes(θ_cp_samples, 2)
                s = StatsBase.mean(θ_cp_samples[:, i])
                μ_samples[i] += s
                θ_cp_samples[:, i] .-= s
            end
        end

        # seen = Set{Int}()
        # sizehint!(seen, k)
        # for l in eachindex(partition)
        #     if partition[l] in seen
        #         θ_cp_samples[l, :] .= θ_c[partition[l], :]
        #     else
        #         push!(seen, partition[l])
        #     end
        # end
        # θ_cs_samples .= θ_s_samples
    else
        # arbitrary rescaling so that sum(θ_cp) = 0
        @views for i in axes(θ_cp_samples, 2)
            s = StatsBase.mean(θ_cp_samples[:, i])
            μ_samples[i] += s
            θ_cp_samples[:, i] .-= s
        end
    end

    if M <: EnumerateThenSample

        return EnumerateThenSampleResult(
            enumerate_results,
            partition_samples,
            AnovaParameterSamples(μ_samples, σ²_samples, g_samples, θ_u_samples, θ_s_samples, θ_cp_samples)
        )
    else

        # should hold when max_cache_size is not reached
        # @assert length(logbf10_dict) <= max_cache_size || length(logbf10_dict) == sampling_stats.no_cache_checks - sampling_stats.no_cache_hits

        return RJMCMCResult(
            partition_samples,
            AnovaParameterSamples(μ_samples, σ²_samples, g_samples, θ_u_samples, θ_s_samples, θ_cp_samples)
        )
    end

end

function initialize_buffer_threaded_local_move(partition)

    probvec = Vector{Float64}(undef, length(partition))
    lookup  = zeros(eltype(partition), length(partition))

    was_computed = Vector{Bool}(undef, length(partition))

    counts = similar(partition)
    partition_cache = similar(partition, length(partition), length(partition))

    return (; probvec, lookup, was_computed, counts, partition_cache)
end

function initialize_buffer_local_move(partition)

    probvec = Vector{Float64}(undef, length(partition))
    lookup  = zeros(eltype(partition), length(partition))

    ρ′  = similar(partition)
    counts = similar(partition)

    return (; probvec, lookup, ρ′, counts)

end

function local_move!(
        partition::AbstractVector{<:Integer}, method, buffer_local_move, j::Integer,
        obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats
    )

    probvec, lookup, ρ′, counts = buffer_local_move

    fast_countmap_partition_incl_zero!(counts, partition)
    no_groups = count(!iszero, counts)
    maybe_add_one = isone(counts[partition[j]]) ? zero(no_groups) : one(no_groups)
    enumerate_to = no_groups + maybe_add_one

    no_cache_checks = enumerate_to
    no_cache_hits   = enumerate_to

    for l in 1:enumerate_to

        copyto!(ρ′, partition)
        ρ′[j] = l
        reduce_model_2_and_clear_lookup!(ρ′, lookup)

        if length(logbf10_dict) < max_size_logbf10_dict

            logml = get!(logbf10_dict, ρ′) do
                no_cache_hits -= 1
                # compute_one_bf(obj, ρ′′, method, priors)[1] + logpdf_model(priors.partition, ρ′′)
                compute_one_bf(obj, ρ′, method, priors)[1] + logpdf_model_distinct(priors.partition, ρ′)
            end
        else

            logml = get(logbf10_dict, ρ′) do
                no_cache_hits -= 1
                # compute_one_bf(obj, ρ′′, method, priors)[1] + logpdf_model(priors.partition, ρ′′)
                compute_one_bf(obj, ρ′, method, priors)[1] + logpdf_model_distinct(priors.partition, ρ′)
            end
        end

        probvec[l] = logml

    end

    probvec_v = view(probvec, 1:enumerate_to)
    probvec_v .= exp.(probvec_v .- LogExpFunctions.logsumexp(probvec_v))

    @assert Distributions.isprobvec(probvec_v) "probvec is not a valid probability vector"
    new_value = rand_categorical(partition, probvec_v)
    moved = partition[j] == new_value
    partition[j] = new_value

    reduce_model_2_and_clear_lookup!(partition, lookup)

    sampling_stats.moved           += moved
    sampling_stats.no_local_moves  += 1
    sampling_stats.no_cache_checks += no_cache_checks
    sampling_stats.no_cache_hits   += no_cache_hits

    return partition

end

function local_move_threaded!(
        partition::AbstractVector{<:Integer}, method, buffer_threaded_local_move, j::Integer,
        obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats
    )

    probvec, lookup, was_computed, counts, partition_cache = buffer_threaded_local_move

    fast_countmap_partition_incl_zero!(counts, partition)
    no_groups = count(!iszero, counts)
    maybe_add_one = isone(counts[partition[j]]) ? zero(no_groups) : one(no_groups)
    enumerate_to = no_groups + maybe_add_one
    sampling_stats.no_cache_checks += enumerate_to

    for l in 1:enumerate_to

        partition_proposal = view(partition_cache, :, l)
        copyto!(partition_proposal, partition)
        partition_proposal[j] = l
        reduce_model_2_and_clear_lookup!(partition_proposal, lookup)

        if haskey(logbf10_dict, partition_proposal)
            probvec[l] = logbf10_dict[partition_proposal]
            was_computed[l] = false
            sampling_stats.no_cache_hits += 1
        else
            was_computed[l] = true
        end
    end

    Threads.@threads for l in 1:enumerate_to

        if was_computed[l]
            partition_proposal = view(partition_cache, :, l)
            probvec[l] = compute_one_bf(obj, partition_proposal, method, priors)[1] + logpdf_model_distinct(priors.partition, partition_proposal)
        end

    end

    if length(logbf10_dict) < max_size_logbf10_dict
        for l in 1:enumerate_to
            if was_computed[l]
                partition_proposal = partition_cache[:, l] # intentionally not a view so logbf10_dict owns the key,
                if length(logbf10_dict) < max_size_logbf10_dict
                    logbf10_dict[partition_proposal] = probvec[l]
                end
            end
        end
    end

    probvec_v = view(probvec, 1:enumerate_to)
    probvec_v .= exp.(probvec_v .- LogExpFunctions.logsumexp(probvec_v))

    @assert Distributions.isprobvec(probvec_v) "probvec is not a valid probability vector"
    new_value = rand_categorical(partition, probvec_v)
    sampling_stats.moved += partition[j] == new_value
    partition[j] = new_value

    reduce_model_2_and_clear_lookup!(partition, lookup)

    sampling_stats.no_local_moves  += 1

    return partition

end

# function initialize_buffer_split_merge_move(partition::AbstractVector{T}) where T <: Integer

#     proposal = similar(partition)
#     lookup = zeros(T, length(partition))

#     return (; proposal, lookup)
# end

#=
function random_pair(j)
    until = binomial(j, 2)
    i0 = rand(1:until)
    count = 0
    for i1 in 1:j
        for i2 in i1+1:j
            count += 1
            if count == i0
                return i1, i2, log(until)
            end
        end
    end
    # cannot happen, but makes the code type stable
    return 0, 0, log(until)
end
function make_merge_proposal!(proposal, current, lookup)

    j = maximum(current)
    i1, i2 = random_pair(j)

    copyto!(proposal, current)
    for i in eachindex(proposal)
        if proposal[i] == i2
            proposal[i] = i1
        end
    end
    reduce_model_2_and_clear_lookup!(proposal, lookup)

end
function log_prob_merge_proposal(current)
    j = maximum(current)
    return -SpecialFunctions.logabsbinomial(j, 2)[1]
end
function make_split_proposal!(proposal, current, lookup)

    # select a partition with more than 1 element
    tb = fast_countmap_partition_incl_zero!(lookup, current)
    j = count(>(one(eltype(tb))), tb)
    i0 = rand(1:j)
    i1 = 0
    for (i, c) in enumerate(tb)
        if c > one(c)
            i1 += 1
        end
        if i1 == i0
            i1 = i
            break
        end
    end

    nj = count(==(i1), current)

    # not sure how to avoid allocating these three lines
    logprobvec = [i == 2 ? 0.0 : -Inf for i in 1:nj]
    d_subproposal = CustomInclusionPartitionDistribution(logprobvec)
    subproposal = rand(d_subproposal)
    # proposal = copy(current)
    # newvalues = [i1, length(proposal)]
    # proposal[proposal .== i1] .= newvalues[subproposal]

    j = 1
    copyto!(proposal, current)
    for i in eachindex(proposal)
        if proposal[i] == i1
            proposal[i] = subproposal[j] == 1 ? i1 : length(proposal)
            j += 1
        end
    end

    fill!(lookup, zero(eltype(lookup)))
    reduce_model_2_and_clear_lookup!(proposal, lookup)

end
function log_prob_split_proposal(current, proposal, lookup)

    tb = fast_countmap_partition_incl_zero!(lookup, current)
    j = count(>(one(eltype(tb))), tb)
    logprob1 = -log(j)

    i0 = findfirst(i -> current[i] != proposal[i], eachindex(current))
    i1 = current[i0]
    nj = count(==(i1), current)

    # again not sure how to avoid these allocations
    subproposal = proposal[current .== i1]
    logprobvec = [i == 2 ? 0.0 : -Inf for i in 1:nj]
    d_subproposal = CustomInclusionPartitionDistribution(logprobvec)

    # TODO: this is actually just 0.0 - log(stirling number of the 2nd kind), no?
    logprob2 = Distributions.logpdf(d_subproposal, EqualitySampler.reduce_model_dct(subproposal))

    fill!(lookup, zero(eltype(lookup)))
    return logprob1 + logprob2

end

function split_merge_move!(current, method, buffer_split_merge_move, obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats)

    proposal, lookup = buffer_split_merge_move
    logtwo = IrrationalConstants.logtwo
    if (allunique(current) || (!allequal(current) && rand() < .5)) # merge

        # @show "merge1", current, proposal, lookup
        make_merge_proposal!(proposal, current, lookup)
        # @show "merge2", current, proposal, lookup
        log_prob_merge = log_prob_merge_proposal(current)                   - (logtwo * !allunique(current))
        # @show "merge3", current, proposal, lookup
        log_prob_split = log_prob_split_proposal(proposal, current, lookup) - (logtwo * !allequal(proposal))
        # @show "merge4", current, proposal, lookup
        log_proposal_ratio = log_prob_split - log_prob_merge

    else

        # @show "split1", current, proposal, lookup
        make_split_proposal!(proposal, current, lookup)
        # @show "split2", current, proposal, lookup
        log_prob_split = log_prob_split_proposal(current, proposal, lookup) - (logtwo * !allequal(current))
        # @show "split3", current, proposal, lookup
        log_prob_merge = log_prob_merge_proposal(proposal)                  - (logtwo * !allunique(proposal))
        # @show "split4", current, proposal, lookup
        log_proposal_ratio = log_prob_merge - log_prob_split

    end

    if haskey(logbf10_dict, current)
        sampling_stats.no_cache_hits += 1
        log_denominator = logbf10_dict[current]
    else
        log_denominator = compute_one_bf(obj, current, method, priors)[1] + logpdf_model_distinct(priors.partition, current)
        length(logbf10_dict) < max_size_logbf10_dict && (logbf10_dict[current] = log_denominator)
    end

    if haskey(logbf10_dict, proposal)
        sampling_stats.no_cache_hits += 1
        log_numerator = logbf10_dict[proposal]
    else
        log_numerator = compute_one_bf(obj, proposal, method, priors)[1] + logpdf_model_distinct(priors.partition, proposal)
        length(logbf10_dict) < max_size_logbf10_dict && (logbf10_dict[proposal] = log_numerator)
    end
    sampling_stats.no_cache_checks += 2

    log_posterior_ratio = log_numerator - log_denominator
    # -randexp()?
    accept = log(rand()) < log_posterior_ratio + log_proposal_ratio
    accept && copyto!(current, proposal)

    sampling_stats.slit_merge_accepted += accept
    sampling_stats.no_split_merge_moves += 1

    return current
end
=#

function sample_partition_rjmcmc_integrated!(partition::AbstractVector{U}, obj, logbf10_dict::Dict{Vector{U}, T},
    method::Union{SampleRJMCMC, SampleIntegrated}, sampling_stats, priors::AbstractPriors, threaded::Bool, max_size_logbf10_dict::Integer) where {T, U}

    split_merge_prob = method.split_merge_prob
    buffer_split_merge_move = initialize_buffer_split_merge_move(partition)

    if threaded
        buffer_threaded_local_move = initialize_buffer_threaded_local_move(partition)
        probvec = buffer_threaded_local_move.probvec
    else
        buffer_local_move = initialize_buffer_local_move(partition)
        probvec = buffer_local_move.probvec
    end

    for j in eachindex(partition)

        if !iszero(split_merge_prob) && rand() < split_merge_prob
            split_merge_move!(       partition, method, buffer_split_merge_move,       obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats)
        else
            if threaded
                local_move_threaded!(partition, method, buffer_threaded_local_move, j, obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats)
            else
                local_move!(         partition, method, buffer_local_move,          j, obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats)
            end
        end
    end

#=
    if threaded

        probvec = Vector{Float64}(undef, length(partition))
        lookup  = zeros(eltype(partition), length(partition))

        was_computed = Vector{Bool}(undef, length(partition))
        no_cache_checks = 0
        no_cache_misses = 0

        counts = similar(partition)
        partition_cache = similar(partition, length(partition), length(partition))

        for j in eachindex(partition)

            fast_countmap_partition_incl_zero!(counts, partition)
            no_groups = sum(!iszero, counts)
            maybe_add_one = isone(counts[partition[j]]) ? zero(no_groups) : one(no_groups)
            enumerate_to = no_groups + maybe_add_one

            # enumerate_to = current_no_groups + 1 - isone(tb[ρ[j]])
            # enumerate_to = length(partition)

            no_cache_checks += enumerate_to

            for l in 1:enumerate_to

                partition_proposal = view(partition_cache, :, l)
                copyto!(partition_proposal, partition)
                partition_proposal[j] = l
                reduce_model_2!(partition_proposal, lookup)
                fill!(lookup, zero(eltype(lookup)))

                if haskey(logbf10_dict, partition_proposal)
                    probvec[l] = logbf10_dict[partition_proposal]
                    was_computed[l] = false
                else
                    was_computed[l] = true
                    no_cache_misses += 1
                end
            end

            Threads.@threads for l in 1:enumerate_to

                if was_computed[l]
                    partition_proposal = view(partition_cache, :, l)
                    probvec[l] = compute_one_bf(obj, partition_proposal, method, priors)[1] + logpdf_model_distinct(priors.partition, partition_proposal)
                end

            end

            if length(logbf10_dict) < max_size_logbf10_dict
                for l in 1:enumerate_to
                    if was_computed[l]
                        no_cache_misses += 1
                        partition_proposal = partition_cache[:, l] # intentionally not a view so logbf10_dict owns the key,
                        if length(logbf10_dict) < max_size_logbf10_dict
                            logbf10_dict[partition_proposal] = probvec[l]
                        end
                    end
                end
            end

            # ρ′  = similar(partition)
            # probvec2 = similar(probvec)
            # for l in eachindex(partition)


            #     ρ′ = copy(partition)
            #     ρ′[j] = l
            #     reduce_model_2!(ρ′, lookup)#, j)
            #     fill!(lookup, zero(eltype(lookup)))

            #     logml = compute_one_bf(obj, ρ′, method, priors)[1] + logpdf_model(priors.partition, ρ′)

            #     probvec2[l] = logml
            # end

            # @assert probvec2 ≈ probvec

            # could use Gumbel trick here?
            # probvec .= exp.(probvec .- LogExpFunctions.logsumexp(probvec))
            # partition[j] = Random.rand(Distributions.Categorical(probvec))
            # partition[j] = rand_categorical(partition, probvec)

            # reduce_model_2!(partition, lookup)
            # fill!(lookup, zero(eltype(lookup)))

            probvec_v = view(probvec, 1:enumerate_to)
            probvec_v .= exp.(probvec_v .- LogExpFunctions.logsumexp(probvec_v))

            @assert Distributions.isprobvec(probvec_v) "probvec is not a valid probability vector"
            partition[j] = rand_categorical(partition, probvec_v)

            reduce_model_2!(partition, lookup)
            fill!(lookup, zero(eltype(lookup)))

        end
        no_cache_hits[] += no_cache_checks - no_cache_misses

    else

        probvec = Vector{Float64}(undef, length(partition))
        lookup  = zeros(eltype(partition), length(partition))

        ρ′  = similar(partition)
        counts = similar(partition)

        no_cache_misses = 0


        for j in eachindex(partition)

            fast_countmap_partition_incl_zero!(counts, partition)
            no_groups = sum(!iszero, counts)
            maybe_add_one = isone(counts[partition[j]]) ? zero(no_groups) : one(no_groups)
            enumerate_to = no_groups + maybe_add_one
            # copyto!(ρ′, ρ)
            # for l in eachindex(partition)
            for l in 1:enumerate_to

                # ρ′[j] = l
                # copyto!(ρ′′, ρ′)
                # reduce_model_2!(ρ′′, lookup)
                # fill!(lookup, zero(eltype(lookup)))

                ρ′ = copy(partition)
                ρ′[j] = l
                reduce_model_2!(ρ′, lookup)#, j)
                fill!(lookup, zero(eltype(lookup)))

                if length(logbf10_dict) < max_size_logbf10_dict

                    logml = get!(logbf10_dict, ρ′) do
                        no_cache_misses += 1
                        # compute_one_bf(obj, ρ′′, method, priors)[1] + logpdf_model(priors.partition, ρ′′)
                        compute_one_bf(obj, ρ′, method, priors)[1] + logpdf_model_distinct(priors.partition, ρ′)
                    end
                else

                    logml = get(logbf10_dict, ρ′) do
                        no_cache_misses += 1
                        # compute_one_bf(obj, ρ′′, method, priors)[1] + logpdf_model(priors.partition, ρ′′)
                        compute_one_bf(obj, ρ′, method, priors)[1] + logpdf_model_distinct(priors.partition, ρ′)
                    end
                end

                probvec[l] = logml

            end

            # could use Gumbel trick here?
            # probvec .= exp.(probvec .- LogExpFunctions.logsumexp(probvec))
            # probvec_v = probvec

            probvec_v = view(probvec, 1:enumerate_to)
            probvec_v .= exp.(probvec_v .- LogExpFunctions.logsumexp(probvec_v))

            @assert Distributions.isprobvec(probvec_v) "probvec is not a valid probability vector"
            partition[j] = rand_categorical(partition, probvec_v)

            reduce_model_2!(partition, lookup)
            fill!(lookup, zero(eltype(lookup)))
        end
        no_cache_hits[] += abs2(length(partition)) - no_cache_misses
    end
=#
end

function rand_categorical(::AbstractVector{U}, probvec::T) where {T, U<:Integer}
    rand(Distributions.DiscreteNonParametric{U, Float64, Base.OneTo{U}, T}(
        Base.OneTo{U}(length(probvec)), probvec; check_args = false
    ))
end

# not sure if this is specific to ANOVA
function instantiate_results(::AbstractSuffstats, ::Enumerate, nmodels::Integer, k::Integer)
    # TODO: probably redundant to have both the log and non-log versions?
    logml                   = Vector{Float64}(undef, nmodels)
    bfs                     = Vector{Float64}(undef, nmodels)
    logbfs                  = Vector{Float64}(undef, nmodels)
    err                     = Vector{Float64}(undef, nmodels)
    log_prior_model_probs   = Vector{Float64}(undef, nmodels)
    log_posterior_probs     = Vector{Float64}(undef, nmodels)
    posterior_probs         = Vector{Float64}(undef, nmodels)
    return EnumerateResult(k, logml, bfs, logbfs, err, log_prior_model_probs, log_posterior_probs, posterior_probs)
    # return (; logml, bfs, logbfs, err, log_prior_model_probs, log_posterior_probs, posterior_probs)
end

function _enumerate_internal!(result_object::EnumerateResult, obj::AbstractSuffstats, logml_h0::Number, modelspace, verbose::Bool, method::Enumerate, priors::AbstractPriors)

    ProgressMeter.@showprogress enabled = verbose for (i, partition) in enumerate(modelspace)

        result_object.log_prior_model_probs[i] = logpdf_model_distinct(priors.partition, partition)
        result_object.logbfs[i], result_object.err[i] = compute_one_bf(obj, partition, method, priors)
        result_object.bfs[i] = exp(result_object.bfs[i])
        result_object.logml[i]  = result_object.logbfs[i] - logml_h0

    end
end

function _enumerate_internal_threaded!(result_object::EnumerateResult, obj::AbstractSuffstats, logml_h0::Number, modelspace, verbose::Bool, method::Enumerate, priors::AbstractPriors)

    # enumerating a matrix might be better.
    modelspace_c = collect(enumerate(modelspace))
    prog = ProgressMeter.Progress(length(modelspace_c), enabled = verbose)

    Threads.@threads for (i, partition) in modelspace_c

        result_object.log_prior_model_probs[i] = logpdf_model_distinct(priors.partition, partition)
        result_object.logbfs[i], result_object.err[i] = compute_one_bf(obj, partition, method, priors)
        result_object.bfs[i] = exp(result_object.bfs[i])
        result_object.logml[i]  = result_object.logbfs[i] - logml_h0

        ProgressMeter.next!(prog)
    end
end

function getQ_Rouder(n_groups::Integer)::Matrix{Float64}
    # X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))
    Σₐ = Matrix{Float64}(LinearAlgebra.I, n_groups, n_groups) .- (1.0 / n_groups)
    _, v::Matrix{Float64} = LinearAlgebra.eigen(Σₐ)
    Q = v[end:-1:1, end:-1:2] # this is what happens in Rouder et al., (2012) eq ...

    @assert isapprox(sum(Q * randn(n_groups-1)), 0.0, atol = 1e-8)

    return Q
end

getQ_rouder_fromObject!(::Nothing, size::Integer) = getQ_Rouder(size)
function getQ_rouder_fromObject!(d::AbstractDict, size::Integer)
    get!(d, size) do
        getQ_Rouder(size)
    end
end
getQ_rouder_fromObject!(d::AbstractVector, size::Integer) = d[size]


function precompute_integrated_log_lik4(obj, partition, partition_size_to_Q)

    obj_p = apply_partition_to_suffstats(obj, partition)

    Q = getQ_rouder_fromObject!(partition_size_to_Q, length(obj_p.n_by_group))

    ỹTỹ = obj.y_var * obj.n
    ỹTX̃ = (obj_p.y_mean_by_group .* obj_p.n_by_group)' * Q - obj_p.y_mean * obj_p.n_by_group' * Q
    # NOTE to self, this is actually a rank-1 downdate to the eigendecomposition given by Q' * LA.Diagonal(obj_p.n_by_group) * Q
    X̃TX̃ = Q' * (LinearAlgebra.Diagonal(obj_p.n_by_group) - obj_p.n_by_group  * obj_p.n_by_group' / obj_p.n) * Q

    λₓ, Qₓ = LinearAlgebra.eigen(LinearAlgebra.Symmetric(X̃TX̃))

    ỹTX̃Qₓ = ỹTX̃ * Qₓ
    return ỹTỹ, λₓ, ỹTX̃Qₓ

end

function log_integrand_bf4(g::AbstractFloat, rscale, N, ỹTỹ, λₓ, ỹTX̃Qₓ)

    log_det_G = length(λₓ) * log(g)
    log_det_Vg = sum(λᵢ->log(λᵢ + 1 / g), λₓ)

    num_ha = ỹTỹ - sum(i-> abs2(ỹTX̃Qₓ[i]) / (λₓ[i] + 1 / g), eachindex(λₓ))
    log_integrand = -1 / 2 * (log_det_G + log_det_Vg) + (N - 1) / 2 * (log(ỹTỹ) - log(num_ha))

    return log_integrand + Distributions.logpdf(Distributions.InverseGamma(rscale / 2, 1 / 2), g)
end

# function log_integrand_bf4(gBrob::Brobdingnag.Brob, rscale, N, ỹTỹ, λₓ, ỹTX̃Qₓ)
#     g = Brobdingnag.asFloat(gBrob)
#     result = log_integrand_bf4(g, rscale, N, ỹTỹ, λₓ, ỹTX̃Qₓ)
#     return Brobdingnag.Brob(result)
# end

function compute_bf4(obj, partition, method::AbstractSamplingMethod{T}, partition_size_to_Q::Union{Nothing, AbstractDict, AbstractVector} = nothing) where T

    all(==(first(partition)), partition) && return 1.0, 0.0

    rscale = method.rscale

    ỹTỹ, λ, ỹTX̃Qₓ = precompute_integrated_log_lik4(obj, partition, partition_size_to_Q)
    N = obj.n

    value, error = QuadGK.quadgk(
        g -> log_integrand_bf4(g, rscale, N, ỹTỹ, λ, ỹTX̃Qₓ),
        zero(T), T(Inf))
    return value::T, error#::T

end

function precompute_integrated_log_lik5(obj, partition)

    obj_p = apply_partition_to_suffstats(obj, partition, false)

    ỹTỹ = obj.y_var * obj.n

    k = length(obj_p.n_by_group)

    v2 = (obj_p.y_mean_by_group .- obj_p.y_mean) .* obj_p.n_by_group
    u2 = (obj_p.n_by_group .- obj_p.n / k) ./ sqrt(obj_p.n)

    B_inv2_diag = similar(obj_p.y_mean_by_group)

    return ỹTỹ, log(ỹTỹ), v2, u2, obj_p.n_by_group, B_inv2_diag

end

function log_integrand_bf5(g::AbstractFloat, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag, mode = zero(g))
    return exp(log_integrand_bf5_inner(g, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag) - mode)
end

function log_integrand_bf5_inner(g::AbstractFloat, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag)

    k = length(n_by_group)

    B_inv2_diag .= (1 .+ g .* n_by_group)
    m = StatsBase.mean(B_inv2_diag)
    log_trace_B = k * (log(m / g)) + log(prod(x-> x / m, B_inv2_diag))

    B_inv2_diag .= g ./ B_inv2_diag


    # A_diag .= n_by_group .+ inv(g)
    # B_inv2_diag .= (g ./ (1 .+ g .* n_by_group)) # old approach
    B = LinearAlgebra.Diagonal(B_inv2_diag)

    temp0 = 1 / k
    temp1 = StatsBase.mean(B_inv2_diag)
    temp3 = LinearAlgebra.dot(u2, B_inv2_diag)
    temp4 = abs2(sum(u2) - temp3) / (k * temp1)


    log_det_G = (k - 1) * log(g)

    # log_det_Vg = sum(λᵢ->log(λᵢ + 1 / g), λₓ)

    # there are ways to optimize this by doing log(prod(B_inv2_diag)) while taking care of underflow and overflow
    # but it's not starightforward. This is the most expensive part of the computation though
    # log_trace_B = -sum(log, B_inv2_diag) # old approach

    # log_det_Vg = LogExpFunctions.logsumexp(log_trace_A - log(A_diag[i]) for i in eachindex(A_diag)) - log(k) +
    #     log1p(temp4 - LinearAlgebra.dot(u2, LinearAlgebra.Diagonal(B_inv2_diag), u2))

    # log_det_Vg = log_trace_A + log(sum(inv, A_diag)) - log(k) +
    #     log1p(temp4 - LinearAlgebra.dot(u2, LinearAlgebra.Diagonal(B_inv2_diag), u2))

    log_det_Vg = log_trace_B + log(sum(B_inv2_diag) / k)# - log(k)

    # reuse A_diag to factor out some common linear algebra operations
    # A_diag .= B_inv2_diag .* u2
    # temp5a = LinearAlgebra.dot(u2, A_diag)
    temp5a = LinearAlgebra.dot(u2, B, u2)

    log_det_Vg += log1p(temp4 - temp5a)


    temp5 = (1 - temp5a + temp4)
    temp6 = LinearAlgebra.dot(v2, B_inv2_diag)
    # temp7 = LinearAlgebra.dot(v2, A_diag)
    temp7 = LinearAlgebra.dot(v2, B, u2)

    # reuse A_diag to factor out some common linear algebra operations
    # A_diag .= B_inv2_diag .* v2

    # dot(ỹTX̃, LinearAlgebra.inv(Vg), ỹTX̃')
    dot_ỹTX̃_inv_Vg_ỹTX̃ = LinearAlgebra.dot(v2, B, v2) - #LinearAlgebra.dot(v2, A_diag) -
        abs2(temp6) * temp0 / temp1 +
        abs2(temp7 - temp6 * temp3 * temp0 / temp1) / temp5

    num_ha = ỹTỹ - dot_ỹTX̃_inv_Vg_ỹTX̃

    log_integrand = -1 / 2 * (log_det_G + log_det_Vg) + (N - 1) / 2 * (log_ỹTỹ - log(num_ha))

    log_result = log_integrand + Distributions.logpdf(Distributions.InverseGamma(rscale / 2, 1 / 2), g)
    return log_result
end

# function log_integrand_bf5(gBrob::Brobdingnag.Brob, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag, mode)
#     g = Brobdingnag.asFloat(gBrob)
#     mmode = Brobdingnag.asFloat(mode)
#     result = log_integrand_bf5_inner(g, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag) - mmode
#     return Brobdingnag.Brob(result)
# end

function log_integrand_bf5(gBrob::T, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag, mode) where T <: Union{LogarithmicNumbers.ULogarithmic, LogarithmicNumbers.Logarithmic}
    g = float(gBrob)
    result = log_integrand_bf5_inner(g, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag) - float(mode)
    return exp(T, result)
end

function get_mode_5(::Type{Float64}, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag)
    # return 0.0

    result = Optim.optimize(
        # limiting cases
        g -> begin
            if iszero(g)
                retval = Inf
            elseif isinf(g)
                retval = Inf
            else
                retval = -log_integrand_bf5_inner(g, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag)
            end
            # @show g, retval
            return retval
        end,
        # TODO: better maximum? perhaps just do change of variables to [0, 1)?
        0.0, 1e6, rel_tol = .1, abs_tol = .1
    )
    # optim_res.minimizer,
    -result.minimum
end

function get_mode_5(::Type{T}, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag) where T
    zero(T)
end

function compute_one_bf(obj::SuffstatsANOVA, partition, ::AbstractSamplingMethod{T}, priors::AnovaPriors) where T

    all(==(first(partition)), partition) && return zero(T), zero(T)

    rscale = priors.rscale

    ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag = precompute_integrated_log_lik5(obj, partition)
    N = obj.n

    mode = get_mode_5(T, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag)

    # try
        value, error = QuadGK.quadgk(
            g -> log_integrand_bf5(g, rscale, N, ỹTỹ, log_ỹTỹ, v2, u2, n_by_group, B_inv2_diag, mode),
            zero(T), T(Inf)
        )
    # catch e
    #     @show obj, partition, rscale
    #     @show e
    #     throw(e)
    # end

    # @show partition, mode, value, error
    new_log_value = mode + log(value)

    return new_log_value, error#::T
    # return value::T, error#::T

end


function logbf_to_logposteriorprob!(out::AbstractVector, logbf::AbstractVector, logprior_model_probs::AbstractVector)
    logsumbfs = LogExpFunctions.logsumexp(logbf .+ logprior_model_probs)
    out .= logbf .+ logprior_model_probs .- logsumbfs
    out
end

function logbf_to_logposteriorprob(logbf::AbstractVector, logprior_model_probs::AbstractVector)
    logbf_to_logposteriorprob!(similar(logbf), logbf, logprior_model_probs)
end

"""

```
_compute_quad_form(y, X, μ, θ_u)
_compute_quad_form(obj, μ, θ_c)
_compute_quad_form_c(obj, μ, θ_c)
```

Computes
```
(y - μ - X_c * Q * θ_u)' * (y - μ - X_c * Q * θ_u)
```
where `y` are the data, `μ` is the grand mean, `X` is the design matrix, `θ_u` is the unconstrained parameter vector, and `θ_c` is the parameter vector with sum to zero constraint.
`_compute_quad_form_c` assumes that the design matrix is centered.
"""
function _compute_quad_form(y, X_c, μ, θ_u)
    n = length(y)
    Q = getQ_Rouder(size(X_c, 2))
    (y - ones(n) * μ - X_c * Q * θ_u)' * (y - ones(n) * μ - X_c * Q * θ_u)
end

function _compute_quad_form(obj, μ, θ_c)

    ss2_y_over_n = obj.y_var + abs2(obj.y_mean)

    obj.n * (
        ss2_y_over_n + abs2(μ) + LinearAlgebra.dot(θ_c, get_XTX_over_n(obj), θ_c) +
        -2 * (
            LinearAlgebra.dot(get_XTy_over_n(obj), θ_c) +
            μ * (
                obj.y_mean -
                LinearAlgebra.dot(LinearAlgebra.diag(get_XTX_over_n(obj)), θ_c)
            )
        )
    )
end

function _compute_quad_form_c(obj, μ, θ_c)
    Xty_over_n = get_XTy_c_over_n(obj)
    XtX_over_n = get_XTX_c_over_n(obj)
    ss2_y_over_n = obj.y_var + abs2(obj.y_mean)

    obj.n * (ss2_y_over_n + abs2(μ) + LinearAlgebra.dot(θ_c, XtX_over_n, θ_c) - 2 * (obj.y_mean * μ + LinearAlgebra.dot(Xty_over_n, θ_c)))
end
# TODO: create unit test for the two functions above!
# _compute_quad_form(y, design_mat_c, θ_u) ≈ _compute_quad_form(ss2_y_over_n, XtX_over_n2, Xty_over_n2, μ, z)

sample_conditional_σ(obj, ss2_y_over_n, μ, θ_c, g) = rand(conditional_σ(obj, ss2_y_over_n, μ, θ_c, g))
function conditional_σ(obj, ss2_y_over_n, μ, θ_c, g)

    p = length(θ_c) - 1
    n = obj.n

    α = (n + p + 1) / 2
    θ = (
        sum(abs2, θ_c) / g + # sum(abs2, θ_c) == sum(abs2, θ_u)
        _compute_quad_form(obj, μ, θ_c)
        #=(
            n * (ss2_y_over_n - 2 * y_mean * μ - 2 * LinearAlgebra.dot(Xty_over_n, z) + abs2(μ)) +
            2 * μ * LinearAlgebra.diag(XtX_over_n)' * z +
            LinearAlgebra.dot(z, XtX_over_n, z)
        )=#
    ) / 2

    return Distributions.InverseGamma(α, θ)
end


sample_conditional_μ(obj, σ², θ_c) = rand(conditional_μ(obj, σ², θ_c))
function conditional_μ(obj, σ², θ_c)
    # NO the second term is not zero by construction, that only holds when all group have equal sample size
    # the use of XtX_over_n is also only valid because there is no ovelap among columns of X (X[i, j] == 1 => X[i, 1:end .!= j] == 0)

    # diag_XtX_over_n = LinearAlgebra.diag(XtX_over_n)
    # offset = sum(i->diag_XtX_over_n[i] / n * θ_s[i], eachindex(θ_s)) # basically dot(θ_s, diag(XtX_over_n)) / n but divide by n first for stability

    offset = LinearAlgebra.dot(obj.n_by_group ./ obj.n, θ_c)

    mean_μ = obj.y_mean - offset# - mean_x' * partition_mat * Q * θ_u # zero by construction # TODO: does that hold in full generality?
    sd_μ = sqrt(σ² / obj.n)
    return Distributions.Normal(mean_μ, sd_μ)
end

sample_conditional_g(σ², θ_u) = rand(conditional_g(σ², θ_u))
function conditional_g(σ², θ_u)

    #TODO: rscale should be used here!

    α = (1 + length(θ_u)) / 2
    θ = 1 / 2 + LinearAlgebra.dot(θ_u, θ_u) / 2σ²
    D_g = Distributions.InverseGamma(α, θ)
    # mean(D_g), g, mean(D_g) - g
    return D_g

end

sample_conditional_θ_u_v!(θ_u_v, obj, Q, μ, σ², g) = Random.rand!(conditional_θ_u_v(obj, Q, μ, σ², g), θ_u_v)
sample_conditional_θ_u_v(obj, Q, μ, σ², g) = rand(conditional_θ_u_v(obj, Q, μ, σ², g))

function conditional_θ_u_v(obj, Q, μ, σ², g)#partition, partition_to_suffstats, partition_size_to_Q)

    # Float64[] is a questionable default
    isone(length(obj.n_by_group)) && return Distributions.MvNormal(Float64[], Matrix{Float64}(undef, 0, 0))

    # probably better is to cache the eigendecomposition of Q' * XtX_over_n * Q
    # right now we're still doing O(p^3) stuff each iteration

    # XtX_over_n, Xty_over_n = partition_to_suffstats[partition]
    # Q = partition_size_to_Q[size(XtX_over_n, 1)]

    # XtX_over_n = get_XTX_c(obj)
    # Xty_over_n = get_yTX_c(obj)

    n = obj.n
    XtX_over_n = get_XTX(obj)
    Xty_over_n = get_XTy_over_n(obj)

    precmat       = 1 / 2σ² * LinearAlgebra.Symmetric(Q' * XtX_over_n * Q + LinearAlgebra.I / g)
    # this works
    # Symmetric(Q' * XtX_over_n * Q + I / g) ≈ Q' * Symmetric(XtX_over_n + I / g) * Q
    # but this can't be simplified further because Q is not invertible
    # inv(Q' * XtX_over_n * Q + I / g) ≈ pinv(Q) * inv(XtX_over_n + I / g) * pinv(Q)'
    # potentialvec  = 1 / 2σ² * (X' * temp)
    # note: the above is not entirely correct, this function can be done in O(p^2) time, but it takes some effort
    # this approach is easy to verify
    potentialvec  = 1 / 2σ² * n * Q' * (Xty_over_n - μ * (obj.n_by_group ./ n))
    covmat        = inv(precmat)
    meanvec       = covmat * potentialvec


    return Distributions.MvNormal(meanvec, covmat)
    # Random.rand!(D_θ_u_v, θ_u_v)

end

#=
function sample_conditional_σ(y_mean, ss2_y_over_n, n, μ, θ_u_v, g, XtX_over_n, Xty_over_n, Q)

    p = length(θ_u_v)

    α = (n + p + 1) / 2

    #=
    θ ≈ (
        (y - μ * ones(n) - design_mat * Q * θ_u)' * (y - μ * ones(n) - design_mat * Q * θ_u) +
        LinearAlgebra.dot(θ_u_v, θ_u_v) / g
        ) / 2
    =#

    # ⋅ === LinearAlgebra.dot
    z = Q * θ_u_v
    # θ = (
    #     n * ss2_y_over_n + abs2(μ) + -2 * (
    #         μ * y_mean + Xty_over_n ⋅ z +
    #         μ * LinearAlgebra.diag(XtX_over_n)' ⋅ z
    #     ) + LinearAlgebra.dot(z, XtX_over_n, z) +
    #     (θ_u_v ⋅ θ_u_v) / g
    # ) / 2

    θ = (
        LinearAlgebra.dot(θ_u_v, θ_u_v) / g +
        (
            n * (ss2_y_over_n - 2 * y_mean * μ - 2 * LinearAlgebra.dot(Xty_over_n, z) + abs2(μ)) +
            2 * μ * LinearAlgebra.diag(XtX_over_n)' * z +
            LinearAlgebra.dot(z, XtX_over_n, z)
        )
    ) / 2

    # θ = (
    #     n * (
    #         ss2_y_over_n + abs2(μ) -2 * (θ_u_v' * Q' * Xty_over_n + mean_y * μ)
    #     ) + LinearAlgebra.dot(Q * θ_u_v, XtX_over_n, Q * θ_u_v) +
    #     LinearAlgebra.dot(θ_u_v, θ_u_v) / g
    # ) / 2

    # if θ < zero(θ)


    #     θ_big = (
    #         big(n) * (
    #             big(ss2_y_over_n) + abs2(big(μ)) -2 * (big.(θ_u_v)' * big.(Q)' * big.(Xty_over_n) + big(mean_y) * big(μ))
    #         ) + LinearAlgebra.dot(big.(Q) * big.(θ_u_v), big.(XtX_over_n), big.(Q) * big.(θ_u_v)) +
    #         LinearAlgebra.dot(big.(θ_u_v), big.(θ_u_v)) / big(g)
    #     ) / 2

    #     # @show mean_y, ss2_y_over_n, n, μ, θ_u_v, g, XtX_over_n, Xty_over_n, Q
    #     @show θ_big
    # end
    D_σ² = Distributions.InverseGamma(α, θ)
    # mean(D_σ²), σ², mean(D_σ²) - σ²
    # σ² * (α - 1)
    return rand(D_σ²)
end

function sample_conditional_μ(mean_y, σ², n, #=y, design_mat, Q, σ², θ_u, g, partition_mat=#)

    # complete the square
    # temp = (y .- design_mat * partition_mat * Q * θ_u)
    # a = length(temp)
    # b = -2sum(temp)
    # c = temp' * temp
    # d = b / 2a
    # e = c - b^2 / 4a
    # a * μ^2 + b * μ + c ≈ a * (μ + d)^2 + e

    # mean_μ = -d
    # sd_μ = sqrt(σ²) * sqrt(1 / a)

    mean_μ = mean_y# - mean_x' * partition_mat * Q * θ_u # zero by construction # TODO: does that hold in full generality?
    sd_μ = sqrt(σ²) * sqrt(1 / n)
    D_μ = Distributions.Normal(mean_μ, sd_μ)
    # mean(D_μ), μ, mean(D_μ) - μ
    return rand(D_μ)

end

function sample_conditional_g(σ², θ_u)

    α = (1 + length(θ_u)) / 2
    θ = 1 / 2 + LinearAlgebra.dot(θ_u, θ_u) / 2σ²
    D_g = Distributions.InverseGamma(α, θ)
    # mean(D_g), g, mean(D_g) - g
    return rand(D_g)

end

function sample_conditional_θ_u_v!(θ_u_v, n, σ², g, XtX_over_n, Xty_over_n, Q)#partition, partition_to_suffstats, partition_size_to_Q)

    # within model move for θ_u_v

    isempty(θ_u_v) && return

    # probably better is to cache the eigendecomposition of Q' * XtX_over_n * Q
    # right now we're still doing O(p^3) stuff each iteration

    # XtX_over_n, Xty_over_n = partition_to_suffstats[partition]
    # Q = partition_size_to_Q[size(XtX_over_n, 1)]

    precmat       = 1 / 2σ² * LinearAlgebra.Symmetric(Q' * XtX_over_n * Q + LinearAlgebra.I / g)
    # this works
    # Symmetric(Q' * XtX_over_n * Q + I / g) ≈ Q' * Symmetric(XtX_over_n + I / g) * Q
    # but this can't be simplified further because Q is not invertible
    # inv(Q' * XtX_over_n * Q + I / g) ≈ pinv(Q) * inv(XtX_over_n + I / g) * pinv(Q)'
    # potentialvec  = 1 / 2σ² * (X' * temp)
    # note: the above is not entirely correct, this function can be done in O(p^2) time, but it takes some effort
    potentialvec  = 1 / 2σ² * n * Q' * Xty_over_n
    covmat        = inv(precmat)
    meanvec       = covmat * potentialvec

    D_θ_u_v = Distributions.MvNormal(meanvec, covmat)
    Random.rand!(D_θ_u_v, θ_u_v)

end
=#

function compute_bf(y, group_idx, partition, rscale = 1.0)
    all(==(first(partition)), partition) && return 1.0, 0.0

    partition2 = indexin(partition, unique(partition))
    group_idx2 = if partition2 != eachindex(partition2)
        partition2[group_idx]
    else
        group_idx
    end

    ỹTỹ, ỹTX̃, X̃TX̃, num_h0 = precompute_integrated_log_lik(y, group_idx2)
    N = length(y)

    value, error = QuadGK.quadgk(
        g -> exp(
            log_integrand_bf(g, num_h0, ỹTỹ, ỹTX̃, X̃TX̃, N) +
            Distributions.logpdf(Distributions.InverseGamma(rscale/2, 1/2), g)
        ), 0.0, Inf)
    return value::Float64, error::Float64

end


function precompute_integrated_log_lik(y, group_idx)

    N, P = length(y), length(unique(group_idx))
    X = zeros(N, P)
    for i in 1:P
        X[group_idx .== i, i] .= 1.0
    end

    # P0 = 1 / N * ones(N) * ones(N)'
    P0 = FillArrays.Fill(1 / N, N, N)

    Q = getQ_Rouder(P)
    X = X * Q

    ỹ = y - P0 * y
    X̃ = X - P0 * X

    ỹTỹ = ỹ'ỹ
    ỹTX̃ = ỹ'X̃
    X̃TX̃ = X̃'X̃
    # gamma_a = SpecialFunctions.loggamma((N-1)/2)

    num_h0 = y' * y - N * StatsBase.mean(y)^2

    return (; ỹTỹ, ỹTX̃, X̃TX̃, num_h0)#gamma_a, N)
end

function log_integrand_bf(g, num_h0, ỹTỹ, ỹTX̃, X̃TX̃, N)
    G = LinearAlgebra.Diagonal(fill(g, size(X̃TX̃, 1)))
    Vg = X̃TX̃ + inv(G)
    num_ha = ỹTỹ - LinearAlgebra.dot(ỹTX̃, LinearAlgebra.inv(Vg), ỹTX̃') # could use invquad from PDmats or so
    log_integrand = -1 / 2 * (LinearAlgebra.logdet(G) + LinearAlgebra.logdet(Vg)) + (N - 1) / 2 * (log(num_h0) - log(num_ha))
    # (y'y - N * mean(y)^2) / (
    #     y_tilde'y_tilde - y_tilde' * X_tilde * inv(Vg) * X_tilde' * y_tilde
    # )
    return log_integrand
end

function logml_H0(y)
    N = length(y)
    a = (N - 1) / 2
    SpecialFunctions.loggamma(a) - (
        a * IrrationalConstants.log2π + 1/2 * log(N) + a * log(sum(abs2, y) - N * StatsBase.mean(y)^2)
    )
end

function logml_H0(obj::SuffstatsANOVA)
    y_var = obj.y_var
    N = obj.n
    a = (N - 1) / 2
    SpecialFunctions.loggamma(a) - (
        a * IrrationalConstants.log2π + log(N) / 2 + a * log(y_var * N)
    )
end

function compute_model_probs2(partition_samples, reduce = false)
    k = size(partition_samples, 1)
    model_counts = OrderedCollections.OrderedDict{Vector{Int}, Int}()
    f = reduce ? reduce_model : identity
    for m in PartitionSpace(k)
        model_counts[f(m)] = 0
    end
    # @show model_counts
    for i in axes(partition_samples, 2)
        model_counts[f(view(partition_samples, :, i))] += 1
    end
    model_probs = OrderedCollections.OrderedDict{Vector{Int}, Float64}(
        m => model_counts[m] / size(partition_samples, 2)
        for m in keys(model_counts)
    )
    model_probs
end


function compute_bf2(y, group_idx, partition, rscale = 1.0)
    all(==(first(partition)), partition) && return 1.0, 0.0

    partition2 = indexin(partition, unique(partition))
    group_idx2 = if partition2 != eachindex(partition2)
        partition2[group_idx]
    else
        group_idx
    end

    ỹTỹ, ỹTX̃, X̃TX̃, num_h0, Vₓ, Sₓ = precompute_integrated_log_lik2(y, group_idx2)
    N = length(y)

    cache_d = similar(Sₓ)
    sqrt_inv_Vg = similar(Sₓ, length(Sₓ), length(Sₓ))

    value, error = QuadGK.quadgk(
        g -> exp(
            log_integrand_bf2(g, num_h0, ỹTỹ, ỹTX̃, X̃TX̃, N, Vₓ, Sₓ, cache_d, sqrt_inv_Vg) +
            Distributions.logpdf(Distributions.InverseGamma(rscale/2, 1/2), g)
        ), 0.0, Inf)
    return value::Float64, error::Float64

end

function precompute_integrated_log_lik2(y, group_idx)

    N, P = length(y), length(unique(group_idx))
    X = zeros(N, P)
    for i in 1:P
        X[group_idx .== i, i] .= 1.0
    end

    # P0 = 1 / N * ones(N) * ones(N)'
    P0 = FillArrays.Fill(1 / N, N, N)

    Q = getQ_Rouder(P)
    X2 = X * Q

    ỹ = y - P0 * y
    X̃ = X2 - P0 * X2

    ỹTỹ = ỹ'ỹ
    ỹTX̃ = ỹ'X̃
    X̃TX̃ = X̃'X̃

    # analytic solution doesn't seem to go anywhere?
    # X̃TX̃ ≈ (X2 - P0 * X2)' * (X2 - P0 * X2)
    # X̃TX̃ ≈ (X * Q - P0 * X * Q)' * (X * Q - P0 * X * Q)
    # X̃TX̃ ≈ Q' * X' * X * Q - Q' * X' * P0 * X * Q -
    #     Q' * X' * P0 * X * Q + Q' * X' * P0 * P0 * X * Q

    Uₓ, Sₓ, Vₓ = LinearAlgebra.svd(X̃)
    # X̃ ≈ Uₓ * LinearAlgebra.Diagonal(Sₓ) * Vₓ'
    # X̃'X̃ ≈ Vₓ * LinearAlgebra.Diagonal(Sₓ)^2 * Vₓ'


    # gamma_a = SpecialFunctions.loggamma((N-1)/2)

    num_h0 = y' * y - N * StatsBase.mean(y)^2

    return (; ỹTỹ, ỹTX̃, X̃TX̃, num_h0, Vₓ, Sₓ)
end

function log_integrand_bf2(g, num_h0, ỹTỹ, ỹTX̃, X̃TX̃, N, Vₓ, Sₓ, cache_d, sqrt_inv_Vg)

    # uncomment the commented code to check against the inefficient version
    # G = LinearAlgebra.Diagonal(fill(g, length(Sₓ)))
    # Vg = X̃TX̃ + inv(G)

    log_det_G = length(Sₓ) * log(g)
    # log_det_G ≈ LinearAlgebra.logdet((G))
    log_det_Vg = sum(sᵢ->log(abs2(sᵢ) + 1 / g), Sₓ)
    # log_det_Vg ≈ LinearAlgebra.logdet(X̃TX̃ + inv(G))

    # Vg ≈ Vₓ * LinearAlgebra.Diagonal(Sₓ .^ 2 .+ 1 ./ g) * Vₓ'

    cache_d .= 1 ./ sqrt.(Sₓ .^ 2 .+ 1 ./ g)

    LinearAlgebra.mul!(sqrt_inv_Vg, LinearAlgebra.Diagonal(cache_d), Vₓ')
    # inv(Vg) ≈ sqrt_inv_Vg' * sqrt_inv_Vg

    LinearAlgebra.mul!(cache_d, sqrt_inv_Vg, ỹTX̃')
    num_ha = ỹTỹ - sum(abs2, cache_d)
    # num_ha ≈ ỹTỹ - LinearAlgebra.dot(ỹTX̃, LinearAlgebra.inv(Vg), ỹTX̃')

    log_integrand = -1 / 2 * (log_det_G + log_det_Vg) + (N - 1) / 2 * (log(num_h0) - log(num_ha))

    return log_integrand
end

function compute_bf3(y, group_idx, partition, rscale = 1.0)
    all(==(first(partition)), partition) && return 1.0, 0.0

    partition2 = indexin(partition, unique(partition))
    group_idx2 = if partition2 != eachindex(partition2)
        partition2[group_idx]
    else
        group_idx
    end

    ỹTỹ, ỹTX̃, X̃TX̃, num_h0, Vₓ, Sₓ, ỹTUₓSₓ = precompute_integrated_log_lik3(y, group_idx2)
    N = length(y)

    value, error = QuadGK.quadgk(
        g -> exp(
            log_integrand_bf3(g, num_h0, N, ỹTỹ, Sₓ, ỹTUₓSₓ) +
            Distributions.logpdf(Distributions.InverseGamma(rscale/2, 1/2), g)
        ), 0.0, Inf)
    return value::Float64, error::Float64

end

# function compute_bf3_brob(y, group_idx, partition, rscale = 1.0)
#     all(==(first(partition)), partition) && return 1.0, 0.0

#     partition2 = indexin(partition, unique(partition))
#     group_idx2 = if partition2 != eachindex(partition2)
#         partition2[group_idx]
#     else
#         group_idx
#     end

#     ỹTỹ, ỹTX̃, X̃TX̃, num_h0, Vₓ, Sₓ, ỹTUₓSₓ = precompute_integrated_log_lik3(y, group_idx2)
#     N = length(y)

#     value, error = QuadGK.quadgk(
#         gBrob -> begin
#             g = Brobdingnag.asFloat(gBrob)
#             result = log_integrand_bf3(g, num_h0, N, ỹTỹ, Sₓ, ỹTUₓSₓ) +
#                 Distributions.logpdf(Distributions.InverseGamma(rscale/2, 1/2), g)
#             return Brobdingnag.Brob(result)
#         end, Brobdingnag.asBrob(0.0), Brobdingnag.asBrob(Inf))
#     return value::Brobdingnag.Brob{Float64}, error::Brobdingnag.Brob{Float64}

# end

function precompute_integrated_log_lik3(y, group_idx)

    N, P = length(y), length(unique(group_idx))
    X = zeros(N, P)
    for i in 1:P
        X[group_idx .== i, i] .= 1.0
    end

    # P0 = 1 / N * ones(N) * ones(N)'
    P0 = FillArrays.Fill(1 / N, N, N)

    Q = getQ_Rouder(P)
    X2 = X * Q

    ỹ = y - P0 * y
    X̃ = X2 - P0 * X2

    ỹTỹ = ỹ'ỹ
    ỹTX̃ = ỹ'X̃
    X̃TX̃ = X̃'X̃

    Uₓ, Sₓ, Vₓ = LinearAlgebra.svd(X̃)
    num_h0 = y' * y - N * StatsBase.mean(y)^2

    ỹTUₓSₓ = ỹ' * Uₓ * LinearAlgebra.Diagonal(Sₓ)

    return (; ỹTỹ, ỹTX̃, X̃TX̃, num_h0, Vₓ, Sₓ, ỹTUₓSₓ)
end

function log_integrand_bf3(g, num_h0, N, ỹTỹ, Sₓ, ỹTUₓSₓ)

    # uncomment the commented code to check against the inefficient version
    # G = LinearAlgebra.Diagonal(fill(g, length(Sₓ)))
    # Vg = X̃TX̃ + inv(G)

    log_det_G = length(Sₓ) * log(g)
    # log_det_G ≈ LinearAlgebra.logdet((G))
    log_det_Vg = sum(sᵢ->log(abs2(sᵢ) + 1 / g), Sₓ)
    # log_det_Vg ≈ LinearAlgebra.logdet(X̃TX̃ + inv(G))

    num_ha = ỹTỹ - sum(i-> abs2(ỹTUₓSₓ[i]) / (abs2(Sₓ[i]) + 1 / g), eachindex(Sₓ))

    # num_ha ≈ ỹTỹ - LinearAlgebra.dot(ỹTX̃, LinearAlgebra.inv(Vg), ỹTX̃')

    log_integrand = -1 / 2 * (log_det_G + log_det_Vg) + (N - 1) / 2 * (log(num_h0) - log(num_ha))

    return log_integrand
end

function group_idx_to_design_mat(group_idx)
    n = length(group_idx)
    k = length(unique(group_idx))
    design_mat = zeros(Int, n, k)
    for i in 1:k
        design_mat[group_idx .== i, i] .= 1
    end
    return design_mat
end

function group_idx_to_design_mat(group_idx::AbstractVector{<:UnitRange})
    n = last(last(group_idx))
    k = length(group_idx)
    design_mat = zeros(Int, n, k)
    for i in 1:k
        design_mat[group_idx[i], i] .= 1
    end
    return design_mat
end


#=
function anova_enumerate(
    y::AbstractVector{<:Number},
    group_idx::AbstractVector{<:Integer},
    π_ρ::AbstractPartitionDistribution;
    verbose::Bool = true,
    threaded::Bool = true,
    useBrob::Bool = false
)

k = length(unique(group_idx))

modelspace = PartitionSpace(k, DistinctPartitionSpace)
nmodels = length(modelspace)

logml                   = Vector{Float64}(undef, nmodels)
bfs                     = Vector{Float64}(undef, nmodels)
err                     = Vector{Float64}(undef, nmodels)
log_prior_model_probs   = Vector{Float64}(undef, nmodels)

logml_h0 = logml_H0(y)
logml[1] = logml_h0

if threaded
    anova_enumerate_internal_threaded!(bfs, logml, err, log_prior_model_probs, y, group_idx, π_ρ, logml_h0, modelspace, verbose, useBrob)
else
    anova_enumerate_internal!(         bfs, logml, err, log_prior_model_probs, y, group_idx, π_ρ, logml_h0, modelspace, verbose, useBrob)
end

logbfs = log.(bfs)
log_posterior_probs = logbf_to_logposteriorprob(logbfs, log_prior_model_probs)
posterior_probs = exp.(log_posterior_probs)
(; posterior_probs, log_posterior_probs, logbfs, bfs, logml, err)
end

function anova_enumerate_internal!(
bfs::Vector{Float64}, logml::Vector{Float64}, err::Vector{Float64}, log_prior_model_probs::Vector{Float64},
y, group_idx, π_ρ, logml_h0, modelspace, verbose::Bool, useBrob::Bool = false
)
ProgressMeter.@showprogress enabled = verbose for (i, partition) in enumerate(modelspace)
    log_prior_model_probs[i] = logpdf_model_distinct(π_ρ, partition)
    if useBrob
        bfs[i], err[i] = compute_bf3_brob(y, group_idx, partition)
    else
        bfs[i], err[i] = compute_bf3(y, group_idx, partition)
    end
    logml[i] = log(bfs[i]) - logml_h0
end
end

function anova_enumerate_internal_threaded!(
bfs::Vector{Float64}, logml::Vector{Float64}, err::Vector{Float64}, log_prior_model_probs::Vector{Float64},
y, group_idx, π_ρ, logml_h0, modelspace, verbose::Bool, useBrob::Bool = false
)

modelspace_c = collect(enumerate(modelspace))
prog = ProgressMeter.Progress(length(modelspace_c), enabled = verbose)

Threads.@threads for (i, partition) in modelspace_c

    log_prior_model_probs[i] = logpdf_model_distinct(π_ρ, partition)
    # bfs[i], err[i] = compute_bf3(y, group_idx, partition)
    if useBrob
        bfs[i], err[i] = compute_bf3_brob(y, group_idx, partition)
    else
        bfs[i], err[i] = compute_bf3(y, group_idx, partition)
    end
    logml[i] = log(bfs[i]) - logml_h0

    ProgressMeter.next!(prog)
end
end

function anova_sample_integrated(
y::AbstractVector{<:Number},
group_idx::AbstractVector{<:Integer},
π_ρ::AbstractPartitionDistribution;
verbose::Bool = true,
no_iter::Integer = 10_000)

k = group_idx |> unique |> length

# this is not logml anymore but bf ρ vs [1, 1, 1, ...], right?
logml_dict = Dict{Vector{Int}, Float64}()
# logml_dict[partition] = compute_bf(y, group_idx, partition)[1]
logml_dict[ones(Int, k)] = 0.0

partition = collect(1:k)
partition_samples = Matrix{Int}(undef, k, no_iter)
ProgressMeter.@showprogress enabled=verbose for it in 1:no_iter

    sample_partition_rjmcmc_integrated!(partition, y, group_idx, π_ρ, logml_dict, no_cache_hits)
    partition_samples[:, it]  = partition

end

return (; partition_samples)

end

function anova_sample(
y::AbstractVector{<:Number},
group_idx::AbstractVector{<:Integer},
π_ρ::AbstractPartitionDistribution;
verbose::Bool = true,
no_iter::Integer = 10_000)

#=

    TODO

    - [x] get the sampling of the partitions to work
    - [ ] rewrite everything to use θ_cs instead of partition_mat * Q * θ_u
    - [ ] initial values? at least for the partition?
    - [ ] better return type?
    - [ ] pass verbose as argument for progressmeter
    - [ ] properly qualify imports
    - [ ] better types for the dictionary than {Vector{Int}, Float64}}?
    - [ ] rewrite a lot if haskey constructions to use get!

=#

k = group_idx |> unique |> length

μ  = mean(y)
σ² = var(y)
n  = length(y)

mean_y = StatsBase.mean(y)
ss2_y_over_n = StatsBase.mean(abs2, y)


g  = 1.0
θ_u = randn(k - 1)
θ_s = similar(θ_u, k)
θ_cs = similar(θ_u, k)

partition = collect(1:k)
Q = getQ_Rouder(k)

μ_samples         = Vector{Float64}(undef, no_iter)
σ²_samples        = Vector{Float64}(undef, no_iter)
g_samples         = Vector{Float64}(undef, no_iter)
θ_u_samples       = Matrix{Float64}(undef, length(θ_u), no_iter)
θ_s_samples       = Matrix{Float64}(undef, k,           no_iter)
θ_cs_samples      = Matrix{Float64}(undef, k,           no_iter)
partition_samples = Matrix{Int}(    undef, k,           no_iter)

# partition_prior = BetaBinomialPartitionDistribution(k, 1, 1)
# π_ρ = UniformPartitionDistribution(k)

# this can be improved upon!
design_mat = zeros(Int, n, k)
for i in 1:k
    design_mat[group_idx .== i, i] .= 1
end

# mean_x = vec(mean(design_mat, dims = 1))
tb_g = StatsBase.countmap(group_idx)
XtX_over_n_full = LinearAlgebra.Diagonal([tb_g[i] for i in axes(design_mat, 2)])
Xty_over_n_full = design_mat' * y ./ n

# type of XtX_over_n could also be Int?
partition_to_suffstats = Dict{Vector{Int}, Tuple{LinearAlgebra.Diagonal{Float64, Vector{Float64}}, Vector{Float64}}}()
partition_to_suffstats[partition] = (XtX_over_n_full, Xty_over_n_full)
partition_size_to_Q = Dict{Int, Matrix{Float64}}(k => Q) # <- could also be Vector{Union{Nothing, ...}} ? probably no benefit

# this is not logml anymore but bf ρ vs [1, 1, 1, ...], right?
logml_dict = Dict{Vector{Int}, Float64}()
# logml_dict[partition] = compute_bf(y, group_idx, partition)[1]
logml_dict[ones(Int, k)] = 0.0

seen = Set{Int}()
sizehint!(seen, k)

ProgressMeter.@showprogress enabled=verbose for it in 1:no_iter

    sample_partition_rjmcmc_integrated!(partition, y, group_idx, π_ρ, logml_dict)

    partition_size = no_distinct_groups_in_partition(partition)
    if !haskey(partition_to_suffstats, partition)

        # group_idx2 = partition[group_idx]
        # design_mat2 = zeros(Int, n, partition_size)
        # for i in 1:partition_size
        #     design_mat2[group_idx2 .== i, i] .= 1
        # end
        # design_mat2' * y ./ n
        # design_mat2' * design_mat2

        XtX_over_n_partition_vec = zeros(Int, partition_size)
        Xty_over_n_partition = zeros(Float64, partition_size)

        partition2 = indexin(partition, unique(partition))
        for i in eachindex(partition2)
            XtX_over_n_partition_vec[partition2[i]] += tb_g[i]
            Xty_over_n_partition[partition2[i]]     += Xty_over_n_full[i]
        end
        partition_to_suffstats[partition] = (LinearAlgebra.Diagonal(XtX_over_n_partition_vec), Xty_over_n_partition)
    end
    XtX_over_n, Xty_over_n = partition_to_suffstats[partition]

    Q = if haskey(partition_size_to_Q, partition_size)
        partition_size_to_Q[partition_size]
    else
        partition_size_to_Q[partition_size] = getQ_Rouder(partition_size)
    end


    θ_u_v = view(θ_u, range(; stop = partition_size - 1))
    sample_conditional_θ_u_v!(θ_u_v, n, σ², g, XtX_over_n, Xty_over_n, Q)

    μ  = sample_conditional_μ(mean_y, σ², n)
    σ² = sample_conditional_σ(mean_y, ss2_y_over_n, n, μ, θ_u_v, g, XtX_over_n, Xty_over_n, Q)
    g  = sample_conditional_g(σ², θ_u_v)

    LinearAlgebra.mul!(view(θ_s, 1:partition_size), Q, θ_u_v)
    empty!(seen)
    for l in eachindex(partition)
        if partition[l] in seen
            θ_cs[l] = θ_s[partition[l]]
        else
            push!(seen, partition[l])
        end
    end


    μ_samples[it]       = μ
    σ²_samples[it]      = σ²
    g_samples[it]       = g
    θ_u_samples[:, it]  = θ_u
    θ_s_samples[:, it]  = θ_s
    θ_cs_samples[:, it] = θ_cs
    partition_samples[:, it]  = partition

end

return (; μ_samples, σ²_samples, g_samples, θ_u_samples, θ_s_samples, θ_cs_samples, partition_samples)

end
=#