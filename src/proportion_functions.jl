function simulate_proportions(partition, n, θ = nothing)

    partition2 = reduce_model_2(partition)
    np = length(unique(partition2))
    if isnothing(θ)
        p = rand(Distributions.Uniform(), np)
    else
        @assert length(θ) == length(partition2)
        p = θ
    end

    if isone(length(n))
        ns = fill(n, length(partition2))
    else
        ns = n
    end

    result = similar(ns, length(partition2))
    for i in eachindex(result)
        result[i] = rand(Distributions.Binomial(ns[i], p[partition2[i]]))
    end
    return (n = ns, k = result, p = p, partition = partition2)
end

struct SuffstatsProportions <: AbstractSuffstats
    ns::Vector{Int}
    ks::Vector{Int}
end
_get_k(obj::SuffstatsProportions) = length(obj.ns)
_get_means(obj::SuffstatsProportions) = obj.ks ./ obj.ns # p
_get_vars( obj::SuffstatsProportions) = obj.ks ./ obj.ns .* (obj.ns .- obj.ks) ./ obj.ns # p * q


function apply_partition_to_suffstats(obj::SuffstatsProportions, partition0::AbstractVector, reduce::Bool = true)

    partition = reduce ? reduce_model_2(partition0) : partition0
    new_length = no_distinct_groups_in_partition(partition)

    ns_new = zeros(eltype(obj.ns), new_length)
    ks_new = zeros(eltype(obj.ks), new_length)
    for i in eachindex(partition)
        ns_new[partition[i]] += obj.ns[i]
        ks_new[partition[i]] += obj.ks[i]
    end

    SuffstatsProportions(ns_new, ks_new)
end

Base.@kwdef struct ProportionPriors{T<:AbstractPartitionDistribution, U<:Number} <: AbstractPriors
    partition::T
    α::U = 1.0
    β::U = 1.0
    function ProportionPriors(partition::T, α::U = 1.0, β::U = 1.0) where {T<:AbstractPartitionDistribution, U<:Number}
        α > zero(α) || throw(DomainError(α, "α must be positive"))
        β > zero(β) || throw(DomainError(β, "β must be positive"))
        new{T, U}(partition, α, β)
    end
end

#=


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


=#
function proportion_test(
    ns::AbstractVector{<:Integer}, ks::AbstractVector{<:Integer}, args...; kwargs...
    )
    proportion_test(SuffstatsProportions(ns, ks), args...; kwargs...)
end


function proportion_test(
    obj::SuffstatsProportions,
    method::Enumerate,
    partition_prior::AbstractPartitionDistribution;
    α::Number = 1.0,
    β::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = true
    )
    _enumerate(obj, method, ProportionPriors(partition_prior, α, β), verbose, threaded)
end

function proportion_test(
    obj::SuffstatsProportions,
    method::SampleIntegrated,
    partition_prior::AbstractPartitionDistribution;
    α::Number = 1.0,
    β::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = true
    )
    _sample_integrated(obj, method, ProportionPriors(partition_prior, α, β), verbose, threaded)
end

function proportion_test(
    obj_full::SuffstatsProportions,
    method::M,
    partition_prior::AbstractPartitionDistribution;
    α::Number = 1.0,
    β::Number = 1.0,
    verbose::Bool = true,
    threaded::Bool = true
    ) where M <: Union{SampleRJMCMC, EnumerateThenSample}

    priors = ProportionPriors(partition_prior, α, β)
    iter = method.iter
    fullmodel_only = !(M <: EnumerateThenSample) && method.fullmodel_only

    !fullmodel_only && _validate_partition_and_suffstats(obj_full, partition_prior)

    k = _get_k(obj_full)

    partition = _get_initial_partition(method, priors, k)

    obj = apply_partition_to_suffstats(obj_full, partition, false)

    θ_p_samples       = Matrix{Float64}(undef, k, iter)
    partition_samples = similar(partition, k, iter)
    θ = similar(θ_p_samples, k)

    if fullmodel_only
        obj_p = obj
        θ_v = view(θ, 1:_get_k(obj_p))
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

    prog = ProgressMeter.Progress(iter; enabled = verbose)
    for it in 1:iter

        if !fullmodel_only

            if M <: EnumerateThenSample
                partition_idx = rand(enumerate_results_for_resampling)
                partition .= @view partition_space[:, partition_idx]
            else
                sample_partition_rjmcmc_integrated!(partition, obj_full, logbf10_dict, method, sampling_stats, priors, threaded, max_size_logbf10_dict)
            end

            partition_samples[:, it]  = partition

            obj_p = apply_partition_to_suffstats(obj_full, partition, false)
            θ_v = view(θ, 1:_get_k(obj_p))
        end

        for j in eachindex(θ_v)
            θ_v[j] = rand(Distributions.Beta(priors.α + obj_p.ks[j], priors.β + obj_p.ns[j] - obj_p.ks[j]))
        end

        @views for j in eachindex(partition)
            θ_p_samples[j, it] = θ_v[partition[j]]
        end

        ProgressMeter.next!(prog)

    end

    if M <: EnumerateThenSample
        return EnumerateThenSampleResult(
            enumerate_results,
            partition_samples,
            ProportionParameterSamples(θ_p_samples)
        )
        # (; θ_p_samples, partition_samples, enumerate_results)
    else
        return RJMCMCResult(
            partition_samples,
            ProportionParameterSamples(θ_p_samples)
        )
        # return (; θ_p_samples, partition_samples)
    end

end

function logml_H0(obj::SuffstatsProportions, α = 1.0, β = 1.0)
    log_marginal_likelihood(sum(obj.ks), sum(obj.ns), α, β)
end

function compute_one_bf(obj::SuffstatsProportions, partition, ::AbstractSamplingMethod{T}, priors::ProportionPriors) where T

    error = T(NaN)
    all(==(first(partition)), partition) && return zero(T), error

    # TODO: must the prior be symmetric across partitions?
    # if yes, this is a pretty annoying restriction
    # if not, then 1 == 2 would imply use the prior of 1 and 2 == 1 would imply use the prior of 2?
    # or should the prior be "averaged" somehow?
    α = priors.α
    β = priors.β

    logml_h0 = logml_H0(obj, α, β)
    obj_p = apply_partition_to_suffstats(obj, partition, false)

    logml_h1 = sum(log_marginal_likelihood(obj_p.ks[i], obj_p.ns[i], α, β) for i in eachindex(obj_p.ns))

    return logml_h1 - logml_h0, error

end

function log_marginal_likelihood(k, n, α, β)
    SpecialFunctions.logbeta(α + k, β + n - k) - SpecialFunctions.logbeta(α, β)
end

#=
# TODO: to use beta or not as a prior object?
function proportions_enumerate(ns, ks,
    π_ρ::AbstractPartitionDistribution;
    π_θ::Distributions.Beta = Distributions.Beta(1.0, 1.0),
    verbose::Bool = true)

    α, β = Distributions.params(π_θ)

    k = length(ns)
    modelspace = PartitionSpace(k, DistinctPartitionSpace)
    nmodels = length(modelspace)

    logml                   = Vector{Float64}(undef, nmodels)
    err                     = zeros(Float64, nmodels)
    log_prior_model_probs   = Vector{Float64}(undef, nmodels)

    ProgressMeter.@showprogress enabled = verbose for (i, m) in enumerate(modelspace)

        log_prior_model_probs = logpdf_model_distinct(π_ρ, m)

        nj = length(unique(m))
        n_temp = [sum(ns[m .== j]) for j in 1:nj]
        k_temp = [sum(ks[m .== j]) for j in 1:nj]
        logml[i] = sum(log_marginal_likelihood.(k_temp, n_temp, α, β))

    end
    logbfs = logml .- logml[1]

    bfs = exp.(logbfs)
    logsumbfs = LogExpFunctions.logsumexp(logbfs .+ log_prior_model_probs)
    log_posterior_probs = logbfs .+ log_prior_model_probs .- logsumbfs
    posterior_probs = exp.(log_posterior_probs)


    return (; posterior_probs, log_posterior_probs, logbfs, bfs, logml, err)
    # return (; bf, logml)
end

function proportion_compute_logbf(ns, ks, ρ, α, β)

    nj = length(unique(ρ))
    n_temp = [sum(ns[ρ .== j]) for j in 1:nj]
    k_temp = [sum(ks[ρ .== j]) for j in 1:nj]
    logml_ρ = sum(log_marginal_likelihood.(k_temp, n_temp, α, β))

    ρ₀ = zeros(Int, length(ρ))
    nj = 1

    logml_ρ = log_marginal_likelihood(sum(ks), sum(ns), α, β)

end


# TODO: can be generalized with a "data object" between anova and this part.
function sample_partition_rjmcmc_integrated_proportion!(ns, ks, ρ, π_ρ, logbf10_dict, α, β)

    ρ′  = similar(ρ)
    ρ′′ = similar(ρ)
    probvec = Vector{Float64}(undef, length(ρ))
    for j in eachindex(ρ)

        copyto!(ρ′, ρ)
        for l in eachindex(ρ)
            ρ′[j] = l
            copyto!(ρ′′, ρ′)
            reduce_model_2!(ρ′′)
            if haskey(logbf10_dict, ρ′′)
                logml = logbf10_dict[ρ′′]
            else
                logml = proportion_compute_logbf(ns, ks, ρ′′, α, β)
                logbf10_dict[copy(ρ′′)] = logml
            end

            probvec[l] = logml + logpdf_model(π_ρ, ρ′)
        end

        probvec .= exp.(probvec .- LogExpFunctions.logsumexp(probvec))

        # could use Gumbel trick here?
        ρ[j] = Random.rand(Distributions.Categorical(probvec))

        reduce_model_2!(ρ)
    end
end

function proportions_sample_integrated(
    ns::AbstractVector{<:Integer},
    ks::AbstractVector{<:Integer},
    π_ρ::AbstractPartitionDistribution;
    π_θ::Distributions.Beta = Distributions.Beta(1.0, 1.0),
    verbose::Bool = true,
    no_iter::Integer = 10_000)

    α, β = Distributions.params(π_θ)

    k = length(ns)
    # this is not logml anymore but bf ρ vs [1, 1, 1, ...], right?
    logbf_dict = Dict{Vector{Int}, Float64}()
    # logml_dict[partition] = compute_bf(y, group_idx, partition)[1]
    logbf_dict[ones(Int, k)] = 0.0

    ρ = collect(1:k)
    partition_samples = Matrix{Int}(undef, k, no_iter)
    ProgressMeter.@showprogress enabled=verbose for it in 1:no_iter

        sample_partition_rjmcmc_integrated_proportion!(ns, ks, ρ, π_ρ, logbf_dict, α, β)
        partition_samples[:, it] = ρ

    end

    return (; partition_samples)


end
=#

#=
function proportion_test2(
    obj::SuffstatsProportions,
    method::SampleIntegrated,
    partition_prior::AbstractPartitionDistribution;
    # TODO: how to pass the beta prior around? inside "method"?
    # as something new? Ideally it's aligned with "rscale" in the ANOVA case!
    # there we could also pass a prior instead of a fixed value?
    π_θ::Distributions.Beta = Distributions.Beta(1.0, 1.0),
    verbose::Bool = true,
    threaded::Bool = true)

    iter = method.iter

    k = length(obj.n_by_group)
    # this is not logml anymore but bf ρ vs [1, 1, 1, ...], right?
    logml_dict = Dict{Vector{Int}, Float64}()
    # logml_dict[ones(Int, k)] = 0.0# + logpdf_model(partition_prior, ones(Int, k))

    partition = if isempty(method.initial_partition)
        collect(1:k)
    else
        Distributions.insupport(UniformPartitionDistribution(k), partition) || throw(ArgumentError("Initial partition is not valid"))
        method.initial_partition
    end
    partition_samples = Matrix{Int}(undef, k, iter)
    partition_size_to_Q  = Dict{Int, Matrix{Float64}}()
    no_cache_hits = Ref{Int}(0)
    ProgressMeter.@showprogress enabled=verbose for it in 1:iter

        sample_partition_rjmcmc_integrated!(partition, obj, partition_prior, logml_dict, method, partition_size_to_Q, no_cache_hits)
        partition_samples[:, it]  = partition

    end

    return (; partition_samples)

end
=#

# TODO: this one is informative for me, but probably should use the integrated step?
function proportions_sample(
    ns::AbstractVector{<:Integer},
    ks::AbstractVector{<:Integer},
    π_ρ::AbstractPartitionDistribution;
    π_θ::Distributions.Beta = Distributions.Beta(1.0, 1.0),
    verbose::Bool = true,
    no_iter::Integer = 10_000
)

    α, β = Distributions.params(π_θ)

    k = length(ns)
    θ_samples         = Matrix{Float64}(undef, k,           no_iter)
    partition_samples = Matrix{Int}(    undef, k,           no_iter)

    θ = rand(k)
    partition = collect(1:k)

    θ2 = similar(θ)
    partition2 = similar(partition)

    seen = Set{Int}()

    log_hits   = log.(ks)
    log_misses = log.(ns .- ks)

    ProgressMeter.@showprogress enabled=verbose for it in axes(θ_samples, 2)

        # between model move
        # TODO: implement this

        # 1. random element to update
        # 2. simulate new values
        # 3. acceptance probability
        # can be taken out of the loop?
        copyto!(partition2, partition)
        copyto!(θ2, θ)
        for i in eachindex(partition)

            # probability moving between is 1 / (k - 1)
            newvalue = rand(setdiff(1:k, partition[i]))
            partition2[i] = newvalue


            # TODO: when the proposal removes a group, then the reverse move is missing the proposal probability!
            idx = findfirst(==(newvalue), view(partition2, 1:k .!= i))

            if isnothing(idx) # add a new group

                post_D = Distributions.Beta(α + ks[i], β + ns[i] - ks[i])
                θ2[i] = rand(post_D)
                log_proposal_density = Distributions.logpdf(post_D, θ2[i])
            else # existing group or removed group
                # add one if needed since findfirst excluded index i
                idx >= i && (idx += 1)
                θ2[i] = θ[idx]

                # if the proposal removes a group then the reverse move needs to account for the probability to sample the removed value
                removes_group = iszero(count(==(partition[i]), partition2))
                if removes_group
                    # reverse move contains the proposal density of the removed value
                    # note the -
                    # the removed value must have existed as a group of size 1
                    log_proposal_density = -Distributions.logpdf(Distributions.Beta(α + ks[i], β + ns[i] - ks[i]), θ[i])
                else
                    # proposal moves between groups but does not add or update parameters, there is no density of the proposed value
                    log_proposal_density = 0.0
                end
            end

            # TODO: Q1 which of these two do we want/ need?
            # A1: should not matter, because the term that differs (binomial(n, k)) cancels in numerator/ denominator
            # nj1 = length(unique(partition))
            # n_temp1 = [sum(ns[partition .== j]) for j in 1:nj1]
            # k_temp1 = [sum(ks[partition .== j]) for j in 1:nj1]
            # θ_temp1 = [θ[findfirst(==(j), partition)] for j in 1:nj1]
            # lpdf_old = sum(logpdf.(Binomial.(n_temp1, θ_temp1), k_temp1))

            # nj2 = length(unique(partition2))
            # partition2b = reduce_model_2(partition2)
            # n_temp2 = [sum(ns[partition2b .== j]) for j in 1:nj2]
            # k_temp2 = [sum(ks[partition2b .== j]) for j in 1:nj2]
            # θ_temp2 = [θ2[findfirst(==(j), partition2b)] for j in 1:nj2]
            # lpdf_new = sum(logpdf.(Binomial.(n_temp2, θ_temp2), k_temp2))
            # lpdf_new - lpdf_old

            # should be the same, no?
            # lpdf_new2 = sum(i->logpdf(Binomial(ns[i], θ2[i]), ks[i]), eachindex(ks))
            # lpdf_old2 = sum(i->logpdf(Binomial(ns[i], θ[i]),  ks[i]), eachindex(ks))
            # lpdf_new2 - lpdf_old2
            # lpdf_new = lpdf_new2
            # lpdf_old = lpdf_old2

            lpdf_new3 = 0.0
            lpdf_old3 = 0.0
            for i in eachindex(ks)
                lpdf_new3 += log(θ2[i]) * ks[i] + log(1 - θ2[i]) * (ns[i] - ks[i])
                lpdf_old3 += log(θ[i])  * ks[i] + log(1 - θ[i])  * (ns[i] - ks[i])
            end
            # lpdf_new3 - lpdf_old3
            lpdf_new = lpdf_new3
            lpdf_old = lpdf_old3


            # TODO: Q2 derive this from scratch! Also check Gottardo & Raftery
            acc = lpdf_new - lpdf_old +
                # logpdf_ratio(partition_prior, partition2, partition) -
                # sum(logpdf.(Binomial.(ns, θ), ks)) - sum(logpdf.(Binomial.(ns, θ2), ks)) +
                logpdf_model(π_ρ, partition2) - logpdf_model(π_ρ, partition) -
                log_proposal_density

            if log(rand()) <= acc
                partition[i] = newvalue
                θ[i] = θ2[i]
                reduce_model_2!(partition)
                copyto!(partition2, partition)
            else
                partition2[i] = partition[i]
                θ2[i]         = θ[i]
            end

        end

        # within model move
        nj = length(unique(partition))
        n_temp = [sum(ns[partition .== j]) for j in 1:nj]
        k_temp = [sum(ks[partition .== j]) for j in 1:nj]

        new_θ = rand.(Distributions.Beta.(α .+ k_temp, β .+ n_temp - k_temp))

        empty!(seen)
        c = 1
        for l in eachindex(partition)
            if partition[l] in seen
                θ[l] = θ[partition[l]]
            else
                θ[l] = new_θ[c]
                push!(seen, partition[l])
                c += 1
            end
        end

        θ_samples[:, it]         = θ
        partition_samples[:, it] = partition
    end
    return (; θ_samples, partition_samples)
end