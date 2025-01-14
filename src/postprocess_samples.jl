
"""
$(TYPEDSIGNATURES)

Compute the proportion of samples where partition[i] == partition[j] ∀i, j
"""
function compute_post_prob_eq(partition_samples::AbstractMatrix)
    # n_samps, n_groups = size(partition_samples)
    n_groups, n_samps = size(partition_samples)
    probs = zeros(Float64, n_groups, n_groups)
    @inbounds for k in axes(partition_samples, 2)
        for j in 1:n_groups-1
            for i in j+1:n_groups
                if partition_samples[i, k] == partition_samples[j, k]
                    probs[i, j] += 1.0
                end
            end
        end
    end
    return probs ./ n_samps
end

compute_post_prob_eq(x::Union{IntegratedResult, RJMCMCResult})             = compute_post_prob_eq(x.partition_samples)

function compute_post_prob_eq(results::EnumerateResult)
    modelspace = PartitionSpace(results.k, EqualitySampler.DistinctPartitionSpace)
    eq_probs = zeros(Float64, results.k, results.k)
    @inbounds for (idx, model) in enumerate(modelspace)

        for j in 1:length(model)-1
            for i in j+1:length(model)
                if model[i] == model[j]
                    eq_probs[i, j] += results.posterior_probs[idx]
                end
            end
        end
    end
    return eq_probs
end

compute_post_prob_eq(x::EnumerateThenSampleResult) = compute_post_prob_eq(x.enumerate_result)

# function compute_post_prob_eq(partition_samples::AbstractArray{T, 3}) where T
# 	n_samps, n_groups, n_chains = size(partition_samples)
# 	probs = zeros(Float64, n_groups, n_groups)
# 	@inbounds for l in axes(partition_samples, 3)
# 		for k in axes(partition_samples, 1)
# 			for j in 1:n_groups-1
# 				for i in j+1:n_groups
# 					if partition_samples[k, i, l] == partition_samples[k, j, l]
# 						probs[i, j] += 1.0
# 					end
# 				end
# 			end
# 		end
# 	end
# 	return probs ./ (n_samps * n_chains)
# end


"""
$(TYPEDSIGNATURES)

Compute how often each partition is visited
"""
function compute_model_counts(partition_samples::AbstractMatrix{T}, add_missing_models::Bool = true) where T<:Integer
    res = Dict{Vector{T}, Int}()
    StatsBase.addcounts!(res, eachcol(partition_samples))
    if add_missing_models
        k = size(partition_samples, 1)
        expected_size = bellnum(k)
        if length(res) != expected_size
            for m in PartitionSpace(k)
                get!(res, m, 0)
                # if !haskey(res, m)
                # 	res[m] = 0
                # end
            end
        end
    end
    return sort!(OrderedCollections.OrderedDict(res); byvalue = true, rev = true)
end

function compute_model_counts(x::Union{IntegratedResult, RJMCMCResult, EnumerateThenSampleResult},
    add_missing_models::Bool = no_groups(x) < 7)
    compute_model_counts(x.partition_samples, add_missing_models)
end

"""
$(TYPEDSIGNATURES)

Compute the posterior probability of each partition
"""
function compute_model_probs(chn::AbstractMatrix, add_missing_models::Bool = true)
    count_models = compute_model_counts(chn, add_missing_models)
    return counts2probs(count_models)
end

function compute_model_probs(x::EnumerateResult{T}, ::Bool = true) where T
    k = x.k
    res = OrderedCollections.OrderedDict{Vector{Int}, T}()
    for (i, m) in enumerate(PartitionSpace(k))
        res[m] = x.posterior_probs[i]
    end
    return res
end

function compute_model_probs(
        x::Union{IntegratedResult, RJMCMCResult, EnumerateThenSampleResult},
        add_missing_models::Bool = no_groups(x) < 7
    )
    compute_model_probs(x.partition_samples, add_missing_models)
end


"""
$(TYPEDSIGNATURES)

Compute the number of models with posterior probability of including 0, ..., k-1 equalities in the model
"""
function compute_incl_counts(x::Union{IntegratedResult, RJMCMCResult}; add_missing_inclusions::Bool = false)
    compute_incl_counts(x.partition_samples, add_missing_inclusions = add_missing_inclusions)
end
function compute_incl_counts(partition_samples::AbstractMatrix{T}; add_missing_inclusions::Bool = false) where T <: Integer
    res = Dict{T, Int}()
    StatsBase.addcounts!(res, map(maximum, eachcol(partition_samples)))
    if (add_missing_inclusions)
        for i in axes(partition_samples, 1)
            get!(res, T(i), zero(Int))
        end
    end
    return sort!(OrderedCollections.OrderedDict(res); byvalue = true, rev = true)
end

"""
$(TYPEDSIGNATURES)

Compute the posterior probability of including 0, ..., k-1 equalities in the model
"""
function compute_incl_probs(partition_samples::AbstractMatrix{T}; add_missing_inclusions::Bool = false) where T <:Integer
    return counts2probs(compute_incl_counts(partition_samples, add_missing_inclusions = add_missing_inclusions))
end

function counts2probs(counts::AbstractDict{T, U}) where {T, U}
    total_visited = sum(values(counts))
    probs = OrderedCollections.OrderedDict{T, float(U)}()
    for (model, count) in counts
        probs[model] = count / total_visited
    end
    return probs
end

"""
$(TYPEDSIGNATURES)

Compute the median posterior probability model/ partition.
Note that this is not the same as the median model/ partition in regression.
For partitions, we find the partition `p` that minimizes `sum(abs2(eq_probs[i, j] - (p[i] == p[j])) for i in eachindex(p), j in eachindex(p))`.
This is done in a heuristic manner, so the result is not guaranteed to be the true mpm.
"""
function get_mpm_partition(eq_probs::AbstractMatrix, threshhold = .5)

    eq_probs_discrete = eq_probs .> threshhold
    mpm_partition = collect(axes(eq_probs, 1))
    for i in axes(eq_probs, 1)
        for j in 1:i
            if isone(eq_probs_discrete[i, j])
                mpm_partition[j] = mpm_partition[i]
            end
        end
    end
    EqualitySampler.reduce_model_2!(mpm_partition)

    return mpm_partition
end

get_mpm_partition(x::AbstractMCMCResult, threshhold = .5) = get_mpm_partition(compute_post_prob_eq(x), threshhold)


"""
$(TYPEDSIGNATURES)

Compute the highest posterior probability model/ partition.
Note that this can be unstable if the result is not obtained via sampling and the modelspace is very large.
"""
function get_hpm_partition(results::EnumerateResult)
    _, idx = findmax(results.log_posterior_probs)
    iter = PartitionSpace(results.k) # TODO: would be nice if this could use the partition_type
    first(Iterators.drop(iter, idx-1))
end

get_hpm_partition(results::EnumerateThenSampleResult) = get_hpm_partition(results.enumerate_result)

function get_hpm_partition(results::Union{IntegratedResult, RJMCMCResult})
    get_hpm_partition(results.partition_samples)
end

function get_hpm_partition(partition_samples::AbstractMatrix{<:Integer})
    collect(StatsBase.mode(eachcol(partition_samples)))
end
