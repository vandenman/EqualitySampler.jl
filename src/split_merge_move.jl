function initialize_buffer_split_merge_move(partition::AbstractVector{T}) where T <: Integer

    proposal = similar(partition)
    lookup = zeros(T, length(partition))

    return (; proposal, lookup)
end

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

#
function split_merge_move_random!(current, method, buffer_split_merge_move, obj, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats)

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
#



function linear_index_to_triangle_index(c::Integer, n::Integer)
    cₙ = 1
    for i in 1:n - 1, j in i+1:n
        if c == cₙ
            return i, j
        end
        cₙ += 1
    end
    return 0, 0
end
function triangle_index_to_linear_index(i1::Integer, i2::Integer, n::Integer)
    c = 1
    for i in 1:n - 1, j in i+1:n
        if i == i1 && j == i2
            return c
        end
        c += 1
    end
    return 0
end

function get_probvec_mean_distances(raw_means)
    k = length(raw_means)
    no_pairs = binomial(k, 2)
    mean_dists = similar(raw_means, no_pairs)
    c = 1
    for i in 1:k - 1, j in i + 1:k
        mean_dists[c] = abs(raw_means[i] - raw_means[j])
        if !iszero(mean_dists[c])
            mean_dists[c] = 1 / mean_dists[c]
        end
        c+=1
    end
    replacement_if_equal_value = 10 * maximum(mean_dists)
    for i in eachindex(mean_dists)
        if iszero(mean_dists[i])
            mean_dists[i] = replacement_if_equal_value
        end
    end

    mean_dists ./= sum(mean_dists)
    return mean_dists
end

function make_informed_merge_proposal!(proposal::AbstractVector{<:Integer}, current::AbstractVector{<:Integer}, lookup::AbstractVector{<:Integer}, suffstats)

    # println("make_informed_merge_proposal!")
    # @show proposal, current
    suffstats_p = EqualitySampler.apply_partition_to_suffstats(suffstats, current)
    k = EqualitySampler._get_k(suffstats_p)

    probvec = get_probvec_mean_distances(_get_means(suffstats_p))
    c_picked = rand(Distributions.Categorical(probvec))

    i1, i2 = linear_index_to_triangle_index(c_picked, k)

    copyto!(proposal, current)
    for i in eachindex(proposal)
        if proposal[i] == i2
            proposal[i] = i1
        end
    end
    EqualitySampler.reduce_model_2_and_clear_lookup!(proposal, lookup)

    @assert proposal != current
    return proposal

end

function log_prob_informed_merge_proposal(current::AbstractVector{<:Integer}, proposal::AbstractVector{<:Integer}, suffstats)

    # println("log_prob_informed_merge_proposal")
    # @show current, proposal
    suffstats_p = EqualitySampler.apply_partition_to_suffstats(suffstats, current)
    k = EqualitySampler._get_k(suffstats_p)

    probvec = get_probvec_mean_distances(_get_means(suffstats_p))

    i_diff = findfirst(i -> current[i] != proposal[i], eachindex(current))
    i2, i1 = current[i_diff], proposal[i_diff]

    c = triangle_index_to_linear_index(i1, i2, k)

    return log(probvec[c])

end

function split_cluster(raw_means::AbstractVector)

    n = length(raw_means)
    if n == 2
        return [1], [2]
    end

    # edge case: all raw means are equal, no meaningful split is possible
    if allequal(raw_means)
        ndiv2 = n÷2
        return collect(1:ndiv2), collect(ndiv2+1:n)
    end

    split_value = StatsBase.median(raw_means)
    # some edge cases
    if all(<(split_value), raw_means)
        split_value = prevfloat(split_value)
    elseif all(>=(split_value), raw_means)
        split_value = nextfloat(split_value)
    end

    idx1 = findall(m -> m <  split_value, raw_means)::Vector{Int}
    idx2 = findall(m -> m >= split_value, raw_means)::Vector{Int}

    return idx1, idx2

end

function get_means_and_sds(suffstats, partition::AbstractVector{<:Integer}, cluster_to_split::Integer)

    idx0 = partition .== cluster_to_split
    raw_means = _get_means(suffstats)[idx0]
    raw_vars  = _get_vars(suffstats)[idx0]

    # split clusters into two groups
    idx1, idx2 = split_cluster(raw_means)
    # split_value = StatsBase.median(raw_means)

    # idx1 = findall(m -> m <  split_value, raw_means)
    # idx2 = findall(m -> m >= split_value, raw_means)

    if isempty(idx1) || isempty(idx2)

        @show suffstats, partition, cluster_to_split

    end

    mean1 = StatsBase.mean(raw_means[i] for i in idx1)
    mean2 = StatsBase.mean(raw_means[i] for i in idx2)

    sd1 = sqrt(StatsBase.mean(raw_vars[i] for i in idx1))
    sd2 = sqrt(StatsBase.mean(raw_vars[i] for i in idx2))

    return mean1, mean2, sd1, sd2

end

function make_informed_split_proposal!(proposal::AbstractVector{<:Integer}, current::AbstractVector{<:Integer}, lookup::AbstractVector{<:Integer}, suffstats)

    # println("make_informed_split_proposal!")
    # @show proposal, current
    suffstats_p = EqualitySampler.apply_partition_to_suffstats(suffstats, current)

    cluster_sizes = EqualitySampler.fast_countmap_partition(current)
    propto = (cluster_sizes .> 1) .* _get_vars(suffstats_p)
    probvec = propto ./ sum(propto)

    cluster_to_split = rand(Distributions.Categorical(probvec))

    if cluster_sizes[cluster_to_split] == 2

        copyto!(proposal, current)
        proposal[findlast(==(cluster_to_split), proposal)] = length(proposal)

    else

        mean1, mean2, sd1, sd2 = get_means_and_sds(suffstats, current, cluster_to_split)
        raw_means = _get_means(suffstats)

        copyto!(proposal, current)
        seen_stay   = false
        seen_switch = false
        c = 1
        log_prob = 0.0
        for i in eachindex(proposal)
            if proposal[i] == cluster_to_split

                if xor(seen_stay, seen_switch) && c == cluster_sizes[cluster_to_split]
                    # force there to be two clusters
                    # @show "deterministic situation", current, proposal, any_different, all_different
                    if seen_stay
                        proposal[i] = length(proposal)
                    end
                else

                    log_alloc_prob1 = Distributions.logpdf(Distributions.Normal(mean1, sd1), raw_means[i])
                    log_alloc_prob2 = Distributions.logpdf(Distributions.Normal(mean2, sd2), raw_means[i])
                    alloc_logits_2_over_1 = log_alloc_prob2 - log_alloc_prob1

                    new_assignment = rand(Distributions.BernoulliLogit(alloc_logits_2_over_1))

                    log_prob += Distributions.logpdf(Distributions.BernoulliLogit(alloc_logits_2_over_1), new_assignment)
                    if isone(new_assignment)
                        proposal[i] = length(proposal)
                        seen_switch = true
                    else
                        seen_stay = true
                    end
                    c += 1
                end
            end
        end
    end

    # @show proposal
    EqualitySampler.reduce_model_2_and_clear_lookup!(proposal, lookup)
    @assert proposal != current
    return proposal
end

function log_prob_informed_split_proposal(current::AbstractVector{<:Integer}, proposal::AbstractVector{<:Integer}, suffstats)

    # NOTE: this should compute
    # p(x_1 == 0, x_2 == 1) + p(x_1 == 1, x_2 == 0) !

    # @show proposal, current
    suffstats_p = EqualitySampler.apply_partition_to_suffstats(suffstats, current)

    cluster_sizes = EqualitySampler.fast_countmap_partition(current)
    propto = (cluster_sizes .> 1) .* _get_vars(suffstats_p)
    probvec = propto ./ sum(propto)

    cluster_to_split = current[findfirst(i-> proposal[i] != current[i], eachindex(current))]

    p1 = probvec[cluster_to_split]

    logp1 = log(p1)

    cluster_sizes[cluster_to_split] == 2 && return logp1

    mean1, mean2, sd1, sd2 = get_means_and_sds(suffstats, current, cluster_to_split)
    raw_means = _get_means(suffstats)

    logp21 = zero(logp1)
    logp22 = zero(logp1)
    seen_switch = false
    seen_stay   = false
    c = 1
    for i in eachindex(proposal)
        if current[i] == cluster_to_split

            if xor(seen_stay, seen_switch) && c == cluster_sizes[cluster_to_split]

                # @show "logprob deterministic situation", current, proposal, seen_stay, seen_switch

                # deterministic situation
                # logp2 += 0.0
                continue
            else

                log_alloc_prob1 = Distributions.logpdf(Distributions.Normal(mean1, sd1), raw_means[i])
                log_alloc_prob2 = Distributions.logpdf(Distributions.Normal(mean2, sd2), raw_means[i])
                # @show Distributions.logcdf(Distributions.Logistic(), log_alloc_prob2 - log_alloc_prob1)

                # TODO: this needs to change to switch & stay
                # logp2 += Distributions.logpdf(Distributions.BernoulliLogit(log_alloc_prob2 - log_alloc_prob1), current[i] != proposal[i])

                logp21 += Distributions.logpdf(Distributions.BernoulliLogit(log_alloc_prob2 - log_alloc_prob1), current[i] != proposal[i])
                logp22 += Distributions.logpdf(Distributions.BernoulliLogit(log_alloc_prob2 - log_alloc_prob1), current[i] == proposal[i])

                if current[i] == proposal[i]
                    seen_stay = true
                    # logp2 += -LogExpFunctions.log1pexp(log_alloc_prob2 - log_alloc_prob1)
                else
                    seen_switch = true
                    # logp2 += -LogExpFunctions.log1pexp(log_alloc_prob1 - log_alloc_prob2)
                end
                c += 1
            end
        end
    end

    return logp1 + LogExpFunctions.logaddexp(logp21, logp22)

end

function split_merge_move_informed!(current, method, buffer_split_merge_move, suffstats, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats)

    proposal, lookup = buffer_split_merge_move
    logtwo = IrrationalConstants.logtwo
    if (allunique(current) || (!allequal(current) && rand() < .5)) # merge

        # @show "merge1", current, proposal, lookup
        make_informed_merge_proposal!(proposal, current, lookup, suffstats)
        # @show "merge2", current, proposal, lookup
        log_prob_merge = log_prob_informed_merge_proposal(current, proposal, suffstats) - (logtwo * !allunique(current))
        # @show "merge3", current, proposal, lookup
        log_prob_split = log_prob_informed_split_proposal(proposal, current, suffstats) - (logtwo * !allequal(proposal))
        # @show "merge4", current, proposal, lookup
        log_proposal_ratio = log_prob_split - log_prob_merge

    else

        # @show "split1", current, proposal, lookup
        make_informed_split_proposal!(proposal, current, lookup, suffstats)
        # @show "split2", current, proposal, lookup
        log_prob_split = log_prob_informed_split_proposal(current, proposal, suffstats) - (logtwo * !allequal(current))
        # @show "split3", current, proposal, lookup
        log_prob_merge = log_prob_informed_merge_proposal(proposal, current, suffstats) - (logtwo * !allunique(proposal))
        # @show "split4", current, proposal, lookup
        log_proposal_ratio = log_prob_merge - log_prob_split

    end

    if haskey(logbf10_dict, current)
        sampling_stats.no_cache_hits += 1
        log_denominator = logbf10_dict[current]
    else
        log_denominator = compute_one_bf(suffstats, current, method, priors)[1] + logpdf_model_distinct(priors.partition, current)
        length(logbf10_dict) < max_size_logbf10_dict && (logbf10_dict[copy(current)] = log_denominator)
    end

    if haskey(logbf10_dict, proposal)
        sampling_stats.no_cache_hits += 1
        log_numerator = logbf10_dict[proposal]
    else
        log_numerator = compute_one_bf(suffstats, proposal, method, priors)[1] + logpdf_model_distinct(priors.partition, proposal)
        length(logbf10_dict) < max_size_logbf10_dict && (logbf10_dict[proposal] = log_numerator)
    end
    sampling_stats.no_cache_checks += 2

    log_posterior_ratio = log_numerator - log_denominator
    # -randexp()?
    accept = log(rand()) < log_posterior_ratio + log_proposal_ratio

    # if accept
    #     @show log_prob_merge, log_prob_split, exp(log_proposal_ratio), exp(log_posterior_ratio), exp(log_posterior_ratio - log_proposal_ratio)
    # end

    accept && copyto!(current, proposal)

    sampling_stats.slit_merge_accepted += accept
    sampling_stats.no_split_merge_moves += 1

    return current
end

function split_merge_move!(current, method, buffer_split_merge_move, suffstats, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats::SamplingStats)
    # split_merge_move_random!(  current, method, buffer_split_merge_move, suffstats, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats)
    split_merge_move_informed!(current, method, buffer_split_merge_move, suffstats, logbf10_dict, priors, max_size_logbf10_dict, sampling_stats)
end