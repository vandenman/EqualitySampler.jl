"""
Find an initial partition using a greedy search.

This finds all overlapping intervals specified by `lower_ci` and `upper_ci` as an adjacency matrix.
Next, it uses a greedy search to approximate this adjacency matrix (i.e., cover the matrix using disjoint complete graphs)
Often, the greedy search procedure needs to choose among a set of improvement that all yield the same immediate improvement.
This decision is done at random, and therefore the procedure is repeated multiple times, specified by `no_repeats`.
"""
function find_initial_partition(lower_ci, upper_ci; verbose::Bool = true, no_repeats::Integer = 10)

    length(lower_ci) == length(upper_ci) || throw(ArgumentError("lower_ci and upper_ci do not have the same length."))
    all(lower_ci[i] <= upper_ci[i] for i in eachindex(upper_ci)) || throw(ArgumentError("Not allow lower_ci[i] <= upper_ci[i]."))
    no_repeats >= zero(no_repeats) || throw(DomainError("no_repeats should be non-negative."))

    k = length(lower_ci)

    # construct adjacency matrix of overlapping intervals
    # this could probably be improved by sorting the intervals first...
    # then we could make some assumptions
    overlap_mat = zeros(Int, k, k)
    for i in axes(overlap_mat, 1), j in i+1:size(overlap_mat, 2)
        overlap_mat[i, j] = overlap_mat[j, i] =
            (lower_ci[j] <= lower_ci[i] <= upper_ci[j]) ||
            (lower_ci[j] <= upper_ci[i] <= upper_ci[j])
    end

    prog = ProgressMeter.Progress(no_repeats; enabled = verbose && !iszero(no_repeats))
    best_cluster, adj_best_cluster = greedy_clustering(overlap_mat)
    error = sum(abs, overlap_mat - adj_best_cluster)
    ProgressMeter.next!(prog)

    for _ in 1:no_repeats
        new_cluster, adj_new_cluster = greedy_clustering(overlap_mat)
        new_error = sum(abs, overlap_mat - adj_best_cluster)
        if new_error < error
            best_cluster, adj_best_cluster, error = new_cluster, adj_new_cluster, new_error
        end
        ProgressMeter.next!(prog)
    end

    return cluster_to_partition(best_cluster), best_cluster, adj_best_cluster, error

end

function greedy_clustering(A::AbstractMatrix{<:Integer})

    n = size(A, 1)
    clusters = [Set([i]) for i in 1:n]
    B = zeros(Int, n, n)
    temp_B = copy(B)

    prev_increase = Inf

    bestpairs  = Matrix{Int}(undef, 2, binomial(n, 2))

    while length(clusters) > 1

        # Find the pair of clusters that minimizes the increase in ||A - B||_F when merged
        idx = 1

        bestvalue = cluster_helper(clusters, 1, 2, A, B, temp_B)
        bestpairs[:, 1] .= (1, 2)

        for i in eachindex(clusters)
            for j in i+1:length(clusters)

                i == 1 && j == 2 && continue

                newvalue = cluster_helper(clusters, i, j, A, B, temp_B)

                if newvalue < bestvalue
                    idx = 1
                    bestpairs[1, idx], bestpairs[2, idx] = i, j
                elseif newvalue == bestvalue
                    idx += 1
                    bestpairs[1, idx], bestpairs[2, idx] = i, j
                end
            end
        end

        # values = map(Combinatorics.combinations(eachindex(clusters), 2)) do (i, j)
        #     combined_cluster = union(clusters[i], clusters[j])
        #     # Form the complete graph for the combined cluster
        #     for u in combined_cluster, v in combined_cluster
        #         temp_B[u, v] = 1
        #         temp_B[v, u] = 1
        #     end
        #     # increase = norm(A - temp_B, 2)
        #     increase = sum(abs, A - temp_B)
        #     # reset the changes
        #     for u in combined_cluster, v in combined_cluster
        #         temp_B[u, v] = B[u, v]
        #         temp_B[v, u] = B[v, u]
        #     end
        #     increase, (i, j)
        # end

        # if the best increase is going up we're getting worse, so bail
        min_increase = bestvalue

        # @show min_increase, prev_increase, view(bestpairs, :, 1:idx)
        prev_increase < min_increase && break
        prev_increase = min_increase

        # pick one pair at random from all that did equally well
        idx = rand(1:idx)
        # idx = 1

        # Merge the clusters
        i, j = bestpairs[1, idx], bestpairs[2, idx]
        clusters[i] = union(clusters[i], clusters[j])
        deleteat!(clusters, j)

        # Update B for the new merged cluster
        for u in clusters[i], v in clusters[i]
            if u != v
                B[u, v] = 1
                B[v, u] = 1
                temp_B[u, v] = 1
                temp_B[v, u] = 1
            end
        end
    end

    return clusters, B
end

function cluster_helper(clusters, i, j, A, B, temp_B)

    combined_cluster = union(clusters[i], clusters[j])
    # Form the complete graph for the combined cluster
    for u in combined_cluster, v in combined_cluster
        if u != v
            temp_B[u, v] = 1
            temp_B[v, u] = 1
        end
    end
    # increase = LinearAlgebra.norm(A - temp_B, 2)
    # increase = sum(abs, A - temp_B)
    increase = StatsBase.L1dist(A, temp_B)
    # reset the changes
    for u in combined_cluster, v in combined_cluster
        if u != v
            temp_B[u, v] = B[u, v]
            temp_B[v, u] = B[v, u]
        end
    end
    return increase
end

function cluster_to_partition(cluster)
    k = sum(length, cluster)
    partition = zeros(Int, k)
    for (i, idx) in enumerate(cluster)
        for j in idx
            partition[j] = i
        end
    end
    @assert all(x->1 <= x <= k, partition)
    return partition
end

