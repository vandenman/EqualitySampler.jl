# TODO:

using EqualitySampler, CairoMakie
using Random
import Distributions
import ColorSchemes
include("plot_partitions (Figure 1).jl")

Random.seed!(16_02_09_04_2025) # 16:02 09-04-2025
K = 10

ρ = sort!(rand(BetaBinomialPartitionDistribution(K, 2, K/4)))#[1, 1, 1, 1, 2, 2, 3, 3]
θ0 = randn(length(unique(ρ)))
θ = θ0[ρ]
obj = simulate_data_one_way_anova(K, 25, θ, ρ, 0.0, 0.5)

obj.data

g_vec = reduce(vcat, [fill(i, length(g)) for (i, g) in enumerate(obj.data.g)])

allcolors = ColorSchemes.viridis[range(0.0, stop = 1.0, length = length(unique(ρ)) + 1)]

w = 450
fig = Figure(size = w .* (2, 3))
ax = Axis(fig[1, 1], title = "raw data", yticks = 1:K)

rainclouds!(ax, g_vec, obj.data.y,
    color = allcolors[indexin(ρ[g_vec], unique(ρ))],
    orientation = :horizontal,
    cloud_width = 1.5,
    markersize = 5
)

fig


suffstats = EqualitySampler.extract_suffstats_one_way_anova(obj.data.y, obj.data.g)

current = copy(ρ)
current_g = current[g_vec]

u_current = unique(current)

ax = Axis(fig[1, 2], title = "current partition",
            yticks = (eachindex(u_current), map(u_current) do i
                "{" * join(findall(==(i), current), ", ") * "}"
            end))

rainclouds!(ax, current_g, obj.data.y,
    color = allcolors[indexin(current_g, unique(current))],
    orientation = :horizontal,
    cloud_width = 1.5,
    markersize = 5
)

fig


suffstats_p = EqualitySampler.apply_partition_to_suffstats(suffstats, current)
k = EqualitySampler._get_k(suffstats_p)

probvec = EqualitySampler.get_probvec_mean_distances(EqualitySampler._get_means(suffstats_p))
c_picked = rand(Distributions.Categorical(probvec))

i1, i2 = EqualitySampler.linear_index_to_triangle_index(c_picked, k)

proposal = copy(current)
for i in eachindex(proposal)
    if proposal[i] == i2
        proposal[i] = i1
    end
end
EqualitySampler.reduce_model_2!(proposal)

probvec_mat = fill(NaN, k, k)
c = 1
for j in 1:k, i in j+1:k
    probvec_mat[i, j] = probvec[c]
    c += 1
end

gl = fig[2, 2] = GridLayout()

ax = Axis(gl[1, 1], title = "Merge proposal distribution", xticks = 2:k, yticks = 1:k-1)
# these two lines should be a legend
# scatter!(ax, 1.8, 7.4, color = :red, marker = :cross, markersize = 10)
# text!(ax, 2, 7, text = ": Sampled pair ($i1, $i2)", fontsize = 10)
hm = heatmap!(ax, 1:k, 1:k, probvec_mat)#, colorrange = (0.0, 1.0))
s = scatter!(ax, i2, i1, color = :red, marker = :xcross, markersize = 15, label = "Sampled clusters: $i1 & $i2")
axislegend(ax, position = :lt)
Colorbar(gl[:, end+1], hm)
fig

gl32 = fig[3, 2] = GridLayout()
ax = Axis(gl32[1, 1], title = "current\npartition")
hidedecorations!(ax)
hidespines!(ax)
plot_one_model!(ax, current, marker = collect.(string.(1:K)))
ax = Axis(gl32[1, 2], title = "proposed\npartition")
hidedecorations!(ax)
hidespines!(ax)
plot_one_model!(ax, proposal)

fig


cluster_sizes = EqualitySampler.fast_countmap_partition(current)
propto = (cluster_sizes .> 1) .* EqualitySampler._get_vars(suffstats_p)
probvec = propto ./ sum(propto)


# what we do
# cluster_to_split = rand(Distributions.Categorical(probvec))
# works better for the visualizaiton
cluster_to_split = argmax(cluster_sizes)


# visualize the first sampling step
gl = fig[1, 3] = GridLayout()
ax = Axis(gl[1, 1], title = "Split (1): Sample a cluster to split")
hm = heatmap!(ax, 1:1, 1:k, reshape(probvec, 1, k))#, colorrange = (0.0, 1.0))
s = scatter!(ax, 1, cluster_to_split, color = :red, marker = :xcross, markersize = 15, label = "Sampled cluster ($i1, $i2)")
axislegend(ax, position = :lt)
Colorbar(gl[:, end+1], hm)
fig



# this branch is never taken in the code because it's trivial
if cluster_sizes[cluster_to_split] == 2

    copyto!(proposal, current)
    proposal[findlast(==(cluster_to_split), proposal)] = length(proposal)

else

    mean1, mean2, sd1, sd2 = EqualitySampler.get_means_and_sds(suffstats, current, cluster_to_split)
    raw_means = EqualitySampler._get_means(suffstats)

    assignment_probs = zeros(length(raw_means), 2)

    proposal = copy(current)
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

                assignment_probs[i, 1] = LogExpFunctions.logistic(log_alloc_prob1)
                assignment_probs[i, 2] = 1 - assignment_probs[i, 1]

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
EqualitySampler.reduce_model_2!(proposal)


gl = fig[2, 3] = GridLayout()

to_update = findall(==(cluster_to_split), current)

new_assignment = EqualitySampler.reduce_model_2!(proposal[to_update])


ax = Axis(gl[1, 1], title = "proposal distribution", xticks = 1:2, yticks = to_update,
    limits = (0.75, 2.25, first(to_update) - 0.25, last(to_update) + 0.25))
hm = heatmap!(ax, 1:2, to_update, assignment_probs[to_update, :]')
scatter!(ax, new_assignment, to_update, color = :red, marker = :xcross, markersize = 15)
Colorbar(gl[:, end+1], hm)


# these two lines should be a legend
# scatter!(ax, 1.8, 7.4, color = :red, marker = :cross, markersize = 10)
# text!(ax, 2, 7, text = ": Sampled pair ($i1, $i2)", fontsize = 10)
# hm = heatmap!(ax, 1:k, 1:k, probvec_mat, colorrange = (0.0, 1.0))
# s = scatter!(ax, i2, i1, color = :red, marker = :xcross, markersize = 15, label = "Sampled pair ($i1, $i2)")
# axislegend(ax, position = :lt)
# Colorbar(gl[:, end+1], hm)
fig

gl32 = fig[3, 3] = GridLayout()
ax = Axis(gl32[1, 1], title = "current\npartition")
hidedecorations!(ax)
hidespines!(ax)
plot_one_model!(ax, current)
ax = Axis(gl32[1, 2], title = "proposed\npartition")
hidedecorations!(ax)
hidespines!(ax)
plot_one_model!(ax, proposal)

fig
