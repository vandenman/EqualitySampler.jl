using EqualitySampler, EqualitySampler.Simulations, DataFrames
import ProgressMeter, Random, AbstractMCMC
using Statistics
include("simulation_helpers.jl")

obs_per_group = 50
hypothesis = :p75
n_groups = 5
offset = 0.2
mcmc_settings = Simulations.MCMCSettings(;iterations = 10_000, burnin = 1000, chains = 1)
mcmc_settings2 = MCMCSettings()
partition_prior = BetaBinomialMvUrnDistribution(n_groups, 1, binomial(n_groups, 2))

n_rep = 100
df_results = DataFrame(
	model           = Vector{Vector{Int}}(undef, n_rep),
	perf_w          = Vector{Int}(undef, n_rep),
	perf_bb         = Vector{Int}(undef, n_rep),
	# perf_bb2        = Vector{Int}(undef, n_rep),
	post_probs_w    = Vector{Matrix{Float64}}(undef, n_rep),
	post_probs_bb   = Vector{Matrix{Float64}}(undef, n_rep),
	# post_probs_bb2  = Vector{Matrix{Float64}}(undef, n_rep),
	data_means      = Vector{Vector{Float64}}(undef, n_rep),
	data_vars       = Vector{Vector{Float64}}(undef, n_rep),
	data_ns         = Vector{Vector{Int}}(undef, n_rep),
)

seeds = 1:100
rng = Random.MersenneTwister()
# i = 9 is a bad case
ProgressMeter.@showprogress for i in 1:n_rep

	Random.seed!(rng, i)
	true_model = sample_true_model(rng, hypothesis, n_groups)
	true_θ = Simulations.normalize_θ(offset, true_model)

	data_obj = Simulations.simulate_data_one_way_anova(rng, n_groups, obs_per_group, true_θ)
	dat = data_obj.data

	fit_w = westfall_test(dat)
	post_probs_w = fit_w.log_posterior_odds_mat

	chain  = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings,  rng = rng, modeltype = :reduced, spl = 0.05)
	# chain2 = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings2, rng = rng, modeltype = :reduced, spl = 0.05)

	partition_samples = MCMCChains.group(chain, :partition).value.data
	post_probs = Simulations.compute_post_prob_eq(partition_samples)

	# partition_samples = MCMCChains.group(chain2, :partition).value.data
	# post_probs2 = Simulations.compute_post_prob_eq(partition_samples)

	perf_w   = prop_incorrect_αβ(post_probs_w, true_model, true)
	perf_bb  = prop_incorrect_αβ(post_probs,   true_model, false)
	# perf_bb2 = prop_incorrect_αβ(post_probs2,   true_model, false)

	df_results[i, :model]          = true_model
	df_results[i, :perf_w]         = perf_w.α_error_prop
	df_results[i, :perf_bb]        = perf_bb.α_error_prop
	# df_results[i, :perf_bb2]       = perf_bb2.α_error_prop
	df_results[i, :post_probs_w]   = post_probs_w
	df_results[i, :post_probs_bb]  = post_probs
	# df_results[i, :post_probs_bb2] = post_probs2

	obs_mean, obs_var, obs_n = Simulations.get_suff_stats(dat)
	df_results[i, :data_means] = obs_mean
	df_results[i, :data_vars] = obs_var
	df_results[i, :data_ns] = obs_n

end

(mean(df_results[!, :perf_w]), mean(df_results[!, :perf_bb]))#, mean(df_results[!, :perf_bb2]))
df_sub = subset(df_results, Cols(:perf_w, :perf_bb) => (w, bb) -> w .> 0 .| bb .> 0)
# df_sub = subset(df_results, Cols(:perf_w, :perf_bb, :perf_bb2) => (w, bb, bb2) -> w .> 0 .|| bb .> 0 .|| bb2 .> 0)

idx_bad = findall(>=(1), df_results.perf_bb)
df_sub = df_results[idx_bad, :]

function compute_obs_mean_abs_diff(obs_mean)
	k = size(obs_mean, 1)
	obs_mean_diff_mat = zeros(k, k)
	for i in 1:k-1, j in i+1:k
		obs_mean_diff_mat[j, i] = abs(obs_mean[i] - obs_mean[j])
		obs_mean_diff_mat[i, j] = obs_mean_diff_mat[j, i]
	end
	obs_mean_diff_mat
end
function compute_obs_mean_diff(obs_mean)
	k = size(obs_mean, 1)
	obs_mean_diff_mat = zeros(k, k)
	for i in 1:k-1, j in i+1:k
		obs_mean_diff_mat[j, i] = obs_mean[i] - obs_mean[j]
		obs_mean_diff_mat[i, j] = obs_mean_diff_mat[j, i]
	end
	obs_mean_diff_mat
end


# function check_consistency_model_pairwise_differences(model, obs_mean)
# 	obs_mean_diff_mat = compute_obs_mean_abs_diff(obs_mean)
# 	for i in axes(obs_mean_diff_mat, 1)
# 		obs_mean_diff_mat[i, i] = Inf
# 	end
# 	_, idx = findmin(obs_mean_diff_mat)
# 	consistent = model[idx[1]] == model[idx[2]]
# 	return consistent
# end

function check_consistency_model_pairwise_differences(model, obs_mean, tol = .3)
	obs_mean_diff_mat = compute_obs_mean_abs_diff(obs_mean)
	for i in axes(obs_mean_diff_mat, 1)
		obs_mean_diff_mat[i, i] = Inf
	end

	cc = EqualitySampler.fast_countmap_partition_incl_zero(model)
	i1, i2 = 0, 0
	for i in eachindex(model)
		if cc[model[i]] > 1
			if i1 == 0
				i1 = i
			else
				i2 = i
			end
		end
	end

	obs_mean_diff_mat_v = view(obs_mean_diff_mat, :, [i1, i2])

	target = obs_mean_diff_mat[i1, i2]
	idx = findall(<(target), obs_mean_diff_mat_v)
	if isnothing(idx)
		return true
	end

	for i in idx
		idx_d = i[2] == 1 ? (i[1], i2) : (i[1], i1)tol
		if obs_mean_diff_mat[idx_d[1], idx_d[2]] >= tol
			return false
		end
	end
	return true
end

df_results.post_probs_bb[1]
df_results.post_probs_bb[9]
compute_obs_mean_abs_diff(df_results.data_means[1])
model, obs_mean = df_sub.model[1], df_sub.data_means[1]
model, obs_mean = df_results.model[1], df_results.data_means[1]
model, obs_mean = df_results.model[51], df_results.data_means[51]
check_consistency_model_pairwise_differences(df_sub.model[1], df_sub.data_means[1], .4)
check_consistency_model_pairwise_differences(df_results.model[1], df_results.data_means[1], .4)

check_consistency_model_pairwise_differences.(df_sub.model, df_sub.data_means)

qb = Simulations.normalize_θ(offset, df_sub.model[1])
qb .- qb[2]
qg = Simulations.normalize_θ(offset, df_results.model[1])
qg .- qg[1]
function get_max_diff_theta(m)
	idx = findfirst(>(1), EqualitySampler.fast_countmap_partition_incl_zero(m))
	true_theta = Simulations.normalize_θ(.2, m)
	maximum(true_theta .- true_theta[idx])
end
get_max_diff_theta.(df_sub.model)
get_max_diff_theta.(df_results.model)

var(qb)
var(qg)

ms = [
	[1, 2, 3, 4],
	[1, 2, 3, 3],
	[1, 2, 2, 2],
	[1, 1, 1, 1]
]
for m in ms
	θ = Simulations.normalize_θ(.2, m)
	θ_diff = LinearAlgebra.LowerTriangular(compute_obs_mean_diff(θ))
	mean_θ_diff = mean(θ_diff[j, i] for i in 1:size(θ_diff, 1)-1 for j in i+1:size(θ_diff, 1))
	println("model = $m")
	println("θ = $θ")
	println("θ_diff")
	display(θ_diff)
	println("mean_θ_diff = $mean_θ_diff")
end

compute_obs_mean_diff(qb)
compute_obs_mean_diff(qg)
mean([qb[i] - qb[j] for i in 1:4 for j in i+1:5])
mean([qg[i] - qg[j] for i in 1:4 for j in i+1:5])

df_results2 = transform(df_results,
	[:model, :data_means] => ((m, d) -> check_consistency_model_pairwise_differences.(m, d, .2)) => :consistent
)
df_results2[!, :i] = axes(df_results2, 1)

sort(df_results2, order(:perf_bb))

df_sub.model[1]
compute_obs_mean_abs_diff(df_sub.data_means[1])

compute_obs_mean_abs_diff(df_sub.data_means[2])

df_sub[1, :model]
df_sub[1, :post_probs_w]
df_sub[1, :post_probs_bb2]
findall(df_results[!, :perf_bb2] .>= 1)

df_results[i, :model]
df_results[i, :post_probs_bb]

partition_samples_2 = Int.(vcat(partition_samples[:, :, 1], partition_samples[:, :, 2], partition_samples[:, :, 3]))
Simulations.compute_post_prob_eq(partition_samples_2)
Simulations.compute_post_prob_eq(partition_samples)

showall(x) = show(stdout, "text/plain", x)
model_probs = EqualitySampler.Simulations.compute_model_probs(partition_samples_2)
showall(model_probs)
EqualitySampler.reduce_model(true_model)
model_probs[join(EqualitySampler.reduce_model(true_model))]

obs_mean, obs_var, obs_n = Simulations.get_suff_stats(dat)
obs_mean
function compute_obs_mean_abs_diff(obs_mean)
	obs_mean_diff_mat = zeros(5, 5)
	for i in 1:4, j in i+1:5
		obs_mean_diff_mat[j, i] = abs(obs_mean[i] - obs_mean[j])
		obs_mean_diff_mat[i, j] = obs_mean_diff_mat[j, i]
	end
	obs_mean_diff_mat
end
#=
5×5 Matrix{Float64}:
0.0       0.218196  0.862712  0.513432  0.383002
0.218196  0.0       0.644517  0.295236  0.601198
0.862712  0.644517  0.0       0.349281  1.24571
0.513432  0.295236  0.349281  0.0       0.896433
0.383002  0.601198  1.24571   0.896433  0.0

The true model is 12325
The hierarchical version fails because:

	- The distance between 1 and 2 is ~0.21.
	  This is (by chance) the smallest difference betwen the means, so these groups are set equal.
	- the next smallest difference is between 2 and 4 which is ~0.29.
	  However, the difference between 1 and 4 is 0.51.
	  So 2 and 4 cannot be equal because 1 and 2 already are equal.

The Westfall approach doesn't care about this because it looks at each pair indepently of the rest.
So the Westfall approach would suggest 1 == 2 and 2 == 4, but not 1 == 4!

=#


showall(sort(model_probs, byvalue=true))
man = 0.0
for k in keys(model_probs)
	if k[2] == k[4]
		man += model_probs[k]
	end
end

mcmc_settings3 = MCMCSettings(;iterations = 5_000, chains = 5, parallel = AbstractMCMC.MCMCThreads())
chain3 = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings3, rng = rng, modeltype = :reduced, spl = 0.05)
partition_samples = MCMCChains.group(chain3, :partition).value.data
post_probs3 = Simulations.compute_post_prob_eq(partition_samples)
partition_samples_mat = reduce(vcat, eachslice(partition_samples, dims = 3))
model_probs = EqualitySampler.Simulations.compute_model_probs(partition_samples_mat)
sort(model_probs, byvalue=true, rev=true)


chain3 = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings3, rng = rng, modeltype = :old, spl = 0.05)
partition_samples = MCMCChains.group(chain3, :partition).value.data
post_probs3 = Simulations.compute_post_prob_eq(partition_samples)
