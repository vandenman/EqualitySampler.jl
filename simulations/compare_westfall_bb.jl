using EqualitySampler, EqualitySampler.Simulations, DataFrames
import ProgressMeter, Random, AbstractMCMC

include("simulation_helpers.jl")

obs_per_group = 50
hypothesis = :p75
n_groups = 5
offset = 0.2
mcmc_settings = Simulations.MCMCSettings(;iterations = 10_000, burnin = 1, chains = 1)
mcmc_settings2 = MCMCSettings()
partition_prior = BetaBinomialMvUrnDistribution(n_groups, 1, binomial(n_groups, 2))

n_rep = 100
df_results = DataFrame(
	model          = Vector{Vector{Int}}(undef, n_rep),
	perf_w         = Vector{Int}(undef, n_rep),
	perf_bb        = Vector{Int}(undef, n_rep),
	perf_bb2       = Vector{Int}(undef, n_rep),
	post_probs_w   = Vector{Matrix{Float64}}(undef, n_rep),
	post_probs_bb  = Vector{Matrix{Float64}}(undef, n_rep),
	post_probs_bb2 = Vector{Matrix{Float64}}(undef, n_rep),
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
	chain2 = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings2, rng = rng, modeltype = :reduced, spl = 0.05)

	partition_samples = MCMCChains.group(chain, :partition).value.data
	post_probs = Simulations.compute_post_prob_eq(partition_samples)

	partition_samples = MCMCChains.group(chain2, :partition).value.data
	post_probs2 = Simulations.compute_post_prob_eq(partition_samples)

	perf_w   = prop_incorrect_αβ(post_probs_w, true_model, true)
	perf_bb  = prop_incorrect_αβ(post_probs,   true_model, false)
	perf_bb2 = prop_incorrect_αβ(post_probs2,   true_model, false)

	df_results[i, :model]          = true_model
	df_results[i, :perf_w]         = perf_w.α_error_prop
	df_results[i, :perf_bb]        = perf_bb.α_error_prop
	df_results[i, :perf_bb2]       = perf_bb2.α_error_prop
	df_results[i, :post_probs_w]   = post_probs_w
	df_results[i, :post_probs_bb]  = post_probs
	df_results[i, :post_probs_bb2] = post_probs2

end

(mean(df_results[!, :perf_w]), mean(df_results[!, :perf_bb]), mean(df_results[!, :perf_bb2]))
df_sub = subset(df_results, Cols(:perf_w, :perf_bb, :perf_bb2) => (w, bb, bb2) -> w .> 0 .|| bb .> 0 .|| bb2 .> 0)

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
obs_mean_diff_mat = zeros(5, 5)
for i in 1:4, j in i+1:5
	obs_mean_diff_mat[j, i] = abs(obs_mean[i] - obs_mean[j])
	obs_mean_diff_mat[i, j] = obs_mean_diff_mat[j, i]
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
