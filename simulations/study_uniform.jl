using EqualitySampler, EqualitySampler.Simulations, MCMCChains
using Plots
import Turing
include("simulation_helpers.jl")

no_groups = 9
no_obs_per_group = 100
true_partition = ones(Int, no_groups)
true_θ = zeros(Float64, no_groups)


mcmc_settings = MCMCSettings(;iterations = 7000, chains = 1)
mcmc_settings2 = MCMCSettings(;iterations = 5000, burnin=1, chains = 1)

spl = 0.00

partition_prior = UniformMvUrnDistribution(no_groups)
# partition_prior = BetaBinomialMvUrnDistribution(no_groups)

# TODO: it looks like the PartitionSampler fails?
# store multiple chains and inspect rhat + ess_per_sec!
# - write a manual logpdf for the problem
# - verify results with values computed by Turing (chain_arr[1][:lp])




nsim = 6
# post_probs_arr = Vector{Matrix{Float64}}(undef, nsim)
# chns = Any[]
res = map(1:nsim) do i
# for i in 1:nsim

	data = simulate_data_one_way_anova(no_groups, no_obs_per_group, true_θ, true_partition)
	dat = data.data

	chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings)
	# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings2, spl = Turing.SMC())
	# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = spl)
	# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = Turing.PG(1))
	partition_samples = Int.(Array(group(chain, :partition)))
	post_probs = compute_post_prob_eq(partition_samples)
	# post_probs_arr[i] = post_probs

	return (;chain, post_probs)

end

chain_arr = [res[i].chain for i in 1:nsim]
post_probs_arr = [res[i].post_probs for i in 1:nsim]

plot(group(chain_arr[1], Symbol("one_way_anova_mv_ss_submodel.θ_r")).value.data[:, :, 1])
plot(group(chain_arr[1], Symbol("one_way_anova_mv_ss_submodel.g")).value.data[:, :, 1])

hcat([summarystats(chn).nt.ess_per_sec for chn in chain_arr]...)
hcat([MCMCChains.wall_duration(chn) for chn in chain_arr]...)
run(`beep_finished.sh 0`)

group(chain_arr[1], :partition)
p1 = plot(group(chain_arr[1], :θ_cs).value.data[:, 1:9, 1])
p2 = plot(group(chain_arr[2], :θ_cs).value.data[:, 1:9, 1])
plot(p1, p2, layout = (1, 2))
plot(group(chain_arr[2], :partition).value.data[:, 8, 1])

data = simulate_data_one_way_anova(no_groups, no_obs_per_group, true_θ, true_partition)
dat = data.data

mcmc_settings2 = MCMCSettings(;iterations = 7000, burnin=1, chains = 1)
chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings2, spl = Turing.SMC())


post_probs_arr[1]
post_probs_arr[3]

# post_probs_arr[4]
hcat(
	any_incorrect.(post_probs_arr, Ref(true_partition)),
	prop_incorrect.(post_probs_arr, Ref(true_partition)),
	prop_incorrect.(post_probs_arr, :null),
	any_incorrect.(post_probs_arr, :null)
)
post_probs_arr[3]
post_probs_arr[6]

exp(logpdf(partition_prior, fill(1, no_groups)))
exp(logpdf(BetaBinomialMvUrnDistribution(no_groups), fill(1, no_groups)))

# Uniform
#  1.0  0.888889  0.888889  1.0
#  1.0  0.527778  0.527778  1.0
#  1.0  0.944444  0.944444  1.0
#  1.0  0.861111  0.861111  1.0
#  1.0  0.944444  0.944444  1.0
#  1.0  0.833333  0.833333  1.0
#  1.0  0.75      0.75      1.0
#  1.0  0.777778  0.777778  1.0
#  1.0  0.805556  0.805556  1.0
#  1.0  0.805556  0.805556  1.0

using Distributions
import LinearAlgebra, FillArrays
function logpdf_unconstrained(θ, obs_mean, obs_var, obs_n, Q, partition = nothing)

	n_groups = length(obs_mean)

	μ_grand = θ[1]
	σ²		= θ[2]
	g		= θ[3]
	θ_r		= view(θ, 4:2+n_groups)

	lpdf  = zero(eltype(θ))
	lpdf += logpdf(EqualitySampler.JeffreysPriorVariance(), σ²)
	lpdf += logpdf(InverseGamma(0.5, 0.5; check_args = false), g)
	# lpdf += logpdf(Turing.filldist(Normal(), n_groups - 1), θ_r)
	lpdf += logpdf(Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1))), θ_r)

	θ_s = Q * (sqrt(g) .* θ_r)

	θ_cs = isnothing(partition) ? θ_s : EqualitySampler.Simulations.average_equality_constraints(θ_s, partition)

	for i in eachindex(obs_mean)
		lpdf += EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end
	return lpdf
end
function logpdf_manual(θ, obs_mean, obs_var, obs_n, Q, partition = nothing, partition_prior = nothing)
	log_abs_jac = log(θ[2]) + log(θ[3])
	log_partition_prior = isnothing(partition) || isnothing(partition_prior) ? 0.0 : logpdf(partition_prior, partition)
	return logpdf_unconstrained(θ, obs_mean, obs_var, obs_n, Q, partition) + log_abs_jac + log_partition_prior
end


no_groups = 9
no_obs_per_group = 100
true_partition = ones(Int, no_groups)
true_θ = zeros(Float64, no_groups)

mcmc_settings = MCMCSettings(;iterations = 5000, burnin = 1, chains = 1)

partition_prior = UniformMvUrnDistribution(no_groups)
data = simulate_data_one_way_anova(no_groups, no_obs_per_group, true_θ, true_partition)
dat = data.data
[mean(view(dat.y, g)) for g in dat.g]

chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings)#, spl = 0.01)
partition_samples = Int.(Array(group(chain, :partition)))
post_probs = compute_post_prob_eq(partition_samples)
compute_post_prob_eq(rand(partition_prior, 10_000)')

chain_bb = anova_test(dat, BetaBinomialMvUrnDistribution(no_groups); mcmc_settings = mcmc_settings)
partition_samples_bb = Int.(Array(group(chain_bb, :partition)))
post_probs_bb = compute_post_prob_eq(partition_samples_bb)
compute_post_prob_eq(rand(BetaBinomialMvUrnDistribution(no_groups), 10_000)')

chain_smc = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = Turing.SMC())
partition_samples_smc = Int.(Array(group(chain_smc, :partition)))
post_probs_smc = compute_post_prob_eq(partition_samples_smc)

function plot_params(chain)
	eq_sampler = Symbol("θ_cs[1]") in names(chain)
	no_plts = eq_sampler ? 6 : 5
	plts = Vector{Plots.Plot}(undef, no_plts)
	for (i, s) in enumerate(("μ_grand", "σ²", "g", "θ_r"))
		ss = eq_sampler ? Symbol("one_way_anova_mv_ss_submodel." * s) : s
		plts[i] = plot(group(chain, ss).value.data[:, :, 1], legend = false, title = s)
	end
	if eq_sampler
		plts[5] = plot(group(chain, :θ_cs).value.data[:, :, 1], title = "θ_cs",      legend = false)
		plts[6] = plot(partition_samples,                       title = "partition", legend = false)
	else
		plts[5] = plot(partition_samples,                       title = "partition", legend = false)
	end

	plts
end

pdf(UniformMvUrnDistribution(no_groups),          ones(Int, no_groups))
pdf(BetaBinomialMvUrnDistribution(no_groups),     ones(Int, no_groups))
pdf(DirichletProcessMvUrnDistribution(no_groups), ones(Int, no_groups))

function plot_lpdf(chain, dat, partition_prior; skip = 50)
	lpdfs = vec(chain[:lp])
	plot(lpdfs[skip:end], legend = :outerright)

	no_groups = length(dat.g)
	obs_mean, obs_var, obs_n, Q = EqualitySampler.Simulations.prep_model_arguments(dat)
	starting_values = EqualitySampler.Simulations.get_starting_values(dat, false)
	init_params = EqualitySampler.Simulations.get_init_params(starting_values...)

	hline!([logpdf_manual(init_params[no_groups+1:end], obs_mean, obs_var, obs_n, Q, true_partition, partition_prior)])

end
plot_lpdf(chain, dat,    UniformMvUrnDistribution(no_groups))
plot_lpdf(chain_bb, dat, BetaBinomialMvUrnDistribution(no_groups))

chain.value.data[1, :, 1]
plts    = plot_params(chain)
plot(plts...)
plts_bb = plot_params(chain_bb)
plot(plts_bb...)

model_full_true_partition = EqualitySampler.Simulations.one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, true_partition)
chain_full = sample(model_full_true_partition, Turing.NUTS(), 1000)
plts_full = plot_params(chain_full)
plot(plts_full...)
plot_lpdf(chain_full, dat, nothing)

logpdf(partition_prior, true_partition)
logpdf(partition_prior, partition)
logpdf_model_distinct(partition_prior, true_partition)
logpdf_model_distinct(partition_prior, partition)

function eval_prior_probs(priors, ks = 2:2:10)
	tb = Matrix{Any}(undef, length(priors), length(ks)+1)
	tb[:, 1] .= priors
	for (i, ng) in enumerate(ks)
		pdf_null = [p === :Westfall ? NaN : pdf(instantiate_prior(p, ng), ones(Int, ng)) for p in priors]
		tb[:, i+1] .= pdf_null
	end
	vcat(hcat("prior", "K=" .* string.(2:2:10)...), tb)
end
priors = get_priors()
eval_prior_probs(priors)

ks = 2:2:10
tb = Matrix{Any}(undef, length(priors), length(ks)+1)
tb[:, 1] .= priors
for (i, ng) in enumerate(ks)
	probs = zeros(ng)
	probs[ng ÷ 2] = 1.0
	D = CustomInclusionMvUrnDistribution(ng, probs)
	partition = rand(D)
	pdf_null = [p === :Westfall ? NaN : pdf_mo(instantiate_prior(p, ng), partition) for p in priors]
	tb[:, i+1] .= pdf_null
end
vcat(hcat("prior", "K=" .* string.(2:2:10)...), tb)


pdf(BetaBinomialMvUrnDistribution(no_groups), true_partition)
pdf(BetaBinomialMvUrnDistribution(no_groups), partition)
pdf_model_distinct(BetaBinomialMvUrnDistribution(no_groups), true_partition)
pdf_model_distinct(BetaBinomialMvUrnDistribution(no_groups), partition)


starting_values = EqualitySampler.Simulations.get_starting_values(dat)
init_params = EqualitySampler.Simulations.get_init_params(starting_values...)

partition = Int.(param_chain.value.data[i, 1:9, 1])

logpdf_manual(init_params[no_groups+1:end], obs_mean, obs_var, obs_n, Q, Int.(init_params[1:no_groups]), UniformMvUrnDistribution(no_groups))

obs_mean, obs_var, obs_n, Q = EqualitySampler.Simulations.prep_model_arguments(dat)

lpdfs = copy(vec(chain[:lp]))

param_chain = MCMCChains.get_sections(chain, :parameters)

lpdf_manual = similar(lpdfs)
for i in eachindex(lpdfs)

	partition = Int.(param_chain.value.data[i, 1:no_groups, 1])
	theta = param_chain.value.data[i, no_groups+1:no_groups+1+no_groups+2, 1]

	lpdf_manual[i] = logpdf_manual(theta, obs_mean, obs_var, obs_n, Q, partition, partition_prior)
end
lpdfs ≈ lpdf_manual
plot(lpdfs[5:end], legend = :outerright)
hline!([logpdf_manual(init_params[no_groups+1:end], obs_mean, obs_var, obs_n, Q, true_partition, UniformMvUrnDistribution(no_groups))])

lpdfs_bb = copy(vec(chain_bb[:lp]))
plot(lpdfs_bb[100:end], legend = :outerright)
hline!([logpdf_manual(init_params[no_groups+1:end], obs_mean, obs_var, obs_n, Q, Int.(init_params[1:no_groups]), BetaBinomialMvUrnDistribution(no_groups))])


c = (
	partition	= Int.(param_chain.value.data[1, 1:9, 1]),
	μ_grand		= param_chain.value.data[1, 10, 1],
	σ²			= param_chain.value.data[1, 11, 1],
	g			= param_chain.value.data[1, 12, 1],
	θ_r			= param_chain.value.data[1, 13:11+no_groups, 1]
)

lp = (partition, c) -> begin
	θ = vcat(c.μ_grand, c.σ², c.g, c.θ_r)
	logpdf_manual(θ, obs_mean, obs_var, obs_n, Q, partition, UniformMvUrnDistribution(no_groups))
	# logpdf_manual(θ, obs_mean, obs_var, obs_n, Q, partition, BetaBinomialMvUrnDistribution(no_groups))
end
function sample_next_values(c, lp)

	n_groups = length(c.partition)
	probvec = zeros(Float64, n_groups)
	nextValues = copy(c.partition)
	cache_idx = 0
	cache_value = -Inf # defined here to extend the scope beyond the if statement and for loop

	#=
		TODO: rather than enumerating all values, this could also enumerate the distinct models
		for example, given partition = [1, 2, 1, 1, 1] and j = 2 it makes no sense for to enumerate i = 1:5
		instead we can recognize that [1, 2, 1, 1, 1] = [1, 3, 1, 1, 1] = ... = [1, 5, 1, 1, 1]
		this implies we would do
		probvec[1] = logpdf(..., partition = [1, 1, 1, 1, 1])    (i = 1)
		probvec[2:5] .= logpdf(..., partition = [1, 2, 1, 1, 1]) (i = 2)
		which should save a bunch of likelihood evaluations whenever the current partition is far a model that implies everything is distinct
	=#

	new_label_log_posterior = 0.0
	new_label_log_posterior_computed = false
	present_labels = EqualitySampler.fast_countmap_partition_incl_zero(nextValues)

	# ~O(k^2) (double look over k) with at worst k * (k-1) likelihood evaluations if all labels are distinct and at best k * 2 likelihood evaluations
	for j in eachindex(probvec)

		oldValue = nextValues[j]
		for i in eachindex(probvec)

			nextValues[j] = i
			if nextValues[j] == cache_idx # cached from previous j

				probvec[i] = cache_value
				@show 1

			elseif !iszero(present_labels[i]) # this label is already among nextValues

				probvec[i] = lp(nextValues, c)
				@show 2

			elseif new_label_log_posterior_computed # this label is entirely new and cached

				probvec[i] = new_label_log_posterior
				@show 3

			else # this label is entirely new and not cached

				new_label_log_posterior = lp(nextValues, c)
				probvec[i] = new_label_log_posterior
				new_label_log_posterior_computed = true
				@show 4

			end

		end

		# why are the probabilities so spikey? is this also true for BetaBinomial & friends?
		probvec_normalized = exp.(probvec .- EqualitySampler.logsumexp_batch(probvec))
		if !Distributions.isprobvec(probvec_normalized)
			@show probvec, probvec_normalized
			@warn "probvec condition not satisfied! trying to normalize once more"
			probvec_normalized ./= sum(probvec_normalized)
		end

		if Distributions.isprobvec(probvec_normalized)
			# decrement the occurence of the old value
			present_labels[oldValue] -= 1
			nextValues[j] = rand(Distributions.Categorical(probvec_normalized))
			# increment the occurence of the newly sampled value
			present_labels[nextValues[j]] += 1

			if !all(>=(0), present_labels) || sum(present_labels) != length(present_labels)
				@show present_labels
				error("This should be impossible!")
			end

		elseif all(isinf, probvec) # not possible to recover from this
			return nextValues
		else
			nextValues[j] = c.partition[j]
		end

		if j != length(probvec)
			cache_idx = nextValues[j+1]
			cache_value = probvec[nextValues[j]]
		end
		new_label_log_posterior_computed = false
	end
	return nextValues

end
