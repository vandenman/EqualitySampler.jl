using EqualitySampler, Turing, Plots

include("../simulations/anovaFunctions.jl")

n_groups = 6
n_obs_per_group = 100

θ_raw = randn(n_groups) .* 3
θ_raw .-= mean(θ_raw)
true_model = [1, 1, 2, 2, 3, 3]
θ_true = average_equality_constraints(θ_raw, true_model)
data = simulate_data_one_way_anova(n_groups, n_obs_per_group, θ_true);

df = data.data
partition_prior = BetaBinomialMvUrnDistribution(n_groups)
obs_mean, obs_var, obs_n, Q = prep_model_arguments(df)

model = one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior)
starting_values = get_starting_values(df)
init_params = get_init_params(starting_values...)

parameters = DynamicPPL.syms(DynamicPPL.VarInfo(model))
continuous_parameters = filter(!=(:partition), parameters)
spl1 = Gibbs(
	HMC(1/32, 20, continuous_parameters...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(model.args.partition_prior), get_logπ(model)))
)
spl2 = Gibbs(
	HMC{Turing.Core.ForwardDiffAD{1}}(1/32, 20, continuous_parameters...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(model.args.partition_prior), get_logπ(model)))
)
chn1 = sample(model, spl1, 1000; init_params = init_params)
chn2 = sample(model, spl2, 1000; init_params = init_params)

fit_eq_model(data.data, prior, spl; mcmc_iterations = 10_000)

priors = (
	full			= nothing,
	uniform 		= UniformMvUrnDistribution(n_groups),
	betabinomial	= BetaBinomialMvUrnDistribution(n_groups),
	Dirichlet		= DirichletProcessMvUrnDistribution(n_groups)
)

samples = map(prior->fit_eq_model(data.data, prior; mcmc_iterations = 10_000), priors)
θ_post_means = map(samp->mean(eachrow(get_θ_cs(samp.model, samp.samples))), samples)

plts = [plot!(plot_retrieval(data.true_values.θ, θ_post_mean), title = nm) for (nm, θ_post_mean) in pairs(θ_post_means)]
plot(plts..., layout = (2, 2))

eq_samples = NamedTuple{(:uniform, :betabinomial, :Dirichlet)}((samples[2], samples[3], samples[4]))
partition_samps = map(samps->Int.(Array(group(samps.samples, :partition))), eq_samples)
eq_tables = map(partition_samp->LA.UnitLowerTriangular(compute_post_prob_eq(partition_samp)), partition_samps)
eq_tables.uniform
eq_tables.betabinomial
eq_tables.Dirichlet

# TODO process results a little bit (only plots!)

data.df
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ_Rouder(n_groups)

iterations = 20_000


# full model
mod_full   = one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q)
samps_full = sample(mod_full, NUTS(), iterations)
θ_cs_full  = get_θ_cs(mod_full, samps_full)

# plot_retrieval(θ_true, vec(mean(θ_cs_full; dims=1)))

# full model 2 | condition, see https://turinglang.github.io/DynamicPPL.jl/stable/#AbstractPPL.condition-Tuple{Model}
mod_full2 = one_way_anova_mv(obs_mean, obs_var, obs_n, Q, UniformMvUrnDistribution(length(obs_mean))) | (partition = collect(1:length(obs_mean)), )
samps_full2 = sample(mod_full2, NUTS(), iterations)
θ_cs_full2 = get_θ_cs(mod_full2, samps_full2)
# plot_retrieval(θ_true, vec(mean(θ_cs_full2; dims=1)))
samps_full2


MCMCChains.wall_duration(samps_full)
MCMCChains.wall_duration(samps_full2)

# constrained model -- uniform prior
mod_eq = one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, UniformMvUrnDistribution(n_groups))

starting_values = get_starting_values(df)
init_params = get_init_params(starting_values...)

samps_eq0 = sample(mod_eq, get_sampler(mod_eq), 10; init_params = init_params)
θ_cs_eq0 = get_θ_cs(mod_eq, samps_eq0)

plot_retrieval(θ_true, vec(mean(θ_cs_eq0; dims=1)))

partition_samps_eq0 = Int.(Array(group(samps_eq0, :partition)))


# makes sense
LA.UnitLowerTriangular(compute_post_prob_eq(partition_samps_eq0))
LA.UnitLowerTriangular(reshape([i==j for i in true_model for j in true_model], n_groups, n_groups))
LA.UnitLowerTriangular(reshape([abs(a-b) for a in θ_true for b in θ_true], n_groups, n_groups))


# PG is about 2x slower than the custom gibbs sampler
continuous_params = filter(!=(:partition), DynamicPPL.syms(DynamicPPL.VarInfo(mod_eq)))
samps_eq0 = sample(mod_eq, get_sampler(mod_eq; ϵ = 0.005), 1000)
samps_eq1 = sample(mod_eq, Gibbs(HMC(0.005, 20, continuous_params...), PG(10, :partition)), 1000)

MCMCChains.wall_duration(samps_eq0)
MCMCChains.wall_duration(samps_eq1)

# constrained model -- Beta-binomial prior
mod_eq = one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, BetaBinomialMvUrnDistribution(n_groups, n_groups, 1))

continuous_params = filter(!=(:partition), DynamicPPL.syms(DynamicPPL.VarInfo(mod_eq)))
spl0 = Gibbs(
	HMC(0.0, 20, continuous_params...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(n_groups, get_logπ(mod_eq)))
)

θ_raw0 = vec(θ_true \ Q ./ sqrt(1.0))

init_params = get_init_params(mod_eq, ones(Int, 6), θ_raw0)
sample(mod_eq, spl0, 10; init_params = init_params)

import Logging
debuglogger = Logging.ConsoleLogger(stderr, Logging.Debug)
Logging.with_logger(debuglogger) do
	sample(mod_eq, spl0, 10; init_params = init_params)
end

sample(mod_eq, MH(), 4; init_params = init_params)


NamedTuple{(:partition, Symbol("one_way_anova_mv_ss_submodel.μ_grand"))}([1, 1, 1, 1, 1, 1], 0.0)
(partition = [1, 1, 1, 1, 1, 1], Symbol("one_way_anova_mv_ss_submodel.μ_grand") = 0.0, g = 1.0, σ² = 1.0, )

DynamicPPL.VarInfo(mod_eq)

dict_2_nt(d::Dict) = NamedTuple{Tuple(keys(d))}(values(d))
function set_init_params(dict::Dict{Symbol, T}, model::DynamicPPL.Model) where T

	varinfo = Turing.VarInfo(model)
	model_keys = Set(Symbol.(keys(varinfo)))

	if any(∉(model_keys), keys(dict))

		bad_keys       = setdiff(keys(dict), model_keys)
		model_keys_str = Dict(model_keys .=> string.(model_keys))
		goodkeys       = setdiff(keys(dict), bad_keys)

		newdict = Dict{Symbol, T}()
		for key in goodkeys
			newdict[key] = dict[key]
		end

		for bad_key in bad_keys
			for (k, v) in model_keys_str
				if endswith(v, string(bad_key))
					newdict[k] = dict[bad_key]
				end
			end
		end

		missing_keys = setdiff(model_keys, keys(newdict))
		if length(missing_keys) > 0
			throw(DomainError(dict, "These keys appear in the model but not in dict: $missing_keys"))
		end

		gooddict = newdict

	else

		gooddict = dict

	end

	model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(dict_2_nt(gooddict)));
	return varinfo[Turing.SampleFromPrior()]

end

θ_raw0 = vec(θ_true \ Q ./ sqrt(1.0))
Q * (sqrt(1.0) .* θ_raw0)
set_init_params(Dict(
	:μ_grand 	=> 0.0,
	:σ²			=> 1.0,
	:θ_r		=> θ_raw0,
	:partition	=> [1, 2, 3, 4, 5, 6],
	:g			=> 1
), mod_eq)




hmc_stepsize = custom_hmc_adaptation(mod_eq, HMC(0.0, 20, continuous_params...), init_theta)

spl1 = Gibbs(HMC(0.0, 20, continuous_params...), PG(10, :partition))
samps1 = sample(mod_eq, spl0, 1000)

MCMCChains.wall_duration(samps0)
MCMCChains.wall_duration(samps1)
θ_cs_eq0 = get_θ_cs(mod_eq, samps0)
plot_retrieval(θ_true, vec(mean(θ_cs_eq0; dims=1)))


sample(mod_eq, spl0, 5)
sample(one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, UniformMvUrnDistribution(n_groups)), Gibbs(
	HMC(0.2, 20, continuous_params...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(n_groups, get_logπ(mod_eq)))
), 5)

sample(mod_eq, Gibbs(HMC(0.2, 20, continuous_params...), PG(10, :partition)), 5)
sample(one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, UniformMvUrnDistribution(n_groups)),
	Gibbs(HMC(0.2, 20, continuous_params...), PG(10, :partition)), 5)

# constrained model -- DirichletProcess prior
mod_eq = one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, DirichletProcessMvUrnDistribution(n_groups, 0.5))
continuous_params = filter(!=(:partition), DynamicPPL.syms(DynamicPPL.VarInfo(mod_eq)))
spl0 = Gibbs(
	HMC(0.003125, 20, continuous_params...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(n_groups, get_logπ(mod_eq)))
)
samps0 = sample(mod_eq, spl0, 1000)

spl1 = Gibbs(HMC(0.003125, 20, continuous_params...), PG(10, :partition))
samps1 = sample(mod_eq, spl0, 1000)

MCMCChains.wall_duration(samps0)
MCMCChains.wall_duration(samps1)



mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_model(df, mcmc_iterations = iterations)
plot(θ_cs_full') # trace plots
hcat(ests, mean_θ_cs_full) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_full, legend = :none); Plots.abline!(1, 0)

# ideally we fix the type instability here
show_code_warntype(model_full)

# equality model with multivariate prior
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, partition_prior = UniformMvUrnDistribution(n_groups))
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constra


