using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import Logging
import ProgressMeter
import Serialization
import DataFrames: DataFrame
import StatsModels: @formula
import Suppressor
import Random

include("../simulations/meansModel_Functions.jl")
include("../simulations/helpersTuring.jl")
include("../simulations/limitedLogger.jl")
include("../simulations/customHMCAdaptation.jl")

function get_logπ(model)
	vari = VarInfo(model)
	mt = vari.metadata
	return function logπ(partition, nt)
		DynamicPPL.setval!(vari, partition, DynamicPPL.VarName(:partition))
		for (key, val) in zip(keys(nt), nt)
			if key !== :partition
				indices = mt[key].vns
				if !(val isa Vector)
					DynamicPPL.setval!(vari, val, indices[1])
				else
					ranges = mt[key].ranges
					for i in eachindex(indices)
						DynamicPPL.setval!(vari, val[ranges[i]], indices[i])
					end
				end
			end
		end
		DynamicPPL.logjoint(model, vari)
	end
end

function get_θ_cs(model, chain)
	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chain, :parameters))
	θ_cs = Matrix{Float64}(undef, length(gen), length(gen[1]))
	for i in eachindex(gen)
		for j in eachindex(gen[i])
			θ_cs[i, j] = gen[i][j]
		end
	end
	return θ_cs
end

function plot_retrieval(true_values, estimated_values)
	p = Plots.plot(legend=false, xlab = "True value", ylab = "Posterior mean")
	Plots.abline!(p, 1, 0)
	scatter!(p, true_values, estimated_values)
end

function get_sampler(model; ϵ::Float64 = 0.0, n_leapfrog::Int = 20)
	parameters = DynamicPPL.syms(DynamicPPL.VarInfo(model))
	if :partition in parameters

		continuous_parameters = filter(!=(:partition), parameters)
		return Gibbs(
			HMC(ϵ, n_leapfrog, continuous_parameters...),
			GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(mod_eq.args.partition_prior), get_logπ(model)))
		)

	else
		return NUTS()
	end
end


@model function one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition = nothing, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	# improper priors on grand mean and variance
	μ_grand 		~ Flat()
	σ² 				~ JeffreysPriorVariance()

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5; check_args = false)

	θ_r ~ MvNormal(LA.Diagonal(Fill(1.0, n_groups - 1)))

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in eachindex(obs_mean)
		Turing.@addlogprob! EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end
	return θ_cs

end

@model function one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	partition ~ partition_prior
	θ_cs = @submodel $(Symbol("one_way_anova_mv_ss_submodel")) one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition, T)
	return θ_cs

end

@model function one_way_anova_mv(obs_mean, obs_var, obs_n, Q, partition_prior, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	# improper priors on grand mean and variance
	μ_grand 		~ Flat()
	σ² 				~ JeffreysPriorVariance()

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5; check_args = false)

	θ_r ~ MvNormal(LA.Diagonal(Fill(1.0, n_groups - 1)))

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# sample a partition
	partition ~ partition_prior

	# constrain θ according to the partition
	θ_cs = average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in eachindex(obs_mean)
		Turing.@addlogprob! EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end
	return θ_cs

end


n_groups = 6
n_obs_per_group = 100

θ_raw = randn(n_groups) .* 3
θ_raw .-= mean(θ_raw)
true_model = [1, 1, 2, 2, 3, 3]
θ_true = average_equality_constraints(θ_raw, true_model)
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, θ_true);
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ_Rouder(n_groups)

iterations = 20_000

# function my_any(f, arr::AbstractArray{T}) where T
# 	hasmethod(f, (T, )) || throw(error("called my_any(f, $(typeof(arr))) but there exists no method f(::$T)!"))
# 	Base.return_types(f, (T, )) === Bool || throw(error("called my_any(f, $(typeof(arr))) but f(::$T) does not return Bool!"))
# 	any(f, arr)
# end
# foo(x::Float64)=x>0
# goo(x::Number)=x>0
# hoo(x::Number)=x

# my_any(foo, collect(1:3))
# my_any(goo, collect(1:3))
# my_any(hoo, collect(1:3))
# any(foo, collect(1:3))
# any(goo, collect(1:3))
# any(hoo, collect(1:3))


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
samps_eq0 = sample(mod_eq, get_sampler(mod_eq; ϵ = 0.005), 1000)
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
samps0 = sample(mod_eq, spl0, 1000)

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

