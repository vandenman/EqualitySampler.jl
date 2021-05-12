#=

	TODO:

		- don't simulate data, just compute summary statistics?

		- fix the issue with running things in parallel

=#

# from a terminal run julia -O3 -t auto simulations/meansModel_simulation_convergence.jl &> log.txt
println("interactive = $(isinteractive())")

if !isinteractive()
	import Pkg
	Pkg.activate(".")
end

using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

# import Suppressor
import Serialization
import Random
import ProgressMeter

if isinteractive()
	include("simulations/meansModel_Functions.jl")
	include("simulations/helpersTuring.jl")
else
	include("meansModel_Functions.jl")
	include("helpersTuring.jl")
end

const dir_for_results = joinpath("simulations", "simulation_results")
!isdir(dir_for_results) && mkdir(dir_for_results)

function get_simulation_params(no_repeats::Int = 1)

	sample_sizes		= [100, 250, 500, 750, 1_000, 10_000]
	no_params			= [5, 10]#, 15, 20]
	no_inequalities		= [20, 40, 60, 80] # percentage
	offset 				= 0.2

	priors = (
		UniformMvUrnDistribution(1),
		BetaBinomialMvUrnDistribution(1, 1, 1),
		# BetaBinomialMvUrnDistribution(1, 1, 2),
		# BetaBinomialMvUrnDistribution(1, 2, 2),
		RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.887)),
		# RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.0))
	)

	repeats = 1:no_repeats

	return Iterators.product((
		sample_sizes		= sample_sizes,
		priors				= priors,
		no_params			= no_params,
		no_inequalities		= no_inequalities,
		offset				= offset,
		repeats				= repeats
	)...)
end

function make_filename(sample_size, prior, no_params, no_inequalities, offset, repeat)

	if prior isa UniformMvUrnDistribution
		priorstring = "uniformMv"
	elseif prior isa BetaBinomialMvUrnDistribution
		priorstring = "BetaBinomialMv_a_$(prior.α)_b_$(prior.β)"
	elseif prior isa RandomProcessMvUrnDistribution
		priorstring = "DirichletProcessMv_a_$(prior.rpm.α)"
	end

	return joinpath(
		dir_for_results,
		"results__n_$(sample_size)__prior_$(priorstring)__no_params_$(no_params)__no_inequalities_$(no_inequalities)__offset_$(offset)__repeat_$(repeat).jls"
	)

end

drop_second(x) = x[1:end .!= 2]
function get_seeds_dict(simulation_options)

	# T = typeof(drop_second(first(simulation_options)))
	T = Tuple{Int64, Int64, Int64, Float64, Int64}
	seeds_dict = Dict{T, Int}()
	next_seed = 0
	for (i, sim) in enumerate(simulation_options)
		# sample_size, prior, no_params, no_inequalities, offset, repeat = sim
		key = drop_second(sim)::T#(sample_size, no_inequalities, no_params, repeat)
		if !haskey(seeds_dict, key)
			seeds_dict[key] = next_seed
			next_seed += 1
		end
	end
	seeds_dict
end

function run_simulation(;mcmc_iterations::Int = 30_000, mcmc_burnin::Int = 10_000)

	simulation_options = get_simulation_params(5)
	seeds_dict = get_seeds_dict(simulation_options)

	println("starting simulation of $(length(simulation_options)) runs with $(Threads.nthreads()) threads")

	Turing.setprogress!(false)

	p = ProgressMeter.Progress(length(simulation_options))
	Threads.@threads for (i, sim) in collect(enumerate(simulation_options))

		sample_size, prior, no_params, no_inequalities, offset, repeat = sim
		filename = make_filename(sample_size, prior, no_params, no_inequalities, offset, repeat)

		ProgressMeter.next!(p; showvalues = [(:file,filename), (:i,i)])

		if !isfile(filename)

			try

				seed = seeds_dict[drop_second(sim)]

				Random.seed!(seed)
				true_model = sample_true_model(no_params, no_inequalities)
				θ_true = get_θ(offset, true_model)

				Random.seed!(seed)
				y, df, D, true_values = simulate_data_one_way_anova(no_params, sample_size, θ_true);

				Random.seed!(seed)
				mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df,
					mcmc_iterations		= mcmc_iterations,
					partition_prior		= prior,
					mcmc_burnin			= mcmc_burnin,
					verbose				= false,
					progress			= false
				)
				incl_probs = LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq))
				included = incl_probs .>= 0.5

				result = Dict(
					# :y				=>		y,
					# :df				=>		df,
					# :D				=>		D,
					:true_values	=>		true_values,
					:mean_θ_cs_eq	=>		mean_θ_cs_eq,
					:θ_cs_eq		=>		θ_cs_eq,
					# :chain_eq		=>		chain_eq,
					# :model_eq		=>		model_eq,
					:incl_probs		=>		incl_probs,
					:included		=>		included,
					:seed			=>		i,
					:true_model		=>		true_model,
					:sim			=>		sim
				)

				Serialization.serialize(filename, result)

			catch e

				@warn "file $filename failed with error $e"

			end

		end
	end
end

run_simulation()