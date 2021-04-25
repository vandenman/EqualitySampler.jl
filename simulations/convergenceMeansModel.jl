# unfinished

const interactive = isinteractive()
println("interactive = $interactive")

if !interactive
	import Pkg
	Pkg.activate(".")
end

using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import Suppressor
import Serialization
import Random
import ProgressMeter

if interactive
	include("simulations/anovaFunctions.jl")
	include("simulations/helpersTuring.jl")
else
	include("anovaFunctions.jl")
	include("helpersTuring.jl")
end

const dir_for_results = joinpath("simulations", "simulation_results")
!isdir(dir_for_results) && mkdir(dir_for_results)

function get_simulation_params(no_repeats::Int = 1)

	sample_sizes		= [100, 250, 500, 750, 1_000]
	no_params			= [5, 10]#, 15, 20]
	no_inequalities		= [20, 50, 80] # percentage
	offset 				= 0.2

	priors = (
		UniformMvUrnDistribution(1),
		# BetaBinomialMvUrnDistribution(1, 1, 1),
		# BetaBinomialMvUrnDistribution(1, 1, 2),
		# BetaBinomialMvUrnDistribution(1, 2, 2),
		# RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.887))
		# RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.0))
	)

	repeats = 1:no_repeats

	return simulation_options = Iterators.product((
		sample_sizes		= sample_sizes,
		priors				= priors,
		no_params			= no_params,
		no_inequalities		= no_inequalities,
		offset				= offset,
		repeats				= repeats
		)...
	), get_true_models(no_params, no_inequalities)
end

function get_true_models(no_params, no_inequalities)

	true_models_file = joinpath(dir_for_results, "true_models.jls")
	isfile(true_models_file) && return Serialization.deserialize(true_models_file)

	# uses a rejection sampler to simulate the true partitions from a uniform distribution over the urns.
	# this is not very efficient and may take a while (especially when p is large)

	rng = Random.seed!(42)
	true_models = Dict{Tuple{Int, Int}, Vector{Int}}()
	for p in no_params
		d = UniformMvUrnDistribution(p)
		for i in no_inequalities
			@show p, i
			s = rand(d)
			while p - count_equalities(s) - 1 != round(Int, (p * i) / 100)
				s = rand(d)
			end
			true_models[(p, i)] = s
		end
	end

	Serialization.serialize(true_models_file, true_models)
	return true_models

end

function get_θ(offset, true_model)
	θ = true_model .* offset
	θ .- mean(θ)
	θ
end

function make_filename(sample_size, prior, no_params, no_inequalities, offset)

	if prior isa UniformMvUrnDistribution
		priorstring = "uniformMv"
	elseif prior isa BetaBinomialMvUrnDistribution
		priorstring = "BetaBinomialMv_a_$(prior.α)_b_$(prior.β)"
	elseif prior isa RandomProcessMvUrnDistribution
		priorstring = "DirichletProcessMv_a_$(prior.RPM.α)"
	end

	return joinpath(
		dir_for_results,
		"results__n_$(sample_size)__prior_$(priorstring)__no_params_$(no_params)__no_inequalities_$(no_inequalities)__offset_$(offset).jls"
	)

end


function run_simulation(;mcmc_iterations::Int = 5_000)

	simulation_options, true_models = get_simulation_params()

	ProgressMeter.@showprogress for (i, sim) in enumerate(simulation_options)

		println("i = $i / $(length(simulation_options))")
		Random.seed!(i)

		sample_size, prior, no_params, no_inequalities, offset, _ = sim
		filename = make_filename(sample_size, prior, no_params, no_inequalities, offset)

		if !isfile(filename)

			Random.seed!(sample_size)
			true_model = true_models[(no_params, no_inequalities)]
			θ_true = get_θ(offset, true_model)

			y, df, D, true_values = simulate_data_one_way_anova(no_params, sample_size, θ_true);

			mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, iterations = mcmc_iterations, partition_prior = prior)
			incl_probs = LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq))
			included = incl_probs .>= 0.5

			result = Dict(
				:y				=>		y,
				:df				=>		df,
				:D				=>		D,
				:true_values	=>		true_values,
				:mean_θ_cs_eq	=>		mean_θ_cs_eq,
				:θ_cs_eq		=>		θ_cs_eq,
				:chain_eq		=>		chain_eq,
				:model_eq		=>		model_eq,
				:incl_probs		=>		incl_probs,
				:included		=>		included,
				:seed			=>		i,
				:true_model		=>		true_model,
				:sim			=>		sim
			)

			Serialization.serialize(filename, result)
		end
	end
end

run_simulation()
# @code_warntype run_simulation()


# benchmark for average_equality_constraints
function average_equality_constraints_orig(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
	ρ_c = similar(ρ)
	# this can be done more efficiently but I'm not sure it matters when length(ρ) is small
	for i in eachindex(ρ)
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	return ρ_c
end

# this one is better but for n_elem is 5 the difference is very, very small (for 50 it's noticeable)
function average_equality_constraints_2(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})

	idx_vecs = [Int[] for _ in eachindex(equal_indices)]
	@inbounds for i in eachindex(equal_indices)
		push!(idx_vecs[equal_indices[i]], i)
	end

	ρ_c = similar(ρ)
	@inbounds for idx in idx_vecs
		isempty(idx) && continue
		ρ_c[idx] .= mean(ρ[idx])
	end

	return ρ_c
end

n_elem = 5
ρ  = randn(n_elem)
ρ .-= mean(ρ)
eq = reduce_model(rand(collect(1:n_elem), n_elem))

@assert average_equality_constraints_orig(ρ, eq) == average_equality_constraints_2(ρ, eq)
@btime average_equality_constraints_orig($ρ, $eq);
@btime average_equality_constraints_2($ρ, $eq);