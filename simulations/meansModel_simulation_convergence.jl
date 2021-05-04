#=

	TODO:

		don't simulate data, just compute summary statistics?

=#

# run julia -O3 -t auto simulations/meansModel_simulation_convergence.jl from a terminal
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

import Suppressor
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

	sample_sizes		= [100, 250, 500, 750, 1_000]
	no_params			= [5, 10, 15, 20]
	no_inequalities		= [20, 50, 80] # percentage
	offset 				= 0.2

	priors = (
		UniformMvUrnDistribution(1),
		BetaBinomialMvUrnDistribution(1, 1, 1),
		BetaBinomialMvUrnDistribution(1, 1, 2),
		BetaBinomialMvUrnDistribution(1, 2, 2),
		RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.887)),
		RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(1.0))
	)

	repeats = 1:no_repeats

	return Iterators.product((
		sample_sizes		= sample_sizes,
		priors				= priors,
		no_params			= no_params,
		no_inequalities		= no_inequalities,
		offset				= offset,
		repeats				= repeats
		)...
	)#, get_true_models(no_params, no_inequalities)
end

function get_true_models(no_params, no_inequalities)

	true_models_file = joinpath(dir_for_results, "true_models.jls")
	isfile(true_models_file) && return Serialization.deserialize(true_models_file)

	# uses a rejection sampler to simulate the true partitions from a uniform distribution over the urns.
	# this is not very efficient and may take a while (especially when p is large)

	# TODO: use BetaBinomialMvUrnDistribution with particular parameters!

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


function run_simulation(;mcmc_iterations::Int = 10_000, mcmc_burnin::Int = 5_000)

	simulation_options = get_simulation_params()

	println("starting simulation of $(length(simulation_options)) runs with $(Threads.nthreads()) threads")

	p = ProgressMeter.Progress(length(simulation_options))
	Threads.@threads for (i, sim) in collect(enumerate(simulation_options))

		sample_size, prior, no_params, no_inequalities, offset, repeat = sim
		filename = make_filename(sample_size, prior, no_params, no_inequalities, offset, repeat)

		ProgressMeter.next!(p; showvalues = [(:file,filename), (:i,i)])

		if !isfile(filename)

			Random.seed!(i)
			true_model = sample_true_model(no_params, no_inequalities)
			θ_true = get_θ(offset, true_model)

			Random.seed!(i)
			y, df, D, true_values = simulate_data_one_way_anova(no_params, sample_size, θ_true);

			Random.seed!(i)
			mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = Suppressor.@suppress fit_model(df, mcmc_iterations = mcmc_iterations, partition_prior = prior, mcmc_burnin = mcmc_burnin)
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


# using BenchmarkTools
# # benchmark for average_equality_constraints
# function average_equality_constraints_orig(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
# 	ρ_c = similar(ρ)
# 	# this can be done more efficiently but I'm not sure it matters when length(ρ) is small
# 	for i in eachindex(ρ)
# 		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
# 	end
# 	return ρ_c
# end

# # this one is better but for n_elem is 5 the difference is very, very small (for 50 it's noticeable)
# function average_equality_constraints_2(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})

# 	idx_vecs = [Int[] for _ in eachindex(equal_indices)]
# 	@inbounds for i in eachindex(equal_indices)
# 		push!(idx_vecs[equal_indices[i]], i)
# 	end

# 	ρ_c = similar(ρ)
# 	@inbounds for idx in idx_vecs
# 		isempty(idx) && continue
# 		ρ_c[idx] .= mean(ρ[idx])
# 	end

# 	return ρ_c
# end

# function average_equality_constraints_3(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})

# 	idx_vecs = [Int[] for _ in eachindex(equal_indices)]
# 	@inbounds for i in eachindex(equal_indices)
# 		push!(idx_vecs[equal_indices[i]], i)
# 	end

# 	ρ_c = similar(ρ)
# 	@inbounds for idx in idx_vecs
# 		isempty(idx) && continue
# 		ρ_c[idx] .= sum(ρ[idx]) / length(idx)
# 	end

# 	return ρ_c
# end

# function average_equality_constraints_4!(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})

# 	idx_vecs = [Int[] for _ in eachindex(equal_indices)]
# 	@inbounds for i in eachindex(equal_indices)
# 		push!(idx_vecs[equal_indices[i]], i)
# 	end

# 	@inbounds for idx in idx_vecs
# 		isempty(idx) && continue
# 		ρ[idx] .= sum(@view ρ[idx]) / length(idx)
# 	end

# 	return ρ
# end

# n_elem = 5
# ρ  = randn(n_elem)
# ρ .-= mean(ρ)
# eq = reduce_model(rand(collect(1:n_elem), n_elem))

# @assert average_equality_constraints_orig(ρ, eq) == average_equality_constraints_2(ρ, eq)
# @assert average_equality_constraints_orig(ρ, eq) == average_equality_constraints_3(ρ, eq)

# @btime average_equality_constraints_orig($ρ, $eq);
# @btime average_equality_constraints_2($ρ, $eq);
# @btime average_equality_constraints_3($ρ, $eq);
# @btime average_equality_constraints_4!($ρ, $eq);

# parallel testing
# a = zeros(1000)

# p = ProgressMeter.Progress(length(a))
# Threads.@threads for i = eachindex(a)
# 	a[i] = Threads.threadid()
# 	sleep(0.125)
# 	ProgressMeter.next!(p)
# end
# a

# function foo(x::Int; kw...)
# 	foo(;kw...)
# end

# function foo(;a, b)
# 	println("$a, $b")
# end

# foo(a = 1, b = 2)
# foo(1; a = 1, b = 2)

a = zeros(1000)

p = ProgressMeter.Progress(length(a))
for i in eachindex(a)
	a[i] = 1.0
	sleep(0.05)
	ProgressMeter.next!(p; showvalues = [(:file,"hoi"), (:i,i)])
end