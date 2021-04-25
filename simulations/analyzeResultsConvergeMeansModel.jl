# unfinished

using EqualitySampler, Plots
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		DataFrames			as DF

import Statistics
import Serialization

function get_files()
	dir_for_results = joinpath("simulations", "simulation_results")
	files = filter!(startswith("results"), readdir(dir_for_results))
	return joinpath.(dir_for_results, files)
end

function get_sim_from_filename(file)
	# this only exists because I forgot to save the simulation settings the first time...
	file = files[1]
	interesting = last(splitdir(file))

	result = parse.(Int, [ match.match for match in eachmatch(r"(\d+)", interesting)])

	# sample_size, prior, no_params, no_inequalities, offset, _ = sim
	return result[1], result[2], result[3]

end

function get_true_model(raw_result)
	# this only exists because I forgot to save the true model the first time...
	θ = round.(raw_result[:true_values][:θ], digits = 2)

	true_model = collect(1:length(θ))
	seen = Set{Int}()
	for i in eachindex(true_model)
		if i ∉ seen
			matches = findall(isapprox(θ[i]), θ)
			true_model[matches] .= i
			union!(seen, matches)
		end
	end
	return true_model
end

function incl_probs_to_model(included)

	no_params = size(included, 1)
	estimated_model = Vector{Int}(1:no_params)

	for i in 1:no_params-1, j in i+1:no_params
		if included[j, i]
			estimated_model[j] = estimated_model[i]
		end
	end

	return reduce_model(estimated_model) # just to be sure
end

function compute_retrieval(true_model, estimated_model)

	@assert length(true_model) == length(estimated_model)

	false_equalities		= 0
	false_inequalities		= 0
	true_equalities			= 0
	true_inequalities		= 0

	no_params = length(true_model)
	for i in 1:no_params-1, j in i+1:no_params
		truthEqual		= true_model[i]			== true_model[j]
		estimatedEqual	= estimated_model[i]	== estimated_model[j]

		if truthEqual
			if estimatedEqual
				true_equalities += 1
			else
				false_inequalities += 1
			end
		else
			if estimatedEqual
				false_equalities += 1
			else
				true_inequalities += 1
			end
		end
	end

	return false_equalities, false_inequalities, true_equalities, true_inequalities

end

function read_results()::DF.DataFrame

	files = get_files()

	df = DF.DataFrame(
		sample_size			= Int[],
		no_params			= Int[],
		no_inequalities		= Int[],
		prior				= String[],
		false_equalities	= Int[],
		false_inequalities	= Int[],
		true_equalities		= Int[],
		true_inequalities	= Int[],
		posterior_means		= Vector{Vector{Float64}}(),
		true_means			= Vector{Vector{Float64}}(),
		correlation			= Float64[],
	)

	for file in files

		raw_result = Serialization.deserialize(file)

		if haskey(raw_result, :sim)
			sample_size, prior, no_params, no_inequalities, offset, _ = raw_result[:sim]
		else
			sample_size, no_params, no_inequalities = get_sim_from_filename(file)
			offset = 0.2
			prior = "uniformMv"

			get_true_model(raw_result)
		end

		true_model = haskey(raw_result, :true_model) ? raw_result[:true_model] : get_true_model(raw_result)

		estimated_model = incl_probs_to_model(raw_result[:included])

		false_equalities, false_inequalities, true_equalities, true_inequalities = compute_retrieval(true_model, estimated_model)

		row = (
			sample_size 		= sample_size,
			no_params			= no_params,
			no_inequalities		= no_inequalities,
			prior				= prior,
			false_equalities	= false_equalities,
			false_inequalities	= false_inequalities,
			true_equalities		= true_equalities,
			true_inequalities	= true_inequalities,
			posterior_means		= vec(raw_result[:mean_θ_cs_eq]),
			true_means			= raw_result[:true_values][:θ],
			correlation			= Statistics.cor(raw_result[:mean_θ_cs_eq], raw_result[:true_values][:θ])[1]
		)

		push!(df, row)

	end

	return df

end

df = read_results()
plot(df[!, :sample_size], df[!, :correlation])
extrema(df[!, :correlation])