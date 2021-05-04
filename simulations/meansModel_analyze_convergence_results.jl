# unfinished

using EqualitySampler, Plots
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		DataFrames			as DF,
		StatsPlots			as SP

import ProgressMeter
import CategoricalArrays: categorical
import Statistics
import Serialization

function get_files()
	dir_for_results = joinpath("simulations", "simulation_results")
	files = filter!(startswith("results"), readdir(dir_for_results))
	return joinpath.(dir_for_results, files)
end

function get_sim_from_filename(file)
	# this only exists because I forgot to save the simulation settings the first time...
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

	return estimated_model
end

function compute_retrieval(true_model, estimated_model)

	@assert length(true_model) == length(estimated_model)


	false_equalities		= 0
	false_inequalities		= 0 # <- examine this to control alpha
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

function file_to_row(file)

	raw_result = Serialization.deserialize(file)

	sample_size::Int, prior, no_params::Int, no_inequalities::Int, offset::Float64, repetition::Int = raw_result[:sim]

	true_model::Vector{Int} = raw_result[:true_model]

	estimated_model::Vector{Int} = incl_probs_to_model(raw_result[:included])

	true_model		= reduce_model(true_model)
	estimated_model	= reduce_model(estimated_model)

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
		repetition			= repetition,
		posterior_means		= vec(raw_result[:mean_θ_cs_eq]),
		true_means			= raw_result[:true_values][:θ],
		correlation			= Statistics.cor(raw_result[:mean_θ_cs_eq], raw_result[:true_values][:θ])[1],
		true_model			= true_model,
		estimated_model		= estimated_model
	)
	row
end

function read_results()::DF.DataFrame

	files = get_files()

	nfiles = length(files)

	df = DF.DataFrame(
		sample_size			= Vector{Int}(undef, nfiles),
		no_params			= Vector{Int}(undef, nfiles),
		no_inequalities		= Vector{Int}(undef, nfiles),
		prior				= Vector{Any}(undef, nfiles),
		false_equalities	= Vector{Int}(undef, nfiles),
		false_inequalities	= Vector{Int}(undef, nfiles),
		true_equalities		= Vector{Int}(undef, nfiles),
		true_inequalities	= Vector{Int}(undef, nfiles),
		repetition			= Vector{Int}(undef, nfiles),
		posterior_means		= Vector{Vector{Float64}}(undef, nfiles),
		true_means			= Vector{Vector{Float64}}(undef, nfiles),
		correlation			= Vector{Float64}(undef, nfiles),
		true_model			= Vector{Vector{Int}}(undef, nfiles),
		estimated_model		= Vector{Vector{Int}}(undef, nfiles)
	)

	p = ProgressMeter.Progress(length(files))
	for (i, file) in enumerate(files)

		df[i, :] = file_to_row(file)
		ProgressMeter.next!(p)

	end

	return df

end

prior_to_string(::UniformMvUrnDistribution) = "Uniform"
prior_to_string(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α = $(d.α), β = $(d.β)"
prior_to_string(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α = $(d.rpm.α)"

# function make_title(subdf)
# 	"prior: $(prior_to_string(subdf[!, :prior][1]))\nparams: $(subdf[!, :no_params][1]) \ninequalities: $((subdf[!, :no_params][1] * subdf[!, :no_inequalities][1]) ÷ 100)"
# end
function make_title(subdf)
	"prior: $(prior_to_string(subdf[!, :prior][1]))\nparams: $(subdf[!, :no_params][1])"
end

function make_subplot(subdf, target = :false_inequalities; legend = false)
	title = make_title(subdf)
	valid = .!isnan.(subdf[!, target])
	!all(valid) && @warn "not all indices were valid for target: $target"

	return scatter(
		subdf[valid, :sample_size],
		subdf[valid, target],
		group = subdf[valid, :no_inequalities],
		title = title,
		markershape = [:circle :hexagon :rect],
		markerstrokewidth = 0.1,
		markersize = 7,
		legend = legend,
	)
end

function make_matrix_plot(grouped_df, target, priors_for_plot, allparams; width = 400)

	npriors	= length(priors_for_plot)
	nparams	= length(allparams)
	plts	= Matrix{Plots.Plot}(undef, nparams, npriors)
	for (i, subdf) in enumerate(grouped_df)

		i1 = findfirst(==(subdf[1, :no_params]), allparams)
		i2 = findfirst(==(subdf[1, :prior]), priors_for_plot)
		plt = make_subplot(subdf, target; legend = i1 == nparams && i2 == 1);

		plts[i1, i2] = plt;
	end

	layout = reverse(size(plts))
	joint_plot = plot(plts..., layout = layout, size = width .* reverse(layout))#, left_margin = 10Plots.PlotMeasures.mm)

	return joint_plot, plts
end

df = read_results()
extrema(df[!, :correlation])

sort(df[!, :correlation])

all_options = unique(df[!, :prior])
priors_for_plot = all_options[[1, 5, 6]]
allparams = sort!(unique(df[!, :no_params]))

DF.sort!(df, :no_params)
dfg = DF.groupby(DF.filter(row->row[:prior] in priors_for_plot, df), [:no_params, :prior])

for target in (:false_equalities, :false_inequalities, :true_equalities, :true_inequalities, :correlation)

	joint_plot, _ = make_matrix_plot(dfg, target, priors_for_plot, allparams)
	savefig(joint_plot, joinpath("figures", "means_model_convergence_$(target).pdf"))

end

# TODO post this online
# using Plots

# function matrixplot(nrows = 5, ncols = 4)
# 	plotmat = Matrix{Plots.Plot}(undef, nrows, ncols)
# 	for i in eachindex(plotmat)
# 		# this is more complicated in reality
# 		plotmat[i] = scatter(rand(10), rand(10))
# 	end
# 	return plotmat
# end

# plotmat = matrixplot()

# plotmat[1, 1]
# width = 300
# this seems to modify plotmat!
# plot(plotmat..., layout = size(plotmat), size = width .* reverse(size(plotmat)))
# plotmat[1, 1]


