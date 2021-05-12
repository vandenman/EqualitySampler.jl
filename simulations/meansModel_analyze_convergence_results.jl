#=

	TODO:


=#

using EqualitySampler, Plots
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		DataFrames			as DF,
		StatsPlots			as SP

import ProgressMeter
import CategoricalArrays: categorical
import Statistics
import Serialization

include("simulations/meansModel_Functions.jl")

function get_files()
	dir_for_results = joinpath("simulations", "simulation_results")
	files = filter!(startswith("results"), readdir(dir_for_results))
	return joinpath.(dir_for_results, files)
end

function file_to_row(file)

	raw_result = Serialization.deserialize(file)

	# sample_size::Int, prior, no_params::Int, no_inequalities::Int, offset::Float64, repetition::Int = raw_result[:sim]
	sample_size, prior, no_params, no_inequalities, offset, repetition = raw_result[:sim]::Tuple{Int, AbstractMvUrnDistribution, Int, Int, Float64, Int}

	true_model = raw_result[:true_model]::Vector{Int}

	estimated_model = incl_probs_to_model(raw_result[:included])::Vector{Int}

	true_model		= reduce_model(true_model)
	estimated_model	= reduce_model(estimated_model)

	# @show true_model, raw_result[:included]
	retrieval_counts = compute_retrieval(true_model, raw_result[:included])
	retrieval_probs  = NamedTuple{keys(retrieval_counts), NTuple{4, Float64}}(values(retrieval_counts) ./ sum(retrieval_counts))

	false_equalities, false_inequalities, true_equalities, true_inequalities						= retrieval_counts
	false_equalities_prob, false_inequalities_prob, true_equalities_prob, true_inequalities_prob	= retrieval_probs

	row = (
		sample_size 				= sample_size,
		no_params					= no_params,
		no_inequalities				= no_inequalities,
		prior						= prior,

		# retrieval_counts			= retrieval_counts,
		# retrieval_probs			= retrieval_probs,

		false_equalities			= false_equalities,
		false_inequalities			= false_inequalities,
		true_equalities				= true_equalities,
		true_inequalities			= true_inequalities,

		false_equalities_prob		= false_equalities_prob,
		false_inequalities_prob		= false_inequalities_prob,
		true_equalities_prob		= true_equalities_prob,
		true_inequalities_prob		= true_inequalities_prob,

		repetition					= repetition,
		posterior_means				= vec(raw_result[:mean_θ_cs_eq]),
		true_means					= raw_result[:true_values][:θ],
		correlation					= Statistics.cor(raw_result[:mean_θ_cs_eq], raw_result[:true_values][:θ])[1],
		true_model					= true_model,
		estimated_model				= estimated_model
	)
	return row
end

function read_results()::DF.DataFrame

	files = get_files()

	nfiles = length(files)

	# retrieval_keys = (:false_equalities, :false_inequalities, :true_equalities, :true_inequalities)

	df = DF.DataFrame(
		sample_size					= Vector{Int}(undef, nfiles),
		no_params					= Vector{Int}(undef, nfiles),
		no_inequalities				= Vector{Int}(undef, nfiles),
		prior						= Vector{AbstractMvUrnDistribution}(undef, nfiles),

		# retrieval_counts			= Vector{NamedTuple{retrieval_keys, NTuple{4, Float64}}}(undef, nfiles),
		# retrieval_probs			= Vector{NamedTuple{retrieval_keys, NTuple{4, Float64}}}(undef, nfiles),

		false_equalities			= Vector{Int}(undef, nfiles),
		false_inequalities			= Vector{Int}(undef, nfiles),
		true_equalities				= Vector{Int}(undef, nfiles),
		true_inequalities			= Vector{Int}(undef, nfiles),

		false_equalities_prob		= Vector{Float64}(undef, nfiles),
		false_inequalities_prob		= Vector{Float64}(undef, nfiles),
		true_equalities_prob		= Vector{Float64}(undef, nfiles),
		true_inequalities_prob		= Vector{Float64}(undef, nfiles),


		repetition					= Vector{Int}(undef, nfiles),
		posterior_means				= Vector{Vector{Float64}}(undef, nfiles),
		true_means					= Vector{Vector{Float64}}(undef, nfiles),
		correlation					= Vector{Float64}(undef, nfiles),
		true_model					= Vector{Vector{Int}}(undef, nfiles),
		estimated_model				= Vector{Vector{Int}}(undef, nfiles)
	)

	p = ProgressMeter.Progress(length(files))
	for (i, file) in enumerate(files)
		# @show file
		try
			df[i, :] = file_to_row(file)
		catch e
			@warn "file $file threw an error: $e"
		end

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

	# yvalue = subdf[valid, target]
	# if length(yvalue) > 1
	# 	yvalue = mean(yvalue)
	# end

	subdf1 = DF.combine(DF.groupby(subdf[valid, :], [:no_inequalities, :sample_size]), target => mean; renamecols=false)

	return scatter(
								categorical(subdf1[!, :sample_size]),
								subdf1[!, target],
		group				=	subdf1[!, :no_inequalities],
		title				=	title,
		markershape			=	[:circle :hexagon :rect],
		markerstrokewidth	=	0.1,
		markersize			=	7,
		legend				=	legend,
		ylims				=	(0, 1),
		yticks				=	0:.2:1.0
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
# extrema(df[!, :correlation])
# sort(df[!, :correlation])


all_options = unique(df[!, :prior])
priors_for_plot = all_options#[[1, 5, 6]]
allparams = sort!(unique(df[!, :no_params]))

DF.sort!(df, :no_params)
dfg = DF.groupby(DF.filter(row->row[:prior] in priors_for_plot, df), [:no_params, :prior])

dfc2 = DF.combine(DF.groupby(DF.filter(row->row[:prior] in priors_for_plot, df), [:no_params, :prior, :no_inequalities]),
	((:false_equalities_prob, :false_inequalities_prob, :true_equalities_prob, :true_inequalities_prob, :correlation) .=> mean)...; renamecols=false)
dfc2g = DF.groupby(dfc2, [:no_params, :prior])

for subdf in dfc2g
	for row in eachrow(subdf)
		@assert sum(view(row, 4:7)) ≈ 1.0
	end
end

for target in (:false_equalities_prob, :false_inequalities_prob, :true_equalities_prob, :true_inequalities_prob, :correlation)

	joint_plot, _ = make_matrix_plot(dfg, target, priors_for_plot, allparams)
	savefig(joint_plot, joinpath("figures", "newsim2", "means_model_convergence_$(target).pdf"))

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


