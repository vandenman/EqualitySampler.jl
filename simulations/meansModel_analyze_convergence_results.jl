#=

	TODO:


=#

if !isinteractive()
	import Pkg
	Pkg.activate(".")
end


using EqualitySampler, Plots
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		DataFrames			as DF,
		StatsPlots			as SP

import ProgressMeter
import CategoricalArrays: categorical
import Statistics
import Serialization
import JLD2
import Logging
import Suppressor

if !isinteractive()
	include("meansModel_Functions.jl")
else
	include("simulations/meansModel_Functions.jl")
end

function get_files()
	dir_for_results = joinpath("simulations", "simulation_results_test")
	files = filter!(startswith("results"), readdir(dir_for_results))
	return joinpath.(dir_for_results, files)
end

function reconstructed_to_tuple(x)
	Tuple(
		if getfield(x, i) isa UInt
			Int(getfield(x, i))
		else
			getfield(x, i)
		end
		for i in 1:nfields(x)
	)
end

function file_to_row(file)

	# raw_result = Serialization.deserialize(file)
	raw_result = Suppressor.@suppress JLD2.load(file)

	# sample_size::Int, prior, no_params::Int, no_inequalities::Int, offset::Float64, repetition::Int = raw_result[:sim]
	sample_size, prior, no_params, no_inequalities, offset, repetition = reconstructed_to_tuple(raw_result["sim"])::Tuple{Int, Tuple, Int, Int, Float64, Int}

	priorname = first(prior)::String

	true_model = raw_result["true_model"]::Vector{Int}

	estimated_model = incl_probs_to_model(raw_result["included"])::Vector{Int}

	true_model		= reduce_model(true_model)
	estimated_model	= reduce_model(estimated_model)

	# @show true_model, raw_result[:included]
	retrieval_counts = compute_retrieval(true_model, raw_result["included"])
	retrieval_probs  = NamedTuple{keys(retrieval_counts), NTuple{4, Float64}}(values(retrieval_counts) ./ sum(retrieval_counts))

	false_equalities, false_inequalities, true_equalities, true_inequalities						= retrieval_counts
	false_equalities_prob, false_inequalities_prob, true_equalities_prob, true_inequalities_prob	= retrieval_probs

	row = (
		sample_size 				= sample_size,
		no_params					= no_params,
		no_inequalities				= no_inequalities,

		# prior						= prior,
		prior						= priorname,

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
		posterior_means				= vec(raw_result["mean_θ_cs_eq"]),
		true_means					= raw_result["true_values"][:θ],
		correlation					= Statistics.cor(raw_result["mean_θ_cs_eq"], raw_result["true_values"][:θ])[1],
		true_model					= true_model,
		estimated_model				= estimated_model,
		filename					= file
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
		# prior						= Vector{AbstractMvUrnDistribution}(undef, nfiles),
		prior						= Vector{String}(undef, nfiles),

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
		estimated_model				= Vector{Vector{Int}}(undef, nfiles),
		filename					= Vector{String}(undef, nfiles)
	)

	p = ProgressMeter.Progress(length(files))
	for (i, file) in enumerate(files)
		# @show file
		try
			df[i, :] = file_to_row(file)
		catch e
			@warn "file $file threw an error: $e"
		end

		div, rem = divrem(i, nfiles ÷ 10)
		if iszero(rem)
			partial_filename = "simulations/partial_results_$(div).jls"
			@info "Saving partial results" partial_filename
			Serialization.serialize(partial_filename, df)
		end

		ProgressMeter.next!(p)

	end

	return df

end

prior_to_string(::UniformMvUrnDistribution) = "Uniform"
prior_to_string(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α = $(d.α), β = $(d.β)"
prior_to_string(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α = $(d.rpm.α)"
function prior_to_string(s::String)

	s == "uniform"			&& return "Uniform"

	s == "betabinom11"		&& return "BetaBinomial (α = 1, β = 1)"
	s == "betabinomk1"		&& return "BetaBinomial (α = k, β = 1)"
	s == "betabinom1k"		&& return "BetaBinomial (α = 1, β = k)"

	# "dppalpha0.5"
	# "dppalpha1"
	# "dppalphak"

	s == "dppalpha0.5"		&& return "DirichletProcess (α = 0.5)"
	s == "dppalpha1"		&& return "BetaBinomial (α = 1)"
	s == "dppalphak"		&& return "BetaBinomial (α = f(k))"


	return s

end

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

	# use median to be slightly robust against misfitting
	subdf1 = DF.combine(DF.groupby(subdf[valid, :], [:no_inequalities, :sample_size]), target => median; renamecols=false)

	return Plots.scatter(
								categorical(subdf1[!, :sample_size]),
								subdf1[!, target],
		group				=	subdf1[!, :no_inequalities],
		title				=	title,
		markershape			=	[:circle :hexagon :rect :square],
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
	for subdf in grouped_df

		i1 = findfirst(==(subdf[1, :no_params]), allparams)
		i2 = findfirst(==(subdf[1, :prior]), priors_for_plot)
		plt = make_subplot(subdf, target; legend = i1 == nparams && i2 == 1);

		plts[i1, i2] = plt;
	end

	layout = reverse(size(plts))
	joint_plot = plot(plts..., layout = layout, size = width .* reverse(layout))#, left_margin = 10Plots.PlotMeasures.mm)

	return joint_plot, plts
end

Logging.LogLevel(Logging.Error)
df = read_results()

if !isinteractive()
	Serialization.serialize("simulations/joined_results.jls", df)
	exit()
end
# extrema(df[!, :correlation])
# sort(df[!, :correlation])

# for i in 1:nfiles
# 	div, rem = divrem(i, nfiles ÷ 10)
# 	if iszero(rem)
# 		partial_filename = "simulations/partial_results_$(div).jls"
# 		@info "Saving partial results" partial_filename
# 		# Serialization.serialize(partial_filename, df)
# 	end
# end

#=

all_options = unique(df[!, :prior])
# priors_for_plot = all_options#[[1, 5, 6]]
priors_for_plot = all_options[vcat(1, 3:7)]
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
	savefig(joint_plot, joinpath("figures", "newsim4", "medians_model_convergence_$(target).pdf"))

end

# using PlotlyBase
gr()
# plotly()

joint_plot, plts = make_matrix_plot(dfg, :false_inequalities_prob, priors_for_plot, allparams)

blankplot = plot(legend=false,grid=false,foreground_color_subplot=:white);

pltsUniform   = vcat(reshape(plts[11:12], 1, 2), [blankplot blankplot; blankplot blankplot]);
pltsBetaBinom = vcat(reshape(plts[[1, 3, 2, 4]],   2, 2), [blankplot blankplot]);
pltsDpp       =      reshape(plts[[5, 7, 9, 6, 8, 10]],  3, 2);
# plot(permutedims(pltsDpp)..., layout = (3, 2))

plotmat2 = hcat(pltsUniform, pltsBetaBinom, pltsDpp);

# joint = plot(plotmat2..., layout = ll);

# # plotly()
# w = 2400
# h =  w ÷ 2
# joint = plot(permutedims(plotmat2)..., layout = (3, 6), size = (w, h));
# path_1 = joinpath("figures", "newsim4", "medians_fancier_false_inequalities_prob.html")
# Plots.savefig(joint, path_1)
# path_2 = joinpath("ISBA2021-Talk", "Figures", "medians_fancier_false_inequalities_prob.html")
# cp(path_1, path_2; force = true)


plot!(plotmat2[1, 4], legend = false);


legendplot = Plots.scatter(
	[0.0],
	reshape(1:4, 1, 4),
	markershape = [:circle :hexagon :rect :square],
	markercolor = reshape(1:4, 1, 4),
	markersize  = fill(10, 1, 4),
	legend = false, border = :none, axis = nothing,
	xlim = (-0.5, 1),
	ylim = (0, 5)
);
Plots.annotate!(
	legendplot,
	[(0.1, i, Plots.text("$(20 * i)% of variables are equal"; halign = :left, pointsize = 14)) for i in 1:4]
);


# size(plotmat2)
plotmat2[3, 1] = legendplot;
plotmat2[3, 2] = blankplot;

# gr()
w = 2400
h = w ÷ 2
joint = plot(permutedims(plotmat2)..., layout = (3, 6), size = (w, h));
path_1 = joinpath("figures", "newsim4", "medians_fancier_false_inequalities_prob.png")
Plots.savefig(joint, path_1)
path_2 = joinpath("ISBA2021-Talk", "Figures", "medians_fancier_false_inequalities_prob.png")
cp(path_1, path_2; force = true)


# w = 400
# joint = plot(permutedims(plotmat2)..., layout = (3, 6), size = (6w, 3w));
# path_0 = joinpath("figures", "newsim4", "medians_fancier_false_inequalities_prob.pdf")
# Plots.savefig(joint, path_0)


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


=#