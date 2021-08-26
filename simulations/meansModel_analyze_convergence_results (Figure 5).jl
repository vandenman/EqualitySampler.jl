#=
	julia -O3 simulations/meansModel_analyze_convergence_results.jl
=#

if !isinteractive()
	import Pkg
	Pkg.activate(".")
end


using EqualitySampler, Plots, LaTeXStrings
import Plots.PlotMeasures: mm
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		DataFrames			as DF#,
		# StatsPlots			as SP


import ProgressMeter, Statistics, Serialization, JLD, Logging, Suppressor
import CategoricalArrays: categorical

import Printf
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

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

reconstructed_to_tuple(x) = x

function file_to_row(file)

	# raw_result = Serialization.deserialize(file)
	raw_result = JLD2.load(file)

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
	# last_partial_result = last(filter(isfile, "simulations/partial_results_" .* string.(1:10) .* ".jls"))

	# if !isempty(last_partial_result)
	# 	df = Serialization.deserialize(last_partial_result)
	# 	files_to_skip = df[filter(i->isassigned(df[:, :filename], i), axes(df, 1)), :filename]
	# 	# setdiff!(files, files_to_skip)
	# 	# @info "this number of files was already read before: " nfiles - length(files)
	# 	# nfiles = length(files)
	# else

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

	# end

	p = ProgressMeter.Progress(length(files))
	# (i, file) = first(enumerate(files))
	for (i, file) in enumerate(files)
		# ugly but then the partial results always work
		# if file ∉ files_to_skip
			# @show file
			try
				df[i, :] = file_to_row(file)
			catch e
				@warn "file $file threw an error: $e"
			end

			# div, rem = divrem(i, nfiles ÷ 10)
			# if iszero(rem)
			# 	partial_filename = "simulations/partial_results_$(div).jls"
			# 	@info "Saving partial results" partial_filename
			# 	Serialization.serialize(partial_filename, df)
			# end
		# end
		ProgressMeter.next!(p)
	end

	return df

end

prior_to_string(::UniformMvUrnDistribution) = "Uniform prior"
prior_to_string(d::BetaBinomialMvUrnDistribution) = "Beta-binomial prior α = $(d.α), β = $(d.β)"
prior_to_string(d::RandomProcessMvUrnDistribution) = "Dirichlet process prior α = $(d.rpm.α)"
function prior_to_string(s::String)

	# s == "uniform"			&& return "Uniform"

	# s == "betabinom11"		&& return "BetaBinomial α=1, β=1"
	# s == "betabinomk1"		&& return "BetaBinomial α=k, β=1"
	# s == "betabinom1k"		&& return "BetaBinomial α=1, β=k"

	# s == "dppalpha0.5"		&& return "DPP α=0.5"
	# s == "dppalpha1"		&& return "DPP α=1"
	# s == "dppalphak"		&& return "DPP α=Gopalan & Berry"

	s == "uniform"			&& return "Uniform prior"

	s == "betabinom11"		&& return "Beta-binomial prior α=1, β=1"
	s == "betabinomk1"		&& return "Beta-binomial prior α=k, β=1"
	s == "betabinom1k"		&& return "Beta-binomial prior α=1, β=k"

	s == "dppalpha0.5"		&& return "Dirichlet process prior α=0.50"
	s == "dppalpha1"		&& return "Dirichlet process prior α=1"
	s == "dppalphak"		&& return "Dirichlet process prior\nα=Gopalan & Berry"

	return s

end

# function make_title(subdf)
# 	"prior: $(prior_to_string(subdf[!, :prior][1]))\nparams: $(subdf[!, :no_params][1]) \ninequalities: $((subdf[!, :no_params][1] * subdf[!, :no_inequalities][1]) ÷ 100)"
# end
function make_title(subdf)
	# latexstring(prior_to_string(subdf[1, :prior]) * "\n", L"K=%$(subdf[1, :no_params])")
	"$(prior_to_string(subdf[!, :prior][1]))\nK=$(subdf[!, :no_params][1])"
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
	subdf1 = DF.combine(DF.groupby(subdf[valid, :], [:no_inequalities, :sample_size]), target => mean; renamecols=false)
	# subdf1 = DF.combine(DF.groupby(subdf[valid, :], [:no_inequalities, :sample_size]), target => median; renamecols=false)

	return Plots.scatter(
									categorical(subdf1[!, :sample_size]),
									subdf1[!, target],
		group					=	subdf1[!, :no_inequalities],
		label					=	["0.20" "0.40" "0.60" "0.80"],
		title					=	title,
		markershape				=	[:circle :rect :star5 :diamond],
		markerstrokewidth		=	0.1,
		markersize				=	7,
		legend					=	legend,
		ylims					=	(0, 0.5),
		yticks					=	0:.1:0.5,
		alpha					=	0.5,
		foreground_color_legend	=	nothing,
		background_color_legend	=	nothing
	)
end

function make_matrix_plot(grouped_df, target, priors_for_plot, allparams; width = 400, dims = nothing)

	npriors	= length(priors_for_plot)
	nparams	= length(allparams)
	if isnothing(dims)
		plts = Matrix{Plots.Plot}(undef, nparams, npriors)
	else
		@assert prod(dims) == length(grouped_df)
		plts = Matrix{Plots.Plot}(undef, dims)
	end
	for (i, subdf) in enumerate(grouped_df)

		i1 = findfirst(==(subdf[1, :no_params]), allparams)
		i2 = findfirst(==(subdf[1, :prior]), priors_for_plot)
		plt = make_subplot(subdf, target; legend = i1 == nparams && i2 == 1);

		# plts[i1, i2] = plt;
		plts[i] = plt;
	end

	layout = reverse(size(plts))
	joint_plot = plot(plts..., layout = layout, size = width .* reverse(layout))#, left_margin = 10Plots.PlotMeasures.mm)

	return joint_plot, plts
end

function generate_anonymous_functions()

	#=
	see https://github.com/JuliaIO/JLD2.jl/issues/334#issuecomment-872136043
	the anonymous functions need to be defined
	=#
	files = get_files()
	safety = 20
	funsymbolSet = Set{Symbol}()
	ProgressMeter.@showprogress for file in files#[1:2]
		m = open(file) do io
			count = 0
			line = readline(io)
			m = match(r"Main.#(\d*)#(\d*)", line)
			while isnothing(m) && count < safety
				count += 1
				line = readline(io)
				m = match(r"Main.#(\d*)#(\d*)", line)
			end
			m
		end

		if !isnothing(m)

			funname0 = "$(m.captures[1])#$(m.captures[2])"
			funname = "var\"$(funname0)\""
			funsymbol = Symbol(funname0)
			code = Meta.parse("$(funname)(x) = x")
			push!(funsymbolSet, funsymbol)

			if !isdefined(Main, funsymbol)
				@info "creating:" funname
				@eval $code
			end
		else
			@info "m was nothing..."
		end
	end
	return funsymbolSet
end
# JLD2.load(files[1])

# Logging.LogLevel(Logging.Error)
# generate_anonymous_functions()
# df = read_results()

# Serialization.serialize("simulations/joined_results_manual.jls", df)
# if !isinteractive()
# 	Serialization.serialize("simulations/joined_results.jls", df)
# 	exit()
# end

df = Serialization.deserialize("simulations/joined_results.jls")


all_options = unique(df[!, :prior])
# priors_for_plot = all_options#[[1, 5, 6]]
priors_for_plot = all_options#[vcat(1, 3:7)]
allparams = sort!(unique(df[!, :no_params]))

DF.sort!(df, :prior)
dfg = DF.groupby(DF.filter(row->row[:prior] in priors_for_plot, df), [:no_params, :prior])
keys(dfg)

dfc2 = DF.combine(DF.groupby(DF.filter(row->row[:prior] in priors_for_plot, df), [:no_params, :prior, :no_inequalities]),
	((:false_equalities_prob, :false_inequalities_prob, :true_equalities_prob, :true_inequalities_prob, :correlation) .=> mean)...; renamecols=false)
dfc2g = DF.groupby(dfc2, [:no_params, :prior])

for subdf in dfc2g
	for row in eachrow(subdf)
		@assert sum(view(row, 4:7)) ≈ 1.0
	end
end

# w = 400
# for target in (:false_equalities_prob, :false_inequalities_prob, :true_equalities_prob, :true_inequalities_prob, :correlation)

# 	_, sub_plts = make_matrix_plot(dfg, target, priors_for_plot, allparams)#, dims = (3, 4))

# 	sub_plts2 = permutedims(reshape(sub_plts, 4, 3));
# 	for (i, plt) in enumerate(sub_plts2)
# 		plot!(plt, legend = isone(i), widen = true);
# 	end
# 	joint_plt = plot(
# 		sub_plts2...,
# 		layout = (4, 3),
# 		size = (3w, 4w)
# 	)
# 	# savefig(joint_plt, joinpath("figures", "simulation_results_test_figures_means", "means_model_convergence_$(target).pdf"))

# end

joint_plot, plts = make_matrix_plot(dfg, :false_inequalities_prob, priors_for_plot, allparams)

# figure for main body of manuscript
w = 420

plts_main = permutedims(reshape(deepcopy(plts[[5, 6, 3, 4, 11, 12]]), 2, 3));
plts_main = plts_main[:, 2:-1:1]
for (i, plt) in enumerate(plts_main)
	plot!(plt, bottom_margin = 5mm, left_margin = 5mm, right_margin = 5mm, top_margin = 5mm, xrotation = 35)
end
for (i, plt) in enumerate(plts_main)
	plot!(plt, legend = isone(i), legendtitle = "Proportion of\ninequalities");
end
plot!(plts_main[1, 1], legend = (.15, .97))

plot!(plts_main[2, 2], xlab = "Sample size");
plot!(plts_main[1, 1], ylab = "P(one or more errors)");
plot!(plts_main[1, 2], ylab = "P(one or more errors)");

# set the y-axis of some plots to 0.2
for i in 1:5
	plot!(plts_main[i], ylim = (0, .2), widen=true)
end

plt_main = plot(
	plts_main...,
	titlefont = font(16),
	layout = (2, 3),
	size = (3w, 2w)#,
	# xlab = "Sample size",
	# ylab = "Proportion false inequalities"
)
savefig(plt_main, joinpath("figures", "simulation_results_test_clean_figures", "simulation_manuscript.pdf"))

# figure for the appendix
plts_appendix = permutedims(reshape(deepcopy(plts), 4, 3));
plts_appendix = plts_appendix[[4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]]
for (i, plt) in enumerate(plts_appendix)
	plot!(plt, bottom_margin = 5mm, left_margin = 7mm, right_margin = 5mm, xrotation = 35)
end
for (i, plt) in enumerate(plts_appendix)
	plot!(plt, legend = isone(i), legendtitle = "Proportion of\ninequalities");
end
plot!(plts_appendix[1, 1], legend = (.18, .97))

for i in (1, 2, 3, 5, 7, 8, 9, 10)
	plot!(plts_appendix[i], ylim = (0, .2), widen=true)
end

plot!(plts_appendix[11], xlab = "Sample size");
for i in 1:3:12
	plot!(plts_appendix[i], ylab = "P(one or more errors)");
end

# plot!(plts_appendix[1, 1], legend = (.95, .99))
plt_appendix = plot(
	plts_appendix...,
	titlefont = font(14),
	layout = (4, 3),
	size = (3w, 4w)
)
savefig(plt_appendix, joinpath("figures", "simulation_results_test_clean_figures", "simulation_appendix.pdf"))


