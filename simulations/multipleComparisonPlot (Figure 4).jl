# from a terminal run julia -O3 -t auto simulations/multipleComparisonPlot.jl

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

import Logging
import ProgressMeter
import Serialization
import DataFrames: DataFrame
import StatsModels: @formula
import Suppressor
import Random

if isinteractive()
	include("simulations/meansModel_Functions.jl")
	include("simulations/helpersTuring.jl")
	include("simulations/limitedLogger.jl")
	include("simulations/customHMCAdaptation.jl")
else
	include("meansModel_Functions.jl")
	include("helpersTuring.jl")
	include("limitedLogger.jl")
	include("customHMCAdaptation.jl")
end

function get_priors()
	return (
		(n_groups)->UniformMvUrnDistribution(n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, n_groups, 1.0),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, 0.5),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, 1),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, :Gopalan_Berry)
	)
end


compute_post_prob_eq_helper(x) = compute_post_prob_eq(x[3])

function any_incorrect(x)

	for j in 1:size(x, 1)-1, i in j+1:size(x, 1)
		if x[i, j] < 0.5
			return true
		end
	end
	return false
end

function prop_incorrect(x)

	count = 0
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		if x[i, j] < 0.5
			count += 1
		end
	end
	return count / (n * (n - 1) / 2)
end

function get_resultsdir()
	results_dir = joinpath("simulations", "results_multiplecomparisonsplot_200", "jls_files")
	!ispath(results_dir) && mkpath(results_dir)
	return results_dir
end

make_filename(results_dir, r, i) = joinpath(results_dir, "repeat_$(r)_groups_$(i).jls")

function run_simulation()

	n_obs_per_group = 100
	repeats = 200
	groups = 2:10

	sim_opts = Iterators.product(1:repeats, eachindex(groups))

	mcmc_iterations = 10_000
	mcmc_burnin = 2_000

	nsim = length(sim_opts)
	println("starting simulation of $(nsim) runs with $(Threads.nthreads()) threads")

	priors = get_priors()

	# results = BitArray(undef, length(priors), length(groups), repeats)
	# results_big = Array{Matrix{Float64}}(undef, length(priors), length(groups), repeats)
	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)

	Logging.disable_logging(Logging.Warn)
	# Logging.global_logger(limited_warning_logger(3))

	results_dir = get_resultsdir()

	# (r, i) = first(sim_opts)
	Threads.@threads for (r, i) in collect(sim_opts)

		filename = make_filename(results_dir, r, i)
		if !isfile(filename)

			n_groups = groups[i]
			true_model = fill(1, n_groups)
			true_θ = get_θ(0.2, true_model)

			_, df, _, _ = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model);

			results = BitArray(undef, length(priors))
			results_big = Array{Matrix{Float64}}(undef, length(priors))

			# (j, fun) = first(enumerate(priors))
			for (j, fun) in enumerate(priors)

				partition_prior = fun(n_groups)
				res = fit_model(df; mcmc_iterations = mcmc_iterations, mcmc_burnin = mcmc_burnin, partition_prior = partition_prior, use_Gibbs = true, hmc_stepsize = 0.0);
				post_probs =  compute_post_prob_eq_helper(res)

				results[j] = any_incorrect(post_probs)
				results_big[j] = post_probs

				# results[j, i, r] = any_incorrect(post_probs)
				# results_big[j, i, r] = post_probs

			end

			Serialization.serialize(filename, (results, results_big))

		end

		ProgressMeter.next!(p)#; showvalues = [(:r, r), (:groups, n_groups)])

	end
	# return results, results_big
end

# path = joinpath("simulations", "results_multiplecomparisonsplot", "multipleComparisonPlot.jls")
# if !isfile(path)
if !isinteractive()
	run_simulation()
	exit()
end

repeats = 200
groups = 2:10
len_priors = length(get_priors())
sim_opts = Iterators.product(1:repeats, eachindex(groups))
results_dir = get_resultsdir()

results = BitArray(undef, len_priors, length(groups), repeats)
results_big = Array{Matrix{Float64}}(undef, len_priors, length(groups), repeats)

ProgressMeter.@showprogress for (r, i) in sim_opts
	filename = make_filename(results_dir, r, i)
	if isfile(filename)

		temp, temp_big = Serialization.deserialize(filename)
		results[:, i, r] .= temp
		results_big[:, i, r] .= temp_big

	else
		@warn "This file does not exist: " filename
	end
end

function make_figure(x, y, ylab, labels; shapes = :auto, kwargs...)
	plot(
		x,
		y,
		markershape			= shapes,
		legend				= :topleft,
		labels				= labels,
		markeralpha			= 0.75,

		ylabel = ylab,
		xlabel = "no. groups",

		ylim = (0, 1.05),
		yticks = 0:.2:1,
		xlim = (1, 11),

		background_color_legend = nothing,
		foreground_color_legend = nothing
		;
		kwargs...
	)
end

size(results)

labels = ["Uniform" "Beta-binomial α=1, β=1" "Beta-binomial α=K, β=1" "DPP α=0.5" "DPP α=1" "DPP α=Gopalan & Berry"]
keep = eachindex(labels)#[1, 2, 3, 5]
labels = reshape(labels[1, keep], 1, length(keep))

shapes = [:circle :rect :star5 :diamond :hexagon :utriangle]

mu = dropdims(mean(results, dims = 3), dims = 3)[keep, :]
# [mean(results[i, g, :]) for i in axes(results, 1), g in axes(results, 2)] == mu
p1 = make_figure(groups, permutedims(mu), "P(one or more errors)", labels; shapes = shapes)
# plot!(p1, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true)
plot!(p1, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true, legend = (0.085, .95))


try2 = prop_incorrect.(results_big)
mu2 = dropdims(mean(try2, dims = 3), dims = 3)[keep, :]
p2 = make_figure(groups, permutedims(mu2), "P(errors)", labels)

figsize = (700, 500)

# using PlotlyJS
plotlyjs()
p12 = plot(plot(p1), plot(p2, legend = false), size = (1500, 500))


figdir = joinpath("figures")
savefig(plot(p1, size = figsize), joinpath(figdir, "one_or_more_errors2.png"))
# savefig(plot(p2, size = figsize), joinpath(figdir, "errors.png"))
savefig(plot(p1, size = figsize), joinpath(figdir, "one_or_more_errors2.pdf"))
# savefig(plot(p2, size = figsize), joinpath(figdir, "errors.pdf"))

savefig(plot(p12, size = (1400, 500)), joinpath(figdir, "multipleComparisonPlot_side_by_side.pdf"))
savefig(plot(p12, size = (1400, 500)), joinpath(figdir, "multipleComparisonPlot_side_by_side.png"))

plotl