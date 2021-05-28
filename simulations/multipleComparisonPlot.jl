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
else
	include("meansModel_Functions.jl")
	include("helpersTuring.jl")
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


function run_simulation()

	n_obs_per_group = 100
	repeats = 100
	groups = 2:10

	sim_opts = Iterators.product(1:repeats, eachindex(groups))

	mcmc_iterations = 10_000
	mcmc_burnin = 2_000

	nsim = length(sim_opts)
	println("starting simulation of $(nsim) runs with $(Threads.nthreads()) threads")

	# TODO: loop over these priors
	priors = (
		(n_groups)->UniformMvUrnDistribution(n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, n_groups, 1.0),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, 1.0, n_groups),
		(n_groups)->RandomProcessMvUrnDistribution(n_groups, Turing.RandomMeasures.DirichletProcess(1.887))
	)

	results = BitArray(undef, length(priors), length(groups), repeats)
	results_big = Array{Matrix{Float64}}(undef, length(priors), length(groups), repeats)
	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)
	Logging.disable_logging(Logging.Info)

	Threads.@threads for (r, i) in collect(sim_opts)

		n_groups = groups[i]
		true_model = fill(1, n_groups)
		true_θ = get_θ(0.2, true_model)

		_, df, _, _ = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model);

		for (j, fun) in enumerate(priors)

			partition_prior = fun(n_groups)
			res = fit_model(df, mcmc_iterations = mcmc_iterations, mcmc_burnin = mcmc_burnin, partition_prior = partition_prior);
			post_probs =  compute_post_prob_eq_helper(res)

			results[j, i, r] = any_incorrect(post_probs)
			results_big[j, i, r] = post_probs

		end

		ProgressMeter.next!(p)#; showvalues = [(:r, r), (:groups, n_groups)])

	end
	return results, results_big
end

path = joinpath("simulations", "results_multiplecomparisonsplot", "multipleComparisonPlot.jls")
if !isfile(path)
	allresult = run_simulation()
	Serialization.serialize(path, allresult)
	results, results_big = allresult
else
	results, results_big = Serialization.deserialize(path);
end
!isinteractive() && exit()

#=
	TODO:
	- rerun with more mcmc samples!
	- plot proportion incorrect instead of totals
=#


function make_figure(x, y, ylab, labels)
	plot(
		x,
		y,
		markershape			= :auto,
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
	)
end

size(results)
labels = ["Uniform" "BetaBinomial (α=1, β=1)" "BetaBinomial (α=no. groups, β=1)" "BetaBinomial (α=1, β=no. groups)" "DPP(α=1.887)"]
keep = [1, 2, 3, 5]
labels = reshape(labels[1, keep], 1, length(keep))

mu = dropdims(mean(results, dims = 3), dims = 3)[keep, :]
# [mean(results[i, g, :]) for i in axes(results, 1), g in axes(results, 2)] == mu
p1 = make_figure(2:10, permutedims(mu), "P(one or more errors)", labels)

try2 = prop_incorrect.(results_big)
mu2 = dropdims(mean(try2, dims = 3), dims = 3)[keep, :]
p2 = make_figure(2:10, permutedims(mu2), "P(errors)", labels)

# figsize = (1200, 800)
figsize = (600, 400)
savefig(plot(p1, size = figsize), joinpath("simulations", "figures_multiplecomparisonsplot", "one_or_more_errors.png"))
savefig(plot(p2, size = figsize), joinpath("simulations", "figures_multiplecomparisonsplot", "errors.png"))

k = 20
included = 0:k-1
plot(included, pdf_incl.(Ref(BetaBinomialMvUrnDistribution(k, 1.0, k)), included))
plot(included, pdf_incl.(Ref(BetaBinomialMvUrnDistribution(k, k, 1.0)), included))
