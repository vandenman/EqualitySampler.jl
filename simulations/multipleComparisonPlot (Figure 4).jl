#=
	to run the simulation, run in a terminal
		julia-1.6.5 --project=simulations -O3 -t auto simulations/multipleComparisonPlot\ \(Figure\ 4\).jl simulate_only

	TODO: split this file into 2 files?

=#

using EqualitySampler, Turing, Plots, Plots.PlotMeasures
import Logging, ProgressMeter, JLD2
include("anovaFunctions.jl")


function get_priors()
	return (
		(n_groups)->UniformMvUrnDistribution(n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, 1.0, 1.0),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, n_groups, 1.0),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, 1.0, n_groups),
		(n_groups)->BetaBinomialMvUrnDistribution(n_groups, 1.0, binomial(n_groups, 2)),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, 0.5),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, 1.0),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, 2.0),
		(n_groups)->DirichletProcessMvUrnDistribution(n_groups, :Gopalan_Berry)
	)
end

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
	results_dir = joinpath("simulations", "results_multiplecomparisonsplot_200_2")
	!ispath(results_dir) && mkpath(results_dir)
	return results_dir
end

make_filename(results_dir, r, i) = joinpath(results_dir, "repeat_$(r)_groups_$(i).jld2")

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

	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)
	Logging.disable_logging(Logging.Warn)

	results_dir = get_resultsdir()

	# (r, i) = first(sim_opts)
	Threads.@threads for (r, i) in collect(sim_opts)

		filename = make_filename(results_dir, r, i)
		if !isfile(filename)

			n_groups = groups[i]
			true_model = fill(1, n_groups)
			true_θ = normalize_θ(0.2, true_model)

			dat, _, _ = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ);

			results = BitArray(undef, length(priors))
			results_big = Array{Matrix{Float64}}(undef, length(priors))

			# (j, fun) = first(enumerate(priors))
			for (j, fun) in enumerate(priors)

				partition_prior = fun(n_groups)
				chn, model = fit_eq_model(dat, partition_prior, nothing; mcmc_iterations = mcmc_iterations, mcmc_burnin = mcmc_burnin);
				partition_samples = Int.(Array(group(chn, :partition)))
				post_probs = compute_post_prob_eq(partition_samples)

				results[j] = any_incorrect(post_probs)
				results_big[j] = post_probs

			end

			JLD2.jldsave(filename; results=results, results_big=results_big)

		end

		ProgressMeter.next!(p)

	end
end

# could also be a const global
simulate_only() = length(ARGS) > 0 && first(ARGS) === "simulate_only"
if (simulate_only())
	println("""
		simulate_only() = true
		get_resultsdir() = $(get_resultsdir())
		pwd() = $(pwd())
	""")
else
	println("simulate_only() = false")
end

if simulate_only()
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

		temp = JLD2.jldopen(filename)
		results[:, i, r] .= temp["results"]
		results_big[:, i, r] .= temp["results_big"]

	else
		fill!(results[:, i, r], 0)
		results_big[:, i, r] .= [zeros(Float64, groups[i], groups[i]) for _ in 1:len_priors]
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

# using PlotlyJS
# plotlyjs()

# gr()
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
p2 = make_figure(groups, permutedims(mu2), "Proportion of errors", labels; shapes = shapes)
plot!(p2, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true)
figsize = (700, 500)

bottom_margin = 5mm
left_margin   = 5mm
p1_2 = plot(p1, legend = (0.085, .95),	bottom_margin = bottom_margin, left_margin = left_margin)

p12 = plot(
	plot(p1, legend = (0.085, .95),	bottom_margin = bottom_margin, left_margin = left_margin),
	plot(p2, legend = false,		bottom_margin = bottom_margin, left_margin = left_margin),
	size = (1500, 500), legendfontsize = 12, titlefontsize = 16, tickfont = 12, guidefontsize = 16
);


figdir = "figures"
# savefig(plot(p1, size = figsize), joinpath(figdir, "one_or_more_errors2.png"))
# savefig(plot(p2, size = figsize), joinpath(figdir, "errors.png"))
# savefig(plot(p1, size = figsize), joinpath(figdir, "one_or_more_errors2.pdf"))
# savefig(plot(p2, size = figsize), joinpath(figdir, "errors.pdf"))

savefig(plot(p12, size = (1000, 500)), joinpath(figdir, "multipleComparisonPlot_side_by_side.pdf"))
# savefig(plot(p12, size = (1000, 500)), joinpath(figdir, "multipleComparisonPlot_side_by_side.png"))


using Gadfly, DataFrames
import Cairo, Fontconfig

# note that "&" must be escaped to "&amp;" for pango, but SVG still fails...
prior_names = ["Uniform", "Beta-binomial α=1, β=1", "Beta-binomial α=K, β=1", "Beta-binomial α=1, β=K", "Beta-binomial α=1, β=binomial(K, 2)", "DPP α=0.5", "DPP α=1", "DPP α=2", "DPP α=Gopalan and Berry"]
any_incorrect_df = DataFrame(
	prior	= repeat(prior_names, inner = length(groups)),
	groups	= repeat(groups, length(prior_names)),
	value	= vec(mean(results, dims = 3))
)

results_big_prop_incorrect = prop_incorrect.(results_big)
prop_incorrect_df = DataFrame(
	prior	= repeat(prior_names, inner = length(groups)),
	groups	= repeat(groups, length(prior_names)),
	value	= vec(mean(results_big_prop_incorrect, dims = 3))
)

function base_plt(df)
	Gadfly.plot(
		df,
		Geom.line(),
		Geom.point,
		Scale.x_continuous(minvalue=0, maxvalue = 10),
		Scale.y_continuous(minvalue=0, maxvalue = 1);
		x=:groups, y=:value, color=:prior, shape=:prior,
		linestyle=[:dash]
	)
end

plt_any_incorrect = base_plt(any_incorrect_df);
push!(plt_any_incorrect, Theme(key_position = :none));
plt_prop_incorrect = base_plt(prop_incorrect_df);
hstack(plt_any_incorrect, plt_prop_incorrect) |> SVG("foo.svg", 30cm, 15cm)

Gadfly.plot(
	DataFrame(prior = rand(["α", "1", "β"], 10), groups=rand(2:10, 10), value=rand(10)),
	Geom.line(),
	Geom.point,
	Scale.x_continuous(minvalue=0, maxvalue = 10),
	Scale.y_continuous(minvalue=0, maxvalue = 1);
	x=:groups, y=:value, color=:prior, shape=:prior,
	linestyle=[:dash]
)
Gadfly.plot(
	any_incorrect_df,
	Geom.line(),
	Geom.point,
	Scale.x_continuous(minvalue=0, maxvalue = 10),
	Scale.y_continuous(minvalue=0, maxvalue = 1);
	x=:groups, y=:value, color=:prior, shape=:prior,
	linestyle=[:dash]
)


# pango does not like &
using Gadfly
prior = ["α", "&", "β"]; groups=[2, 3, 4]; value=[1.0, 2.0, 3.0]
Gadfly.plot(Geom.point;	x=groups, y=value, color=prior)
import Cairo, Fontconfig
Gadfly.plot(Geom.point;	x=groups, y=value, color=prior)
prior = ["α", "&amp;", "β"]
Gadfly.plot(Geom.point;	x=groups, y=value, color=prior)

# @model function gdemo(x, y)
# 	s² ~ InverseGamma(2, 3)
# 	m ~ Normal(0, sqrt(s²))
# 	x ~ Normal(m, sqrt(s²))
# 	y ~ Normal(m, sqrt(s²))
# end

# #  Run sampler, collect results
# e0 = gdemo(1.5, 2)
# e1 = HMC(0.1, 5)
# chn = sample(e0, MH(), 1000) # <- type unstable
