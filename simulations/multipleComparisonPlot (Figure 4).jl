#=
	to run the simulation, run in a terminal
		julia-1.7.2 --project=simulations -O3 --threads 8 --check-bounds=no simulations/multipleComparisonPlot\ \(Figure\ 4\).jl simulate_only

	TODO: split this file into 2 files?
		- one for running the simulation
			-	rerun simulations when any rhat is NaN or above 1.05?
		- one for creating the figure

=#

using EqualitySampler, EqualitySampler.Simulations, MCMCChains
import ProgressMeter, JLD2, Turing, Logging

#region simulation functions
function instantiate_prior(symbol::Symbol, k::Integer)

	symbol == :uniform				&&	return UniformMvUrnDistribution(k)
	symbol == :BetaBinomial11		&&	return BetaBinomialMvUrnDistribution(k, 1.0, 1.0)
	symbol == :BetaBinomialk1		&&	return BetaBinomialMvUrnDistribution(k, k, 1.0)
	symbol == :BetaBinomial1k		&&	return BetaBinomialMvUrnDistribution(k, 1.0, k)
	symbol == :BetaBinomial1binomk2	&&	return BetaBinomialMvUrnDistribution(k, 1.0, binomial(k, 2))
	symbol == :DirichletProcess0_5	&&	return DirichletProcessMvUrnDistribution(k, 0.5)
	symbol == :DirichletProcess1_0	&&	return DirichletProcessMvUrnDistribution(k, 1.0)
	symbol == :DirichletProcess2_0	&&	return DirichletProcessMvUrnDistribution(k, 2.0)
	symbol == :DirichletProcessGP	&&	return DirichletProcessMvUrnDistribution(k, :Gopalan_Berry)

end

function get_priors()
	return (
		:uniform,
		:BetaBinomial11,
		:BetaBinomialk1,
		:BetaBinomial1k,
		:BetaBinomial1binomk2,
		:DirichletProcess0_5,
		:DirichletProcess1_0,
		:DirichletProcess2_0,
		:DirichletProcessGP,
		:Westfall
	)
end

function get_reference_and_comparison(hypothesis::Symbol, values_are_log_odds::Bool = false)
	if values_are_log_odds
		reference =  0.0
		comparison = hypothesis === :null ? !isless : isless
	else
		reference =  0.5
		comparison = hypothesis === :null ? isless : !isless
	end
	return reference, comparison
end

function any_incorrect(x, hypothesis::Symbol, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(hypothesis, values_are_log_odds)
	for j in 1:size(x, 1)-1, i in j+1:size(x, 1)
		if comparison(x[i, j], reference)
			return true
		end
	end
	return false
end

function prop_incorrect(x, hypothesis::Symbol, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(hypothesis, values_are_log_odds)
	count = 0
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		if comparison(x[i, j], reference)
			count += 1
		end
	end
	return count / (n * (n - 1) ÷ 2)
end

function get_resultsdir()
	results_dir = joinpath("simulations", "results_multiplecomparisonsplot_200_7")
	!ispath(results_dir) && mkpath(results_dir)
	return results_dir
end

make_filename(results_dir, r, i, hypothesis) = joinpath(results_dir, "repeat_$(r)_groups_$(i)_H_$(hypothesis).jld2")

function get_hyperparams()
	n_obs_per_group = 100
	repeats = 1:2# 30
	groups = 2:10
	hypothesis=(:null, :full)
	return n_obs_per_group, repeats, groups, hypothesis
end

function validate_r_hat(chn, tolerance = 1.05)
	rhats = MCMCChains.summarystats(chn).nt.rhat
	any(isnan, rhats) && return true, NaN
	any(>(tolerance), rhats) && return true, mean(rhats)
	return false, 0.0
end

function run_simulation()

	n_obs_per_group, repeats, groups, hypotheses = get_hyperparams()

	sim_opts = Iterators.product(repeats, eachindex(groups), hypotheses)

	priors = get_priors()

	nsim = length(sim_opts)
	println("starting simulation of $(nsim) runs with $(length(priors)) priors and $(Threads.nthreads()) threads")

	mcmc_settings = MCMCSettings(;iterations = 10_000, burnin = 2_000, chains = 1)
	# mcmc_settings = MCMCSettings(;iterations = 200, burnin = 100, chains = 1)

	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)
	Logging.disable_logging(Logging.Warn)

	results_dir = get_resultsdir()

	# (r, i, hypothesis) = first(sim_opts)
	Threads.@threads for (r, i, hypothesis) in collect(sim_opts)
	# for (r, i, hypothesis) in collect(sim_opts)

		filename = make_filename(results_dir, r, i, hypothesis)
		if !isfile(filename)

			# @show i, r, hypothesis
			n_groups = groups[i]
			true_model = hypothesis === :null ? fill(1, n_groups) : collect(1:n_groups)
			true_θ = normalize_θ(0.2, true_model)

			data_obj = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ);
			dat = data_obj.data

			results = BitArray(undef, length(priors))
			results_big = Array{Matrix{Float64}}(undef, length(priors))
			results_rhat = BitArray(undef, length(priors))

			# (j, prior) = first(enumerate(priors))
			for (j, prior) in enumerate(priors)

				@show r, i, hypothesis, j, prior

				if prior === :Westfall

					result = westfall_test(dat)
					log_posterior_odds_mat = result.log_posterior_odds_mat
					results[j] = any_incorrect(log_posterior_odds_mat, hypothesis, true)
					results_big[j] = Matrix(log_posterior_odds_mat)

				else

					partition_prior = instantiate_prior(prior, n_groups)
					for bad_rhat_count in 1:5
					# while any_bad_rhats && bad_rhat_count <= 5
					# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = :PG)
					# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = Turing.SMC())
						chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings)
						any_bad_rhats, mean_rhat_value = validate_r_hat(chain)
						if any_bad_rhats
							@error "This run had an bad r-hat:" r, i, hypothesis, j, prior bad_rhat_count, mean_rhat_value
						else

							partition_samples = Int.(Array(group(chain, :partition)))
							post_probs = compute_post_prob_eq(partition_samples)

							results[j] = any_incorrect(post_probs, hypothesis)
							results_big[j] = post_probs
							results_rhat[j] = any_bad_rhats
						end
					end
				end
			end

			JLD2.jldsave(filename; results=results, results_big=results_big, results_rhat = results_rhat, run = (;r, i, hypothesis))

		end

		ProgressMeter.next!(p)

	end
end
#endregion

#region analysis functions
function load_results()

	_, repeats, groups, hypotheses = get_hyperparams()
	repeats = 1:maximum(repeats)
	no_repeats = length(repeats)
	no_hypotheses = length(hypotheses)
	len_priors = length(get_priors())
	sim_opts = Iterators.product(repeats, eachindex(groups), hypotheses)
	results_dir = get_resultsdir()

	results = BitArray(undef, len_priors, length(groups), no_repeats, no_hypotheses)
	results_big = Array{Matrix{Float64}}(undef, len_priors, length(groups), no_repeats, no_hypotheses)

	ProgressMeter.@showprogress for (r, i, hypothesis) in sim_opts
		filename = make_filename(results_dir, r, i, hypothesis)

		hypothesis_idx = hypothesis === :null ? 1 : 2
		if isfile(filename)
			@show filename

			temp = JLD2.jldopen(filename)
			results[:, i, r, hypothesis_idx] .= temp["results"]
			results_big[:, i, r, hypothesis_idx] .= temp["results_big"]

		else
			fill!(results[:, i, r, hypothesis_idx], 0)
			results_big[:, i, r, hypothesis_idx] .= [zeros(Float64, groups[i], groups[i]) for _ in 1:len_priors]
			@warn "This file does not exist: " filename
		end
	end

	return results, results_big

end



#endregion

# could also be a const global
simulate_only() = length(ARGS) > 0 && first(ARGS) === "simulate_only"
if (simulate_only())
	println("""
		simulate_only() = true
		get_resultsdir() = $(get_resultsdir())
		pwd() = $(pwd())
		threads = $(Threads.nthreads())
	""")
else
	println("simulate_only() = false")
end

if simulate_only()
	run_simulation()
	exit()
end

results, results_big = load_results()


function make_figure(x, y, ylab, labels; shapes = :auto, kwargs...)
	Plots.plot(
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

using Plots, Plots.PlotMeasures
# using PlotlyJS
# plotlyjs()

# gr()

groups = 2:10
labels = string.(collect(get_priors()))
# labels = ["Uniform", "Beta-binomial α=1, β=1", "Beta-binomial α=K, β=1", "Beta-binomial α=1, β=K", "Beta-binomial α=1, β=binomial(K, 2)", "DPP α=0.5", "DPP α=1", "DPP α=2", "DPP α=Gopalan & Berry"]
keep = eachindex(labels)#[1, 2, 3, 5]
labels = reshape(labels[keep], 1, length(keep))

shapes = :auto#[:circle :rect :star5 :diamond :hexagon :utriangle]

mu = dropdims(mean(view(results, :, :, :, 1), dims = 3), dims = 3)[keep, :]
# [mean(results[i, g, :]) for i in axes(results, 1), g in axes(results, 2)] == mu
p1 = make_figure(groups, permutedims(mu), "P(one or more errors)", labels; shapes = shapes)
# plot!(p1, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true)
Plots.plot!(p1, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true, legend = (0.085, .95))

mu_full = dropdims(mean(view(results, :, :, :, 2), dims = 3), dims = 3)[keep, :]
p1_full = make_figure(groups, permutedims(mu_full), "P(one or more errors)", labels; shapes = shapes)
Plots.plot!(p1_full, xlabel = "Number of groups", xticks = 2:10, xlim = (2, 10), ylim = (0, 1), widen = true, legend = (0.085, .95))



# TODO: why does results_big have #undef values?
try2 = Array{Float64}(undef, size(results_big))
for it in CartesianIndices(results_big)
	try2[it] = prop_incorrect(view(results_big, it)[1], isone(it[4]) ? :null : :full, labels[it[1]] === :Westfall)
end
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


using Gadfly, DataFrames, Compose
import Cairo, Fontconfig

function vstack2(plots::Vector{Plot}; spacing::Float64=0.0, heights::Vector{<:Real}=Float64[])
	n = length(plots)
	heights==Float64[] && (heights = fill(1/n, n))
	sumh = cumsum([0;heights])
	vpos = sumh + spacing.*[0; 1:n-1; n-1]
	M = [(context(0,v,1,h), render(p)) for (v,h,p) in zip(vpos[1:n], heights, plots)]
	return compose(context(units=UnitBox(0,0,1,vpos[n+1])), M...)
end

function hstack2(plots::Vector{Plot}; spacing::Float64=0.0, widths::Vector{<:Real}=Float64[])
	n = length(plots)
	isempty(widths) && (widths = fill(1/n, n))
	sumw = cumsum([0;widths])
	vpos = sumw + spacing.*[0; 1:n-1; n-1]
	M = [(context(v,0,w,1), render(p)) for (v,w,p) in zip(vpos[1:n], widths, plots)]
	return compose(context(units=UnitBox(0,0,vpos[n+1], 1)), M...)
end

function base_plt(df, ylabel)
	Gadfly.plot(
		df,
		Geom.line(),
		Geom.point,
		Scale.x_continuous(minvalue = 2, maxvalue = 10),
		Scale.y_continuous(minvalue = 0, maxvalue = 1),
		Guide.xticks(; ticks = collect(2:10)),
		Guide.yticks(; ticks = collect(0.0:0.2:1.0)),
		Guide.xlabel("Groups"),
		Guide.ylabel(ylabel),
		Theme(
			key_position = :none,
			point_size = 4pt,
			major_label_font_size = 16pt,
			minor_label_font_size = 14pt,
			key_title_font_size   = 14pt,
			key_label_font_size   = 12pt
		);
		x=:groups, y=:value,
		color=:prior, shape=:prior,
		linestyle=[:dash]
		# color=:prior_family, shape=:prior_index, group=:prior,
		# linestyle=:prior_index
	)
end

# note that "&" must be escaped to "&amp;" for pango, but SVG still fails...
prior_names  = ["Uniform", "BB α=1, β=1", "BB α=K, β=1", "BB α=1, β=K", "BB α=1, β=binomial(K, 2)", "DPP α=0.5", "DPP α=1", "DPP α=2", "DPP α=Gopalan &amp; Berry"]
# this would be nice, but the legend doesn't cooperate
prior_family = ["Uniform", "BB", "BB", "BB", "BB", "DPP", "DPP", "DPP", "DPP"]
prior_index  = [ 1       , 1,    2,    3,    4,    1,     2,     3,     4    ]

# prior_names = ["Uniform", "Beta-binomial α=1, β=1", "Beta-binomial α=K, β=1", "Beta-binomial α=1, β=K", "Beta-binomial α=1, β=binomial(K, 2)", "DPP α=0.5", "DPP α=1", "DPP α=2", "DPP α=Gopalan &amp; Berry"]
any_incorrect_df = DataFrame(
	prior	= repeat(prior_names, length(groups)),
	prior_family = repeat(prior_family, length(groups)),
	prior_index  = repeat(prior_index, length(groups)),
	groups	= repeat(groups, inner = length(prior_names)),
	value	= vec(mean(results, dims = 3)),
)

results_big_prop_incorrect = prop_incorrect.(results_big)
prop_incorrect_df = DataFrame(
	prior	= repeat(prior_names, length(groups)),
	prior_family = repeat(prior_family, length(groups)),
	prior_index  = repeat(prior_index, length(groups)),
	groups	= repeat(groups, inner = length(prior_names)),
	value	= vec(mean(results_big_prop_incorrect, dims = 3))
)

plt_any_incorrect = base_plt(any_incorrect_df, "P(one or more errors)");
plt_prop_incorrect = base_plt(prop_incorrect_df, "Proportion of errors");

legend_plt = Gadfly.plot(
	prop_incorrect_df,
	color=:prior, shape=:prior,
	# color=:prior_family, shape=:prior_index, group=:prior,
	# linestyle=:prior_index,
	Geom.blank,
	Guide.shapekey(title = "Prior"; pos=[0.0w,-0.15h]),
	# Guide.colorkey(title = "Prior"; pos=[0.0w,-0.15h]),
	# Guide.(title = "Prior"; pos=[0.0w,-0.15h]),
	Theme(
		point_size = 4pt,
		key_title_font_size   = 14pt,
		key_label_font_size   = 11pt
	)
);

joined_plt = hstack2([plt_any_incorrect, plt_prop_incorrect, legend_plt]; widths= [1, 1, .6]);
joined_plt |> PDF("foo2.pdf", 30cm, 15cm)


Gadfly.plot(
	DataFrame(prior = rand(["α", "1", "β"], 10), groups=rand(2:10, 10), value=rand(10)),
	Geom.line(),
	Geom.point,
	Scale.x_continuous(minvalue=0, maxvalue = 10),
	Scale.y_continuous(minvalue=0, maxvalue = 1),
	Guide.xticks(;ticks = [0, 2, 6, 8, 10]);
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

prior_names = ["Uniform", "Beta-binomial α=1, β=1", "Beta-binomial α=K, β=1", "Beta-binomial α=1, β=K", "Beta-binomial α=1, β=binomial(K, 2)", "DPP α=0.5", "DPP α=1", "DPP α=2", "DPP α=Gopalan and Berry"]
joined_df = DataFrame(
	prior	= repeat(prior_names, 2length(groups)),
	groups	= repeat(groups; inner = length(prior_names), outer = 2),
	value	= vcat(vec(mean(results, dims = 3)), vec(mean(results_big_prop_incorrect, dims = 3))),
	panel	= repeat(["left", "right"], inner = prod(size(results)[1:2]))
)

Gadfly.plot(
	joined_df,
	x=:groups, y=:value, color=:prior, shape=:prior, xgroup=:panel,
	Geom.subplot_grid(Geom.line, Geom.point),
	linestyle=[:dash],
	Scale.x_continuous(minvalue=0, maxvalue = 10),
	Scale.y_continuous(minvalue=0, maxvalue = 1),
	Guide.xlabel("Group")
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


library(ggplot2)
dat <- data.frame(
  x = 1:10,
  y = rep(1:5, 2),
  panel = rep(c("left", "right"), each = 5)
)
ggplot(data = dat, mapping = aes(x = x, y = y)) +
  geom_point() +
  facet_grid(cols=vars(panel))
using Gadfly, DataFrames, Cairo, Fontconfig
dat = DataFrame(
	x=repeat(1:5,2),
	y=1:10,
	panel=repeat(["left", "right"], inner=5)
)
Gadfly.plot(dat,
	x=:x, y=:y, xgroup=:panel,
	Geom.subplot_grid(Geom.point),
	Guide.xlabel("Group")
) |> PNG("subplot_grid_example.png", 600Gadfly.px, 400Gadfly.px)
