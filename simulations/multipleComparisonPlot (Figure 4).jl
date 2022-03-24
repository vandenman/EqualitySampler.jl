#=
	to run the simulation, run in a terminal
		julia-1.7.2 --project=simulations -O3 --threads 8 --check-bounds=no simulations/multipleComparisonPlot\ \(Figure\ 4\).jl simulate_only

	TODO: split this file into 2 files?
		- one for running the simulation
			-	rerun simulations when any rhat is NaN or above 1.05?
		- one for creating the figure

=#

using EqualitySampler, EqualitySampler.Simulations, MCMCChains, Random
import ProgressMeter, JLD2, Turing, Logging

#region simulation functions
function instantiate_prior(symbol::Symbol, k::Integer)
	# this works nicely with jld2 but it's not type stable

	symbol == :uniform				&&	return UniformMvUrnDistribution(k)
	symbol == :BetaBinomial11		&&	return BetaBinomialMvUrnDistribution(k, 1.0, 1.0)
	symbol == :BetaBinomialk1		&&	return BetaBinomialMvUrnDistribution(k, k, 1.0)
	symbol == :BetaBinomial1k		&&	return BetaBinomialMvUrnDistribution(k, 1.0, k)
	symbol == :BetaBinomial1binomk2	&&	return BetaBinomialMvUrnDistribution(k, 1.0, binomial(k, 2))
	symbol == :DirichletProcess0_5	&&	return DirichletProcessMvUrnDistribution(k, 0.5)
	symbol == :DirichletProcess1_0	&&	return DirichletProcessMvUrnDistribution(k, 1.0)
	symbol == :DirichletProcess2_0	&&	return DirichletProcessMvUrnDistribution(k, 2.0)
	# symbol == :DirichletProcessGP	&&
	return DirichletProcessMvUrnDistribution(k, :Gopalan_Berry)

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
	results_dir = joinpath("simulations", "results_multiplecomparisonsplot_200_8")
	!ispath(results_dir) && mkpath(results_dir)
	return results_dir
end

make_filename(results_dir, r, i, hypothesis) = joinpath(results_dir, "repeat_$(r)_groups_$(i)_H_$(hypothesis).jld2")

function get_hyperparams()
	n_obs_per_group = 100
	# repeats = 1:70
	repeats = 1:200
	groups = 2:10
	hypothesis=(:null, :full)
	return n_obs_per_group, repeats, groups, hypothesis
end

function validate_r_hat(chn, tolerance = 1.2)
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

	mcmc_settings = MCMCSettings(;iterations = 5_000, burnin = 1_000, chains = 1)
	# mcmc_settings = MCMCSettings(;iterations = 200, burnin = 100, chains = 1)

	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)
	Logging.disable_logging(Logging.Warn)
	# Logging.disable_logging(Logging.Debug)

	results_dir = get_resultsdir()

	# how often to restart a run when the target rhats are not met
	max_retries = 10

	trngs = [MersenneTwister(i) for i in 1:Threads.nthreads()];

	# (iteration, (r, i, hypothesis)) = first(enumerate(sim_opts))
	Threads.@threads for (iteration, (r, i, hypothesis)) in collect(enumerate(sim_opts))
	# for (iteration, (r, i, hypothesis)) in enumerate(sim_opts)

		filename = make_filename(results_dir, r, i, hypothesis)
		if !isfile(filename)

			# @show i, r, hypothesis
			n_groups = groups[i]
			true_model = hypothesis === :null ? fill(1, n_groups) : collect(1:n_groups)
			true_θ = normalize_θ(0.2, true_model)

			rng = trngs[Threads.threadid()]
			Random.seed!(rng, iteration)
			data_obj = simulate_data_one_way_anova(rng, n_groups, n_obs_per_group, true_θ);
			dat = data_obj.data

			results = BitArray(undef, length(priors))
			results_big = Array{Matrix{Float64}}(undef, length(priors))
			results_rhat = Vector{Int}(undef, length(priors))

			# j = 7; prior = priors[j]
			# (j, prior) = first(enumerate(priors))
			for (j, prior) in enumerate(priors)

				# @show iteration, r, i, hypothesis, j, prior

				if prior === :Westfall

					result = westfall_test(dat)
					log_posterior_odds_mat = result.log_posterior_odds_mat
					results[j] = any_incorrect(log_posterior_odds_mat, hypothesis, true)
					results_big[j] = Matrix(log_posterior_odds_mat)
					results_rhat[j] = 0

				else

					partition_prior = instantiate_prior(prior, n_groups)
					for retry in 1:max_retries

						Random.seed!(rng, iteration + retry)
						# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = :PG)
						# chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = Turing.SMC())
						chain = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, rng = rng, spl = 0.0)
						any_bad_rhats, mean_rhat_value = validate_r_hat(chain)
						if any_bad_rhats && retry != max_retries
							# @error "This run had a bad r-hat:" r, i, hypothesis, j, prior, any_bad_rhats, mean_rhat_value, retry
						else

							partition_samples = Int.(Array(group(chain, :partition)))
							post_probs = compute_post_prob_eq(partition_samples)

							results[j] = any_incorrect(post_probs, hypothesis)
							results_big[j] = post_probs
							results_rhat[j] = retry
							break
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
	results_rhat = Array{Float64}(undef, len_priors, length(groups), no_repeats, no_hypotheses)

	p = ProgressMeter.Progress(length(sim_opts))
	generate_showvalues(filename) = () -> [(:filename,filename)]
	# ProgressMeter.@showprogress
	for (r, i, hypothesis) in sim_opts
		filename = make_filename(results_dir, r, i, hypothesis)

		hypothesis_idx = hypothesis === :null ? 1 : 2
		if isfile(filename)
			# @show filename

			temp = JLD2.jldopen(filename)
			results[:, i, r, hypothesis_idx] .= temp["results"]
			results_big[:, i, r, hypothesis_idx] .= temp["results_big"]
			results_rhat .= temp["results_rhat"]

		else
			results[:, i, r, hypothesis_idx] .= false
			results_big[:, i, r, hypothesis_idx] .= [zeros(Float64, groups[i], groups[i]) for _ in 1:len_priors]
			results_rhat[:, i, r, hypothesis_idx] .= 0.0
			@warn "This file does not exist: " filename
		end
		ProgressMeter.next!(p; showvalues = generate_showvalues(filename))
	end

	return results, results_big, results_rhat

end

function load_results_as_df()

	_, repeats, groups, hypotheses = get_hyperparams()
	priors = collect(get_priors())
	repeats = 1:maximum(repeats)
	no_groups     = length(groups)
	no_repeats    = length(repeats)
	no_hypotheses = length(hypotheses)
	no_priors    = length(priors)
	sim_opts = Iterators.product(repeats, eachindex(groups), hypotheses)

	n_sim = length(sim_opts) * no_priors
	df = DataFrame(
		repeats        = repeat(repeats,             inner = no_priors,                                outer = no_groups * no_hypotheses),
		groups         = repeat(groups,              inner = no_priors * no_repeats,                   outer = no_hypotheses),
		hypotheses     = repeat(collect(hypotheses), inner = no_priors * no_repeats * no_groups        #= outer = 1=#),
		prior          = repeat(priors,                      no_repeats * no_groups * no_hypotheses),
		any_incorrect  = -1 .* ones(Int, n_sim),
		prop_incorrect = -1.0 .* ones(Float64, n_sim),
		results_rhat   = -1.0 .* ones(Float64, n_sim)
	)

	results_dir = get_resultsdir()

	p = ProgressMeter.Progress(length(sim_opts))
	generate_showvalues(filename) = () -> [(:filename,filename)]
	# ProgressMeter.@showprogress
	rowRange = 1:no_priors
	values_are_log_odds = priors .=== :Westfall
	# (r, i, hypothesis) = first(sim_opts)
	for (r, i, hypothesis) in sim_opts

		filename = make_filename(results_dir, r, i, hypothesis)

		if isfile(filename)

			temp = JLD2.jldopen(filename)
			df[rowRange, :any_incorrect]  .= temp["results"]
			df[rowRange, :results_rhat]   .= temp["results_rhat"]
			df[rowRange, :prop_incorrect] .= prop_incorrect.(temp["results_big"], hypothesis, values_are_log_odds)

		else
			@warn "This file does not exist: " filename
		end
		rowRange = rowRange .+ no_priors
		ProgressMeter.next!(p; showvalues = generate_showvalues(filename))
	end

	return df

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

using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf

results_df = load_results_as_df()

reduced_results_df = @chain results_df begin
	groupby(Cols(:prior, :groups, :hypotheses))
	combine(:any_incorrect => mean, :prop_incorrect => mean)
	groupby(:hypotheses)
end

lambda_results_df = @chain results_df begin
	unstack([:repeats, :groups, :prior], :hypotheses, :prop_incorrect)
	transform(
		[:null, :full] => ((n, f) -> 0.50 * n + 0.50 * f) => :lambda_0_50,
		[:null, :full] => ((n, f) -> 0.95 * n + 0.05 * f) => :lambda_0_95
	)
	groupby(Cols(:prior, :groups))
	combine(:lambda_0_50 => mean, :lambda_0_95 => mean)
end

function get_labels(priors)
	# lookup = Dict(
	# 	:uniform              => "Uniform",
	# 	:BetaBinomial11       => "Beta",
	# 	:BetaBinomialk1       => "Beta-binomial α=K, β=1",
	# 	:BetaBinomial1k       => "Beta-binomial α=1, β=K",
	# 	:BetaBinomial1binomk2 => "Beta-binomial α=K, β=binomial(K,2)",
	# 	:DirichletProcess0_5  => "DPP α=0.5",
	# 	:DirichletProcess1_0  => "DPP α=1",
	# 	:DirichletProcess2_0  => "DPP α=2",
	# 	:DirichletProcessGP   => "DPP α=Gopalan & Berry",
	# 	:Westfall             => "Westfall"
	# )
	lookup = Dict(
		:uniform				=> "Uniform",
		:BetaBinomial11			=> "BB α=1, β=1",
		:BetaBinomialk1			=> "BB α=K, β=1",
		:BetaBinomial1k			=> "BB α=1, β=K",
		:BetaBinomial1binomk2	=> "BB α=1, β=binom(K,2)",
		# :BetaBinomial1binomk2	=> L"\mathrm{BB}\,\,\alpha=K, \beta=\binom{K}{2}",
		:DirichletProcess0_5	=> "DPP α=0.5",
		:DirichletProcess1_0	=> "DPP α=1",
		:DirichletProcess2_0	=> "DPP α=2",
		# :DirichletProcessGP		=> "DPP α=Gopalan & Berry",
		:DirichletProcessGP		=> "DPP α=G&B",
		:Westfall				=> "Westfall"
	)
	priors_set = sort!(unique(priors))
	return reshape([lookup[prior] for prior in priors_set], 1, length(priors_set))
end

function get_colors(priors, alpha = 0.75)
	# colors = alphacolor.(distinguishable_colors(4, RGB(.4,.5,.6)), alpha)
	# colors = alphacolor.(ColorSchemes.seaborn_colorblind6[1:4], alpha)
	# lookup = Dict(
	# 	:uniform              => colors[2],
	# 	:BetaBinomial11       => colors[3],
	# 	:BetaBinomialk1       => colors[3],
	# 	:BetaBinomial1k       => colors[3],
	# 	:BetaBinomial1binomk2 => colors[3],
	# 	:DirichletProcess0_5  => colors[4],
	# 	:DirichletProcess1_0  => colors[4],
	# 	:DirichletProcess2_0  => colors[4],
	# 	:DirichletProcessGP   => colors[4],
	# 	:Westfall             => colors[1]
	# )
	# return [lookup[prior] for prior in priors]
	colors = alphacolor.(ColorSchemes.seaborn_colorblind[1:10], alpha)
	lookup = Dict(
		:uniform              => colors[1],
		:BetaBinomial11       => colors[2],
		:BetaBinomialk1       => colors[3],
		:BetaBinomial1k       => colors[4],
		:BetaBinomial1binomk2 => colors[5],
		:DirichletProcess0_5  => colors[6],
		:DirichletProcess1_0  => colors[7],
		:DirichletProcess2_0  => colors[8],
		:DirichletProcessGP   => colors[9],
		:Westfall             => colors[10]
	)
	return [lookup[prior] for prior in priors]
end

function get_shapes(priors)
	lookup = Dict(
		:uniform              => :rect,
		:BetaBinomial11       => :utriangle,
		:BetaBinomialk1       => :rtriangle,
		:BetaBinomial1k       => :ltriangle,
		:BetaBinomial1binomk2 => :dtriangle,
		:DirichletProcess0_5  => :star4,
		:DirichletProcess1_0  => :star5,
		:DirichletProcess2_0  => :star6,
		:DirichletProcessGP   => :star8,
		:Westfall             => :circle
	)
	return [lookup[prior] for prior in priors]
end

function make_figure(df, y_symbol; kwargs...)
	# colors1 = get_colors(df[!, :prior], .8)
	colors2 = get_colors(df[!, :prior], .5)
	shapes = get_shapes(df[!, :prior])
	offset_size = 0.1
	offset_lookup = Dict(
		((:BetaBinomial11, :BetaBinomialk1, :BetaBinomial1k, :BetaBinomial1binomk2) .=> -offset_size)...,
		((:DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcess2_0, :DirichletProcessGP) .=> offset_size)...,
		((:uniform, :Westfall) .=> 0.0)...
	)
	offset = [offset_lookup[prior] for prior in df[!, :prior]]
	Plots.plot(
		df[!, :groups] .+ offset,
		df[!, y_symbol],
		group = df[!, :prior],

		linecolor			= colors2,
		markercolor			= colors2,
		markershape			= shapes,
		# markerstrokealpha	= 0.0,
		markerstrokewidth	= 0,
		# markerstrokecolor	= colors1,
		labels				= get_labels(df[!, :prior]),
		markersize			= 7,
		markeralpha			= 0.75,

		ylim   = (0, 1.05),
		yticks = 0:.2:1,
		xlim   = (1, 11),

		background_color_legend = nothing,
		foreground_color_legend = nothing
		;
		kwargs...
	)
end

p_null_any  = make_figure(reduced_results_df[(hypotheses=:null,)], :any_incorrect_mean;  xlabel = "",           ylabel = "P(one or more errors)", title = "Null model",  legend = :topleft);
p_null_prop = make_figure(reduced_results_df[(hypotheses=:null,)], :prop_incorrect_mean; xlabel = "no. groups", ylabel = "Proportion of errors",  title = "",            legend = false#=:topleft=#);
p_full_any  = make_figure(reduced_results_df[(hypotheses=:full,)], :any_incorrect_mean;  xlabel = "",           ylabel = "",                      title = "Full  model", legend = false#=:right=#);
p_full_prop = make_figure(reduced_results_df[(hypotheses=:full,)], :prop_incorrect_mean; xlabel = "no. groups", ylabel = "",                      title = "",            legend = false#=:topright=#);

joined_plot = plot(
	p_null_any,
	p_full_any,
	p_null_prop,
	p_full_prop,
	layout = (2, 2),
	left_margin = 4mm
);
savefig(plot(joined_plot, size = (900, 900)), joinpath("figures", "multipleComparisonPlot_4x4.pdf"))

make_title(λ) = @sprintf "%.2f * (errors | null model) +\n%.2f * (errors | full model)    " λ 1 - λ
p_null_prop2  = make_figure(reduced_results_df[(hypotheses=:null,)], :prop_incorrect_mean; xlabel = "",           ylabel = "Proportion of errors",            title = "Null model",     legend = :topleft);
p_full_prop2  = make_figure(reduced_results_df[(hypotheses=:full,)], :prop_incorrect_mean; xlabel = "",           ylabel = "",                                title = "Full model",     legend = false#=:topright=#);
p_lambda_0_95 = make_figure(lambda_results_df,                       :lambda_0_95_mean;    xlabel = "No. groups", ylabel = "Weighted proportion of errors",   title = make_title(0.95), legend = false#=:right=#);
p_lambda_0_50 = make_figure(lambda_results_df,                       :lambda_0_50_mean;    xlabel = "No. groups", ylabel = "",                                title = make_title(0.50), legend = false#=:right=#);

joined_plot_lambda = plot(
	p_null_prop2,
	p_full_prop2,
	p_lambda_0_95,
	p_lambda_0_50,
	layout = (2, 2),
	left_margin = 4mm
);
savefig(plot(joined_plot_lambda, size = (900, 900)), joinpath("figures", "multipleComparisonPlot_lambda_4x4.pdf"))

reduced_results_df_null = filter(:hypotheses => ==(:null), reduced_results_df)
plot(
	reduced_results_df_null[!, :groups],
	reduced_results_df_null[!, :any_incorrect_mean],
	group = reduced_results_df_null[!, :prior],

	markershape			= shapes,
	legend				= :topleft,
	# labels				= labels,
	markeralpha			= 0.75,

	xlabel = "no. groups",
	ylabel = "P(one or more errors)",
	ylim   = (0, 1.05),
	yticks = 0:.2:1,
	xlim   = (1, 11),

	background_color_legend = nothing,
	foreground_color_legend = nothing
)
@chain reduced_results_df begin
	plot(x = :groups)
end
results_df |>
	groupby(Cols(:prior, :groups, :hypotheses))
plot()


results, results_big, results_rhat = load_results()


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
