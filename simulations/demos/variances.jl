#=

	TODO:

	- increase line thickness of the density plots
	- add x-axis label

	- make 1 function for the density plot (individual plot)
	- make 1 function for the density plot (combined plot)
	- make 1 function for the heatmap plot
	- the part that make a single trace plot should just make trace plots for all continuous parameters

=#

using EqualitySampler, EqualitySampler.Simulations, Turing, Plots, FillArrays, Plots.PlotMeasures, Colors
using Chain
using MCMCChains
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel, VarInfo
import Distributions
import KernelDensity
import Printf
import ColorSchemes

include("../simulation_helpers.jl")
include("lkj_prior.jl")
include("variance_demo_functions.jl")

#region initial values
function get_starting_values(data_1, data_2; kwargs...)
	obs_mean_1, obs_cov_chol_1, _ = get_normal_dense_chol_suff_stats(data_1)
	obs_mean_2, obs_cov_chol_2, _ = get_normal_dense_chol_suff_stats(data_2)
	obs_cov_1 = obs_cov_chol_1'obs_cov_chol_1
	obs_cov_2 = obs_cov_chol_2'obs_cov_chol_2
	obs_sd_1 = sqrt.(LA.diag(obs_cov_1))
	obs_sd_2 = sqrt.(LA.diag(obs_cov_2))
	get_starting_values(obs_mean_1, obs_mean_2, obs_sd_1, obs_sd_2, obs_cov_chol_1, obs_cov_chol_2; kwargs...)
end

function get_starting_values(obs_mean_1, obs_mean_2, obs_sd_1, obs_sd_2, obs_cov_chol_1, obs_cov_chol_2; include_partition = true, target_quantile = .3)
	τ_sample = sum(obs_sd_1) + sum(obs_sd_2)
	ρ_sample = [obs_sd_1; obs_sd_2] ./ τ_sample
	obs_R_chol_1 = obs_cov_chol_1 * LA.Diagonal(1 ./ obs_sd_1)
	obs_R_chol_2 = obs_cov_chol_2 * LA.Diagonal(1 ./ obs_sd_2)
	if include_partition

		partition_diff = [abs(a - b) for a in ρ_sample, b in ρ_sample]
		diff_vals = filter(!iszero, triangular_to_vec(LA.UpperTriangular(partition_diff)))
		target = SB.quantile(diff_vals, target_quantile)
		adj_mat = abs.(partition_diff) .< target
		partition_sample = reduce_model(map(x->findfirst(isone, x), eachcol(adj_mat)))

		return (;
			obs_mean_1, obs_mean_2,
			τ_sample, ρ_sample, obs_R_chol_1, obs_R_chol_2,
			partition_sample
		)

	else

		return (;
			obs_mean_1, obs_mean_2,
			τ_sample, ρ_sample, obs_R_chol_1, obs_R_chol_2
		)

	end
end

function starting_values_to_init_params(starting_values, model)

	nt = (
		partition									= starting_values.partition_sample,
		var"varianceMANOVA_suffstat.μ_m"			= starting_values.obs_mean_1,
		var"varianceMANOVA_suffstat.μ_w"			= starting_values.obs_mean_2,
		var"varianceMANOVA_suffstat.ρ"				= starting_values.ρ_sample,
		var"varianceMANOVA_suffstat.τ"				= starting_values.τ_sample,
		var"varianceMANOVA_suffstat.manual_lkj_m.y"	= bounded_to_unbounded(starting_values.obs_R_chol_1),
		var"varianceMANOVA_suffstat.manual_lkj_w.y"	= bounded_to_unbounded(starting_values.obs_R_chol_2)
	)
	varinfo = Turing.VarInfo(model)
	model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(nt));
	init_params = varinfo[Turing.SampleFromPrior()]::Vector{Float64}
	return init_params
end
#endregion

#region Turing models
function validate_cholesky(L, ::Type{T} = Float64) where T
	return LA.det(Σ_chol_m) < eps(T)
end

@model function varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m,
										obs_mean_w, obs_cov_chol_w, n_w,
										partition = nothing, η = 1.0, ::Type{T} = Float64) where T

	p = length(obs_mean_m)

	μ_m ~ filldist(Flat(), p)
	μ_w ~ filldist(Flat(), p)

	ρ ~ Dirichlet(Ones(2p))
	τ ~ JeffreysPriorStandardDeviation()

	ρ_constrained = isnothing(partition) ? ρ : EqualitySampler.Simulations.average_equality_constraints(ρ, partition)

	σ_m = τ .* view(ρ_constrained,   1: p)
	σ_w = τ .* view(ρ_constrained, p+1:2p)

	# Until Turing properly supports sampling LKJCholesky this is unfortunately necessary
	DynamicPPL.@submodel prefix="manual_lkj_m" R_chol_m = manual_lkj3(p, η, T)
	DynamicPPL.@submodel prefix="manual_lkj_w" R_chol_w = manual_lkj3(p, η, T)

	Σ_chol_m = LA.UpperTriangular(R_chol_m * LA.Diagonal(σ_m))
	Σ_chol_w = LA.UpperTriangular(R_chol_w * LA.Diagonal(σ_w))

	# Until Turing has something like a multi_normal_cholesky this is unfortunately necessary
	if LA.det(Σ_chol_m) < eps(T) || LA.det(Σ_chol_w) < eps(T) || any(i->Σ_chol_m[i, i] < 0, 1:p) || any(i->Σ_chol_w[i, i] < 0, 1:p)
		if T === Float64 && (any(i->Σ_chol_m[i, i] < zero(T), 1:p) || any(i->Σ_chol_w[i, i] < zero(T), 1:p))
			@show τ, ρ_constrained, σ_m, σ_w, ρ, partition
			@warn "bad Real!"
		else
			@warn "bad Diff!"
		end
		Turing.@addlogprob! -Inf
		return (σ_m, σ_w, Σ_chol_m, Σ_chol_w)
	end

	Turing.@addlogprob! logpdf_mv_normal_chol_suffstat(obs_mean_m, obs_cov_chol_m, n_m, μ_m, Σ_chol_m)
	Turing.@addlogprob! logpdf_mv_normal_chol_suffstat(obs_mean_w, obs_cov_chol_w, n_w, μ_w, Σ_chol_w)

	return (σ_m, σ_w, Σ_chol_m, Σ_chol_w)

end

@model function varianceMANOVA_suffstat_equality_selector(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w,
	partition_prior, η = 1.0, ::Type{T} = Float64) where T

	partition ~ partition_prior
	DynamicPPL.@submodel prefix="varianceMANOVA_suffstat" (σ_m, σ_w, Σ_chol_m, Σ_chol_w) = varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, partition, η, T)

	return (σ_m, σ_w, Σ_chol_m, Σ_chol_w, partition)

end

function logpdf_manual(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, partition_prior,
	μ_m, μ_w, τ, ρ, partition, R_chol_m, R_chol_w, η = 1.0)

	p = length(obs_mean_m)

	res = 0.0
	res += logpdf(Dirichlet(Ones(2p)), ρ)
	res += logpdf(JeffreysPriorStandardDeviation(), τ)
	res += logpdf(partition_prior, partition)
	res += logpdf(LKJCholesky(p, η), R_chol_m)
	res += logpdf(LKJCholesky(p, η), R_chol_w)

	ρ_constrained = isnothing(partition) ? ρ : EqualitySampler.Simulations.average_equality_constraints(ρ, partition)

	σ_m = τ .* view(ρ_constrained,   1: p)
	σ_w = τ .* view(ρ_constrained, p+1:2p)

	Σ_chol_m = LA.UpperTriangular(R_chol_m * LA.Diagonal(σ_m))
	Σ_chol_w = LA.UpperTriangular(R_chol_w * LA.Diagonal(σ_w))

	res += logpdf_mv_normal_chol_suffstat(obs_mean_m, obs_cov_chol_m, n_m, μ_m, Σ_chol_m)
	res += logpdf_mv_normal_chol_suffstat(obs_mean_w, obs_cov_chol_w, n_w, μ_w, Σ_chol_w)

	return res

end

#endregion

#region simulated data
# let's test the model on simulated data
n, p = 1_000, 5

μ_1 = randn(p)
μ_2 = randn(p)
τ   = 40# 5p#50.0 # this should scale with p, e.g., 10p or so
partition = rand(UniformMvUrnDistribution(2p))
ρ_constrained = EqualitySampler.Simulations.average_equality_constraints(rand(Dirichlet(ones(2p))), partition)
σ_1, σ_2 = τ .* view(ρ_constrained,   1: p), τ .* view(ρ_constrained, p+1:2p)

R_chol_1, R_chol_2 = rand(Distributions.LKJCholesky(p, 1.0), 2)
Σ_chol_1, Σ_chol_2 = LA.UpperTriangular(R_chol_1.U * LA.Diagonal(σ_1)), LA.UpperTriangular(R_chol_2.U * LA.Diagonal(σ_2))
Σ_1, Σ_2 = Σ_chol_1'Σ_chol_1, Σ_chol_2'Σ_chol_2

D_1, D_2 = MvNormal(μ_1, Σ_1), MvNormal(μ_2, Σ_2)
data_1, data_2 = rand(D_1, n), rand(D_2, n)

obs_mean_1, obs_cov_chol_1, n_1 = get_normal_dense_chol_suff_stats(data_1)
obs_mean_2, obs_cov_chol_2, n_2 = get_normal_dense_chol_suff_stats(data_2)
obs_cov_1 = obs_cov_chol_1'obs_cov_chol_1
obs_cov_2 = obs_cov_chol_2'obs_cov_chol_2
obs_sd_1 = sqrt.(LA.diag(obs_cov_1))
obs_sd_2 = sqrt.(LA.diag(obs_cov_2))

[abs(i - j) for i in [obs_sd_1; obs_sd_2], j in [obs_sd_1; obs_sd_2]]
[i - j for i in [obs_sd_1; obs_sd_2], j in [obs_sd_1; obs_sd_2]]
[i == j for i in partition, j in partition]

observed_values = (obs_mean_1, obs_mean_2, obs_sd_1, obs_sd_2, triangular_to_vec(obs_cov_chol_1), triangular_to_vec(obs_cov_chol_2))
true_values     = (μ_1, μ_2, σ_1, σ_2, triangular_to_vec(Σ_chol_1), triangular_to_vec(Σ_chol_2))

mod_var_ss_full = varianceMANOVA_suffstat(obs_mean_1, obs_cov_chol_1, n_1, obs_mean_2, obs_cov_chol_2, n_2)
chn_full = sample(mod_var_ss_full, NUTS(), 5_000)

post_means_full = extract_means_continuous_params(mod_var_ss_full, chn_full)
# plot_retrieval(observed_values, post_means_full, string.(keys(post_means_full)))
# plot_retrieval(true_values,     post_means_full, string.(keys(post_means_full)))
plot_retrieval2(observed_values, post_means_full, string.(keys(post_means_full)))
plot_retrieval2(true_values,     post_means_full, string.(keys(post_means_full)))

# the model with equality constraints converges slower than the model without and requires more samples
n_iter = 50_000; n_burn = n_iter ÷ 5
partition_prior = BetaBinomialMvUrnDistribution(2p, 1, 1)
mod_var_ss_eq = varianceMANOVA_suffstat_equality_selector(obs_mean_1, obs_cov_chol_1, n_1, obs_mean_2, obs_cov_chol_2, n_2, partition_prior)

starting_values = get_starting_values(data_1, data_2)
init_params = starting_values_to_init_params(starting_values, mod_var_ss_eq)
@assert init_params[1:2p] ≈ starting_values.partition_sample
# τ_sample = sum(obs_sd_1) + sum(obs_sd_2)
# ρ_sample = [obs_sd_1; obs_sd_2] ./ τ_sample
# obs_R_chol_1 = obs_cov_chol_1 * LA.Diagonal(1 ./ obs_sd_1)
# obs_R_chol_2 = obs_cov_chol_2 * LA.Diagonal(1 ./ obs_sd_2)

# # TODO: is this a good way to get a sample estimate for the partition?
# partition_diff = [abs(a - b) for a in ρ_sample, b in ρ_sample]
# diff_vals = filter(!iszero, triangular_to_vec(LA.UpperTriangular(partition_diff)))
# target = SB.quantile(diff_vals, .35)
# adj_mat = abs.(partition_diff) .< target
# partition_sample = reduce_model(map(x->findfirst(isone, x), eachcol(adj_mat)))
# reduce_model(partition)

# nt = (
# 	partition									= partition_sample,
# 	var"varianceMANOVA_suffstat.μ_m"			= obs_mean_1,
# 	var"varianceMANOVA_suffstat.μ_w"			= obs_mean_2,
# 	var"varianceMANOVA_suffstat.ρ"				= ρ_sample,
# 	var"varianceMANOVA_suffstat.τ"				= τ_sample,
# 	var"varianceMANOVA_suffstat.manual_lkj_m.y"	= bounded_to_unbounded(obs_R_chol_1),
# 	var"varianceMANOVA_suffstat.manual_lkj_w.y"	= bounded_to_unbounded(obs_R_chol_2)
# )
# varinfo = Turing.VarInfo(mod_var_ss_eq)
# mod_var_ss_eq(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(nt));
# init_params = varinfo[Turing.SampleFromPrior()]::Vector{Float64}

spl = EqualitySampler.Simulations.get_sampler(mod_var_ss_eq, :custom, 0.0)
chn_eq = sample(mod_var_ss_eq, spl, n_iter; init_params = init_params)
run(`beep_finished.sh`)

post_means_eq = extract_means_continuous_params(mod_var_ss_eq, chn_eq[n_burn:end, :, :])
plot_retrieval(observed_values,  post_means_eq,   string.(keys(post_means_eq)))
plot_retrieval(true_values,      post_means_eq,   string.(keys(post_means_eq)))
plot_retrieval2(observed_values, post_means_eq, string.(keys(post_means_full)))
plot_retrieval2(true_values,     post_means_eq, string.(keys(post_means_full)))

partition_samples = extract_partition_samples(mod_var_ss_eq, chn_eq)
post_probs_eq = compute_post_prob_eq(view(partition_samples, n_burn:n_iter, :))
prop_incorrect_αβ(post_probs_eq, partition, false) # <- not good!
post_probs_eq .+ 10 .* [p == q for p in partition, q in partition]
plot(partition_samples)

var_samples_full = extract_σ_samples(mod_var_ss_full, chn_full)
var_samples_eq   = extract_σ_samples(mod_var_ss_eq,   chn_eq[n_burn:end, :, :])

plot(var_samples_full[:, 1])
plot(var_samples_eq[:, 1])

density_est_full = compute_density_estimate(var_samples_full)
density_est_eq   = compute_density_estimate(var_samples_eq)

xr = extrema(density_est_full.x)
xr = floor(xr[1]), ceil(xr[2])
xticks = range(xr[1], xr[2], length = 5)
xlim = extrema(xticks)
p_top = plot(density_est_full.x, density_est_full.y, legend = true, xticks = xticks, xlim = xlim)
p_bot = plot(density_est_eq.x,   density_est_eq.y,   legend = true, xticks = xticks, xlim = xlim)
plot(p_top, p_bot, layout = (2, 1), size = 600 .* (2, 1))

# The trick is using more MCMC samples, more burnin, and good starting values.



#endregion

#region data example
big5_data = DF.DataFrame(CSV.File(joinpath("simulations", "demos", "data", "personality_variability.csv")))

df_m = @chain big5_data begin
	DF.filter(:sex => ==("male"), _)
	DF.select(_, setdiff(names(_), vcat("sex", filter(startswith("self_"), names(_)))))
end

df_w = @chain big5_data begin
	DF.filter(:sex => ==("female"), _)
	DF.select(_, setdiff(names(_), vcat("sex", filter(startswith("self_"), names(_)))))
end

data_m = permutedims(Matrix(df_m))
data_w = permutedims(Matrix(df_w))

obs_mean_m, obs_cov_chol_m, n_m = get_normal_dense_chol_suff_stats(data_m)
obs_mean_w, obs_cov_chol_w, n_w = get_normal_dense_chol_suff_stats(data_w)
obs_cov_m = obs_cov_chol_m'obs_cov_chol_m
obs_cov_w = obs_cov_chol_w'obs_cov_chol_w
obs_sd_m = sqrt.(LA.diag(obs_cov_m))
obs_sd_w = sqrt.(LA.diag(obs_cov_w))

observed_values = (obs_mean_m, obs_mean_w, obs_sd_m, obs_sd_w, triangular_to_vec(obs_cov_chol_m), triangular_to_vec(obs_cov_chol_w))

LA.isposdef(obs_cov_m)
LA.isposdef(obs_cov_w)

@assert logpdf_mv_normal_suffstat(obs_mean_m, obs_cov_m, n_m, obs_mean_m, obs_cov_m) ≈ loglikelihood(MvNormal(obs_mean_m, obs_cov_m), data_m)

mod_var_ss_full = varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w)
chn_full = sample(mod_var_ss_full, NUTS(), 5_000)
post_means_full = extract_means_continuous_params(mod_var_ss_full, chn_full)
post_mean_μ_m, post_mean_μ_w, post_mean_σ_m, post_mean_σ_w, post_mean_Σ_chol_m, post_mean_Σ_chol_w = post_means_full

plot_retrieval2(observed_values, post_means_full, string.(keys(post_means_full)))

partition_prior = BetaBinomialMvUrnDistribution(10, 1, 10)
mod_var_ss_eq = varianceMANOVA_suffstat_equality_selector(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, partition_prior)

starting_values = get_starting_values(data_m, data_w; target_quantile = .3)
hcat(starting_values.partition_sample, starting_values.ρ_sample, [obs_sd_m; obs_sd_w])
init_params = starting_values_to_init_params(starting_values, mod_var_ss_eq)
@assert starting_values.partition_sample ≈ init_params[1:10]

n_iter = 50_000
n_burn = 5_000
spl = EqualitySampler.Simulations.get_sampler(mod_var_ss_eq, :custom, 0.0)
chn_eq = sample(mod_var_ss_eq, spl, n_iter; init_params = init_params)
run(`beep_finished.sh`)

# gen = generated_quantities(mod_var_ss_eq, Turing.MCMCChains.get_sections(chn_eq, :parameters))

# all_keys = keys(VarInfo(mod_var_ss_eq).metadata)
# # handles submodels
# key_μ_m = all_keys[findfirst(x->occursin("μ_m", string(x)), all_keys)]
# key_μ_w = all_keys[findfirst(x->occursin("μ_w", string(x)), all_keys)]

# vec(mean(group(chn_eq, key_μ_m).value.data, dims = 1))

post_means_eq = extract_means_continuous_params(mod_var_ss_eq, chn_eq[2n_burn:end, :, :])
post_mean_μ_m, post_mean_μ_w, post_mean_σ_m, post_mean_σ_w, post_mean_Σ_chol_m, post_mean_Σ_chol_w = post_means_eq
plot_retrieval2(observed_values, post_means_eq, string.(keys(post_means_eq)))

partition_samples = extract_partition_samples(mod_var_ss_eq, chn_eq)
post_probs_eq = compute_post_prob_eq(view(partition_samples, n_burn:n_iter, :))
plot(partition_samples)

var_samples_full = extract_σ_samples(mod_var_ss_full, chn_full)
var_samples_eq   = extract_σ_samples(mod_var_ss_eq,   chn_eq[n_burn:end, :, :])

plot(var_samples_full[:, 1])
plot(var_samples_eq[:, 1])

density_est_full = compute_density_estimate(var_samples_full)
density_est_eq   = compute_density_estimate(var_samples_eq)

labels_temp = replace.(names(df_m), "other_" => "")
legend_labels = ["     Men - " .* labels_temp; "Women - " .* labels_temp]
legend_labels_short = ["M-" .* first.(labels_temp); "W-" .* first.(labels_temp)]

# use the same colors for the big 5 variables, and different line styles for men vs women
accent_color(x) = range(x, colorant"white", length = 20)[9]#[12]
# vcat(ColorSchemes.seaborn_colorblind[1:5], accent_color.(ColorSchemes.seaborn_colorblind[1:5]))
# linecolor = [ColorSchemes.seaborn_colorblind[1:5]; accent_color.(ColorSchemes.seaborn_colorblind[1:5])]
basecolors = theme_palette(:auto)[1:5] # ColorSchemes.seaborn_colorblind[1:5]
linecolor = [accent_color.(basecolors); basecolors]
linestyle = repeat([:solid, :dash], inner = 5)
linewidth = 1.5

xr = extrema(vcat(extrema(density_est_full.x)..., extrema(density_est_eq.x)...))
xr = floor(xr[1]), ceil(xr[2])
xticks = range(xr[1], xr[2], length = 5)
xlim = extrema(xticks)
xlim = (2.9, 5.5)
xticks = 3:5
p_top = plot(density_est_full.x, density_est_full.y, linewidth = linewidth, linecolor = permutedims(linecolor), linestyle = permutedims(linestyle),  legend = false, xticks = xticks, xlim = xlim, title = "Full model", labels = permutedims(legend_labels_short))
p_bot = plot(density_est_eq.x,   density_est_eq.y,   linewidth = linewidth, linecolor = permutedims(linecolor), linestyle = permutedims(linestyle), legend = false, xticks = xticks, xlim = xlim, title = "Model averaged", xlab = "σ")
plot(p_top, p_bot, layout = (2, 1))

plt_legend = plot(zeros(1, 10); showaxis = false, grid = false, axis = nothing,
	foreground_color_legend = nothing, background_color_legend = nothing, left_margin = 0mm, labels = permutedims(legend_labels_short))

joined_density_plots = plot(p_top, p_bot, plt_legend, size = (2, 1) .* 400,
	layout = @layout [grid(2,1) a{0.1w}]);

plt_ylabel = plot([0 0]; ylab = "Density", showaxis = false, grid = false, axis = nothing, legend = false, left_margin = -6mm, right_margin = 6mm, ymirror=true)
left_panel = plot(plt_ylabel, p_top, p_bot, plt_legend,
	bottom_margin = 3mm,
	layout = @layout [a{0.00001w} grid(2, 1) a{0.1w}]
);

# plot(p_top, legend = (.95, .95), legend_background_color=:transparent, legend_foreground_color=:transparent)

left_panel = plot(plt_ylabel, plot(p_top, legend = (.9, .95), legend_background_color=:transparent, legend_foreground_color=:transparent),
	p_bot,
	bottom_margin = 3mm,
	layout = @layout [a{0.00001w} grid(2, 1)]
);


# heatmap
x_nms = legend_labels_short
color_gradient = cgrad(cgrad(ColorSchemes.magma)[0.15:0.01:1.0])
eq_table = copy(post_probs_eq)
for i in axes(eq_table, 1), j in i:size(eq_table, 2)
	eq_table[i, j] = NaN
end
annotations = []
for i in 1:9, j in i+1:10
	z = eq_table[11-i, 11-j]
	col = color_gradient[1 - z]
	push!(annotations,
		(
			10 - j + 0.5,
			i - 0.5,
			Plots.text(
				round_2_decimals(z),
				8, col, :center
			)
		)
	)
end

right_panel = heatmap(x_nms, reverse(x_nms), Matrix(eq_table)[10:-1:1, :],
	aspect_ratio = 1, showaxis = false, grid = false, color = color_gradient,
	clims = (0, 1),
	title = "Posterior probability of pairwise equality",
	annotate = annotations,
	xmirror = false);


joined_plot = plot(left_panel, right_panel, layout = (1, 2), size = (2, 1) .* 500);
savefig(joined_plot, "figures/demo_variances_2panel_plot_newcolors.pdf")
#endregion
