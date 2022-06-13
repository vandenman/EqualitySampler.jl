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

include("../simulation_helpers.jl")
include("lkj_prior.jl")
include("variance_demo_functions.jl")


#region Turing models
function validate_cholesky(L, ::Type{T} = Float64)
	return LA.det(Σ_chol_m) < eps(T)
end

@model function varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w,
					partition = nothing, η = 1.0, ::Type{T} = Float64) where T

	p = length(obs_mean_m)

	μ_m ~ filldist(Flat(), p)
	μ_w ~ filldist(Flat(), p)

	ρ ~ Dirichlet(Ones(2p))
	τ ~ JeffreysPriorStandardDeviation()

	ρ_constrained = isnothing(partition) ? ρ : EqualitySampler.Simulations.average_equality_constraints(ρ, partition)

	σ_m = τ .* view(ρ_constrained,   1: p)
	σ_w = τ .* view(ρ_constrained, p+1:2p)

	DynamicPPL.@submodel prefix="manual_lkj_m" R_chol_m = manual_lkj3(p, η, T)
	DynamicPPL.@submodel prefix="manual_lkj_w" R_chol_w = manual_lkj3(p, η, T)

	Σ_chol_m = LA.UpperTriangular(R_chol_m * LA.Diagonal(σ_m))
	Σ_chol_w = LA.UpperTriangular(R_chol_w * LA.Diagonal(σ_w))

	if LA.det(Σ_chol_m) < eps(T) || LA.det(Σ_chol_w) < eps(T) || any(i->Σ_chol_m[i, i] < 0, 1:p) || any(i->Σ_chol_w[i, i] < 0, 1:p)
		if T === Float64 && (any(i->Σ_chol_m[i, i] < zero(T), 1:p) || any(i->Σ_chol_w[i, i] < zero(T), 1:p))
			@show τ, ρ_constrained, σ_m, σ_w, ρ, partition
			error("bad!")
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

#endregion

#region simulated data
# let's test the model on simulated data
n, p = 1_000, 3

μ_1 = randn(p)
μ_2 = randn(p)
τ   = 10.0 # with 1.0 everything is too small
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
chn_full = sample(mod_var_ss_full, NUTS(), 15_000)
gen = generated_quantities(mod_var_ss_full, Turing.MCMCChains.get_sections(chn_full, :parameters))

post_means_full = extract_means_continuous_params(mod_var_ss_full, chn_full)
plot_retrieval(observed_values, post_means_full, string.(keys(post_means_full)))
plot_retrieval(true_values,     post_means_full, string.(keys(post_means_full)))

partition_prior = BetaBinomialMvUrnDistribution(2p, 1, 1)
mod_var_ss_eq = varianceMANOVA_suffstat_equality_selector(obs_mean_1, obs_cov_chol_1, n_1, obs_mean_2, obs_cov_chol_2, n_2, partition_prior)
spl = EqualitySampler.Simulations.get_sampler(mod_var_ss_eq, :custom, 0.0)
chn_eq = sample(mod_var_ss_eq, spl, 10_000; discard_initial=5000)

# params = DynamicPPL.syms(DynamicPPL.VarInfo(mod_var_ss_eq))
# spl2 = Gibbs(
# 	HMC(0.0, 20, filter(!=(:partition), params)...),
# 	MH(:partition)
# )
# chn_eq = sample(mod_var_ss_eq, spl2, 10_000; discard_initial=5000)
run(`beep_finished.sh 0`)

# @profview sample(mod_var_ss_eq, spl, 500)
# view_profile()

post_means_eq = extract_means_continuous_params(mod_var_ss_eq, chn_eq)
plot_retrieval(observed_values, post_means_eq, string.(keys(post_means_eq)))
plot_retrieval(true_values,     post_means_eq, string.(keys(post_means_eq)))
# these do not look good for p = 5!

partition_samples = extract_partition_samples(mod_var_ss_eq, chn_eq)
post_probs_eq = compute_post_prob_eq(partition_samples)
prop_incorrect_αβ(post_probs_eq, partition, false) # <- not good!
plot(partition_samples)

var_samples_full = extract_σ_samples(mod_var_ss_full, chn_full)
var_samples_eq   = extract_σ_samples(mod_var_ss_eq,   chn_eq)

plot(var_samples_full[:, 1])
plot(var_samples_eq[2000:end, 1]) # <- does NOT look good!

density_est_full = compute_density_estimate(var_samples_full)
density_est_eq   = compute_density_estimate(var_samples_eq)

plot(density_est_full.x, density_est_full.y, legend = true, xticks = 0:.25:1.5, xlim = (0, 1.5))
plot(density_est_eq.x,   density_est_eq.y, legend = true, xticks = 0:.25:1.5, xlim = (0, 1.5))

# heatmap
# x_nms = journal_data[!, :journal]# names(eq_table)[1]
# color_gradient = cgrad(cgrad(ColorSchemes.magma)[0.15:0.01:1.0])
# annotations = []
# for i in 1:7, j in i+1:8
# 	z = eq_table[9-i, 9-j]
# 	col = color_gradient[1 - z]
# 	push!(annotations,
# 		(
# 			8 - j + 0.5,
# 			i - 0.5,
# 			Plots.text(
# 				round_2_decimals(z),
# 				8, col, :center
# 			)
# 		)
# 	)
# end

# right_panel = heatmap(x_nms, reverse(x_nms), Matrix(eq_table)[8:-1:1, :],
# 	aspect_ratio = 1, showaxis = false, grid = false, color = color_gradient,
# 	clims = (0, 1),
# 	title = "Posterior probability of pairwise equality",
# 	#=colorbar_ticks = 0:.2:1, <- only works with pyplot =#
# 	annotate = annotations,
# 	xmirror = false);



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

obs_mean_m, obs_cov_m, n_m = get_suff_stats(data_m)
obs_mean_w, obs_cov_w, n_w = get_suff_stats(data_w)
obs_sd_m = sqrt.(LA.diag(obs_cov_m))
obs_sd_w = sqrt.(LA.diag(obs_cov_w))
obs_cov_chol_m = LA.cholesky(obs_cov_m).U
obs_cov_chol_w = LA.cholesky(obs_cov_w).U

LA.isposdef(obs_cov_m)
LA.isposdef(obs_cov_w)

@assert logpdf_mv_normal_suffstat(obs_mean_m, obs_cov_m, n_m, obs_mean_m, obs_cov_m) ≈ loglikelihood(MvNormal(obs_mean_m, obs_cov_m), data_m)

mod_var_ss_full = varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w)
chn_full = sample(mod_var_ss_full, NUTS(), 5_000)
gen = generated_quantities(mod_var_ss_full, Turing.MCMCChains.get_sections(chn, :parameters))

post_mean_μ_m      = vec(mean(group(chn, :μ_m).value.data, dims = 1))
post_mean_μ_w      = vec(mean(group(chn, :μ_w).value.data, dims = 1))
post_mean_σ_m      = mean(x[1] for x in gen)
post_mean_σ_w      = mean(x[2] for x in gen)
post_mean_Σ_chol_m = mean(triangular_to_vec(x[3]) for x in gen)
post_mean_Σ_chol_w = mean(triangular_to_vec(x[4]) for x in gen)

temp_hcat(x, y) = hcat(x, y, x .- y)

temp_hcat(post_mean_μ_m, obs_mean_m)
temp_hcat(post_mean_μ_w, obs_mean_w)
temp_hcat(post_mean_σ_m, obs_sd_m)
temp_hcat(post_mean_σ_w, obs_sd_w)
temp_hcat(post_mean_Σ_chol_m, triangular_to_vec(obs_cov_chol_m))
temp_hcat(post_mean_Σ_chol_w, triangular_to_vec(obs_cov_chol_w))

partition_prior = BetaBinomialMvUrnDistribution(10, 10, 1)
mod_var_ss_eq = varianceMANOVA_suffstat_equality_selector(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, partition_prior)

# adapted from https://discourse.julialang.org/t/get-list-of-parameters-from-turing-model/66278/8
continuous_params = filter(!=(:partition), DynamicPPL.syms(DynamicPPL.VarInfo(mod_var_ss_eq)))
spl = Gibbs(
	HMC(0.0, 20, continuous_params...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(10, get_logπ(mod_var_ss_eq)))
)
chn = sample(mod_var_ss_eq, spl, 1000)
gen = generated_quantities(mod_var_ss_eq, Turing.MCMCChains.get_sections(chn, :parameters))

partition_samples = Matrix{Int}(undef, length(gen[1][end]), length(gen))
for i in eachindex(gen)
	partition_samples[:, i] .= reduce_model(gen[i][end])
end

#endregion
