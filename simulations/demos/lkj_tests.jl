using Turing, MCMCChains, FillArrays, Distributions, LinearAlgebra
using KernelDensity, Plots
import DynamicPPL: @submodel
import PDMats
# import DistributionsAD
# import EqualitySampler


include("lkj_prior.jl")

"""
Marginal distributions of lkj samples

adpated from https://mjskay.github.io/ggdist/reference/lkjcorr_marginal.html
"""
function dlkjcorr_marginal(x, K, η, log = false)
	D = Beta(η - 1 + K / 2, η - 1 + K / 2)
	return log ? logpdf(D, (x + 1) / 2) - log(2) : pdf(D, (x + 1) / 2) / 2
end

function sample_lkj(K, η, no_samples)

	mod_manual_LKJ = manual_lkj3(K, η)
	chn = sample(mod_manual_LKJ, NUTS(), no_samples)
	gen = generated_quantities(mod_manual_LKJ, MCMCChains.get_sections(chn, :parameters))

	# form the sampled correlation matrices from the cholesky factors
	Rs = map(x-> x'x, gen)

	# extract the samples for the individual correlations
	ρs = Matrix{Float64}(undef, no_samples, K * (K - 1) ÷ 2)
	for i in axes(ρs, 1)
		ρs[i, :] .= [Rs[i][m, n] for m in 2:size(Rs[i], 1) for n in 1:m-1]
	end

	return ρs, chn
end

no_samples = 10_000
chn1 = sample(manual_lkj(10, 1.0),  NUTS(), no_samples)
chn2 = sample(manual_lkj2(10, 1.0), NUTS(), no_samples)
chn3 = sample(manual_lkj3(10, 1.0), NUTS(), no_samples)
chns_tp = (chn1, chn2, chn3)
map(MCMCChains.wall_duration, chns_tp)
map(x->mean(MCMCChains.summarystats(x).nt.ess_per_sec), chns_tp)


Ks = (3, 4, 5)
ηs = (1.0, 2.0, 5.0)
no_samples = 10_000
results = Vector{NamedTuple}()
for (K, η) in Iterators.product(Ks, ηs)
	@show K, η
	ρs, _ = sample_lkj(K, η, no_samples)
	push!(results, (K = K, η = η, ρs = ρs))
end

plts = Vector{Plots.Plot}(undef, length(results))
xcoords = -1.0:0.05:1.0 # for true density
for i in eachindex(plts)

	K, η, ρs = results[i]
	density_estimate = kde(view(ρs, :, 1); npoints = 2^12, boundary = (-1, 1))
	plt = Plots.plot(density_estimate.x, density_estimate.density, title = "K = $K, η = $η", legend = false, label = "sampled")
	Plots.plot!(plt, xcoords, dlkjcorr_marginal.(xcoords, K, η), label = "theoretical")
	plts[i] = plt

end

legend = Plots.plot([0 0], showaxis = false, grid = false, label = ["sampled" "theoretical"], axis = nothing);
Plots.plot(plts..., legend, layout = @layout [grid(3,3) a{0.2w}])
# plot(plts..., layout = (length(Ks), length(ηs)))

# use the LKJ to fit a normal

function perf_plot(model, chn, true_μ, true_σ, true_ρs)

	est_μ = MCMCChains.summarystats(group(chn, :μ)).nt.mean
	est_σ = MCMCChains.summarystats(group(chn, :σ)).nt.mean

	gen = generated_quantities(model, MCMCChains.get_sections(chn, :parameters))
	Rs = map(x -> x'x, gen)
	ρs = Matrix{Float64}(undef, length(Rs), p * (p - 1) ÷ 2)
	for i in axes(ρs, 1)
		ρs[i, :] .= [Rs[i][m, n] for m in 2:size(Rs[i], 1) for n in 1:m-1]
	end
	est_ρs = mean(eachrow(ρs))

	plt_μ = scatter(true_μ, est_μ, title = "μ")
	Plots.abline!(plt_μ, 1, 0, legend=false)

	plt_σ = scatter(true_σ, est_σ, title = "σ")
	Plots.abline!(plt_σ, 1, 0, legend=false)

	plt_ρs = scatter(true_ρs, est_ρs, title = "ρ")
	Plots.abline!(plt_ρs, 1, 0, legend=false)

	plot(plt_μ, plt_σ, plt_ρs, layout = grid(1, 3))

end

n, p = 1_000, 4

true_means  = collect(1.0:p)
true_R_chol = rand(LKJCholesky(p, 1.0, 'U'))
true_R      = true_R_chol.U'true_R_chol.U
true_sd     = collect(1.0:p)
true_S_chol = true_R_chol.U * Diagonal(true_sd)
true_S      = true_S_chol'true_S_chol

true_ρs = [true_R[m, n] for m in 2:size(true_R, 1) for n in 1:m-1]

true_S_chol2 = Cholesky(Matrix(true_S_chol), :U, 0)
true_S_pdm = PDMats.PDMat(true_S_chol2)
true_d = MvNormal(true_means, true_S_pdm)

x = rand(true_d, n)

# This works but is really slow
@model function mvnormal_turing(x, η, ::Type{T} = Float64) where T

	p, n = size(x)
	μ ~ MvNormal(Zeros(p), Ones(p))
	σ ~ filldist(InverseGamma(5.0, 4.0), p)
	DynamicPPL.@submodel prefix="manual_lkj" R_chol = manual_lkj3(p, η, T)

	if LinearAlgebra.det(R_chol) < sqrt(eps(T))
		if T === Float64
			@show R_chol
			@warn "bad real cholesky!"
		else
			@warn "bad dual cholesky!"
		end
		Turing.@addlogprob! -Inf
	else

		Σ_pdm = PDMats.PDMat(Cholesky(Matrix(R_chol * Diagonal(σ)), :U, 0))
		for i in 1:n
			x[:, i] ~ MvNormal(μ, Σ_pdm)
		end

	end

	return (R_chol, )
end

mod_mvnormal0 = mvnormal_turing(x, 1.0)
samps0 = sample(mod_mvnormal0, NUTS(), 1000) # slow
perf_plot(mod_mvnormal0, samps0, true_means, true_sd, true_ρs)

# arguably something that should live in Distributions.jl
function manual_loglikelihood(d::Distributions.AbstractMvNormal, ss::Distributions.MvNormalStats)
	μ, Σ = Distributions.params(d)
	x̄, S, n = ss.m, ss.s2, ss.tw
	# should check for length mismatches
	p = length(x̄)
	return (
		-n / 2 * (
			p * log(2pi) +
			LinearAlgebra.logdet(Σ) +
			PDMats.invquad(Σ, x̄ .- μ) +
			# PDMats.X_invA_Xt(Σ, (x̄ .- μ)') +
			# (x̄ .- μ)' / Matrix(Σ) * (x̄ .- μ) +
			LinearAlgebra.tr(Σ \ (S ./ ss.tw))
		)
	)
end

ss = Distributions.suffstats(MvNormal, x)
@assert loglikelihood(true_d, x) ≈ manual_loglikelihood(true_d, ss)

@model function mvnormal_suffstats(ss, η = 1.0, ::Type{T} = Float64) where T

	p = length(ss.m)
	μ ~ MvNormal(Zeros(p), Ones(p))
	σ ~ filldist(InverseGamma(5.0, 4.0), p)
	DynamicPPL.@submodel prefix="manual_lkj" R_chol = manual_lkj3(p, η, T)

	check_zero = T !== Float64 && any(iszero, view(R_chol, diagind(R_chol)))
	if check_zero
		# check_zero && @warn "dual cholesky contains zero on diagonal!"
		Turing.@addlogprob! -Inf
	else
		Σ_pdm = PDMats.PDMat(Cholesky(Matrix(R_chol * Diagonal(σ)), :U, 0))
		Turing.@addlogprob! manual_loglikelihood(MvNormal(μ, Σ_pdm), ss)
	end

	return R_chol
end

mod_mvnormal1 = mvnormal_suffstats(ss, 1.0)
samps1 = sample(mod_mvnormal1, NUTS(), 1000)
perf_plot(mod_mvnormal1, samps1, true_means, true_sd, true_ρs)


samps1.wall_duration

obs_mean, obs_cov_chol, n = get_normal_dense_chol_suff_stats(x)
prec_chol = inv(true_S_chol)
# inv(Diagonal(true_sd)) * inv(true_R_chol.U)
# Diagonal(1 ./ true_sd) * inv(true_R_chol.U)
# Diagonal(1 ./ true_sd) / true_R_chol.U
# Diagonal(1 ./ true_sd) \ true_R_chol.U
# Diagonal(1 ./ true_sd) * inv(true_R_chol.U)


loglikelihood(MvNormal(true_means, true_S), x) ≈ logpdf_mv_normal_chol_suffstat(obs_mean, obs_cov_chol, n, true_means, true_S_chol) ≈ logpdf_mv_normal_precision_chol_suffstat(obs_mean, obs_cov_chol, n, true_means, prec_chol)

mod_mvnormal = mvnormal_suffstats(obs_mean, obs_cov_chol, n)
samps = sample(mod_mvnormal, NUTS(), 5_000)

perf_plot(mod_mvnormal, samps, true_means, true_sd, true_ρs)