using EqualitySampler, Turing, Plots, FillArrays, Plots.PlotMeasures, Colors
using Chain
using MCMCChains
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel

import Printf
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

include("simulations/silentGeneratedQuantities.jl")
include("simulations/helpersTuring.jl")

function logpdf_mv_normal_suffstat(x̄, S, n, μ, Σ)
	if !LA.isposdef(Σ)
		@show Σ
		return -Inf
	end
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			LA.logdet(Σ) +
			(x̄ .- μ)' / Σ * (x̄ .- μ) +
			LA.tr(Σ \ S)
		)
	)
end

function logpdf_mv_normal_chol_suffstat(x̄, S_chol::LA.UpperTriangular, n, μ, Σ_chol::LA.UpperTriangular)
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			2LA.logdet(Σ_chol) +
			sum(x->x^2, (x̄ .- μ)' / Σ_chol) +
			sum(x->x^2, S_chol / Σ_chol)
		)
	)
end

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

# see https://github.com/TuringLang/Turing.jl/issues/1629
quad_form_diag(M, v) = LA.Symmetric((v .* v') .* (M .+ M') ./ 2)

# although a MANOVA means "Multiple Analysis of Variance", here we mean that we compare the variance
@model function varianceMANOVA(data_m, data_w, η = 1.0)

	n_m, n_w = size(data_m, 2), size(data_w, 2)
	p = size(data_m, 1)

	μ_m ~ MvNormal(ones(Float64, p))
	μ_w ~ MvNormal(ones(Float64, p))

	R_m ~ LKJ(p, η)
	R_w ~ LKJ(p, η)

	ρ ~ Dirichlet(ones(2p))
	τ ~ InverseGamma(0.1, 0.1)

	# apply the equality model here to ρ.

	σ_m = τ .* view(ρ,   1: p)
	σ_w = τ .* view(ρ, p+1:2p)

	# TODO: form the cholesky decompositions instead of full covariance matrices?
	Σ_m = quad_form_diag(R_m, σ_m)
	Σ_w = quad_form_diag(R_w, σ_w)

	if !LA.isposdef(Σ_m) || !LA.isposdef(Σ_w)
		Turing.@addlogprob! -Inf
		return (σ_m, σ_w, Σ_m, Σ_w)
	end

	for i in 1:n_m
		data_m[:, i] ~ MvNormal(μ_m, Σ_m)
	end

	for i in 1:n_w
		data_w[:, i] ~ MvNormal(μ_w, Σ_w)
	end
	# for col in eachcol(data_m)
	# 	col ~ MvNormal(μ_m, Σ_m)
	# end

	# for col in eachcol(data_w)
	# 	col ~ MvNormal(μ_w, Σ_w)
	# end

	return (σ_m, σ_w, Σ_m, Σ_w)

end

@model function varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, η = 1.0, ::Type{T} = Float64) where T

	p = size(data_m, 1)

	μ_m ~ MvNormal(ones(Float64, p))
	μ_w ~ MvNormal(ones(Float64, p))

	ρ ~ Dirichlet(ones(2p))
	τ ~ InverseGamma(0.1, 0.1)

	# apply the equality model here to ρ.
	σ_m = τ .* view(ρ,   1: p)
	σ_w = τ .* view(ρ, p+1:2p)

	# TODO: form the cholesky decompositions instead of full covariance matrices?
	# R_m ~ LKJ(p, η)
	# R_w ~ LKJ(p, η)
	# Σ_m = quad_form_diag(R_m, σ_m)
	# Σ_w = quad_form_diag(R_w, σ_w)

	R_chol_m = @submodel $(Symbol("manual_lkj")) manual_lkj(p, 1.0, false, T)
	R_chol_w = @submodel $(Symbol("manual_lkj")) manual_lkj(p, 1.0, false, T)
	Σ_chol_m = LA.UpperTriangular(R_chol_m * LA.Diagonal(σ_m))
	Σ_chol_w = LA.UpperTriangular(R_chol_w * LA.Diagonal(σ_w))
	if LA.det(Σ_chol_m) < eps(T) || LA.det(Σ_chol_w) < eps(T)
		# @show R_chol_m, R_chol_w, σ_m, σ_w
		Turing.@addlogprob! -Inf
		return (σ_m, σ_w, Σ_chol_m, Σ_chol_w)
	end

	Turing.@addlogprob! logpdf_mv_normal_chol_suffstat(obs_mean_m, obs_cov_chol_m, n_m, μ_m, Σ_chol_m)
	Turing.@addlogprob! logpdf_mv_normal_chol_suffstat(obs_mean_w, obs_cov_chol_w, n_w, μ_w, Σ_chol_w)

	return (σ_m, σ_w, Σ_chol_m, Σ_chol_w)

end

get_suff_stats(x) = begin
	n = size(x, 2)
	obs_mean = vec(mean(x, dims = 2))
	obs_cov  = cov(x') .* ((n - 1) / n)
	return obs_mean, obs_cov, n
end

obs_mean_m, obs_cov_m, n_m = get_suff_stats(data_m)
obs_mean_w, obs_cov_w, n_w = get_suff_stats(data_w)
obs_sd_m = sqrt.(LA.diag(obs_cov_m))
obs_sd_w = sqrt.(LA.diag(obs_cov_w))
obs_cov_chol_m = LA.cholesky(obs_cov_m).U
obs_cov_chol_w = LA.cholesky(obs_cov_w).U

LA.isposdef(obs_cov_m)
LA.isposdef(obs_cov_w)

# ff = Distributions.suffstats(Distributions.MvNormal, data_m)


logpdf_mv_normal_suffstat(obs_mean_m, obs_cov_m, n_m, obs_mean_m, obs_cov_m)
loglikelihood(MvNormal(obs_mean_m, obs_cov_m), data_m)

mod_var_ss_full = varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w)
spl = NUTS()#HMC(0.00, 20)
chn = sample(mod_var_ss_full, spl, 1_000)

gen = generated_quantities(mod_var_ss_full, Turing.MCMCChains.get_sections(chn, :parameters))

function triangular_to_vec(x::LA.UpperTriangular{T, Matrix{T}}) where T
	p = size(x, 1)
	result = Vector{T}(undef, p * (p + 1) ÷ 2)
	idx = 1
	for i in 1:p
		for j in 1:i
			result[idx] = x[j, i]
			idx += 1
		end
	end
	result
end
triangular_to_vec(Σ_chol_m)

post_mean_μ_m      = vec(mean(group(chn, :μ_m).value.data, dims = 1))
post_mean_μ_w      = vec(mean(group(chn, :μ_w).value.data, dims = 1))
post_mean_σ_m      = mean(x[1] for x in gen)
post_mean_σ_w      = mean(x[2] for x in gen)
post_mean_Σ_chol_m = mean(triangular_to_vec(x[3]) for x in gen)
post_mean_Σ_chol_w = mean(triangular_to_vec(x[4]) for x in gen)

hcat(post_mean_μ_m, obs_mean_m, post_mean_μ_m .- obs_mean_m)
hcat(post_mean_μ_w, obs_mean_w, post_mean_μ_w .- obs_mean_w)
hcat(post_mean_σ_m, obs_sd_m,   post_mean_σ_m .- obs_sd_m)
hcat(post_mean_σ_w, obs_sd_w,   post_mean_σ_w .- obs_sd_w)
hcat(post_mean_Σ_chol_m, obs_cov_chol_m, post_mean_Σ_chol_m .- obs_cov_chol_w)
hcat(post_mean_Σ_chol_w, obs_cov_chol_w, post_mean_Σ_chol_w .- obs_cov_chol_m)

	vec(mean(group(chn, :μ_m).value.data, dims = 1)),
	obs_mean_m
)
vec(gen[1][3])
hcat(
	mean(triangular_to_vec(x[3]) for x in gen),
	triangular_to_vec(obs_cov_chol_m)
)
hcat(
	mean(triangular_to_vec(x[4]) for x in gen),
	triangular_to_vec(obs_cov_chol_w)
)


(R_chol_m, R_chol_w, σ_m, σ_w) = ([1.0 -0.13700879180205505 -0.08709729528087903 -0.3526799851828268 -0.16609925852289315; -0.13700879180205505 1.0 0.7407414513582602 -0.31073702850840296 0.30865624629527866; -0.08709729528087903 0.7407414513582602 1.0000000000000002 -0.7602533797542129 -0.15000458911847728; -0.3526799851828268 -0.31073702850840296 -0.7602533797542129 0.9999999999999997 0.3900196787215228; -0.16609925852289315 0.30865624629527866 -0.15000458911847728 0.3900196787215228 1.0], [1.0 0.5962534059779934 -0.6800277351005118 -0.6052380263598311 0.47085671210089225; 0.5962534059779934 1.0 -0.4905423968004018 -0.5554234031799439 0.4034820781085538; -0.6800277351005118 -0.4905423968004018 1.0 0.5475132489358331 -0.2224925538132741; -0.6052380263598311 -0.5554234031799439 0.5475132489358331 1.0 -0.023161513325564098; 0.47085671210089225 0.4034820781085538 -0.2224925538132741 -0.023161513325564098 1.0000000000000002], [2.035829896214002, 0.08430108193964772, 0.05838618921404039, 0.64830633613398, 0.09253287689002059], [0.2177831522501285, 0.4199858167724204, 0.35158292355918797, 0.532944130429116, 0.09521255833014333])

mod_var_full = varianceMANOVA(data_m, data_w)
spl = HMC(0.01, 10)
chn = sample(mod_var_full, spl, 1_000)


@model function manual_lkj(K, eta, give_cholesky::Bool = true, ::Type{T} = Float64) where T

	alpha = eta + (K - 2) / 2
	r_tmp ~ Beta(alpha, alpha)

	r12 = 2 * r_tmp - 1
	R = Matrix{T}(undef, K, K)
	R[1, 1] = one(T)
	R[1, 2] = r12
	R[2, 2] = sqrt(one(T) - r12^2)

	if K > 2

		y = Vector{T}(undef, K-2)
		z = Vector{Vector{T}}(undef, K-2)

		for m in 2:K-1

			i = m - 1
			alpha -= 0.5
			y[i] ~ Beta(m / 2, alpha)
			z[i] ~ MvNormal(ones(m))

			R[1:m, m+1] .= sqrt(y[i]) .* z[i] ./ sqrt(sum(x->x^2, z[i]))
			R[m+1, m+1] = sqrt(1 - y[i])

		end
	end

	return give_cholesky ? LA.UpperTriangular(R) : LA.UpperTriangular(R)'LA.UpperTriangular(R)

end

mod_manual_LKJ = manual_lkj(5, 1.0)
spl = HMC(0.0, 20)# NUTS()
chn = sample(mod_manual_LKJ, spl, 100)

# see https://discourse.julialang.org/t/turing-jl-warnings-when-running-generated-quantities/64698/2
gen = generated_quantities(mod_manual_LKJ, Turing.MCMCChains.get_sections(chn, :parameters))
gen[1]'gen[1]
# Rs = map(x-> x'x, gen) # all correlation matrices

@model function mvnormal_manual(obs_mean, obs_cov_chol, n, ::Type{T} = Float64) where T

	p, n = size(x)
	μ ~ MvNormal(fill(10.0, p))
	σ ~ filldist(InverseGamma(1, 1), p)

	Rchol = @submodel $(Symbol("manual_lkj")) manual_lkj(p, 1.0, false, T)

	Σchol = LA.UpperTriangular(Rchol * LA.Diagonal(σ_true))
	Turing.@addlogprob! logpdf_mv_normal_chol_suffstat(obs_mean, obs_cov_chol, n, μ, Σchol)

	return (μ, σ, Rchol'Rchol, Σchol'Σchol)
end

@model function mvnormal_naive(x, ::Type{T} = Float64) where T

	p, n = size(x)
	μ ~ MvNormal(fill(10.0, p))
	σ ~ filldist(InverseGamma(1, 1), p)

	R = LKJ(p, 1.0)

	Σ = quad_form_diag(R, σ)
	if !LA.isposdef(Σ)
		Turing.@addlogprob! -Inf
		return (μ, σ, R, Σ)
	end

	for i in 1:n
		x[:, i] ~ MvNormal(μ, Σ)
	end
	return (μ, σ, R, Σ)
end


# see https://github.com/TuringLang/Turing.jl/issues/1629
quad_form_diag(M, v) = LA.Symmetric((v .* v') .* (M .+ M') ./ 2)

p, n = 5, 1000
μ_true = randn(p)
R_true = rand(LKJ(p, 1.0))
σ_true = rand(InverseGamma(4, 2), p)
Σ_true = quad_form_diag(R_true, σ_true)
# LA.Diagonal(σ_true) * R_true * LA.Diagonal(σ_true)
Rchol_true = LA.cholesky(R_true).U
Rchol_true'Rchol_true .- R_true
Σchol_true = LA.cholesky(Σ_true).U
V = Rchol_true * LA.Diagonal(σ_true)
V'V .- Σ_true
LA.logdet(Σchol_true)
LA.logdet(Rchol_true) + sum(log.(σ_true))


x = rand(MvNormal(μ_true, Σ_true), n)

obs_mean, obs_cov, n = get_suff_stats(x)
obs_cov_chol = LA.cholesky(obs_cov).U

mod_mvnormal = mvnormal_manual(obs_mean, obs_cov_chol, n)
chn_manual = sample(mod_mvnormal, NUTS(), 1000)
chn_naive  = sample(mod_mvnormal, NUTS(), 1000)

MCMCChains.wall_duration(chn_manual)
MCMCChains.wall_duration(chn_naive)


gg = generated_quantities(mod_mvnormal, Turing.MCMCChains.get_sections(chn, :parameters))
mean(x[1] for x in gg)
mean(x[2] for x in gg)
mean(x[3] for x in gg)

mean(x[1] for x in gg) .- μ_true
mean(x[2] for x in gg) .- σ_true
mean(x[3] for x in gg) .- R_true
mean(x[4] for x in gg) .- Σ_true

hcat(mean(x[1] for x in gg), μ_true)
hcat(mean(x[2] for x in gg), σ_true)
hcat(mean(x[3] for x in gg), R_true)
hcat(mean(x[4] for x in gg), Σ_true)


Rchol = LA.cholesky(R_true).U
Rchol'Rchol

LA.Diagonal(σ_true) * R_true * LA.Diagonal(σ_true)
LA.Diagonal(σ_true) * Rchol' * Rchol * LA.Diagonal(σ_true)
(LA.Diagonal(σ_true) * Rchol') * (Rchol * LA.Diagonal(σ_true))
tmp = LA.Diagonal(σ_true) * Rchol'
LA.Symmetric(tmp * tmp')
Σ_true


Rs[1]
Rchol_2_R(x) = LA.UpperTriangular(x)' * LA.UpperTriangular(x)
Rchol_2_R(gen[1])

ee = LA.UpperTriangular(gen[1])
ee[2, 1]


T = Float64
r12 = 0.5
K = 5
m = 2
(y[i], z[i]) = (0.7578720687023597, [0.5925628763325705, 0.673869131356593])


@model function LKJ_test(p, η = 1.0)
	R_m ~ LKJ(p, η)
end
LKJ_mod = LKJ_test(2, 1)
spl = HMC(0.01, 20)
chn = sample(LKJ_mod, spl, 1_000)

# mod = varianceMANOVA(data_m, data_w)
# spl = NUTS()
# chn = sample(mod, spl, 1_000)






#region testing likelihood simplifications
x = rand(MvNormal(ones(5)), 10)
obs_mean, obs_cov, n = get_suff_stats(x)

μ_test = zero(obs_mean)
Σ_test = rand(Wishart(10, one(obs_cov)))

obs_cov_chol = LA.cholesky(obs_cov)
Σ_chol = LA.cholesky(Σ_test)

LA.isposdef(S)


loglikelihood(MultivariateNormal(μ_test, Σ_test), x)
logpdf_mv_normal_suffstat(obs_mean, obs_cov, n, μ_test, Σ_test)
logpdf_mv_normal_chol_suffstat(obs_mean, obs_cov_chol, n, μ_test, Σ_chol)

@btime loglikelihood(MultivariateNormal(μ_test, Σ_test), x)
@btime logpdf_mv_normal_suffstat(obs_mean, obs_cov, n, μ_test, Σ_test)
@btime logpdf_mv_normal_chol_suffstat(obs_mean, obs_cov_chol, n, μ_test, Σ_chol)
#endregion
