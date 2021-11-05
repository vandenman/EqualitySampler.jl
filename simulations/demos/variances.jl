using EqualitySampler, Turing, Plots, FillArrays, Plots.PlotMeasures, Colors
using Chain
using MCMCChains
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel, VarInfo
import Distributions

import Printf

#region functions

include("simulations/helpersTuring.jl")

round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

# include("simulations/helpersTuring.jl")
# function parameters_to_vec(partition, nt)
# 	v = Vector{Float64}(undef, sum(length, nt))
# 	start = 1
# 	for (key::Symbol, val::Union{Float64, Vector{Float64}, Vector{Int}}) in zip(keys(nt), nt)
# 		if key === :partition
# 			v[start:start+length(partition) - 1] .= Float64.(partition)
# 			start += length(partition) - 1
# 		else
# 			if val isa Vector{Float64}
# 				v[start:start+length(val) - 1] .= val
# 				start += length(val) - 1
# 			elseif val isa Float64
# 				v[start] = val
# 				start += 1
# 			else
# 				throw(DomainError("unexpected type in tuple", nt))
# 			end
# 		end
# 	end
# 	return v
# end

# function get_logπ(model, other_params)
# 	vari = VarInfo(model)
# 	logπ_internal = Turing.Inference.gen_logπ(vari, DynamicPPL.SampleFromPrior(), model)
# 	return function logπ(partition, nt)
# 		# @show partition, nt
# 		v = parameters_to_vec(partition, NamedTuple{other_params}(nt))
# 		logπ_internal(v)
# 	end
# end

# function get_logπ2(model)
# 	vari = VarInfo(model)
# 	return function logπ(partition, nt)
# 		DynamicPPL.setval!(vari, partition, DynamicPPL.VarName(:partition))
# 		for (key, val) in zip(keys(nt), nt)
# 			if key !== :partition
# 				DynamicPPL.setval!(vari, val, DynamicPPL.VarName(key))
# 			end
# 		end
# 		DynamicPPL.logjoint(model, vari)
# 	end
# end

function get_logπ(model)
	vari = VarInfo(model)
	mt = vari.metadata
	return function logπ(partition, nt)
		DynamicPPL.setval!(vari, partition, DynamicPPL.VarName(:partition))
		for (key, val) in zip(keys(nt), nt)
			if key !== :partition
				indices = mt[key].vns
				if !(val isa Vector)
					DynamicPPL.setval!(vari, val, indices[1])
				else
					ranges = mt[key].ranges
					for i in eachindex(indices)
						DynamicPPL.setval!(vari, val[ranges[i]], indices[i])
					end
				end
			end
		end
		DynamicPPL.logjoint(model, vari)
	end
end

function average_equality_constraints!(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer})
	idx_vecs = [Int[] for _ in eachindex(partition)]
	@inbounds for i in eachindex(partition)
		push!(idx_vecs[partition[i]], i)
	end

	@inbounds for idx in idx_vecs
		isempty(idx) && continue
		ρ[idx] .= mean(ρ[idx])
	end
	return ρ
end

average_equality_constraints(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer}) = average_equality_constraints!(copy(ρ), partition)


# function logpdf_mv_normal_suffstat(x̄, S, n, μ, Σ)
# 	if !LA.isposdef(Σ)
# 		@show Σ
# 		return -Inf
# 	end
# 	d = length(x̄)
# 	return (
# 		-n / 2 * (
# 			d * log(2pi) +
# 			LA.logdet(Σ) +
# 			(x̄ .- μ)' / Σ * (x̄ .- μ) +
# 			LA.tr(Σ \ S)
# 		)
# 	)
# end

# function logpdf_mv_normal_chol_suffstat(x̄, S_chol::LA.UpperTriangular, n, μ, Σ_chol::LA.UpperTriangular)
# 	d = length(x̄)
# 	return (
# 		-n / 2 * (
# 			d * log(2pi) +
# 			2LA.logdet(Σ_chol) +
# 			sum(x->x^2, (x̄ .- μ)' / Σ_chol) +
# 			sum(x->x^2, S_chol / Σ_chol)
# 		)
# 	)
# end

# see https://github.com/TuringLang/Turing.jl/issues/1629
quad_form_diag(M, v) = LA.Symmetric((v .* v') .* (M .+ M') ./ 2)

get_suff_stats(x) = begin
	n = size(x, 2)
	obs_mean = vec(mean(x, dims = 2))
	obs_cov  = cov(x') .* ((n - 1) / n)
	return obs_mean, obs_cov, n
end

function triangular_to_vec(x::LA.UpperTriangular{T, Matrix{T}}) where T
	# essentially vec(x) but without the zeros
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

function extract_means_continuous_params(model, chn)

	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chn, :parameters))

	vi = VarInfo(mod_var_ss_eq)
	mt = vi.metadata
	# handles submodels
	mu_keys = sort(collect(filter(x->occursin("μ", string(x)), keys(mt))))

	return (
		μ_1 = vec(mean(group(chn, mu_keys[1]).value.data, dims = 1)),
		μ_2 = vec(mean(group(chn, mu_keys[2]).value.data, dims = 1)),
		σ_1 = mean(x[1] for x in gen),
		σ_2 = mean(x[2] for x in gen),
		Σ_1 = mean(triangular_to_vec(x[3]) for x in gen),
		Σ_2 = mean(triangular_to_vec(x[4]) for x in gen)
	)
end

function extract_partition_samples(model, chn)

	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chn, :parameters))
	partition_samples = Matrix{Int}(undef, length(gen[1][end]), length(gen))
	for i in eachindex(gen)
		partition_samples[:, i] .= reduce_model(gen[i][end])
	end
	return partition_samples
end

#endregion

#region Turing models

@model function manual_lkj(K, eta, ::Type{T} = Float64) where T

	alpha = eta + (K - 2) / 2
	r_tmp ~ Beta(alpha, alpha)

	r12 = 2 * r_tmp - 1
	R = Matrix{T}(undef, K, K)
	R[1, 1] = one(T)
	R[1, 2] = r12
	R[2, 2] = sqrt(one(T) - r12^2.0)

	if K > 2

		y = Vector{T}(undef, K-2)
		# z = Vector{Vector{T}}(undef, K-2)
		z ~ MvNormal(ones(K * (K - 1) ÷ 2 - 1))
		z_idx = 0
		# @show y, z
		for m in 2:K-1

			z_i = view(z, z_idx + 1:z_idx + m)
			z_idx += m

			i = m - 1
			alpha -= 0.5
			y[i] ~ Beta(m / 2, alpha)

			# R[1:m, m+1] .= sqrt(y[i]) .* z[i] ./ LA.norm(z[i])
			R[1:m, m+1] .= sqrt(y[i]) .* z_i ./ LA.norm(z_i)
			# LA.normalize does not work because of https://github.com/JuliaDiff/ForwardDiff.jl/issues/175
			# R[1:m, m+1] .= sqrt(y[i]) .* LA.normalize(z[i])
			R[m+1, m+1] = sqrt(1 - y[i])

		end
	end

	# return give_cholesky ? LA.UpperTriangular(R) : LA.UpperTriangular(R)'LA.UpperTriangular(R)
	return LA.UpperTriangular(R)

end

# although MANOVA means "Multiple Analysis of Variance", here we mean that we are actually comparing variances and not means
@model function varianceMANOVA(data_m, data_w, η = 1.0)

	# this model specification does not work because the

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

	# this fails in Turing, it's better to form the cholesky decompositions instead of the full covariance matrices
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

	return (σ_m, σ_w, Σ_m, Σ_w)

end

@model function varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w,
					partition = nothing, η = 1.0, ::Type{T} = Float64) where T

	p = length(obs_mean_m)

	μ_m ~ MvNormal(ones(Float64, p))
	μ_w ~ MvNormal(ones(Float64, p))

	ρ ~ Dirichlet(ones(2p))
	#Todo improper prior
	τ ~ InverseGamma(0.1, 0.1)

	ρ_constrained = isnothing(partition) ? ρ : average_equality_constraints(ρ, partition)

	σ_m = τ .* view(ρ_constrained,   1: p)
	σ_w = τ .* view(ρ_constrained, p+1:2p)

	R_chol_m = @submodel $(Symbol("manual_lkj_m")) manual_lkj(p, η, T)
	R_chol_w = @submodel $(Symbol("manual_lkj_w")) manual_lkj(p, η, T)

	Σ_chol_m = LA.UpperTriangular(R_chol_m * LA.Diagonal(σ_m))
	Σ_chol_w = LA.UpperTriangular(R_chol_w * LA.Diagonal(σ_w))
	if LA.det(Σ_chol_m) < eps(T) || LA.det(Σ_chol_w) < eps(T)
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
	(σ_m, σ_w, Σ_chol_m, Σ_chol_w) = @submodel $(Symbol("varianceMANOVA_suffstat")) varianceMANOVA_suffstat(obs_mean_m, obs_cov_chol_m, n_m, obs_mean_w, obs_cov_chol_w, n_w, partition, η, T)

	return (σ_m, σ_w, Σ_chol_m, Σ_chol_w, partition)

end

#endregion

#region simulated data
# let's test the model on simulated data
n, p = 10_000, 5

μ_1 = randn(p)
μ_2 = randn(p)
τ   = 1.0
partition = rand(UniformMvUrnDistribution(2p))
ρ_constrained = average_equality_constraints(rand(Dirichlet(ones(2p))), partition)
σ_1, σ_2 = τ .* view(ρ_constrained,   1: p), τ .* view(ρ_constrained, p+1:2p)

R_chol_1, R_chol_2 = rand(Distributions.LKJCholesky(p, 1.0), 2)
Σ_chol_1, Σ_chol_2 = LA.UpperTriangular(R_chol_1.U * LA.Diagonal(σ_1)), LA.UpperTriangular(R_chol_2.U * LA.Diagonal(σ_2))
Σ_1, Σ_2 = Σ_chol_1'Σ_chol_1, Σ_chol_2'Σ_chol_2

D_1, D_2 = MvNormal(μ_1, Σ_1), MvNormal(μ_2, Σ_2)
data_1, data_2 = rand(D_1, n), rand(D_2, n)

obs_mean_1, obs_cov_1, n_1 = get_suff_stats(data_1)
obs_mean_2, obs_cov_2, n_2 = get_suff_stats(data_2)
obs_sd_1 = sqrt.(LA.diag(obs_cov_1))
obs_sd_2 = sqrt.(LA.diag(obs_cov_2))
obs_cov_chol_1 = LA.cholesky(obs_cov_1).U
obs_cov_chol_2 = LA.cholesky(obs_cov_2).U

observed_values = (obs_mean_1, obs_mean_2, obs_sd_1, obs_sd_2, triangular_to_vec(obs_cov_chol_1), triangular_to_vec(obs_cov_chol_2))
true_values     = (μ_1, μ_2, σ_1, σ_2, triangular_to_vec(Σ_chol_1), triangular_to_vec(Σ_chol_2))

mod_var_ss_full = varianceMANOVA_suffstat(obs_mean_1, obs_cov_chol_1, n_1, obs_mean_2, obs_cov_chol_2, n_2)
chn_full = sample(mod_var_ss_full, NUTS(), 5_000)
gen = generated_quantities(mod_var_ss_full, Turing.MCMCChains.get_sections(chn_full, :parameters))

post_mean_μ_1      = vec(mean(group(chn_full, :μ_m).value.data, dims = 1))
post_mean_μ_2      = vec(mean(group(chn_full, :μ_w).value.data, dims = 1))
post_mean_σ_1      = mean(x[1] for x in gen)
post_mean_σ_2      = mean(x[2] for x in gen)
post_mean_Σ_chol_1 = mean(triangular_to_vec(x[3]) for x in gen)
post_mean_Σ_chol_2 = mean(triangular_to_vec(x[4]) for x in gen)

function plot_retrieval(obs, estimate, nms)
	@assert length(obs) == length(estimate) == length(nms)
	plts = Vector{Plots.Plot}(undef, length(obs))
	for (i, (o, e, nm)) in enumerate(zip(obs, estimate, nms))
		plt = scatter(o, e, title = nm, legend = false)
		Plots.abline!(plt, 1, 0)
		plts[i] = plt
	end
	nr = isqrt(length(obs))
	nc = ceil(Int, length(obs) / nr)
	plot(plts..., layout = (nr, nc))
end
plot_retrieval(
	observed_values,
	(post_mean_μ_1, post_mean_μ_2, post_mean_σ_1, post_mean_σ_2, post_mean_Σ_chol_1, post_mean_Σ_chol_2),
	("μ_1", "μ_2", "σ_1", "σ_2", "Σ_1", "Σ_2")
)

plot_retrieval(
	true_values,
	(post_mean_μ_1, post_mean_μ_2, post_mean_σ_1, post_mean_σ_2, post_mean_Σ_chol_1, post_mean_Σ_chol_2),
	("μ_1", "μ_2", "σ_1", "σ_2", "Σ_1", "Σ_2")
)

# temp_hcat(x, y) = hcat(x, y, x .- y)
# temp_hcat(post_mean_μ_1, obs_mean_1)
# temp_hcat(post_mean_μ_2, obs_mean_2)
# temp_hcat(post_mean_σ_1, obs_sd_1)
# temp_hcat(post_mean_σ_2, obs_sd_2)
# temp_hcat(post_mean_Σ_chol_1, triangular_to_vec(obs_cov_chol_1))
# temp_hcat(post_mean_Σ_chol_2, triangular_to_vec(obs_cov_chol_2))

# temp_hcat(post_mean_μ_1, μ_1)
# temp_hcat(post_mean_μ_2, μ_2)
# temp_hcat(post_mean_σ_1, σ_1)
# temp_hcat(post_mean_σ_2, σ_2)
# temp_hcat(post_mean_Σ_chol_1, triangular_to_vec(Σ_chol_1))
# temp_hcat(post_mean_Σ_chol_2, triangular_to_vec(Σ_chol_2))

partition_prior = BetaBinomialMvUrnDistribution(p, p, 1)
mod_var_ss_eq = varianceMANOVA_suffstat_equality_selector(obs_mean_1, obs_cov_chol_1, n_1, obs_mean_2, obs_cov_chol_2, n_2, partition_prior)
continuous_params = filter(!=(:partition), DynamicPPL.syms(DynamicPPL.VarInfo(mod_var_ss_eq)))
spl = Gibbs(
	HMC(0.0, 20, continuous_params...),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(10, get_logπ(mod_var_ss_eq)))
)
chn_eq = sample(mod_var_ss_eq, spl, 5_000)

post_means_eq = extract_means_continuous_params(mod_var_ss_eq, chn_eq)
plot_retrieval(observed_values, post_means_eq, string.(keys(post_means_eq)))

partition_samples = extract_partition_samples(mod_var_ss_eq, chn_eq)

compute_post_prob_eq()


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

#region other random stuff, maybe delete this?
@model function gdemo_0(z0)
	x  ~ Normal(0, 1)
	z0 ~ Normal(x, 1)
	return x
end

@model function gdemo_1(z1)
	x = @submodel gdemo_0 gdemo_0(z1)
	y ~ Normal(0, 1)
	z1 ~ Normal(x + y)
	return x, y
end

@model function gdemo_2(z2)
	x, y = @submodel gdemo_1 gdemo_1(z2)
	z ~ Normal(0, 1)
	z2 ~ Normal(x + y + z)
	return x, y, z
end

# 0 submodels
sample(gdemo_0(2), HMC(0.05, 20), 100)
sample(gdemo_0(2), NUTS(), 100)


# 1 nested submodel
gdemo_1_instance = gdemo_1()
spl = HMC(0.0, 20)
chn = sample(gdemo_1_instance, spl, 100)
gen = generated_quantities(gdemo_1_instance, Turing.MCMCChains.get_sections(chn, :parameters))

# 1 nested submodel - gibbs sampler
spl = Gibbs(HMC(0.05, 20, Symbol("gdemo_0.x")), HMC(0.05, 20, :y))
chn = sample(gdemo_1_instance, spl, 100)

# 2 nested submodels
gdemo_2_instance = gdemo_2()
spl = HMC(0.0, 20)
chn = sample(gdemo_2_instance, spl, 100)

# 2 nested submodel - gibbs sampler
spl = Gibbs(HMC(0.05, 20, Symbol("gdemo_1.gdemo_0.x")), HMC(0.05, 20, Symbol("gdemo_1.y")), HMC(0.05, 20, :z))
chn = sample(gdemo_2_instance, spl, 10)


spl = Gibbs(
	HMC(0.0, 20, Symbol("gdemo_2.gdemo_3.z")),
	HMC(0.0, 20, Symbol("gdemo_2.y")),
	HMC(0.0, 20, :z)
)
chn = sample(gdemo_1_instance, spl, 100)



# not sure what was going on below here


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


@model function mmm(d, eta)
	R ~ LKJCholesky(d, eta)
end

m_instance = mmm(3, 1.0)
samps = sample(m_instance, NUTS(), 100)

using LinearAlgebra
import DynamicPPL, Bijectors
### ---------------- DynamicPPL stuff -----------------------------

DynamicPPL.vectorize(d::LKJCholesky, r::Cholesky) =  copy(vec( r.factors ))


function DynamicPPL.reconstruct(d::LKJCholesky, val::AbstractVector)
    return  LinearAlgebra.Cholesky(reshape(copy(val), size(d)), 'L', 0)
end



### ---------------- Bijector stuff -----------------------------

struct LKJCholBijector <: Bijector{2} end

function (b::LKJCholBijector)(x::Cholesky)
    return LinearAlgebra.Cholesky(Array(_link_w_lkj_chol(x.U.data)') + zero(x.U.data), 'L', 0)
end

(b::LKJCholBijector)(X::AbstractArray{<:Cholesky}) = map(b, X)

function (ib::Inverse{<:LKJCholBijector})(y::Cholesky)
    return LinearAlgebra.Cholesky(Array(_inv_link_w_lkj_chol(y.U.data)'), 'L', 0)
end

(ib::Inverse{<:LKJCholBijector})(Y::AbstractArray{<:Cholesky}) = map(ib, Y)


function Bijectors.logabsdetjac(::Inverse{LKJCholBijector}, y::Cholesky)
    K = LinearAlgebra.checksquare(y)

    result = float(zero(eltype(y)))
    for j in 2:K, i in 1:(j - 1)
        @inbounds abs_y_i_j = abs((y.L)[i, j])
        result += (K - i + 1) * (logtwo - (abs_y_i_j + StatsFuns.log1pexp(-2 * abs_y_i_j)))
    end

    return result
end
function Bijectors.logabsdetjac(b::LKJCholBijector, X::Cholesky)
    #=
    It may be more efficient if we can use un-contraint value to prevent call of b
    It's recommended to directly call
    `logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})`
    if possible.
    =#
    return -Bijectors.logabsdetjac(inv(b), (b(X)))
end
function Bijectors.logabsdetjac(b::LKJCholBijector, X::AbstractArray{<:Cholesky})
    return Bijectors.mapvcat(X) do x
        Bijectors.logabsdetjac(b, x)
    end
end




function _inv_link_w_lkj_chol(y)
    K = LinearAlgebra.checksquare(y)
    w = zero(y)

    @inbounds for j in 1:K
        w[1, j] = 1
        for i in 2:j
            z = tanh(y[i-1, j])
            tmp = w[i-1, j]

            w[i-1, j] = z * tmp

            w[i, j] = tmp * sqrt(1 - min(typeof(z)(oneminsq), z^2))
        end
        for i in (j+1):K
            w[i, j] = 0
        end
    end
    return w
end


function _link_w_lkj_chol(w)
    K = LinearAlgebra.checksquare(w)

    z = similar(w) # z is also UpperTriangular.
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    # This block can't be integrated with loop below, because w[1,1] != 0.
    @inbounds z[1, 1] = 0

    @inbounds for j=2:K
        tmp_w = max(typeof(w[1,j])(-onemin), min(typeof(w[1,j])(onemin), w[1, j]))
        z[1, j] = atanh(tmp_w)
        tmp = sqrt(1 - tmp_w^2)
        for i in 2:(j - 1)
            p = w[i, j] / tmp
            p = max(typeof(p)(-onemin), min(typeof(p)(onemin), p))

            tmp *= sqrt(1 - p^2)
            z[i, j] = atanh(p)
        end
        z[j, j] = 0
    end
    return z
end

Bijectors.bijector(d::LKJCholesky) = LKJCholBijector()


Turing.Utilities.FlattenIterator(name, value::Cholesky) = Turing.Utilities.FlattenIterator(Symbol(name), Array(value.L))
#endregion