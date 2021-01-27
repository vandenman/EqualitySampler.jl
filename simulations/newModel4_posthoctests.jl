# for linear models
using GLM, StatsBase, StatsModels, DataFrames
# for Bayesian inference
using Turing, DynamicPPL, FillArrays, LinearAlgebra
# for plots
using StatsPlots, Plots
import Colors # doesn't really need to be imported with using
include("src/loglikelihood.jl")
include("src/helperfunctions.jl")
include("src/newApproach4.jl")

#region helper function 
function trace_plots_nuisance(chain)
	params = string.(chain.name_map.parameters)
	filter!(!startswith("equal_indices"), params)
	subplots = (
		plot(chain[p], title = "$p",	legend = false)	for p in params
	)
	return subplots
end

function get_θ_cs(model, chain)
	
	gen = generated_quantities(model, chain)
	θ_cs_est = Matrix{Float64}(undef, length(gen), length(gen[1][1]))
	for i in eachindex(gen)
		θ_cs_est[i, :] .= gen[i][1]
	end
	return θ_cs_est
end

trace_plots_θ_cs(model, chain) = trace_plots_θ_cs(model, chain, get_θ_cs(model, chain))
function trace_plots_θ_cs(model, chain, θ_cs_est)
	
	n_groups = size(θ_cs_est)[2]
	color_options = Colors.distinguishable_colors(n_groups, [RGB(1,1,1), RGB(0,0,0)], dropseed=true);
	eq_samples = mapslices(reduce_model, Int.(get_eq_samples(chain)), dims = 2)

	subplots = (
		plot(view(θ_cs_est, :, i), title = "θ_cs_$i", legend = false, 
				linecolor = color_options[eq_samples[:, i]])
			for i in axes(θ_cs_est, 2)
	)
	return subplots
end

trace_plots(model, chain) = trace_plots(model, chain, get_θ_cs(model, chain))
function trace_plots(model, chain, θ_cs_est)

	subplots_nuisance = trace_plots_nuisance(chain)
	subplotsθ_cs = trace_plots_θ_cs(model, chain, θ_cs_est)
	plot(Iterators.flatten((subplots_nuisance, subplotsθ_cs))...)
	
end

density_θ_cs(model, chain) = density_θ_cs(get_θ_cs(model, chain))
function density_θ_cs(θ_cs_est)
	θ_samples = DataFrame(
		θ_cs = vec(θ_cs_est),
		dim = vcat(fill.(axes(θ_cs_est, 2), size(θ_cs_est)[1])...)
	)
	@df θ_samples density(:θ_cs, group = :dim)
end

function getQ(n_groups::Int)
	Σₐ = Matrix{Float64}(I, n_groups, n_groups) .- (1.0 / n_groups)
	_ , v = eigen(Σₐ)
	return v[end:-1:1, end:-1:2]
end

function average_equality_constraints(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
	# only makes sense when selecting on a simplex
	ρ_c = similar(ρ)
	for i in eachindex(ρ)
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	return ρ_c
end

function select_equality_constraints(θ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
	θ_c = similar(θ)
	for i in eachindex(θ_c)
		θ_c[i] = θ[findfirst(==(equal_indices[i]), equal_indices)]
	end
	return θ_c
end
#endregion

#region model definitions without ρ
@model function anova_model_raw(y, X, Q, uniform = true, indicator_state = ones(Int, size(X)[2]), ::Type{T} = Float64) where {T}

	# TODO: this only works for a 1-way anova
	n, n_groups = size(X)
	# sample equalities among the sds
	equal_indices = TArray{Int}(n_groups)
	equal_indices .= indicator_state
	for i in 1:n_groups
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end
	indicator_state .= equal_indices

	σ² ~ InverseGamma(1, 1)
	μ_grand ~ Normal(0, 1)
	# θ ~ filldist(Normal(0, 1), n_groups)
	# # constrain θ according to the sampled equalities
	# θ_c = select_equality_constraints(θ, equal_indices)
	# # subtract the mean of θ to ensure the sum to zero constraint
	# θ_cs = θ_c .- mean(θ_c)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	# constrain θ according to the sampled equalities
	# TODO: first check if full model works as intended!
	θ_cs = θ_s#average_equality_constraints(θ_s, equal_indices)
	
	# definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	y ~ MvNormal(μ_grand .+ σ .* (X * θ_cs), σ)
	return (θ_cs, θ_s)
end

@model function anova_model_ss(obs_mean, obs_var, obs_n, Q, uniform = true, indicator_state = ones(Int, size(X)[2]), ::Type{T} = Float64) where {T}

	# TODO: this only works for a 1-way anova

	n_groups = length(obs_mean)
	# sample equalities among the sds
	equal_indices = TArray{Int}(n_groups)
	equal_indices .= indicator_state
	for i in 1:n_groups
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end
	indicator_state .= equal_indices

	σ² ~ InverseGamma(1, 1)
	μ_grand ~ Normal(0, 1)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	# constrain θ according to the sampled equalities
	# TODO: first check if full model works as intended!
	θ_cs = θ_s#average_equality_constraints(θ_s, equal_indices)
	
	# definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + θ_cs[i], σ, obs_n[i])
	end
	return (θ_cs, θ_s)
end
#endregion

#region model definitions with ρ

@model function anova_model_rho_ss(obs_mean, obs_var, obs_n, uniform = true, indicator_state = ones(Int, size(X)[2]), ::Type{T} = Float64) where {T}

	#= TODO: 

		this can be done way easier!
		first implement the full model with the regular sum to zero constraint
		afterward, average the groups that are equal just like with ρ!

	=#
	# TODO: this only works for a 1-way anova

	n_groups = length(obs_mean)
	# sample equalities among the sds
	equal_indices = TArray{Int}(n_groups)
	equal_indices .= indicator_state
	for i in 1:n_groups
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end
	indicator_state .= equal_indices

	σ² ~ InverseGamma(1, 1)
	μ_grand ~ Normal(0, 1)
	ρ ~ Dirichlet(ones(Int, n_groups))
	ρ_c = average_equality_constraints(ρ, equal_indices)
	θ_cs = quantile.(Normal(0, 1), ρ_c)
	
	# definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + θ_cs[i], σ, obs_n[i])
	end
	return (θ_cs, θ_c)
end


#endregion

#region simulate data
n_groups = 4
n_obs_per_group = 250
n_obs = n_groups * n_obs_per_group
μ_grand_true = 5.0
θ_true_val = 2.0
θ_true = (1:n_groups) .* θ_true_val
θ_true = collect(θ_true) .- mean(θ_true)
σ_true = 1.0

g = Vector{Int}(undef, n_obs)
for (i, r) in enumerate(Iterators.partition(1:n_obs, ceil(Int, n_obs / n_groups)))
	g[r] .= i
end

D = MvNormal(μ_grand_true .+ σ_true .* θ_true[g], σ_true)
y = vec(rand(D, 1))

df = DataFrame(:y => y, :g => categorical(g))
combine(groupby(df, :g), :y => mean, :y => x->mean(x .- μ_grand_true))


# get design matrix without intercept bit with full dummy coding 
#TODO: maybe an indicator vector is easier (and more efficient) like in the data simulation
X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))
#endregion

#region fit frequentist linear model
mod = fit(LinearModel, @formula(y ~ 1 + g), df)#, contrasts = Dict(:g => StatsModels.FullDummyCoding()))
coefs = copy(coef(mod))
# for default coding
ests = similar(coefs)
ests[1] = coefs[1] - mean(df[!, :y])
ests[2:end] .= coefs[2:end] .+ coefs[1] .- mean(df[!, :y])
scatter(ests, θ_true, legend = :none)
Plots.abline!(1, 0)
# for FullDummyCoding -- not sure why that suddenly gives an error about inverting things
# s = mean(coefs[2:end])
# ests = similar(coefs)
# ests[1] = coefs[1] + s
# ests[2:end] = coefs[2:end] .- s
# scatter(ests[2:end], θ_true, legend = :none)
# Plots.abline!(1, 0)
#endregion

#region model on raw data
uniform_prior = true
prior_distribution = uniform_prior ? UniformConditionalUrnDistribution(ones(Int, n_groups), 1) : BetaBinomialConditionalUrnDistribution(ones(Int, n_groups), 1)
Q = getQ(size(X)[2])
mod_raw = anova_model_raw(y, X, Q, uniform_prior)

# study prior
chn_raw_prior = sample(mod_raw, Prior(), 20_000)
visualize_helper(prior_distribution, chn_raw_prior)

gen = generated_quantities(mod_raw, chn_raw_prior)
θ_cs_prior = Matrix{Float64}(undef, length(gen), n_groups)
for i in eachindex(gen)
	θ_cs_prior[i, :] .= gen[i][1]
end
density(θ_cs_prior[:, 1])
mean(θ_cs_prior, dims = 1)

# study posterior
spl = Gibbs(PG(100, :equal_indices), HMC(0.0005, 20, :σ, :μ_grand, :θ))
chn_eq_raw = sample(mod_raw, spl, 50_000)

# Trace plots of nuisance parameters
trace_plots(mod_raw, chn_eq_raw)

θ_cs_post = get_θ_cs(mod_raw, chn_eq_raw)
trace_plots(mod_raw, chn_eq_raw, θ_cs_post)
density_θ_cs(θ_cs_post)
mean(θ_cs_post, dims = 1)

# posterior model space
post_model_probs = sort(compute_model_probs(chn_eq_raw), byvalue=true)
df_post = DataFrame(
	model = collect(keys(post_model_probs)),
	postprob = collect(values(post_model_probs)),
	no_incl = count_equalities.(collect(keys(post_model_probs)))
)

# posterior parameter space after applying model constraints
mean(θ_cs_est, dims = 1)
# density(θ_cs_est[:, 1])

θ_samples = DataFrame(
	θ_cs = vec(θ_cs_est),
	dim = vcat(fill.(axes(θ_cs_est, 2), size(θ_cs_est)[1])...)
)
@df θ_samples density(:θ_cs, group = :dim)
combine(groupby(θ_samples, :dim), :θ_cs => x->mean(@view x[10_000:end]))

#endregion

#region sufficient statistics

temp_df = combine(groupby(df, :g), :y => mean, :y => x -> mean(xx -> xx^2, x), :y => length)
obs_means = temp_df[!, "y_mean"]
obs_var = temp_df[!, "y_function"]
obs_n = temp_df[!, "y_length"]
Q = getQ(length(obs_means))
sum(Q * randn(length(obs_means) - 1))


uniform_prior = true
prior_distribution = uniform_prior ? UniformConditionalUrnDistribution(ones(Int, n_groups), 1) : BetaBinomialConditionalUrnDistribution(ones(Int, n_groups), 1)
mod_ss = anova_model_ss(obs_means, obs_var, obs_n, Q, uniform_prior)

# study prior
chn_ss_prior = sample(mod_ss, Prior(), 20_000)
visualize_helper(prior_distribution, chn_ss_prior)

θ_cs_prior = get_θ_cs(mod_ss, chn_ss_prior)
density_θ_cs(θ_cs_prior)
mean(θ_cs_prior, dims = 1)
trace_plots(mod_ss, chn_ss_prior, θ_cs_prior)

# study posterior
spl = Gibbs(PG(50, :equal_indices), HMC(0.05, 20, :σ², :μ_grand, :θ))
chn_ss_post = sample(mod_ss, spl, 20_000)

θ_cs_post = get_θ_cs(mod_ss, chn_ss_post)
trace_plots(mod_ss, chn_ss_post, θ_cs_post)
density_θ_cs(θ_cs_post)
mean(θ_cs_post, dims = 1)

visualize_helper(prior_distribution, chn_ss_post)

post_model_probs = sort(compute_model_probs(chn_ss_post), byvalue=true)
df_ss_post = DataFrame(
	model = collect(keys(post_model_probs)),
	postprob = collect(values(post_model_probs)),
	no_incl = count_equalities.(collect(keys(post_model_probs)))
)

k = 5
Dρ = Dirichlet(ones(k))
ρ  = rand(Dρ, 10000)
θc = similar(ρ)
for i in axes(ρ, 2)
	θc[:, i] .= quantile.(Normal(0, 1), ρ[:, i])
end
density(θc[1, :])
plot!(x->pdf(Normal(0, 1), x), -4:.01:4)

function geteigen(X::AbstractMatrix)
	F = eigen(X)
	reverse!(F.values)
	k = length(F.values)
	F.vectors .= F.vectors[k:-1:1, k:-1:1]
	F
end

k = 5
Σₐ = Matrix{Float64}(I, k, k) .- (1 / k)
F = geteigen(Σₐ)
Qₐ = F.vectors[1:k, 1:k-1]
Qₐ'
Qₐ * Diagonal(F.values[1:k-1]) * Qₐ'

oo = randn(k-1)

X * Qₐ * Qₐ' * oo
X * Qₐ * oo
X * (Qₐ' \ oo)

α
αa1 = 1

o2 = Qₐ * oo
sum(o2)
o3 = copy(o2)
o3[[1, 4]] .= mean(o3[[1, 4]])
sum(o3)

inv(Qₐ')

Qₐ' \ oo
sum(Qₐ' \ oo)
solve(Qₐ', oo)
Qₐ * Qₐ'

Qₐ * Qₐ' * oo
oo2 = randn(k - 1)
mean(Qₐ * oo2)
X * Qₐ * oo2

oo2 = 10 .* randn(100 - 1)
oo3 = oo2 .- mean(oo2)
oo4 = copy(oo3)
oo4[1] = mean(oo3[[1, 3]])
oo4[3] = mean(oo3[[1, 3]])
sum(oo4)

#endregion