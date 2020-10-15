# this example works

using Turing, DynamicPPL, LinearAlgebra

function multivariate_normal_likelihood_0(obs_mean, obs_var, pop_mean, pop_sds, n)
	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
	pop_prec = 1 ./ (pop_sds .^2)
	return - n / 2 * (
		2 * sum(log, pop_sds) +
		length(pop_sds) * log(2 * float(pi)) +
		dot(obs_var,  pop_prec) -
		2 * dot(obs_mean, pop_prec .* pop_mean) +
		dot(pop_mean, pop_prec .* pop_mean)
	)
end

@model function model(obs_mean, obs_mean_sq, n, ::Type{T} = Float64) where {T}

	k = length(obs_mean)

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ = Vector{T}(undef, k)
	for i in 1:k
		μ[i] ~ Normal(0, 1)
	end
	# k * τ .* ρ2 gives the precisions of each group
	σ = 1 ./ sqrt.(k * τ .* ρ)

	Turing.@addlogprob! multivariate_normal_likelihood_0(obs_mean, obs_mean_sq, μ, σ, n)
	# for i in 1:n
	# 	x[:, i] ~ MvNormal(μ, σ)
	# end
	return (σ, )

end

# simulate data
n = 1000
k = 3
sds = Float64[1, 3, 5]
D = MvNormal(sds)
x = rand(D, n)
obsmu  = mean(x, dims = 2)
obsmu2 = mean(x .^ 2, dims = 2)
sqrt.(var(x, dims = 2))

@assert sum(logpdf(D, x)) ≈ multivariate_normal_likelihood_0(obsmu, obsmu2, zeros(3), sds, n)
pmu, psd = randn(3), abs.(randn(3))
@assert sum(logpdf(MvNormal(pmu, psd), x)) ≈ multivariate_normal_likelihood_0(obsmu, obsmu2, pmu, psd, n)

# fit model
mod = model(obsmu, obsmu2, n)
spl = HMC(0.01, 10)
chn = sample(mod, spl, 10_000)

gen = generated_quantities(mod, chn)
σs = Matrix(undef, length(gen), k)
for i in eachindex(gen)
	σs[i, :] .= gen[i][1]
end
mean(σs, dims = 1) # should be equal to sds, but it's not
collect(mean(first(x)[j] for x in gen) for j in 1:k)


# @model function model2(x, ::Type{T} = Float64) where {T}

# 	k, n = size(x)
# 	μ = Vector{T}(undef, k)
# 	σ = Vector{T}(undef, k)
# 	for i in 1:k
# 		μ[i] ~ Normal(0, 1)
# 		σ[i] ~ InverseGamma(1, 1)
# 	end
# 	x ~ MvNormal(μ, σ)
# 	# for i in 1:n
# 	# 	x[:, i] ~ MvNormal(μ, σ)
# 	# end

# end

# mod2 = model2(x)
# spl2 = HMC(0.01, 10)
# chn2 = sample(mod2, spl2, 10_000)
