using LinearAlgebra
using QuadGK, HCubature
N = 10
p = 4

y = randn(N)
X = randn(N, p)

θ = randn(p)
μ = randn()
σ = abs(randn())
g = abs(randn())

P0 = 1/ N * ones(N) * ones(N)'

tmp0 = (y - ones(N)*μ - X*θ)
integrand0 = tmp0'tmp0

integrandRouder = (y - ones(N)*μ)' * (I-P0) * (y - ones(N)*μ) + N * (μ - ones(N)' * (y - X*θ) / N)^2 #- ones(N)' * (y - X*θ) / N

integrand1 = μ^2 * N - μ*2*ones(N)' * (y - X*θ) - 2*y'*X*θ + (X*θ)' * X*θ + y'y


integrand1_fn = μ-> exp(-1 / 2σ^2 * (μ^2 * N - μ*2*ones(N)' * (y - X*θ) - 2*y'*X*θ + (X*θ)' * X*θ + y'y))
quadgk(integrand0_fn, -Inf, Inf)

a = N / 2σ^2
b = (ones(N)' * (y - X*θ)) / σ^2
# c = -(- 2*y'*X*θ + (X*θ)' * X*θ + y'y) * 1 / 2σ^2
c = (2*y'*X*θ - (X*θ)' * X*θ - y'y) * 1 / 2σ^2

sqrt(pi/a) * exp(b^2 / 4a + c)

integrand1_manual = sqrt(pi*2σ^2 / N) * exp(1 / 2σ^2 * ((ones(N)' * (y - X*θ))^2 / N + (2*y'*X*θ - (X*θ)' * X*θ - y'y)))

(ones(N)' * (y - X*θ))^2
ones(N)' * (y - X*θ) * (y - X*θ)' * ones(N)
ones(N)' * y * y' * ones(N) - ones(N)' * 2*y*(X*θ)' * ones(N) + ones(N)' * X*θ * (X*θ)' * ones(N)




-((ones(N)' * (y - X*θ))^2 / N + (2*y'*X*θ - (X*θ)' * X*θ - y'y))
ỹ = (I-P0) * y
X̃ = (I-P0) * X
(ỹ - X̃*θ)' * (ỹ - X̃*θ)

integrand_dosRouder = exp(-(ỹ - X̃*θ)' * (ỹ - X̃*θ) / 2σ^2)
integrand_dos       = exp(((ones(N)' * (y - X*θ))^2 / N + (2*y'*X*θ - (X*θ)' * X*θ - y'y))/ 2σ^2)

Vg = X̃'X̃
Q2 = ỹ'ỹ + (θ - Vg \ X̃'*ỹ)' * Vg * (θ - Vg \ X̃'*ỹ) - ỹ' * X̃ / Vg * X̃'*ỹ


integrand_dos_fn2(qq) = begin
	aa = exp(-(ỹ - X̃*qq)' * (ỹ - X̃*qq) / 2σ^2)
	# if isnan(aa)
	# elseif iszero(aa)
	# 	@show θ
	# end
	@show θ, aa
	aa
end
hcubature(integrand_dos_fn2, fill(-1e3, p), fill(1e3, p))
integrand_dos_fn2(fill(0.0,p))

Cubature.hcubature(integrand_dos_fn2, fill(-1e3, p), fill(1e3, p))

import SpecialFunctions, LinearAlgebra
function integrated_log_lik(y, X, g)
	N, p = size(X)


	P0 = 1/ N * ones(N) * ones(N)'

	ỹ = (I-P0) * y
	X̃ = (I-P0) * X

	G = LinearAlgebra.Diagonal(fill(g, p))
	Vg = X̃'X̃ + G


	a = (N - 1) / 2
	b = ỹ'ỹ - ỹ' * X̃ / Vg * X̃'*ỹ

	#=a * log(2) +=#
	SpecialFunctions.logabsgamma(a) - (
		a * log(2*pi) + (log(N) + LinearAlgebra.logabsdet(G)[1] + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
	)

end

using Turing, EqualitySampler, Plots

function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

	G = LinearAlgebra.Diagonal(fill(g, length(ỹTX̃)))
	Vg = X̃TX̃ + inv(G)

	a = (N - 1) / 2
	b = ỹTỹ - ỹTX̃ / Vg * ỹTX̃'

	return gamma_a - (
		a * log(2*pi) + (log(N) + LinearAlgebra.logabsdet(G)[1] + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
	)

end

# function integrated_log_lik(ỹTỹ, ỹ, X, gamma_a, N, g, Q, partition)

# 	Ρ = EqualitySampler.Simulations.get_equalizer_matrix_from_partition(partition)
# 	X̃ = X * Ρ * Q
# 	ỹTX̃ = ỹ'X̃
# 	X̃TX̃ = X̃'X̃

# 	return integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

# end


function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)

	Ρ = EqualitySampler.Simulations.get_equalizer_matrix_from_partition(partition)
	B = Q'Ρ*Q
	ỹTX̃ = ỹTX̃ * B
	X̃TX̃ = B * X̃TX̃ * B

	return integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)
end

function precompute_integrated_log_lik(dat)

	y = dat.y
	N, P = length(y), length(dat.g)
	X0 = zeros(N, P)
	for (i, idx) in enumerate(dat.g)
		X0[idx, i] .= 1
	end

	P0 = 1 / N * ones(N) * ones(N)'

	Q = EqualitySampler.Simulations.getQ_Rouder(P)
	X = X0 * Q

	ỹ = (LinearAlgebra.I-P0) * y
	X̃ = (LinearAlgebra.I-P0) * X

	ỹTỹ = ỹ'ỹ
	ỹTX̃ = ỹ'X̃
	X̃TX̃ = X̃'X̃
	gamma_a = SpecialFunctions.logabsgamma((N-1)/2)[1]

	P0X = (LinearAlgebra.I-P0) * X0
	return (; ỹ, P0X, ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
end

data = EqualitySampler.Simulations.simulate_data_one_way_anova(3, 50)
dat = data.data
ỹ, X, ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = precompute_integrated_log_lik(dat)
Q = EqualitySampler.Simulations.getQ_Rouder(length(dat.g))
integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, n, 1.0)
integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, n, 1.0, Q, collect(1:length(dat.g)))
# integrated_log_lik2(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, n, 1.0, Q, collect(1:length(dat.g)))

# b0 = @benchmark integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃,  gamma_a, n, 1.0)
# b1 = @benchmark integrated_log_lik(ỹTỹ, ỹ,   X,    gamma_a, n, 1.0, Q, collect(1:length(dat.g)))
# b2 = @benchmark integrated_log_lik2(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, n, 1.0, Q, collect(1:length(dat.g)))
# judge(median(b2), median(b1))
# judge(median(b2), median(b0))


@model function integrated_full_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)

	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

end

@model function integrated_partition_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, Q, partition_prior)

	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	partition ~ partition_prior
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)

end

no_iter = Int(1e5)
mcmc_settings = EqualitySampler.Simulations.MCMCSettings(;iterations = no_iter, burnin=1, chains=1, thinning=1, parallel= Turing.MCMCSerial())
fit_all = anova_test(dat, nothing; mcmc_settings = mcmc_settings)

mod_integrated = integrated_full_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
fit_integrated = sample(mod_integrated, NUTS(), no_iter)

samples_g_all = vec(fit_all[:g].data)
samples_g_int = vec(fit_integrated[:g].data)

h = 0.01
probs = h:h:1-h
q_g_all = quantile(samples_g_all, probs)
q_g_int = quantile(samples_g_int, probs)
q_g_all .- q_g_int
plot(probs, [q_g_all q_g_int])

partition_prior = UniformMvUrnDistribution(length(dat.g))
fit_eq_all = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = 0.01)

mod_eq_integrated = integrated_partition_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, Q, partition_prior)
spl_eq_integrated = EqualitySampler.Simulations.get_sampler(mod_eq_integrated, :custom, 0.01)
fit_eq_integrated = sample(mod_eq_integrated, spl_eq_integrated, no_iter)

Turing.MCMCChains.wall_duration(fit_eq_all)
Turing.MCMCChains.wall_duration(fit_eq_integrated)

samples_g_eq_all = vec(fit_eq_all[Symbol("one_way_anova_mv_ss_submodel.g")].data)
samples_g_eq_int = vec(fit_eq_integrated[:g].data)

h = 0.01
probs = h:h:1-h
q_g_eq_all = quantile(samples_g_eq_all, probs)
q_g_eq_int = quantile(samples_g_eq_int, probs)
q_g_eq_all .- q_g_eq_int
plot(probs, [q_g_eq_all q_g_eq_int])

partition_samples_all = MCMCChains.group(fit_eq_all,        :partition).value.data
partition_samples_int = MCMCChains.group(fit_eq_integrated, :partition).value.data
post_probs_eq_all = EqualitySampler.Simulations.compute_post_prob_eq(partition_samples_all)
post_probs_eq_int = EqualitySampler.Simulations.compute_post_prob_eq(partition_samples_int)