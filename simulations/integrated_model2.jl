using Turing, EqualitySampler, EqualitySampler.Simulations, Plots, BenchmarkTools
import LinearAlgebra, SpecialFunctions, Printf, FillArrays, Random

function plot_quantiles(x, y)
	h = 0.01
	probs = h:h:1-h
	q_x = quantile(x, probs)
	q_y = quantile(y, probs)

	p1 = plot(probs, [q_x q_y], title = Printf.@sprintf("mean difference = %.3f", mean(q_x .- q_y)), legend = :topleft, labels = ["q(x)" "q(y)"])
	p2 = plot(q_x, q_y, legend=:topleft, label = "q(x) vs q(y)")
	Plots.abline!(p2, 1, 0, label = "abline y=x")
	plot(p1, p2, layout = (1, 2))
end


no_groups = 5
true_partition = fill(1, no_groups)
true_θ = Simulations.normalize_θ(.2, true_partition)
data = EqualitySampler.Simulations.simulate_data_one_way_anova(no_groups, 100, true_θ)
dat = data.data

ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = Simulations.precompute_integrated_log_lik(dat)
Q = EqualitySampler.Simulations.getQ_Rouder(length(dat.g))

@assert Simulations.integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, 1.0) ≈ Simulations.integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, 1.0, Q, collect(1:length(dat.g)))

G = LinearAlgebra.Diagonal(fill(g, length(ỹTX̃)))
X̃TX̃inv = inv(X̃TX̃)
B = LinearAlgebra.I + X̃TX̃inv * G

target = inv(X̃TX̃ + invG)


mcmc_settings = EqualitySampler.Simulations.MCMCSettings(;iterations = Int(1e3), burnin=1, chains=4, thinning=1, parallel = Turing.MCMCThreads())
fit_all = anova_test(dat, nothing; mcmc_settings = mcmc_settings)
fit_int = anova_test(dat, nothing; mcmc_settings = mcmc_settings, modeltype = :reduced)

(Turing.MCMCChains.wall_duration(fit_all), Turing.MCMCChains.wall_duration(fit_int))

samples_g_all = vec(fit_all[:g].data)
samples_g_int = vec(fit_int[:g].data)
plot_quantiles(samples_g_all, samples_g_int)


partition_prior = UniformMvUrnDistribution(length(dat.g))
fit_eq_all  = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = 0.01)
fit_eq_all2 = anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, spl = 0.05, modeltype = :reduced)

(Turing.MCMCChains.wall_duration(fit_eq_all), Turing.MCMCChains.wall_duration(fit_eq_all2))

samples_g_eq_all  = vec(fit_eq_all[Symbol("one_way_anova_mv_ss_submodel.g")].data)
samples_g_eq_all2 = vec(fit_eq_all2[:g].data)
plot_quantiles(samples_g_eq_all, samples_g_eq_all2)

partition_samples_all  = MCMCChains.group(fit_eq_all,        :partition).value.data
partition_samples_all2 = MCMCChains.group(fit_eq_all2,       :partition).value.data
post_probs_eq_all  = EqualitySampler.Simulations.compute_post_prob_eq(partition_samples_all)
post_probs_eq_all2 = EqualitySampler.Simulations.compute_post_prob_eq(partition_samples_all2)

no_groups = 9
true_partition = fill(1, no_groups)
true_θ = Simulations.normalize_θ(.2, true_partition)
data = EqualitySampler.Simulations.simulate_data_one_way_anova(no_groups, 100, true_θ)
dat = data.data

ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = Simulations.precompute_integrated_log_lik(dat)
Q = EqualitySampler.Simulations.getQ_Rouder(length(dat.g))

y = dat.y
N, P = length(y), length(dat.g)
X = zeros(N, P)
for (i, idx) in enumerate(dat.g)
	X[idx, i] .= 1.0
end

# P0 = 1 / N * ones(N) * ones(N)'
P0 = FillArrays.Fill(1 / N, N, N)

Q = EqualitySampler.Simulations.getQ_Rouder(P)
X = X * Q

# ỹ = (LinearAlgebra.I-P0) * y
# X̃ = (LinearAlgebra.I-P0) * X
# avoids forming LinearAlgebra.I-P0
ỹ = y - P0 * y
X̃ = X - P0 * X
Vg = copy(X̃'X̃)# + invG
for i in axes(Vg, 1)
	Vg[i, i] += (1 / g)
end
ỹTỹ - ỹTX̃ / Vg * ỹTX̃'

G = LinearAlgebra.Diagonal(fill(g, size(X̃, 2)))

ỹ' / (LinearAlgebra.I + X̃*G*X̃') * ỹ

function integrated_log_lik2(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

	# G = LinearAlgebra.Diagonal(fill(g, length(ỹTX̃)))
	# Vg = X̃TX̃ + inv(G)
	# invG = LinearAlgebra.Diagonal(fill(1 / g, length(ỹTX̃)))
	Vg = copy(X̃TX̃)# + invG
	@inbounds for i in axes(X̃TX̃, 1)
		Vg[i, i] += 1 / g
	end

	a = (N - 1) / 2
	b = ỹTỹ - ỹTX̃ / Vg * ỹTX̃'

	logabsdet_g = g^length(ỹTX̃)

	return @inbounds gamma_a - (
		a * log(2*pi) + (log(N) + logabsdet_g + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
	)

end
g = 1.0
b0 = @benchmark Simulations.integrated_log_lik($ỹTỹ, $ỹTX̃, $X̃TX̃, $gamma_a, $N, $g)
b1 = @benchmark            integrated_log_lik2($ỹTỹ, $ỹTX̃, $X̃TX̃, $gamma_a, $N, $g)
judge(median(b1), median(b0))


no_groups = 15
true_partition = fill(1, no_groups)
true_θ = Simulations.normalize_θ(.2, true_partition)
data = EqualitySampler.Simulations.simulate_data_one_way_anova(no_groups, 100, true_θ)
dat = data.data

ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = Simulations.precompute_integrated_log_lik(dat)
Q = EqualitySampler.Simulations.getQ_Rouder(length(dat.g))

using Profile
@profview (for i in 1:200 Simulations.integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, 1.0, Q, true_partition) end)

using LinearAlgebra
logabsdet(factorize(X̃TX̃).F)


#=

function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

	# G = LinearAlgebra.Diagonal(fill(g, length(ỹTX̃)))
	invG = LinearAlgebra.Diagonal(fill(1 / g, length(ỹTX̃)))
	Vg = X̃TX̃ + invG

	a = (N - 1) / 2
	b = ỹTỹ - ỹTX̃ / Vg * ỹTX̃'

	return @inbounds gamma_a - (
		a * log(2*pi) + (log(N) - LinearAlgebra.logabsdet(invG)[1] + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
		# a * log(2*pi) + (log(N) + LinearAlgebra.logabsdet(G)[1] + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
	)

end

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

	# P0 = 1 / N * ones(N) * ones(N)'
	P0 = FillArrays.Fill(1 / N, N, N)

	Q = EqualitySampler.Simulations.getQ_Rouder(P)
	X = X0 * Q

	# ỹ = (LinearAlgebra.I-P0) * y
	# X̃ = (LinearAlgebra.I-P0) * X
	# avoids forming LinearAlgebra.I-P0
	ỹ = y - P0 * y
	X̃ = X - P0 * X

	ỹTỹ = ỹ'ỹ
	ỹTX̃ = ỹ'X̃
	X̃TX̃ = X̃'X̃
	gamma_a = SpecialFunctions.logabsgamma((N-1)/2)[1]

	return (; ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
end

@model function integrated_full_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)

	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

end

@model function integrated_partition_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, Q, partition_prior)

	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	partition ~ partition_prior
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)

end


# partition_prior = UniformMvUrnDistribution(length(dat.g))
# df = dat
# obs_mean, obs_var, obs_n, Q = Simulations.prep_model_arguments(df)
# model = Simulations.one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior)
# starting_values = Simulations.get_starting_values(df)
# init_params = Simulations.get_init_params(starting_values...)
# spl = 0.01
# mcmc_sampler = Simulations.get_mcmc_sampler_anova(spl, model, init_params)
# chain = Simulations.sample_model(model, mcmc_sampler, mcmc_settings, Random.GLOBAL_RNG; init_params = init_params)

# constrained_samples = Simulations.get_generated_quantities(model, chain)
# parameter_name = "θ_cs"

# newdims = (size(chain, 1), size(constrained_samples, 2), size(chain, 3))
# temp =
# reshape(temp, )
# constrained_chain = MCMCChains.setrange(
# 	MCMCChains.Chains(reshape(constrained_samples, newdims), collect(Symbol(parameter_name, "["* string(i) * "]") for i in axes(constrained_samples, 2))),
# 	range(chain)
# )
# combined_chain = hcat(chain, constrained_chain)

=#

