using Turing, FillArrays # Turing v0.14.10

function multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
	# unlike the commented version above, this one doesn't depend on LinearAlgebra
	result = length(obs_mean) * log(2 * float(pi))
	for i in eachindex(obs_mean)
		pop_prec_i = 1.0 / pop_sds[i] / pop_sds[i]
		result +=
			2 * log(pop_sds[i]) +
			obs_var[i] * pop_prec_i +
			(pop_mean[i] - 2 * obs_mean[i]) * pop_mean[i] * pop_prec_i

	end
	return - n / 2 * result
end


@model function normal_mean_variance(x)
	k = size(x)[1]
	s ~ filldist(Gamma(1, 1), k)
	m ~ filldist(Normal(0, 5), k)
	for i in axes(x, 2)
		x[:, i] ~ MvNormal(m, s)
	end
	return m, s
end

@model function normal_mean_variance_ss(obs_mean, obs_m2, n)
	k = length(obs_mean)
	s ~ filldist(Gamma(1, 1), k)
	m ~ filldist(Normal(0, 5), k)
	Turing.@addlogprob! multivariate_normal_likelihood(obs_mean, obs_m2, m, s, n)
	return m, s
end

D = MvNormal(collect(1.0:2:5), collect(1.0:2:5))
n = 100
x = rand(D, n);
obs_mean = mean(x, dims = 2)
obs_m2 = mean(x->x^2, x, dims = 2)

# the two are equal
sum(logpdf(D, x))
multivariate_normal_likelihood(obs_mean, obs_m2, mean(D), sqrt.(var(D)), n)

model = normal_mean_variance(x)
model_ss = normal_mean_variance_ss(obs_mean, obs_m2, n)

chn    = sample(model,    HMC(0.01, 10), 8_000, discard_initial = 2_000, n_adapts = 2_000, drop_warmup = true);
chn_ss = sample(model_ss, HMC(0.01, 10), 8_000, discard_initial = 2_000, n_adapts = 2_000, drop_warmup = true);

s    = summarystats(chn)
s_ss = summarystats(chn_ss)
