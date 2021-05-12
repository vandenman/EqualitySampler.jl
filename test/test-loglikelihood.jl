import Distributions: Normal, MvNormal, loglikelihood, logpdf
import Statistics: mean, var


@testset "Multivariate normal likelihood" begin

	n = 50
	for p in 5:10

		pop_mu = randn(p)
		pop_sds = abs.(randn(p))

		D = MvNormal(pop_mu, pop_sds)
		x = rand(D, n)

		obs_mean = mean(x, dims = 2)
		obs_var  = mean(x .^ 2, dims = 2)

		refvalue	= loglikelihood(D, x)
		testvalue	= EqualitySampler._multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n)
		@test refvalue ≈ testvalue
	end

end

@testset "Univariate normal loglikelihood" begin

	n = 100

	mus = randn(5)
	sigmas = abs.(randn(5))

	for (mu, sigma) in zip(mus, sigmas)

		D = Normal(mu, sigma)
		x = rand(D, n)

		obs_mean = mean(x)
		obs_var  = var(x)

		D2 = NormalSuffStat(obs_var, mu, sigma^2, n)

		refvalue	= loglikelihood(D, x)
		testvalue	= @inferred logpdf(D2, obs_mean)

		@test refvalue ≈ testvalue

	end
end


#= Benchmarks

# benchmark the multivariate approach:
n, p = 10_000, 30
pop_mu = randn(p)
pop_sds = abs.(randn(p))

D = MvNormal(pop_mu, pop_sds)
x = rand(D, n)

obs_mean = mean(x, dims = 2)
obs_var  = mean(x .^ 2, dims = 2)

refvalue	= loglikelihood(D, x)
testvalue	= EqualitySampler._multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n)
m1 = median(@benchmark loglikelihood(D, x))
m2 = median(@benchmark EqualitySampler._multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n))
judge(m2, m1)

# benchmark the univariate approach:
n = 1000
pop_mu = 1.5
pop_sds = 2.3

D = Normal(pop_mu, pop_sds)
x = rand(D, n)

obs_mean = mean(x)
obs_var  = var(x)
pop_var  = pop_sds^2

refvalue	= loglikelihood(D, x)
testvalue	= EqualitySampler._univariate_normal_likelihood(obs_mean, obs_var, n, pop_mu, pop_sds^2)
m1 = median(@benchmark loglikelihood(D, x))
m2 = median(@benchmark EqualitySampler._univariate_normal_likelihood(obs_mean, obs_var, n, pop_mu, pop_var))
judge(m2, m1)

=#