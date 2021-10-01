import Distributions: Normal, MvNormal, loglikelihood, logpdf, Wishart
import Statistics: mean, var
import LinearAlgebra: Diagonal, I, cholesky

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

@testset "Multivariate diagonal covariance matrix likelihood" begin

	n = 50
	for p in 5:10

		pop_mu = randn(p)
		pop_sds = abs.(randn(p))

		D = MvNormal(pop_mu, Diagonal(map(abs2, pop_sds)))
		# D = MvNormal(pop_mu, Diagonal(pop_sds))
		x = rand(D, n)

		obs_mean = mean(x, dims = 2)
		obs_var  = mean(x .^ 2, dims = 2)

		refvalue	= loglikelihood(D, x)
		testvalue	= EqualitySampler._multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n)
		@test refvalue ≈ testvalue
	end

end

@testset "Multivariate dense covariance matrix loglikelihood" begin

	n = 50
	for p in 5:10

		pop_mu = randn(p)
		pop_Σ  = rand(Wishart(p, Matrix{Float64}(I, p, p)))
		pop_Σ_chol = cholesky(pop_Σ).U

		D = MvNormal(pop_mu, pop_Σ)

		x = rand(D, n)
		refvalue = loglikelihood(D, x)

		obs_mean, obs_cov, n = get_normal_dense_suff_stats(x)
		D1 = MvNormalDenseSuffStat(obs_cov, n, pop_mu, pop_Σ)
		testvalue1 = logpdf(D1, obs_mean)
		@test refvalue ≈ testvalue1

		obs_mean, obs_cov_chol, n = get_normal_dense_chol_suff_stats(x)
		D2 = MvNormalCholDenseSuffStat(obs_cov_chol, n, pop_mu, pop_Σ_chol)
		testvalue2 = logpdf(D2, obs_mean)
		@test refvalue ≈ testvalue2

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