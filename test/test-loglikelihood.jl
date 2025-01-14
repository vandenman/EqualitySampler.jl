import Distributions: Normal, MvNormal, loglikelihood, logpdf, Wishart, suffstats
import Statistics: mean, var
import LinearAlgebra: Diagonal, I, cholesky

@testset "Univariate normal loglikelihood" begin

	n = 100

	mus = randn(5)
	sigmas = abs.(randn(5))

	for (mu, sigma) in zip(mus, sigmas)

		d = Normal(mu, sigma)
		x = rand(d, n)

		ss = suffstats(Normal, x)

		refvalue	= loglikelihood(d, x)
		testvalue	= @inferred loglikelihood_suffstats(d, ss)

		@test refvalue ≈ testvalue

	end
end

@testset "Multivariate diagonal covariance matrix likelihood" begin

	n = 50
	for p in 5:10

		pop_mu = randn(p)
		pop_sds = abs.(randn(p))

		d = MvNormal(pop_mu, Diagonal(map(abs2, pop_sds)))
		x = rand(d, n)

		ss = suffstats(MvNormal, x)

		refvalue	= loglikelihood(d, x)
		testvalue	= @inferred loglikelihood_suffstats(d, ss)
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


		ss = suffstats(MvNormal, x)
		testvalue = @inferred loglikelihood_suffstats(D, ss)

		@test refvalue ≈ testvalue

	end
end
#= Benchmarks
using BenchmarkTools

# benchmark the multivariate approach:
n, p = 10_000, 30
pop_mu = randn(p)
pop_sds = abs.(randn(p))

d = MvNormal(pop_mu, pop_sds)
x = rand(d, n)

ss = suffstats(MvNormal, x)

refvalue	= loglikelihood(d, x)
testvalue	= loglikelihood_suffstats(d, ss)
b1 = @benchmark loglikelihood($d, $x)
b2 = @benchmark loglikelihood_suffstats($d, $ss)
m1 = median(b1)
m2 = median(b2)
judge(m2, m1)

# benchmark the univariate approach:
n = 1000
pop_mu = 1.5
pop_sds = 2.3

d = Normal(pop_mu, pop_sds)
x = rand(d, n)

ss = suffstats(Normal, x)

refvalue	= loglikelihood(d, x)
testvalue	= loglikelihood_suffstats(d, ss)
b1 = @benchmark loglikelihood($d, $x)
b2 = @benchmark loglikelihood_suffstats($d, $ss)
m1 = median(b1)
m2 = median(b2)
judge(m2, m1)


=#