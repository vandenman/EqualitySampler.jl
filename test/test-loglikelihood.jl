using Test
import Distributions: MvNormal, loglikelihood
import Statistics: mean
println(pwd())
if endswith(pwd(), "bfvartest")
	cd("julia")
elseif endswith(pwd(), "test")
	cd("../")
end
println(pwd())
include(joinpath(pwd(), "loglikelihood.jl"))


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
		testvalue	= _multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n)
		@test refvalue ≈ testvalue
	end

end

# benchmark the two approaches:
# n, p = 10_000, 30
# pop_mu = randn(p)
# pop_sds = abs.(randn(p))

# D = MvNormal(pop_mu, pop_sds)
# x = rand(D, n)

# obs_mean = mean(x, dims = 2)
# obs_var  = mean(x .^ 2, dims = 2)

# refvalue	= loglikelihood(D, x)
# testvalue	= multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n)
# m1 = median(@benchmark loglikelihood(D, x))
# m2 = median(@benchmark multivariate_normal_likelihood(obs_mean, obs_var, pop_mu, pop_sds, n))
# judge(m2, m1)