using Turing, Plots, StatsBase

@model function linear_regression_uniform(x, y)

	nfeatures = size(x, 2)
	indicator = tzeros(Int, nfeatures)
	for i in eachindex(indicator)
		indicator[i] ~ Bernoulli(0.5)
	end

	# Example code from website
	
	# Set the priors on our coefficients.
	coefficients ~ MvNormal(nfeatures, sqrt(10))
	
	# Calculate all the mu terms.
	mu = x * (coefficients .* indicator) # <- use indicator variable
	y ~ MvNormal(mu, 1.0)

end

n = 500
p = 3

x = randn(n, p)

coefficients = 3 .+ 0.5 .* randn(p)
indicator = rand(0:1, p)
residual = randn(n) # <- implies population σ₂ = 1
residual ./= sqrt(var(residual))

y = x * (coefficients .* indicator) .+ residual

mod = linear_regression_uniform(x, y)
spl = Gibbs(HMC(0.05, 10, :coefficients), PG(20, :indicator))
chain = sample(mod, spl, 3_000);

samples_coefficients = chain["coefficients[" .* string.(1:p) .* "]"].value.data
samples_indicator    = chain["indicator[" .* string.(1:p) .* "]"].value.data
mean(samples_indicator, dims = 1)
indicator

# looks fine
coefficients_posterior_means = vec(mean(samples_coefficients .* samples_indicator, dims = 1))
scatter(coefficients .* indicator, coefficients_posterior_means, legend=false)
Plots.abline!(1, 0)

no_samples = 10_000
prior_chain = sample(mod, Prior(), no_samples);
samples_indicator = reshape(prior_chain["indicator[" .* string.(1:p) .* "]"].value.data, no_samples, p)
mean(samples_indicator, dims = 1)
no_inclusions = vec(sum(samples_indicator, dims = 2))
bar(sort(countmap(no_inclusions)), legend = false)


@model function linear_regression_beta_binomial(x, y)

	nfeatures = size(x, 2) # number of predictors

	indicator = TArray(Int, nfeatures)
	# this biases the prior towards including no predictors
	indicator .= 0
	# this biases the prior towards including all predictors
	indicator .= 1
	# this somewhat mitigates the bias but is not correct
	indicator .= rand(0:1, nfeatures)

	# BetaBinomial prior on the number of included predictors
	for i in eachindex(indicator)
		no_incl = sum(indicator[1:p .!= i])
		# probability of model where indicator[i] = 0
		prob0 = pdf(BetaBinomial(p, 1, 1), no_incl)     / binomial(p, no_incl)
		# probability of model where indicator[i] = 1
		prob1 = pdf(BetaBinomial(p, 1, 1), no_incl + 1) / binomial(p, no_incl + 1)
		# normalize probabilities
		prob = prob1 / (prob0 + prob1)
		indicator[i] ~ Bernoulli(prob)
	end

	# Example code from website for linear regression

	# Set the priors on our coefficients.
	coefficients ~ MvNormal(nfeatures, sqrt(10))
	
	# Calculate all the mu terms.
	mu = x * (coefficients .* indicator) # <- use indicator variable
	y ~ MvNormal(mu, 1.0)

end

mod2 = linear_regression_beta_binomial(x, y)


prior_chain = sample(mod2, Prior(), no_samples);
samples_indicator = reshape(prior_chain["indicator[" .* string.(1:p) .* "]"].value.data, no_samples, p)
mean(samples_indicator, dims = 1)
no_inclusions = vec(sum(samples_indicator, dims = 2))
bar(sort(countmap(no_inclusions)), legend = false)

Turing.proposal
Turing.Inference.proposal()


vi = VarInfo()
mod2(Random.AbstractRNG, vi, SampleFromPrior())


sample, state = Turing.Inference.step(Random.AbstractRNG, mod2, Prior())


