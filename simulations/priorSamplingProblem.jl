using Turing, Plots, StatsBase

# Turing model
@model function example_model(k, indicatorState, α = 1.0, β = 1.0, initialize_zero = true)

	indicator = TArray(Int, k)
	indicator .= indicatorState
	# println("indicator is $indicator")

	# if initialize_zero
	# 	# this biases the prior towards including no predictors
	# 	indicator .= 0
	# else
	# 	# this biases the prior towards including all predictors
	# 	indicator .= 1
	# end
	# this somewhat mitigates the bias but is not correct
	# indicator .= rand(0:1, k)

	# BetaBinomial prior on the number of included predictors
	D = BetaBinomial(k, α, β)
	for i in eachindex(indicator)
		no_incl = sum(indicator[1:k .!= i])
		# probability of model where indicator[i] = 0
		prob0 = pdf(D, no_incl)     / binomial(k, no_incl)
		# probability of model where indicator[i] = 1
		prob1 = pdf(D, no_incl + 1) / binomial(k, no_incl + 1)
		# normalize probabilities
		prob = prob1 / (prob0 + prob1)
		indicator[i] ~ Bernoulli(prob)
	end

	indicatorState .= indicator
	# println("indicatorState is $indicatorState")
end

# manual version
function sample_manual(k, no_samples, α = 1.0, β = 1.0)
	indicator_samples = Matrix{Int}(undef, no_samples, k)
	indicator_samples[1, :] .= 1
	for l in 2:no_samples
		indicator = view(indicator_samples, l, :)
		indicator .= indicator_samples[l - 1, :]
		D = BetaBinomial(k, α, β)
		# same as Turing model
		for i in eachindex(indicator)
			no_incl = sum(indicator[1:k .!= i])
			prob0 = pdf(D, no_incl)     / binomial(k, no_incl)
			prob1 = pdf(D, no_incl + 1) / binomial(k, no_incl + 1)
			prob = prob1 / (prob0 + prob1)
			indicator[i] = rand(Bernoulli(prob), 1)[1]
		end
	end
	return indicator_samples
end

function compute_incl_probs(samples)
	no_inclusions = vec(sum(Int, samples, dims = 2))
	raw_counts = countmap(no_inclusions)
	total_counts = sum(values(raw_counts))
	probs = Dict{Int, Float64}()
	for (key, value) in sort(raw_counts)
		probs[key] = value / total_counts
	end
	return probs
end

k = 5               # no. predictors
α = 3.0             # hyperparameter of BetaBinomial
β = 9.0             # hyperparameter of BetaBinomial
no_samples = 40_000      # no. samples
D = BetaBinomial(k, α, β)

indicator_samples_manual = sample_manual(k, no_samples, α, β)
no_inclusions_manual = compute_incl_probs(indicator_samples_manual)
plot_manual = bar(no_inclusions_manual, legend = false);
scatter!(plot_manual, 0:k, pdf(D, 0:k));

indicatorState = ones(Int, k)
modd = example_model(k, indicatorState, α, β)
spl = Prior()
prior_chain = sample(modd, spl, no_samples);

indicator_samples_Turing = reshape(prior_chain["indicator[" .* string.(1:k) .* "]"].value.data, no_samples, k)
no_inclusions_Turing = compute_incl_probs(indicator_samples_Turing)
plot_Turing = bar(no_inclusions_Turing, legend = false);
scatter!(plot_Turing, 0:k, pdf(D, 0:k));

prior_chain = sample(example_model(k, α, β, false), Prior(), no_samples);
indicator_samples_Turing = reshape(prior_chain["indicator[" .* string.(1:k) .* "]"].value.data, no_samples, k)
no_inclusions_Turing = compute_incl_probs(indicator_samples_Turing)
plot_Turing2 = bar(no_inclusions_Turing, legend = false);
scatter!(plot_Turing2, 0:k, pdf(D, 0:k));

plot_joint = plot(plot_manual, plot_Turing, plot_Turing2, layout = grid(3, 1), title = ["manual sampling" "Turing initialized to zero" "Turing initialized to one"],
		size = (600, 1200));
# png(plot_joint, "prior.png")


# Turing model
# @macroexpand @model function example_model(k, α = 1.0, β = 1.0, initialize_zero = true)

# 	indicator = tzeros(Int, k)

# 	# BetaBinomial prior on the number of included predictors
# 	D = BetaBinomial(k, α, β)
# 	for i in eachindex(indicator)
# 		no_incl = sum(indicator[1:k .!= i])
# 		# probability of model where indicator[i] = 0
# 		prob0 = pdf(D, no_incl)     / binomial(k, no_incl)
# 		# probability of model where indicator[i] = 1
# 		prob1 = pdf(D, no_incl + 1) / binomial(k, no_incl + 1)
# 		# normalize probabilities
# 		prob = prob1 / (prob0 + prob1)
# 		indicator[i] ~ Bernoulli(prob)
# 	end
# end

Turing.VarInfo
Turing.vi(example_model(k, α, β))

ee = example_model(k, α, β)

ee
vi = Turing.VarInfo()
ee(Random.AbstractRNG, vi, Prior())


