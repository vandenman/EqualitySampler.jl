using EqualitySampler, Plots#, Turing
include("simulations/plotFunctions.jl")

updateDistribution(::UniformConditionalUrnDistribution, urns, j) = UniformConditionalUrnDistribution(urns, j)
updateDistribution(D::BetaBinomialConditionalUrnDistribution, urns, j) = BetaBinomialConditionalUrnDistribution(urns, j, D.α, D.β)

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		urns = ones(Int, k)
		for j in 1:k
			D = updateDistribution(D, urns, j)
			urns[j] = rand(D, 1)[1]
		end
		sampled_models[:, i] .= reduce_model(urns)
	end
	return sampled_models
end

nrand = 100_000
urns = collect(1:4)
D = UniformConditionalUrnDistribution(urns, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)
# png(pjoint, "modelspace uniform $k.png")

nrand = 100_000
urns = collect(1:5)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 4, 2)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)

# pretty theoretical Plots
function model_pmf(D, yaxis = :log)

	k = length(D)
	allmodels = generate_distinct_models(k)
	almodels_str = [join(col) for col in eachcol(allmodels)]
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0
	transparent = Colors.RGBA(0, 0, 0, 0)
	plt = plot(almodels_str, fill(maximum(y), length(y)), label = nothing, color = transparent, yaxis = yaxis)
	for i in 1:k
		yval = y[cumulative_sum + incl_size[i]]
		ycoords = [yval, yval]
		if isone(incl_size[i])
			xcoords = [xstart + 0.5, xstart + 0.5]
			scatter!(plt, xcoords, ycoords, m = 4, label = string(incl_size[i]), yaxis = yaxis);
		else
			xcoords = [xstart, xstart + incl_size[i]]
			plot!(plt, xcoords, ycoords, lw = 4, label = string(incl_size[i]), yaxis = yaxis);
		end
		cumulative_sum += incl_size[i]
		xstart += incl_size[i]
	end
	return plt
end

k = 4
Duniform = UniformConditionalUrnDistribution(ones(Int, k))
DBetaBinom = BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, 1, 1)
model_pmf(Duniform)
model_pmf(DBetaBinom)
model_pmf(BetaBinomialConditionalUrnDistribution(ones(Int, 6), 1, 3, 1))
model_pmf(BetaBinomialConditionalUrnDistribution(ones(Int, 12), 1, 1, 1))

model_probs = expected_model_probabilities(Duniform)
incl_probs  = expected_inclusion_probabilities(Duniform)

model_size = [count_equalities(col) for col in eachcol(allmodels)]
p = plot(almodels_str, model_probs, legend = false)
fo
p = scatter(almodels_str[1], model_probs[1])


foo(Duniform, almodels_str)
