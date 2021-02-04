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

# Theoretical Plots
function model_pmf(D; yaxis = :log, legend = true, xrotation = 45)

	k = length(D)
	allmodels = generate_distinct_models(k)
	almodels_str = [join(col) for col in eachcol(allmodels)]
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0
	transparent = Colors.RGBA(0, 0, 0, 0)
	plt = plot(almodels_str, fill(maximum(y), length(y)), label = nothing, color = transparent,
				yaxis = yaxis, legend = legend, xrotation = xrotation)
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

function incl_pmf(D::AbstractConditionalUrnDistribution; yaxis = :log)
	k = length(D)
	plt = scatter(0:k-1, expected_inclusion_probabilities(D), m = 4, yaxis = yaxis, legend = false);
	return plt
end

k = 4
Duniform = UniformConditionalUrnDistribution(ones(Int, k))
DBetaBinom = BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, 1, 1)

yaxis_scale = :none
p1 = model_pmf(Duniform, legend = false, yaxis = yaxis_scale);
p2 = model_pmf(DBetaBinom, legend = false, yaxis = yaxis_scale);
plot!(p1, title = "Uniform", ylab = "Probabilty", xlab = "Model");
plot!(p2, title = "Beta-binomial (α = $(DBetaBinom.α), β = $(DBetaBinom.β))", xlab = "Model");

p3 = incl_pmf(Duniform, yaxis = yaxis_scale);
p4 = incl_pmf(DBetaBinom, yaxis = yaxis_scale);
plot!(p3, xlab = "No. inequalities", ylab = "Probabilty")
plot!(p4, xlab = "No. inequalities")

w = 360
joint = plot(p1, p2, p3, p4, layout = (2, 2), size = (2w, 2w))
savefig(joint, joinpath("figures", "prior.pdf"))