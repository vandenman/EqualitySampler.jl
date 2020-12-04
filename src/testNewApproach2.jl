# how does an induced prior look like on the equalities when you have a uniform prior
# on the model space?
using Plots
include("src/newApproach2.jl")

#= 

	TODO:
		- clean up visualization function

=#

function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end

function count_equalities(sampled_models)

	k, n = size(sampled_models)
	result = Vector{Int}(undef, n)
	for i in axes(sampled_models, 2)
		result[i] = k - length(unique(view(sampled_models, :, i)))
	end
	return result
end

function plot_modelspace(D, probs_models_by_incl)

	k = length(D)
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0
	plt = bar(probs_models_by_incl, legend=false, yaxis=:log);
	for i in 1:k
		yval = y[cumulative_sum + incl_size[i]]
		ycoords = [yval, yval]
		if incl_size[i] == 1
			xcoords = [xstart + 0.5, xstart + 0.5]
			scatter!(plt, xcoords, ycoords, m = 4);
		else
			xcoords = [xstart, xstart + incl_size[i]]
			plot!(plt, xcoords, ycoords, lw = 4);
		end
		cumulative_sum += incl_size[i]
		xstart += incl_size[i]
	end
	return plt
end

function plot_inclusionprobabilities(D, probs_equalities)
	k = length(D.urns)
	plt = bar(sort(probs_equalities), legend=false, yaxis=:log);
	scatter!(plt, 0:k-1, expected_inclusion_probabilities(D), m = 4);
	return plt
end

function plot_expected_vs_empirical(D, probs_models_by_incl)
	x, y = expected_model_probabilities(D), collect(values(probs_models_by_incl))
	ablinecoords = [extrema([extrema(x)..., extrema(y)...])...]
	plt = plot(ablinecoords, ablinecoords, lw = 3, legend = false, yaxis=:log, xaxis=:log);
	plot!(plt, x, y, seriestype = :scatter);
end

updateDistribution(D::UniformConditionalUrnDistribution, urns, j) = UniformConditionalUrnDistribution(urns, j)
updateDistribution(D::BetaBinomialConditionalUrnDistribution, urns, j) = BetaBinomialConditionalUrnDistribution(urns, j, D.α, D.β)

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	# sampled_models_orig = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		for j in 1:k
			D = updateDistribution(D, urns, j)
			urns[j] = rand(D, 1)[1]
		end
		# sampled_models_orig[:, i] .= urns
		sampled_models[:, i] .= reduce_model(urns)
	
	end
	return sampled_models
end

function get_empirical_model_probabilities(sampled_models)
	count_models = countmap(vec(mapslices(x->join(Int.(x)), sampled_models, dims = 1)))
	probs_models = counts2probs(count_models)
	return sort(probs_models, by = x->count_equalities(x))
end

function get_empirical_inclusion_probabilities(sampled_models)
	no_equalities = count_equalities(sampled_models)
	counts_equalities = countmap(no_equalities)
	return counts2probs(counts_equalities)
end

# UniformConditionalUrnDistribution
nrand = 500_000
urns = collect(1:7)
D = UniformConditionalUrnDistribution(urns, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
			  size = (600, 1200))
png(pjoint, "modelspace uniform $k.png")

# BetaBinomialConditionalUrnDistribution
nrand = 500_000
urns = collect(1:7)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 5, 3)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
              size = (600, 1200))
png(pjoint, "modelspace betabinomial k=$(length(D)) alpha=$(D.α) beta=$(D.β) .png")
# savefig(pjoint, "modelspace betabinomial $k.pdf")



x = generate_distinct_models(3)
mapslices(parametrize_Gopalan_Berry, x, dims = 1)
collect(parametrize_Gopalan_Berry(x[:, i]) for i in axes(x, 2))
