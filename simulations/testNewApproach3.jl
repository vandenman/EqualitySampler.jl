using Plots
include("src/newApproach3.jl")
include("src/helperfunctions.jl")

updateDistribution(::UniformConditionalUrnDistribution, urns, index) = UniformConditionalUrnDistribution(urns, index)
updateDistribution(::BetaBinomialConditionalUrnDistribution, urns, index) = BetaBinomialConditionalUrnDistribution(urns, index, D.α, D.β)

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	# sampled_models_orig = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		fill!(urns, 1)
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
	return sort(counts2probs(counts_equalities))
end

# UniformConditionalUrnDistribution
nrand = 40_000
urns = collect(1:6)
D = UniformConditionalUrnDistribution(urns)

sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
			  size = (600, 1200))
# png(pjoint, "modelspace uniform $k.png")

# BetaBinomialConditionalUrnDistribution
nrand = 50_000
urns = collect(1:3)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 1, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
              size = (600, 1200))
# png(pjoint, "modelspace betabinomial k=$(length(D)) alpha=$(D.α) beta=$(D.β) .png")
# savefig(pjoint, "modelspace betabinomial $k.pdf")



x = generate_distinct_models(3)
mapslices(parametrize_Gopalan_Berry, x, dims = 1)
collect(parametrize_Gopalan_Berry(x[:, i]) for i in axes(x, 2))

# multivariate distributions
D = UniformMvUrnDistribution(3)
sampled_models = mapslices(reduce_model, rand(D, 100_000), dims=1);

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
			  size = (600, 1200))


D = BetaBinomialMvUrnDistribution(3)
sampled_models = mapslices(reduce_model, rand(D, 100_000), dims=1);

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
			  size = (600, 1200))


# DirichletMultinomial
D = Distributions.DirichletMultinomial(3, [1, 1, 1])
function marginalPMF(D::Distributions.DirichletMultinomial, i1::Int, i2::Int)
	k = length(D)
	probequal = 0.0
	probdiff  = 0.0
	for i in 1:k
		iterators = fill(0:k,k)
		loopIdx = Iterators.product(iterators...)
		for it in loopIdx
			# @show it
			add = pdf(D, collect(it))
			if it[i1] == it[i2]
				probequal += add
			else
				probdiff += add
			end
		end
	end
	return probequal / (probequal + probdiff)
end

function expected_inclusion_probabilities(D::Distributions.DirichletMultinomial)
	# TODO: this is very inefficient
	k = length(D)
	allmodels = Iterators.product(fill(0:k-1, k)...)
	probs = zeros(Float64, k)
	for it in allmodels
		mod = collect(it)
		no_inclusions = count_equalities(mod) + 1
		probs[no_inclusions] += pdf(D, mod)
	end
	return probs
end

function expected_model_probabilities(D::Distributions.DirichletMultinomial)
	k = length(D)
	models = generate_distinct_models(k) .- 1
	probs = vec(mapslices(x-> pdf(D, x), models, dims = 1))
	return probs ./ sum(probs)
end
sampled_models = mapslices(reduce_model, rand(D, 100_000) .+ 1, dims=1);

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)
p1 = plot_modelspace(D, empirical_model_probs);
p2 = plot_inclusionprobabilities(D, empirical_inclusion_probs);
p3 = plot_expected_vs_empirical(D, empirical_model_probs);
pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
			  size = (600, 1200))


