using EqualitySampler
import StatsBase: countmap
import Distributions: pdf
k = 5
d_betabinomial_process = BetaBinomialProcessMvUrnDistribution(k)
obs = rand(d, 100_000)
obs_u = mapslices(reduce_model, obs; dims = 1)

function structure_type(x)

	no_eqs = count_equalities(x)

	if no_eqs == 0
		return 1
	elseif no_eqs == 1
		return 2
	elseif no_eqs == 2
		if maximum(values(countmap(x))) == 2
			return 3
		else
			return 4
		end
	elseif no_eqs == 3
		if maximum(values(countmap(x))) == 3
			return 5
		else
			return 6
		end
	elseif no_eqs == 4
		return 7
	end
	return -1
end

unique_structures = reduce_model.([
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 4],
	[1, 2, 2, 3, 3],
	[1, 1, 1, 2, 3],
	[1, 1, 1, 2, 2],
	[1, 1, 1, 1, 2],
	[1, 1, 1, 1, 1]
])
structure_type.(unique_structures)

count(==([1, 1, 1, 1, 1]), eachcol(obs_u))

obs_structures = sort(countmap(structure_type.(eachcol(obs_u))))
hcat(1:7, values(obs_structures) ./ sum(values(obs_structures)), count_equalities.(unique_structures))

probs0 = Vector{Float64}(undef, k)
probs = Matrix{Float64}(undef, k, 7)
for j in 1:7
	for i in 1:k
		EqualitySampler._pdf_helper!(probs0, d_betabinomial_process, i, unique_structures[j])
		known_values = view(unique_structures[j], 1:i-1)
		idx = unique_structures[j][i]
		# if unique_structures[j][i] ∈ known_values
		# 	idx = unique_structures[j][i]
		# 	# idx = findall(==(unique_structures[j][i]), known_values)
		# else
		# 	idx = setdiff(1:k, known_values)
		# end
		probs[i, j] = sum(view(probs0, idx))
	end
end
combs = count_combinations.(unique_structures)# / sum(count_combinations.(unique_structures))
ccc, sizes = count_set_partitions_given_partition_size(k)

probs2 = prod.(eachcol(probs))
probs2 .* combs .* reverse(ccc)
probs2 / sum(probs2)


probs = [prod([
	EqualitySampler._pdf_helper(d_betabinomial_process, i, unique_structures[j])[unique_structures[j][i]]
	for i in 1:k
])
for j in eachindex(unique_structures)] #count_combinations(unique_structures[1])
probs ./ sum(probs)

d_dirichlet    = DirichletProcessMvUrnDistribution(k, 0.5)
d_betabinomial = BetaBinomialMvUrnDistribution(k)
pdf.(Ref(d_dirichlet),            unique_structures)
pdf.(Ref(d_betabinomial),         unique_structures)
pdf.(Ref(d_betabinomial_process), unique_structures)