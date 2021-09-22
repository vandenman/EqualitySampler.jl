using EqualitySampler
import StatsBase: countmap
using Distributions
using Statistics

function structure_type(x)
	length(x) == 3 && return structure_type_3(x)
	length(x) == 4 && return structure_type_4(x)
	return structure_type_5(x)
end

structure_type_3(x) = count_equalities(x) + 1
function structure_type_4(x)
	u = count_equalities(x)
	# 0, 1 => 1, 2
	u <= 1 && return u + 1
	if u == 2
		maximum(values(countmap(x))) == 2 && return 3
		return 4
	end
	return 5
end


function structure_type_5(x)

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

to_prob(x) = x ./ sum(x)

function _pdf_helper2(d::Union{AbstractConditionalUrnDistribution, AbstractMvUrnDistribution}, index, complete_urns)

	k = length(complete_urns)
	result = zeros(Float64, length(complete_urns))
	_pdf_helper2!(result, d, index, complete_urns)
	return result

end
function _pdf_helper2!(result, d::BetaBinomialProcessMvUrnDistribution, index, complete_urns)

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	index_already_sampled = 1:index - 1
	n0 = k

	# no_duplicated = count_equalities(view(urns, index_already_sampled))
	v_known_urns = view(complete_urns, index_already_sampled)
	r = length(Set(v_known_urns))
	n = n0 - (index - r - 1)

	model_probs_by_incl = exp.(EqualitySampler.log_model_probs_by_incl(d))

	num = den = 0.0
	for k in 1:n0
		num += model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    )
		den += model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1)
	end
	probEquality = num / (num + den / r)

	current_counts = countmap(v_known_urns)
	total_counts = sum(values(current_counts))
	no_inequality_options = (k - length(current_counts)) #length(current_counts) == the number of distinct elements in v_known_urns

	for i in 1:k
		if haskey(current_counts, i)
			# probability of an equality
			result[i] = probEquality * current_counts[i] / total_counts
		else
			# probability of an inequality
			result[i] = (1.0 - probEquality) / no_inequality_options
		end
	end
	# for (key, val) in current_counts
	# 	result[key] = probEquality * val / total_counts
	# end
	return result
end

unique_structures_dict = Dict{Int, Vector{Vector{Int}}}(
	3 => reduce_model.([
		[1, 2, 3],
		[1, 2, 2],
		[1, 1, 1]
	]),
	4 => reduce_model.([
		[1, 2, 3, 4],
		[1, 2, 3, 3],
		[1, 1, 3, 3],
		[1, 1, 1, 4],
		[1, 1, 1, 1],
	]),
	5 => reduce_model.([
		[1, 2, 3, 4, 5],
		[1, 2, 3, 4, 4],
		[1, 2, 2, 3, 3],
		[1, 1, 1, 2, 3],
		[1, 1, 1, 2, 2],
		[1, 1, 1, 1, 2],
		[1, 1, 1, 1, 1]
	])
)

k = 4
# d_betabinomial_process = UniformMvUrnDistribution(k) #
d_betabinomial_process = BetaBinomialProcessMvUrnDistribution(k)
obs = rand(d_betabinomial_process, 200_000)
obs_u = mapslices(reduce_model, obs; dims = 1)

unique_structures = unique_structures_dict[k]
structure_type.(unique_structures)
count_equalities.(unique_structures)

count(==(ones(Int, k)), eachcol(obs_u))

it = generate_all_models(k)
all_models = Matrix{Int}(undef, k, k^k)
for (i, m) in enumerate(it)
	all_models[:, i] .= m
end

obs_all_model_probs = to_prob([count(==(col), eachcol(obs)) for col in eachcol(all_models)])

unique_models = mapslices(reduce_model, generate_distinct_models(k); dims = 1)
obs_model_probs = to_prob([count(==(col), eachcol(obs_u)) for col in eachcol(unique_models)])
sum(obs_model_probs)
to_prob(compute_model_pdf.(eachcol(unique_models), Ref(d_betabinomial_process)))

# to match this one we don't need count_combinations
emp_model_probs = [count(==(unique_structures[i]), eachcol(obs_u)) for i in eachindex(unique_structures)]
to_prob(emp_model_probs .* reverse(ccc))

obs_structures = sort(countmap(structure_type.(eachcol(obs_u))))
hcat(eachindex(unique_structures), to_prob(values(obs_structures)), count_equalities.(unique_structures), emp_model_probs)

function compute_model_pdf(x, d)
	k = length(d)
	probs0 = Vector{Float64}(undef, k)
	res = 0.0
	for i in 1:k
		EqualitySampler._pdf_helper!(probs0, d, i, x)
		res += log(probs0[x[i]])
	end
	return exp(res)
end

obs_structure_dict = sort(Dict{Int, Vector{Int}}())
for i in axes(obs, 2)
	ee = structure_type(view(obs, :, i))
	if haskey(obs_structure_dict, ee)
		push!(obs_structure_dict[ee], i)
	else
		obs_structure_dict[ee] = [i]
	end
end

length.(values(obs_structure_dict))
mean(view(obs_all_model_probs, obs_structure_dict[1]))

Set(sort(obs[:, j]) for j in obs_structure_dict[4])


compute_model_pdf.(unique_structures, Ref(d_betabinomial_process))

all_models_structure_dict = sort(Dict{Int, Vector{Int}}())
for i in axes(all_models, 2)
	ee = structure_type(view(all_models, :, i))
	if haskey(all_models_structure_dict, ee)
		push!(all_models_structure_dict[ee], i)
	else
		all_models_structure_dict[ee] = [i]
	end
end
theoretical_probs = compute_model_pdf.(eachcol(all_models), Ref(d_betabinomial_process))
theoretical_probs_by_structure = [sum(theoretical_probs[all_models_structure_dict[j]]) for j in eachindex(all_models_structure_dict)]
length.(values(all_models_structure_dict))

join.(eachcol(all_models[:, all_models_structure_dict[4]]))
hcat(join.(eachcol(all_models[:, all_models_structure_dict[4]])),
	theoretical_probs[all_models_structure_dict[4]]
)
# this is the key!
# "2221"  0.00297619
# "1131"  0.00396825
# because the probability of the odd one (1 and 3) differs depending on the sizes of the already sampled ones!

d_dirichlet = DirichletProcessMvUrnDistribution(k, .5)
pdf_model_distinct(d_dirichlet, [1, 1, 1, 2])
pdf_model_distinct(d_dirichlet, [1, 1, 3, 1])

function sample_dirichlet_manual(n, k, α)
	result = Matrix{Int}(undef, k, n)
	for i in 1:n
		result[1, i] = rand(1:k)
		for j in 2:k
			if rand() < (α / (α + j - 1))
				result[j, i] = rand(setdiff(1:k, result[1:j-1, i]))
			else
				probvec = values(countmap(result[1:j-1, i])) ./ (α + j - 1)
				result[j, i] = result[rand(Categorical(to_prob(probvec))), i]
			end
		end
	end
	result
end
# samps = rand(d_dirichlet, 100_000)
samps = sample_dirichlet_manual(100_000, k, 0.5)
samps_u = mapslices(reduce_model, samps; dims = 1)
samps_u_counts = sort(countmap(eachcol(samps_u)))
samps_u_probs  = sort(Dict(keys(samps_u_counts) .=> values(samps_u_counts) ./ size(samps, 2)))
hcat(collect(keys(samps_u_probs)), pdf_model_distinct.(Ref(d_dirichlet), keys(samps_u_probs)))

function solvequadratic(a, b, c)
	d = sqrt(b^2 - 4a*c)
	(-b - d) / 2a, (-b + d) / 2a
end
solvequadratic(1.0, den, index - 1 - num / (num + den))


probs0 = Vector{Float64}(undef, k)
probs = Matrix{Float64}(undef, k, length(unique_structures))
for j in eachindex(unique_structures)
	for i in 1:k
		EqualitySampler._pdf_helper!(probs0, d_betabinomial_process, i, unique_structures[j])
		# @assert _pdf_helper2(d_betabinomial_process, i, unique_structures[j]) ≈ probs0
		@assert sum(probs0) ≈ 1.0 atol=1e-6
		idx = unique_structures[j][i]
		probs[i, j] = probs0[idx]

		# known_values = view(unique_structures[j], 1:i-1)
		# probs[i, j] = sum(probs0[2:3])
		# if unique_structures[j][i] ∈ known_values
		# 	idx = unique_structures[j][i]
		# 	# idx = findall(==(unique_structures[j][i]), known_values)
		# else
		# 	idx = setdiff(1:k, known_values)
		# end
		# probs[i, j] = sum(view(probs0, idx))
		# probs[i, j] = sum(view(probs0, idx))
	end
end
# ppp = probs[i, j]
# probs[i, j] = 1.25ppp
prod.(eachcol(probs))
mm = sort(countmap(structure_type.(eachcol(all_models))))
nn = hcat(
	to_prob(values(obs_structures)) ./ values(mm),
	prod.(eachcol(probs)) .* combs .* reverse(ccc),
	prod.(eachcol(probs))
)

prod.(eachcol(probs))[4] * 48

hcat(
	to_prob(values(obs_structures)) ./ values(mm),
	prod.(eachcol(probs))
)

logunsignedstirlings1(k, k - count_equalities(unique_structures[4]))

nn[:, 1] ./ nn[:, 2]
combs .* reverse(ccc) .* prod.(eachcol(probs))
combs .* reverse(ccc) .* prod.(eachcol(probs))
values(mm) ./ sum(values(mm))
prod.(eachcol(probs))



obs_all_model_probs

combs = count_combinations.(unique_structures)# / sum(count_combinations.(unique_structures))
ccc, sizes = count_set_partitions_given_partition_size(k)

length.(Set(sort(obs[:, j]) for j in obs_structure_dict[i]) for i in 1:length(obs_structure_dict))
length.(Set(sort(obs_u[:, j]) for j in obs_structure_dict[i]) for i in 1:length(obs_structure_dict))
length.(Set(obs_u[:, j] for j in obs_structure_dict[i]) for i in 1:length(obs_structure_dict))


start = unique_structures[4]
[EqualitySampler._pdf_helper(d_betabinomial_process, i, start)[start[i]] for i in 1:k]

EqualitySampler._pdf_helper!(probs0, d_betabinomial_process, 4, start)


combs .* reverse(ccc) .* prod.(eachcol(probs))

all_models
obs_all_model_probs

mean_obs_probs = [
	sum()
]

sum(reverse(ccc) .* combs)
prod.(eachcol(probs)) .* reverse(ccc) .* combs

probs2 = prod.(eachcol(probs))
to_prob(to_prob(probs2) .* reverse(ccc))
to_prob(to_prob(probs2) .* reverse(ccc))

hcat(combs, unique_structures, reverse(ccc), reverse(sizes))

probs2 = prod.(eachcol(probs))
probs3 = probs2 .* combs .* reverse(ccc)
probs3 / sum(probs3)


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