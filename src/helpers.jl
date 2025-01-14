# function counts2probs(counts::Dict{T, Int}) where T
# 	total_visited = sum(values(counts))
# 	probs = Dict{T, Float64}()
# 	for (model, count) in counts
# 		probs[model] = count / total_visited
# 	end
# 	return probs
# end

function count_equalities(sampled_models::AbstractMatrix{<:Integer})
	map(count_equalities, eachcol(sampled_models))
end

function empirical_model_probabilities(sampled_models::AbstractMatrix)
	count_models = StatsBase.countmap(vec(mapslices(x->join(Int.(x)), sampled_models, dims = 1)))
	probs_models = counts2probs(count_models)
	return sort!(OrderedCollections.OrderedDict(probs_models), by=x->count_parameters(x))
end

function empirical_equality_probabilities(sampled_models::AbstractMatrix)
	no_equalities = count_equalities(sampled_models)
	counts_equalities = StatsBase.countmap(no_equalities)
	return sort!(OrderedCollections.OrderedDict(counts2probs(counts_equalities)), by=x->count_equalities(x))
end

function empirical_no_parameters_probabilities(sampled_models::AbstractMatrix)
	no_parameters = map(count_parameters, eachcol(sampled_models))
	counts_parameters = StatsBase.countmap(no_parameters)
	return sort!(OrderedCollections.OrderedDict(counts2probs(counts_parameters)))
end
