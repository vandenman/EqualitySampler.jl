function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end

function count_equalities(sampled_models::AbstractMatrix)

	k, n = size(sampled_models)
	result = Vector{Int}(undef, n)
	for i in axes(sampled_models, 2)
		result[i] = k - length(unique(view(sampled_models, :, i)))
	end
	return result
end

function empirical_model_probabilities(sampled_models::AbstractMatrix)
	count_models = countmap(vec(mapslices(x->join(Int.(x)), sampled_models, dims = 1)))
	probs_models = counts2probs(count_models)
	return sort!(OrderedDict(probs_models), by=x->count_equalities(x))
end

function empirical_inclusion_probabilities(sampled_models::AbstractMatrix)
	no_equalities = count_equalities(sampled_models)
	counts_equalities = countmap(no_equalities)
	return sort!(OrderedDict(counts2probs(counts_equalities)))
end
