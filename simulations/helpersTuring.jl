import OrderedCollections: OrderedDict
import StatsBase: countmap

get_eq_ind_nms(samples) = filter(x->startswith(string(x), "partition"), samples.name_map.parameters)
function get_eq_samples(samples)
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	return reshape(samples[eq_ind_nms].value.data, :, s[2])
end


"""
	compute the proportion of samples where partition[i] == partition[j] ∀i, j
"""
function compute_post_prob_eq(samples)
	samps = get_eq_samples(samples)
	n_samps, n_groups = size(samps)
	probs = zeros(Float64, n_groups, n_groups)
	for row in eachrow(samps)
		for j in eachindex(row)
			idx = j .+ findall(==(row[j]), row[j+1:end])
			probs[idx, j] .+= 1.0
		end
	end
	return probs ./ n_samps
end

"""
	compute how often each model is visited
"""
function compute_model_counts(chn, add_missing_models::Bool = true)
	eq_samples = get_eq_samples(chn)
	res = countmap(vec(mapslices(x->join(reduce_model(Int.(x))), eq_samples, dims = 2)))
	if add_missing_models
		k = size(eq_samples)[2]
		expected_size = count_distinct_models(k)
		if length(res) != expected_size
			allmodels = vec(mapslices(x->join(reduce_model(x)), generate_distinct_models(k), dims = 1))
			for m in allmodels
				if !haskey(res, m)
					res[m] = 0
				end
			end
		end
	end
	return sort(res)
end

"""
	compute the posterior probability of each model
"""
function compute_model_probs(chn, add_missing_models::Bool = true)
	count_models = compute_model_counts(chn, add_missing_models)
	total_visited = sum(values(count_models))
	probs_models = Dict{String, Float64}()
	for (model, count) in count_models
		probs_models[model] = count / total_visited
	end
	return sort!(OrderedDict(probs_models), by=x->count_equalities(x))
end

"""
	compute the posterior probability of including 0, ..., k-1 equalities in the model
"""
function compute_incl_probs(chn; add_missing_inclusions::Bool = true)
	eq_samples = get_eq_samples(chn)
	k = size(eq_samples)[2]
	inclusions_per_model = vec(mapslices(x->k - length(Set(x)), eq_samples, dims = 2))
	count = countmap(inclusions_per_model)
	if add_missing_inclusions && length(count) != k
		for i in 1:k
			if !haskey(count, i)
				count[i] = 0
			end
		end
	end
	return sort(counts2probs(count))
end

function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end


function get_posterior_means_mu_sigma(model, chn)
	s = summarystats(chn)
	μ = s.nt.mean[startswith.(string.(s.nt.parameters), "μ")]
	gen = generated_quantities(model, chn)
	σ = collect(mean(first(x)[j] for x in gen) for j in 1:k)
	return (μ, σ)
end


