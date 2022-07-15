#region sampling related stuff
function get_logπ(model)
	vari = DynamicPPL.VarInfo(model)
	mt = vari.metadata
	return function logπ(partition, nt)
		DynamicPPL.setval!(vari, partition, DynamicPPL.VarName(:partition))
		for (key, val) in zip(keys(nt), nt)
			if key !== :partition
				indices = mt[key].vns
				if !(val isa Vector)
					DynamicPPL.setval!(vari, val, indices[1])
				else
					ranges = mt[key].ranges
					for i in eachindex(indices)
						DynamicPPL.setval!(vari, val[ranges[i]], indices[i])
					end
				end
			end
		end
		DynamicPPL.logjoint(model, vari)
	end
end

function get_sampler(model, discrete_sampler::Symbol = :custom, ϵ::Float64 = 0.0, n_leapfrog::Int = 20)
	parameters = DynamicPPL.syms(DynamicPPL.VarInfo(model))
	if :partition in parameters

		continuous_parameters = filter(!=(:partition), parameters)
		if discrete_sampler === :custom
			return Turing.Gibbs(
				Turing.HMC(ϵ, n_leapfrog, continuous_parameters...),
				Turing.GibbsConditional(:partition, PartitionSampler(length(model.args.partition_prior), get_logπ(model)))
			)
		else
			no_groups = length(model.args.obs_mean)::Int
			return Turing.Gibbs(
				Turing.HMC(ϵ, n_leapfrog, continuous_parameters...),
				Turing.PG(no_groups, :partition)
			)
		end

	else
		return Turing.NUTS()
	end
end
#endregion

get_eq_ind_nms(samples::MCMCChains.Chains) = filter(x->startswith(string(x), "partition"), samples.name_map.parameters)
function get_eq_samples(samples::MCMCChains.Chains)
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	return reshape(samples[eq_ind_nms].value.data, :, s[2])
end


"""
	compute the proportion of samples where partition[i] == partition[j] ∀i, j
"""
compute_post_prob_eq(chn::MCMCChains.Chains) = compute_post_prob_eq(get_eq_samples(chn))
function compute_post_prob_eq(partition_samples::AbstractMatrix)
	n_samps, n_groups = size(partition_samples)
	probs = zeros(Float64, n_groups, n_groups)
	@inbounds for k in axes(partition_samples, 1)
		for j in 1:n_groups-1
			for i in j+1:n_groups
				if partition_samples[k, i] == partition_samples[k, j]
					probs[i, j] += 1.0
				end
			end
		end
	end
	return probs ./ n_samps
end

function compute_post_prob_eq(partition_samples::AbstractArray{T, 3}) where T
	n_samps, n_groups, n_chains = size(partition_samples)
	probs = zeros(Float64, n_groups, n_groups)
	@inbounds for l in axes(partition_samples, 3)
		for k in axes(partition_samples, 1)
			for j in 1:n_groups-1
				for i in j+1:n_groups
					if partition_samples[k, i, l] == partition_samples[k, j, l]
						probs[i, j] += 1.0
					end
				end
			end
		end
	end
	return probs ./ (n_samps * n_chains)
end


"""
	compute how often each model is visited
"""
compute_model_counts(chn::MCMCChains.Chains, add_missing_models::Bool = true) = compute_model_counts(get_eq_samples(chn), add_missing_models)

function compute_model_counts(partition_samples::AbstractMatrix, add_missing_models::Bool = true)
	res = StatsBase.countmap(vec(mapslices(x->join(reduce_model(Int.(x))), partition_samples, dims = 2)))
	if add_missing_models
		k = size(partition_samples)[2]
		expected_size = bellnum(k)
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
function compute_model_probs(chn::Union{MCMCChains.Chains, AbstractMatrix}, add_missing_models::Bool = true)
	count_models = compute_model_counts(chn, add_missing_models)
	total_visited = sum(values(count_models))
	probs_models = Dict{String, Float64}()
	for (model, count) in count_models
		probs_models[model] = count / total_visited
	end
	return sort!(OrderedCollections.OrderedDict(probs_models), by=x->count_equalities(x))
end

"""
	compute the posterior probability of including 0, ..., k-1 equalities in the model
"""
compute_incl_probs(chn::MCMCChains.Chains; add_missing_inclusions::Bool = true) = compute_incl_probs(get_eq_samples(chn); add_missing_inclusions=add_missing_inclusions)
function compute_incl_probs(partition_samples::AbstractMatrix; add_missing_inclusions::Bool = true)
	k = size(partition_samples)[2]
	inclusions_per_model = vec(mapslices(x->k - no_distinct_groups_in_partition(x)), partition_samples, dims = 2)
	count = StatsBase.countmap(inclusions_per_model)
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


function get_posterior_means_mu_sigma(model, chn::MCMCChains.Chains)
	s = summarystats(chn)
	μ = s.nt.mean[startswith.(string.(s.nt.parameters), "μ")]
	gen = generated_quantities(model, chn)
	σ = collect(mean(first(x)[j] for x in gen) for j in 1:k)
	return (μ, σ)
end


