# TEMPORARILY COPIED FROM DynamicPPL (basically a version of generated_quantities that is always silent)
using DynamicPPL

function _apply2!(kernel!, vi::AbstractVarInfo, values, keys)
	keys_strings = map(string, DynamicPPL.collectmaybe(keys))
	num_indices_seen = 0

	for vn in Base.keys(vi)
		indices_found = kernel!(vi, vn, values, keys_strings)
		# @show vn, indices_found
		if indices_found !== nothing
			num_indices_seen += length(indices_found)
		end
	end

	# if length(keys) > num_indices_seen
	#     # Some keys have not been seen, i.e. attempted to set variables which
	#     # we were not able to locate in `vi`.
	#     # Find the ones we missed so we can warn the user.
	#     unused_keys = DynamicPPL._find_missing_keys(vi, keys_strings)
	#     @warn "the following keys were not found in `vi`, and thus `kernel!` was not applied to these: $(unused_keys)"
	# end

	return vi
end

function generated_quantities2(model, chain)
	varinfo = DynamicPPL.VarInfo(model)
	iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
	return map(iters) do (sample_idx, chain_idx)
		setval_and_resample2!(varinfo, chain, sample_idx, chain_idx)
		model(varinfo)
	end
end
setval_and_resample2!(vi, x) = _apply!(_setval_and_resample_kernel!, vi, values(x), keys(x))
function setval_and_resample2!(vi, chains, sample_idx::Int, chain_idx::Int)
	return _apply2!(_setval_and_resample_kernel2!, vi, chains.value[sample_idx, :, chain_idx], keys(chains))
end

function _setval_and_resample_kernel2!(vi, vn, values, keys)
	indices = findall(Base.Fix1(DynamicPPL.subsumes_string, string(vn)), keys)
	if !isempty(indices)
		val = reduce(vcat, values[indices])
		DynamicPPL.setval!(vi, val, vn)
		DynamicPPL.settrans!(vi, false, vn)
	else
		# Ensures that we'll resample the variable corresponding to `vn` if we run
		# the model on `vi` again.
		DynamicPPL.set_flag!(vi, vn, "del")
	end

	return indices
end

