
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

function triangular_to_vec(x::LA.UpperTriangular{T, Matrix{T}}) where T
	# essentially vec(x) but without the zeros
	p = size(x, 1)
	result = Vector{T}(undef, p * (p + 1) ÷ 2)
	idx = 1
	@inbounds for i in 1:p
		for j in 1:i
			result[idx] = x[j, i]
			idx += 1
		end
	end
	result
end

function extract_means_continuous_params(model, chn)

	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chn, :parameters))

	all_keys = keys(VarInfo(model).metadata)
	# handles submodels
	key_μ_m = all_keys[findfirst(x->occursin("μ_m", string(x)), all_keys)]
	key_μ_w = all_keys[findfirst(x->occursin("μ_w", string(x)), all_keys)]

	return (
		μ_1 = vec(mean(group(chn, key_μ_m).value.data, dims = 1)),
		μ_2 = vec(mean(group(chn, key_μ_w).value.data, dims = 1)),
		σ_1 = mean(x[1] for x in gen),
		σ_2 = mean(x[2] for x in gen),
		Σ_1 = mean(triangular_to_vec(x[3]) for x in gen),
		Σ_2 = mean(triangular_to_vec(x[4]) for x in gen)
	)
end

function extract_partition_samples(model, chn)

	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chn, :parameters))
	partition_samples = Matrix{Int}(undef, length(gen), length(gen[1][end]))
	for i in eachindex(gen)
		partition_samples[i, :] .= reduce_model(gen[i][end])
	end
	return partition_samples
end

function extract_σ_samples(model, chn)

	gen = generated_quantities(model, Turing.MCMCChains.get_sections(chn, :parameters))
	k = length(gen[1][1])
	σ_samples = Matrix{Float64}(undef, length(gen), 2k)
	@inbounds for i in eachindex(gen)
		σ_samples[i, 1:k]    .= gen[i][1]
		σ_samples[i, k+1:2k] .= gen[i][2]
	end
	return σ_samples
end


#region plot functions
function plot_retrieval(obs, estimate, nms, size = :default)
	@assert length(obs) == length(estimate) == length(nms)
	plts = Vector{Plots.Plot}(undef, length(obs))
	for (i, (o, e, nm)) in enumerate(zip(obs, estimate, nms))
		plt = scatter(o, e, title = nm, legend = false)
		Plots.abline!(plt, 1, 0)
		plts[i] = plt
	end
	nc = isqrt(length(obs))
	nr = ceil(Int, length(obs) / nc)

	plot(plts..., layout = (nr, nc), size = size === :default ? 250 .* (nc, nr) : size)
end

function plot_retrieval2(obs, estimate, nms, size = :default)
	@assert length(obs) == length(estimate) == length(nms)
	plts = Vector{Plots.Plot}(undef, length(obs) ÷ 2)
	for (i, idx) in enumerate(1:2:length(obs))
		plt = plot()
		Plots.abline!(plt, 1, 0, color = :grey, label = "")
		# plt = scatter(obs[idx], estimate[idx], title = nms[idx], legend = isone(i) ? :topleft : false)
		scatter!(plt, obs[idx], estimate[idx], title = nms[idx], legend = isone(i) ? :topleft : false, label = "m")
		scatter!(plt, obs[idx+1], estimate[idx+1], label = "w")
		plts[i] = plt
	end
	nplt = length(obs) ÷ 2
	nc = isqrt(nplt)
	nr = ceil(Int, nplt / nc)

	plot(plts..., layout = (nr, nc), size = size === :default ? 250 .* (nc, nr) : size)
end


function compute_density_estimate(mat, npoints = 2^12)
	no_groups = size(mat, 2)
	x = Matrix{Float64}(undef, npoints, no_groups)
	y = Matrix{Float64}(undef, npoints, no_groups)

	for (i, col) in enumerate(eachcol(mat))
		k = KernelDensity.kde(col; npoints = npoints)#, boundary = (0, Inf))
		x[:, i] .= k.x
		y[:, i] .= k.density
	end

	return (x = x, y = y)
end
#endregion
