struct TrueValues{T<:Real, U<:AbstractVector{T}, W<:AbstractVector{<:Integer}}
	μ::T
	σ::T
	θ::U
	partition::W
end

function normalize_θ(offset::AbstractFloat, true_model::Vector{T}) where T<:Integer

	copy_model = copy(true_model)
	current_max = copy_model[1]

	for i in eachindex(copy_model)
		if copy_model[i] > current_max
			copy_model[copy_model .== i] .= current_max
			current_max += 1
		elseif copy_model[i] == i
			current_max += 1
		end
	end

	θ = copy_model .* offset
	return θ .- mean(θ)
end

struct AnovaSimulatedData
	data::SimpleDataSet
	distribution::Distributions.MvNormal
	true_values::TrueValues
end

simulate_data_one_way_anova(
	n_groups::Integer,
	n_obs_per_group::Integer,
	θ::AbstractVector{<:AbstractFloat} = Float64[],
	partition::AbstractVector{<:Integer} = 1:n_groups,
	μ::AbstractFloat = 0.0,
	σ::AbstractFloat = 1.0
) = simulate_data_one_way_anova(Random.GLOBAL_RNG, n_groups, n_obs_per_group, θ, partition, μ, σ)

function simulate_data_one_way_anova(
	rng::Random.AbstractRNG,
	n_groups::Integer,
	n_obs_per_group::Integer,
	θ::AbstractVector{<:AbstractFloat} = Float64[],
	partition::AbstractVector{<:Integer} = 1:n_groups,
	μ::AbstractFloat = 0.0,
	σ::AbstractFloat = 1.0
)

	if isempty(θ)
		# θ = 2 .* randn(n_groups)
		θ = 0.2 .* partition
	end

	length(θ) != n_groups && throw(error("length(θ) != n_groups"))

	n_obs = n_groups * n_obs_per_group
	θc = θ .- mean(θ)

	g = Vector{UnitRange{Int}}(undef, n_groups)
	for i in eachindex(g)
		g[i] = 1 + n_obs_per_group * (i - 1) : n_obs_per_group * i
	end
	# g = Vector{Int}(undef, n_obs)
	# for (i, r) in enumerate(Iterators.partition(1:n_obs, ceil(Int, n_obs / n_groups)))
	# 	g[r] .= i
	# end

	g_big = Vector{UInt8}(undef, n_obs) # max 255 groups, should be fine
	for (i, idx) in enumerate(g)
		g_big[idx] .= i
	end
	D = Distributions.MvNormal(μ .+ σ .* view(θc, g_big), abs2(σ) * LinearAlgebra.I)
	y = rand(rng, D)

	dat = SimpleDataSet(y, g)

	true_values = TrueValues(μ, σ, θc, partition)

	return AnovaSimulatedData(dat, D, true_values)

end

function fit_lm(X::AbstractMatrix, y::AbstractVector)
	fit = GLM.lm(X, y)#::GLM.LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}
	return GLM.coef(fit), GLM.confint(fit), fit
end

function fit_lm(df::SimpleDataSet)
	design_matrix = zeros(Int, length(df.y), length(df.g))
	for (i, idx) in enumerate(df.g)
		design_matrix[idx, i] .= 1
	end
	return fit_lm(design_matrix, df.y)
end

function fit_lm(df, formula = StatsModels.@formula(y ~ 1 + g))

	# TODO: don't use a dataframe but use the X, y directly?
	fit = GLM.lm(formula, df)#::StatsModels.TableRegressionModel{GLM.LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}, Matrix{Float64}}
	#, contrasts = Dict(:g => StatsModels.FullDummyCoding()))
	# transform the coefficients to a grand mean and offsets
	coefs = GLM.coef(fit)
	ests = similar(coefs)::Vector{Float64}
	y = df.y::Vector{Float64}
	ests[1] = coefs[1] - mean(y)
	ests[2:end] .= coefs[2:end] .+ coefs[1] .- mean(y)

	cis = GLM.confint(fit)
	cis[1, :] .-= GLM.coef(fit)[1] # subtract reference level value

	return ests, cis, fit
end

function get_suff_stats(df::SimpleDataSet)
	return [Distributions.suffstats(Distributions.Normal, view(df.y, idx)) for idx in df.g]
end

function get_suff_stats(df::DataFrames.DataFrame)

	_, obs_mean, obs_var, obs_n = eachcol(
		DataFrames.combine(DataFrames.groupby(df, :g), :y => mean, :y => var, :y => length)
	)

	return obs_mean, obs_var, obs_n
end

function getQ_Rouder(n_groups::Integer)::Matrix{Float64}
	# X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))
	Σₐ = Matrix{Float64}(LinearAlgebra.I, n_groups, n_groups) .- (1.0 / n_groups)
	_, v::Matrix{Float64} = LinearAlgebra.eigen(Σₐ)
	Q = v[end:-1:1, end:-1:2] # this is what happens in Rouder et al., (2012) eq ...

	@assert isapprox(sum(Q * randn(n_groups-1)), 0.0, atol = 1e-8)

	return Q
end

function average_equality_constraints!(destination::AbstractVector{T}, ρ::AbstractVector{T}, partition::AbstractVector{U}) where {T<:Real, U<:Integer}

	# ~ O(2K), whereas a double for loop would be O(K^2)
	idx_vecs = [U[] for _ in eachindex(partition)]
	@inbounds for i in eachindex(partition)
		push!(idx_vecs[partition[i]], i)
	end

	@inbounds for idx in idx_vecs
		isempty(idx) && continue
		destination[idx] .= mean(ρ[idx])
	end
	return destination
end

average_equality_constraints(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer}) = average_equality_constraints!(similar(ρ), ρ, partition)

function get_equalizer_matrix_from_partition(partition::AbstractVector{<:Integer})
	k = length(partition)
	mmm = Matrix{Float64}(undef, k, k)
	get_equalizer_matrix_from_partition!(mmm, partition)
	return mmm
end

function get_equalizer_matrix_from_partition!(mmm, partition::AbstractVector{<:Integer})
	@inbounds for i in axes(mmm, 2)
		v = count(==(partition[i]), partition)
		mmm[:, i] .= (1.0 / v) .* (partition .== partition[i])
	end
end

function get_equalizer_eigenvectors_from_partition(partition)
	cm = EqualitySampler.fast_countmap_partition_incl_zero(partition)
	cm0 = filter(!iszero, cm)
	mat = zeros(length(partition), length(cm0))
	# @show partition
	column = 1
	for i in eachindex(cm)
		if !iszero(cm[i])
			value = sqrt(cm[i]) / cm[i]
			for j in eachindex(partition)
				if partition[j] == i
					mat[j, column] = value
				end
			end
			column += 1
		end
	end
	mat
end


function build_get_equalizer_matrix_from_partition_with_cache()
	local _cache = Dict{Vector{Int}, Matrix{Float64}}()
	function _inner(partition::Vector{Int})
		reduced_partition = reduce_model(partition)
		if haskey(_cache, reduced_partition)
			return _cache[reduced_partition]
		else
			res = get_equalizer_matrix_from_partition(partition)
			_cache[reduced_partition] = res
			return res
		end
	end
end

function get_starting_values(df, full_model_partition=true)

	coefs, cis, fit = fit_lm(df)

	adj_mat = [
		Int(cis[i, 1] <= cis[j, 2] && cis[j, 1] <= cis[i, 2])
		for i in axes(cis, 1), j in axes(cis, 1)
	]

	n_groups = size(cis, 1)
	Q = getQ_Rouder(n_groups)

	y = df.y
	if full_model_partition
		partition_start	= collect(axes(Q, 1))
	else
		partition_start	= map(x->findfirst(isone, x), eachcol(adj_mat))::Vector{Int}
	end
	n_partitions	= EqualitySampler.no_distinct_groups_in_partition(partition_start)
	σ_start			= var(GLM.residuals(fit))
	μ_start			= mean(y)
	θ_c_start		= isone(n_partitions) ? zeros(n_groups) : average_equality_constraints(coefs, partition_start)
	# this is faster than Q \ θ_c_start which uses a qr decomposition
	θ_start_0		= LinearAlgebra.pinv(Q) * θ_c_start
	if full_model_partition
		g_start = 1.0
	else
		g_start = isone(n_partitions) ? 1.0 : var(θ_start_0; corrected = false)
	end
	θ_start			= vec(θ_start_0 ./ sqrt.(g_start))

	# @assert Q * (sqrt(g_start) .* θ_start) ≈ θ_c_start

	return (partition = partition_start, θ = θ_start, μ = μ_start, σ = σ_start, g = g_start)

end

function get_θ_cs(model, chain)
	gen = DynamicPPL.generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
	θ_cs = Matrix{Float64}(undef, length(gen), length(gen[1]))
	for i in eachindex(gen)
		for j in eachindex(gen[i])
			θ_cs[i, j] = gen[i][j]
		end
	end
	return vec(mean(θ_cs, dims = 2)), θ_cs
end

function get_init_params(model, partition::Vector{Int}, θ_r::Vector{Float64}, μ = 0.0, σ² = 1.0, g = 1.0)
	nt = (
		partition									= partition,
		var"one_way_anova_mv_ss_submodel.μ_grand"	= μ,
		var"one_way_anova_mv_ss_submodel.σ²"		= σ²,
		var"one_way_anova_mv_ss_submodel.g"			= g,
		var"one_way_anova_mv_ss_submodel.θ_r"		= θ_r
	)
	varinfo = Turing.VarInfo(model)
	model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(nt));
	init_params = varinfo[Turing.SampleFromPrior()]::Vector{Float64}
	return init_params

end

function get_init_params(partition::Vector{Int}, θ_r::Vector{Float64}, μ = 0.0, σ² = 1.0, g = 1.0)
	#  get_init_params(model, ...) also works and can be used to verify the order,
	# but that does a lot of type unstabled and complicated things that essentially boil down to this
	return vcat(partition, μ, σ², g, θ_r)
end

function get_init_params(θ_r::Vector{Float64}, μ::Float64 = 0.0, σ² = 1.0, g = 1.0)
	return vcat(μ, σ², g, θ_r)
end

DynamicPPL.@model function one_way_anova_mv_ss_submodel(suff_stats_vec, Q, partition = nothing, ::Type{T} = Float64) where {T}

	n_groups = length(suff_stats_vec)

	# improper priors on grand mean and variance
	μ_grand 		~ Turing.Flat()
	σ² 				~ JeffreysPriorVariance()

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)

	# θ_r ~ Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1)))
	θ_r ~ Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1)))

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	# @inbounds
	for i in eachindex(suff_stats_vec)
		Turing.@addlogprob! loglikelihood_suffstats(Distributions.Normal(μ_grand + sqrt(σ²) * θ_cs[i], sqrt(σ²)), suff_stats_vec[i])
	end

	return θ_cs

end

DynamicPPL.@model function one_way_anova_mv_ss_eq_submodel(suff_stats_vec, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractPartitionDistribution}

	partition ~ partition_prior
	DynamicPPL.@submodel prefix=false θ_cs = one_way_anova_mv_ss_submodel(suff_stats_vec, Q, partition, T)
	return θ_cs

end

function precompute_integrated_log_lik(dat)

	y = dat.y
	N, P = length(y), length(dat.g)
	X = zeros(N, P)
	for (i, idx) in enumerate(dat.g)
		X[idx, i] .= 1.0
	end

	# P0 = 1 / N * ones(N) * ones(N)'
	P0 = FillArrays.Fill(1 / N, N, N)

	Q = EqualitySampler.Simulations.getQ_Rouder(P)
	X = X * Q

	# ỹ = (LinearAlgebra.I-P0) * y
	# X̃ = (LinearAlgebra.I-P0) * X
	# avoids forming LinearAlgebra.I-P0
	ỹ = y - P0 * y
	X̃ = X - P0 * X

	ỹTỹ = ỹ'ỹ
	ỹTX̃ = ỹ'X̃
	X̃TX̃ = X̃'X̃
	gamma_a = SpecialFunctions.loggamma((N-1)/2)

	return (; ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
end


# function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

# 	invG = LinearAlgebra.Diagonal(fill(1 / g, length(ỹTX̃)))
# 	Vg = X̃TX̃ + invG

# 	a = (N - 1) / 2
# 	b = ỹTỹ - @inbounds (ỹTX̃ / Vg * ỹTX̃')[1]

# 	return @inbounds gamma_a - (
# 		a * log(2*pi) + (log(N) - LinearAlgebra.logabsdet(invG)[1] + LinearAlgebra.logabsdet(Vg)[1]) / 2 + a * log(b)
# 	)

# end

# function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)

# 	Ρ = EqualitySampler.Simulations.get_equalizer_matrix_from_partition(partition)
# 	B = Q'Ρ*Q
# 	ỹTX̃ = ỹTX̃ * B
# 	X̃TX̃ = B * X̃TX̃ * B

# 	return integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)
# end

function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)

	invG = LinearAlgebra.Diagonal(fill(1 / g, length(ỹTX̃)))
	Vg = X̃TX̃ + invG

	Vg_chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Vg))

	# based on https://github.com/JuliaStats/PDMats.jl/blob/87277241153cde10fa6ec82086e2af2f050240f4/src/pdmat.jl#L108-L112
	z = ỹTX̃ / Vg_chol.U
	b = ỹTỹ - LinearAlgebra.dot(z, z) # does dot work with AD?

	a = (N - 1) / 2
	logabsdet_g = length(ỹTX̃) * log(g)

	return @inbounds gamma_a - (
		a * log(2*pi) + (log(N) + logabsdet_g + 2*LinearAlgebra.logabsdet(Vg_chol.U)[1]) / 2 + a * log(b)
	)

end

function integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)#, ỹTX̃_B_s, X̃TX̃_B_s)

	no_distinct = EqualitySampler.no_distinct_groups_in_partition(partition)
	if isone(no_distinct)
		return integrated_log_lik(ỹTỹ, zero(ỹTX̃), zero(X̃TX̃), gamma_a, N, g)
	elseif no_distinct == length(partition)
		return integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)
	end

	Ρeigvec = get_equalizer_eigenvectors_from_partition(partition)
	B = Q' * Ρeigvec
	B = B * B'

	# X̃TX̃ = B * X̃TX̃ * B
	# simplifies to
	# B * X̃TX̃
	# since B (and Ρ) is idempotent we have
	# B * X̃TX̃ * B = B * B * X̃TX̃ = B * X̃TX̃
	return integrated_log_lik(ỹTỹ, ỹTX̃ * B, X̃TX̃ * B, gamma_a, N, g)
end


DynamicPPL.@model function integrated_full_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g)
end

DynamicPPL.@model function integrated_partition_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, Q, partition_prior)
	g ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	partition ~ partition_prior
	Turing.@addlogprob! integrated_log_lik(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, g, Q, partition)
end

function prep_model_arguments(df)
	suff_stats_vec = get_suff_stats(df)
	n_groups = length(suff_stats_vec)
	Q = getQ_Rouder(n_groups)
	return suff_stats_vec, Q
end

# for full model
get_mcmc_sampler_anova(spl::Turing.Inference.InferenceAlgorithm, args...) = spl
get_mcmc_sampler_anova(::Symbol, model) = get_sampler(model)

# for equalities
get_mcmc_sampler_anova(spl::Real, model, _) = get_sampler(model, :custom, spl)
function get_mcmc_sampler_anova(spl::Symbol, model, init_params)
	ϵ = brute_force_ϵ(model, spl; init_params = init_params)
	return get_sampler(model, spl, ϵ)
end

function fit_full_model(
		df
		;
		spl = :custom,
		mcmc_settings::MCMCSettings = MCMCSettings(),
		modeltype::Symbol = :old,
		rng::Random.AbstractRNG = Random.GLOBAL_RNG
	)

	if modeltype === :old
		suff_stats_vec, Q = prep_model_arguments(df)
		model = one_way_anova_mv_ss_submodel(suff_stats_vec, Q)

		starting_values = get_starting_values(df, true)
		init_params = get_init_params(Base.structdiff(starting_values, (partition=nothing, ))...)
	else#if modeltype === :reduced

		ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = precompute_integrated_log_lik(df)
		model = integrated_full_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N)
		starting_values = get_starting_values(df, true)
		init_params = [starting_values.g]

	end

	mcmc_sampler = get_mcmc_sampler_anova(spl, model)
	chain = sample_model(model, mcmc_sampler, mcmc_settings; init_params = init_params)::MCMCChains.Chains

	if modeltype === :old
		return combine_chain_with_generated_quantities(model, chain, "θ_cs")
	else
		return chain
	end

end

function fit_eq_model(
		df,
		partition_prior::EqualitySampler.AbstractPartitionDistribution
		;
		spl = :custom,
		mcmc_settings::MCMCSettings = MCMCSettings(),
		eq_model::Symbol = :old,
		rng::Random.AbstractRNG = Random.GLOBAL_RNG
	)

	if eq_model === :reduced
		ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N = precompute_integrated_log_lik(df)
		Q = getQ_Rouder(length(df.g))
		model = integrated_partition_model(ỹTỹ, ỹTX̃, X̃TX̃, gamma_a, N, Q, partition_prior)
		starting_values = get_starting_values(df, false)
		init_params = vcat(starting_values.g, starting_values.partition)
	else
		suff_stats_vec, Q = prep_model_arguments(df)
		model = one_way_anova_mv_ss_eq_submodel(suff_stats_vec, Q, partition_prior)
		starting_values = get_starting_values(df)
		init_params = get_init_params(starting_values...)
	end

	mcmc_sampler = get_mcmc_sampler_anova(spl, model, init_params)

	chain = sample_model(model, mcmc_sampler, mcmc_settings, rng; init_params = init_params)::MCMCChains.Chains

	if eq_model === :reduced
		return chain
	else
		return combine_chain_with_generated_quantities(model, chain, "θ_cs")
	end
end

function get_generated_quantities(model, chain)

	gen = DynamicPPL.generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
	gen_mat = Matrix{Float64}(undef, length(gen), length(gen[1]))
	for i in eachindex(gen)
		for j in eachindex(gen[i])
			gen_mat[i, j] = gen[i][j]
		end
	end
	return gen_mat
end

function combine_chain_with_generated_quantities(model, chain, parameter_name::AbstractString)

	constrained_samples = get_generated_quantities(model, chain)
	new_dims = (size(chain, 1), size(constrained_samples, 2), size(chain, 3))

	# my reshape-foo is not good enough to avoid this
	reshaped_constrained_samples = Array{Float64}(undef, new_dims)
	for i in 1:size(chain, 3)
		idx = 1 + size(chain, 1) * (i - 1) : size(chain, 1) * i
		reshaped_constrained_samples[:, :, i] .= view(constrained_samples, idx, :)
	end

	constrained_chain = MCMCChains.setrange(
		MCMCChains.Chains(reshaped_constrained_samples, collect(Symbol(parameter_name, "["* string(i) * "]") for i in axes(constrained_samples, 2))),
		range(chain)
	)

	combined_chain = hcat(chain, constrained_chain)

	return combined_chain
end

"""
	$(TYPEDSIGNATURES)

Using the formula `f` and data frame `df` fit a one-way ANOVA.
"""
anova_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame, args...; kwargs...) = anova_test(SimpleDataSet(f, df), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

Using the vector `y` and grouping variable `g` fit a one-way ANOVA.
"""
anova_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:Integer}, args...; kwargs...) = anova_test(SimpleDataSet(y, g), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

Using the vector `y` and grouping variable `g` fit a one-way ANOVA.
Here `g` is a vector of UnitRanges where each element indicates the group membership of `y`.
"""
anova_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:UnitRange{<:Integer}}, args...; kwargs...) = anova_test(SimpleDataSet(y, g), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

# Arguments:
- `df` a DataFrame or SimpleDataSet.
- `partition_prior::Union{Nothing, AbstractPartitionDistribution}`, either nothing (i.e., fit the full model) or a subtype of `AbstractPartitionDistribution`.

# Keyword arguments
- `spl`, overwrite the sampling algorithm passed to Turing. It's best to look at the source code for the parameter names and so on.
- `mcmc_settings`, settings for sampling.
- `modeltype`, `:old` indicated all parameters are sampled whereas `reduced` indicates only `g` and the partitions are sampled using an integrated representation of the posterior.
- `rng` a random number generator.
"""
function anova_test(
	df::Union{SimpleDataSet, DataFrames.DataFrame},
	partition_prior::Union{Nothing, AbstractPartitionDistribution},
	;
	spl = :custom,
	mcmc_settings::MCMCSettings = MCMCSettings(),
	modeltype::Symbol = :old,
	rng = Random.GLOBAL_RNG
)
	# TODO: dispatch based on Nothing vs AbstractPartitionDistribution?
	if isnothing(partition_prior)
		return fit_full_model(df;                spl = spl, mcmc_settings = mcmc_settings, modeltype = modeltype, rng = rng)
	else
		return fit_eq_model(df, partition_prior; spl = spl, mcmc_settings = mcmc_settings, eq_model = modeltype, rng = rng)
	end
end
