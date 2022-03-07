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


function simulate_data_one_way_anova(
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
	D = Distributions.MvNormal(μ .+ σ .* view(θc, g_big), σ)
	y = rand(D)

	dat = SimpleDataSet(y, g)

	true_values = TrueValues(μ, σ, θc, partition)

	return (data=dat, distribution=D, true_values=true_values)

end

function fit_lm(X::AbstractMatrix, y::AbstractVector)
	fit = GLM.lm(X, y)::GLM.LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}
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
	fit = GLM.lm(formula, df)::StatsModels.TableRegressionModel{GLM.LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}, Matrix{Float64}}
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
	obs_mean = [mean(df.y[idx]) for idx in df.g]
	obs_var  = [var(df.y[idx]) for idx in df.g]
	obs_n    = length.(df.g)
	return obs_mean, obs_var, obs_n
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

function get_starting_values(df)

	coefs, cis, fit = fit_lm(df)

	adj_mat = [
		Int(cis[i, 1] <= cis[j, 2] && cis[j, 1] <= cis[i, 2])
		for i in axes(cis, 1), j in axes(cis, 1)
	]

	n_groups = size(cis, 1)
	Q = getQ_Rouder(n_groups)

	y = df.y
	partition_start	= map(x->findfirst(isone, x), eachcol(adj_mat))::Vector{Int}
	n_partitions	= length(unique(partition_start))
	σ_start			= var(GLM.residuals(fit))
	μ_start			= mean(y)
	θ_c_start		= isone(n_partitions) ? zeros(n_groups) : average_equality_constraints(coefs, partition_start)
	# this is faster than Q \ θ_c_start which uses a qr decomposition
	θ_start_0		= LinearAlgebra.pinv(Q) * θ_c_start
	g_start			= isone(n_partitions) ? 1.0 : var(θ_start_0)
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

# function plot_retrieval(true_values, estimated_values)
# 	p = Plots.plot(legend=false, xlab = "True value", ylab = "Posterior mean")
# 	Plots.abline!(p, 1, 0)
# 	scatter!(p, true_values, estimated_values)
# end

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
	return vcat(
		partition,
		μ,
		σ²,
		g,
		θ_r
	)
end

DynamicPPL.@model function one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition = nothing, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	# improper priors on grand mean and variance
	μ_grand 		~ Turing.Flat()
	σ² 				~ JeffreysPriorVariance()

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)

	θ_r ~ Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1)))

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in eachindex(obs_mean)
		# TODO: how to access this function?
		Turing.@addlogprob! EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end

	return θ_cs

end

DynamicPPL.@model function one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	partition ~ partition_prior
	DynamicPPL.@submodel prefix="one_way_anova_mv_ss_submodel" θ_cs = one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition, T)
	return θ_cs

end

function prep_model_arguments(df)
	obs_mean, obs_var, obs_n = get_suff_stats(df)
	n_groups = length(obs_mean)
	Q = getQ_Rouder(n_groups)
	return obs_mean, obs_var, obs_n, Q
end

function fit_full_model(
		df
		;
		spl = nothing,
		mcmc_settings::MCMCSettings = MCMCSettings()
	)

	obs_mean, obs_var, obs_n, Q = prep_model_arguments(df)
	model = one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q)
	mcmc_sampler = isnothing(spl) ? get_sampler(model) : spl

	chain = sample_model(model, mcmc_sampler, mcmc_settings)::MCMCChains.Chains
	return combine_chain_with_generated_quantities(model, chain, "θ_cs")

end

function fit_eq_model(
		df,
		partition_prior::EqualitySampler.AbstractMvUrnDistribution
		;
		spl = nothing,
		mcmc_settings::MCMCSettings = MCMCSettings()
	)

	obs_mean, obs_var, obs_n, Q = prep_model_arguments(df)
	model = one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior)
	starting_values = get_starting_values(df)
	init_params = get_init_params(starting_values...)

	if isnothing(spl)
		ϵ = brute_force_ϵ(model; init_params = init_params)
		mcmc_sampler = get_sampler(model, ϵ)
	else
		mcmc_sampler = spl
	end

	chain = sample_model(model, mcmc_sampler, mcmc_settings)::MCMCChains.Chains

	return combine_chain_with_generated_quantities(model, chain, "θ_cs")
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

	constrained_chain = MCMCChains.setrange(
		MCMCChains.Chains(constrained_samples, collect(Symbol(parameter_name, "["* string(i) * "]") for i in axes(constrained_samples, 2))),
		range(chain)
	)

	combined_chain = hcat(chain, constrained_chain)

	return combined_chain
end

"""
	$(TYPEDSIGNATURES)

Using the formula `f` and data frame `df` fit a one-way ANOVA.
If `partition_prior` is specified as a keyword argument then equalities among the levels of the grouping variable are sampled.
"""
anova_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame, args...; kwargs...) = anova_test(SimpleDataSet(f, df), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

Using the vector `y` and grouping variable `g` fit a one-way ANOVA.
If `partition_prior` is specified as a keyword argument then equalities among the levels of the grouping variable are sampled.
"""
anova_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:Integer}, args...; kwargs...) = anova_test(SimpleDataSet(y, g), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

Using the vector `y` and grouping variable `g` fit a one-way ANOVA.
Here `g` is a vector of UnitRanges where each element indicates the group membership of `y`.

If `partition_prior` is specified as a keyword argument then equalities among the levels of the grouping variable are sampled.
"""
anova_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:UnitRange{<:Integer}}, args...; kwargs...) = anova_test(SimpleDataSet(y, g), args...; kwargs...)

"""
	$(TYPEDSIGNATURES)

positional arguments:
- `partition_prior::Union{Nothing, AbstractMvUrnDistribution}`, either nothing (i.e., fit the full model) or a subtype of `AbstractMvUrnDistribution`.

keyword arguments:

- `spl = nothing`, a custom Turing sampler. `nothing` implies a default sampler is used which for the the full model defaults to `NUTS()` and for the equality selector to a Gibbs sampler with HMC for continuous parameters and Metropolis for discrete parameters.
- `mcmc_iterations::Integer = 10_000`, the number of post warmup MCMC samples.
- `mcmc_burnin::Integer = 1_000`, the number of initial MCMC samples to discard.
- `mcmc_chains::Integer = 3`, the number of MCMC chains to sample.
- `parallel::AbstractMCMC.AbstractMCMCEnsemble = Turing.MCMCSerial`, should the chains be sampled in parallel? Possible values are, No parallization (`Turing.MCMCSerial``), paralellization through multiple julia processes (`Turing.MCMCDistributed`), and paralellization through multithreading (`Turing.MCMCThreads`).
"""
function anova_test(
	df::Union{SimpleDataSet, DataFrames.DataFrame},
	partition_prior::Union{Nothing, AbstractMvUrnDistribution},
	;
	spl = nothing,
	mcmc_settings::MCMCSettings = MCMCSettings()
)
	# TODO: dispatch based on Nothing vs AbstractMvUrnDistribution?
	if isnothing(partition_prior)
		return fit_full_model(df;                spl = spl, mcmc_settings = mcmc_settings)
	else
		return fit_eq_model(df, partition_prior; spl = spl, mcmc_settings = mcmc_settings)
	end
end

# using EqualitySampler
# y = randn(100)
# g = rand(1:5, 100)
# df = EqualitySampler.Simulations.SimpleDataSet(y, g)
# fit0 = anova_test(y, g; partition_prior=nothing)
# fit1 = anova_test(df; partition_prior=nothing)

# @edit anova_test(df; partition_prior=nothing)