using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		NamedArrays			as NA,
		GLM

import DataFrames: DataFrame
import StatsModels: @formula
import Random
import ProgressMeter

# include("simulations/plotFunctions.jl") # <- unused?
include("simulations/helpersTuring.jl")
include("simulations/silentGeneratedQuantities.jl")
include("simulations/meansModel_Functions.jl")
include("simulations/limitedLogger.jl")
include("simulations/customHMCAdaptation.jl")

make_relative!(x) = x ./= minimum(x)
make_relative(x) = make_relative!(copy(x))
function summarize_timings(x, relative::Bool = false)
	mat = hcat(
		sum(timings, dims = 2),
		mean(timings, dims = 2),
		median(timings, dims = 2),
		minimum(timings, dims = 2),
		maximum(timings, dims = 2)
	)
	if relative
		for i in axes(mat, 2)
			make_relative!(view(mat, :, i))
		end
	end
	NA.NamedArray(mat,
		(
			"timing " .* string.(axes(mat, 1)),
			[
				"sum",
				"mean",
				"median",
				"minimum",
				"maximum"
			]
		)
	)
end

# TODO: use right-haar prior on variance by looking at Flat(): https://github.com/TuringLang/Turing.jl/blob/57e0512e3b63e4622c755dc6d933b51eb76d4483/src/stdlib/distributions.jl#L1-L10

@model function test_model(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# pass the prior like this to the model?
	partition ~ partition_prior

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)
	# θ_r ~ filldist(Normal(), n_groups - 1)
	θ_r ~ MvNormal(n_groups - 1, 1.0)
	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + sqrt(σ²) * θ_cs[i], σ², obs_n[i])
	end
	return (θ_cs, )

end

# MUCH slower!
@model function test_model2(y, g_idx, n_groups, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# pass the prior like this to the model?
	partition ~ partition_prior

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)
	# θ_r ~ filldist(Normal(), n_groups - 1)
	θ_r ~ MvNormal(n_groups - 1, 1.0)
	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	y ~ MvNormal(μ_grand .+ sqrt(σ²) * θ_cs[g_idx], sqrt(σ²))
	# for i in 1:n_groups
	# 	obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + sqrt(σ²) * θ_cs[i], σ², obs_n[i])
	# end
	return (θ_cs, )

end

function getQ_Stan(K::Integer)
	# Stan approach: https://mc-stan.org/docs/2_18/stan-users-guide/parameterizing-centered-vectors.html
	A = Matrix(LA.Diagonal(ones(K)))
	for i in 1:K-1
		A[K, i] = -1;
	end
	A[K,K] = 0;
	return LA.qr(A).Q[:, 1:K-1]
end

K = n_groups
Q1 = getQ(K)
Q2 = getQ_Stan(K)

u  = randn(K-1)
u0 = Q1 * u
u1 = Q2 * u

@assert isapprox(sum(u0), 0.0; atol = 1e-6)
@assert isapprox(sum(u1), 0.0; atol = 1e-6)


n_groups = 5
n_obs_per_group = 10_000
# true_model = collect(1:n_groups)
true_model = reduce_model(sample_true_model(n_groups, 50))
true_θ = get_θ(0.2, true_model)

y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model, 0.0, 1.0);

obs_mean, obs_var, obs_n = get_suff_stats(df)
Q_Stan = getQ_Stan(n_groups)
# mod2 = test_model2(y, df[!, :g], n_groups, Q, UniformMvUrnDistribution(n_groups))
mod2 = test_model(obs_mean, obs_var, obs_n, Q_Stan, UniformMvUrnDistribution(n_groups))

mcmc_iterations = 20_000
mcmc_burnin     = 5_000
nsim = 20
timings = Matrix{Float64}(undef, 4, nsim)

Turing.setprogress!(false)
Logging.with_logger(limited_warning_logger(2)) do

	ProgressMeter.@showprogress for i in axes(timings, 2)

		resGibbsStuff1 = fit_model(df, mcmc_iterations = mcmc_iterations, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = true, hmc_stepsize = 0.0);
		resGibbsStuff2 = fit_model(df, mcmc_iterations = mcmc_iterations, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = true, hmc_stepsize = 0.0, model = test_model);
		resGibbsStuff3 = fit_model(df, mcmc_iterations = mcmc_iterations, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = true, hmc_stepsize = 0.0, model = mod2);
		resGibbsStuff4 = fit_model(df, mcmc_iterations = mcmc_iterations, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = true, hmc_stepsize = 0.0, model = mod2, custom_hmc_adaptation = true);


		timings[1, i] = MCMCChains.wall_duration(resGibbsStuff1[3])
		timings[2, i] = MCMCChains.wall_duration(resGibbsStuff2[3])
		timings[3, i] = MCMCChains.wall_duration(resGibbsStuff3[3])
		timings[4, i] = MCMCChains.wall_duration(resGibbsStuff4[3])
	end
end

summarize_timings(timings)
summarize_timings(timings, true)

partition_prior = UniformMvUrnDistribution(n_groups)
model = resGibbsStuff2[4]
spl0  = resGibbsStuff2[5]
init_theta = get_initial_values(model, obs_mean, obs_var, obs_n, Q, partition_prior)
vi    = VarInfo(model)
DynamicPPL.initialize_parameters!(vi, init_theta, spl)
spl   = Sampler(spl0, model)

custom_hmc_adaptation(model, spl, vi)

# ee = Sampler(spl, model)
# ee.alg.algs

# @edit sample(model, spl, 10)

# DynamicPPL.initialstep(Random.GLOBAL_RNG, model, spl, vi)

custom_hmc_adaptation(model, spl0, init_theta; max_n_iters = 100)

function custom_hmc_adaptation(model, spl0, init_theta; max_n_iters = 100, kwargs...)

	vi    = VarInfo(model)
	spl   = Sampler(spl0, model)
	DynamicPPL.initialize_parameters!(vi, init_theta, spl)

	algs = spl.alg.algs
	i = 0
	samplers = map(algs) do alg
		i += 1
		if i == 1
			prev_alg = algs[end]
		else
			prev_alg = algs[i-1]
		end
		rerun = Turing.Inference.gibbs_rerun(prev_alg, alg)
		selector = DynamicPPL.Selector(Symbol(typeof(alg)), rerun)
		Sampler(alg, model, selector)
	end

	# Add Gibbs to gids for all variables.
	for sym in keys(vi.metadata)
		vns = getfield(vi.metadata, sym).vns

		for vn in vns
			# update the gid for the Gibbs sampler
			DynamicPPL.updategid!(vi, vn, spl)

			# try to store each subsampler's gid in the VarInfo
			for local_spl in samplers
				DynamicPPL.updategid!(vi, vn, local_spl)
			end
		end
	end

	rng = Random.GLOBAL_RNG
	spl_hmc = samplers[2]

	link!(vi, spl_hmc)
	model(rng, vi, spl_hmc)

	# Extract parameters.
	theta = vi[spl_hmc]

	# Create a Hamiltonian.
	metricT = Turing.Inference.getmetricT(spl_hmc.alg)
	metric = metricT(length(theta))
	∂logπ∂θ = Turing.Inference.gen_∂logπ∂θ(vi, spl_hmc, model)
	logπ = Turing.Inference.gen_logπ(vi, spl_hmc, model)
	hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

	ϵ, _, i1 = find_good_stepsize2(rng, hamiltonian, theta; max_n_iters = max_n_iters, kwargs...)
	safety = 1
	maxsafety = 50
	ϵ_min = ϵ
	while i1 == max_n_iters && safety < maxsafety
		ϵ, _, i1 = find_good_stepsize2(rng, hamiltonian, theta; max_n_iters = max_n_iters, kwargs...)
		safety += 1
		if ϵ < ϵ_min
			ϵ_min = ϵ
		end
	end
	if safety == maxsafety
		ϵ = min_ϵ
	end
	@info "Found initial step size" ϵ safety i1
	return ϵ
end

function find_good_stepsize2(
    rng::Random.AbstractRNG,
    h::AHMC.Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100,
	DEBUG::Bool = false
) where {T<:Real}
    # Initialize searching parameters
    ϵ′ = ϵ = T(0.1)
    a_min, a_cross, a_max = T(0.25), T(0.5), T(0.75) # minimal, crossing, maximal accept ratio
    d = T(2.0)
    # Create starting phase point
    r = rand(rng, h.metric) # sample momentum variable
    z = AHMC.phasepoint(h, θ, r)
    H = AHMC.energy(z)

    # Make a proposal phase point to decide direction

    z′, H′ = AHMC.A(h, z, ϵ)
    ΔH = H - H′ # compute the energy difference; `exp(ΔH)` is the MH accept ratio
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
	i0 = 0
    for _ = 1:max_n_iters
		i0 += 1
        # `direction` being  `1` means MH ratio too high
        #     - this means our step size is too small, thus we increase
        # `direction` being `-1` means MH ratio too small
        #     - this means our step szie is too large, thus we decrease
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        z′, H′ = AHMC.A(h, z, ϵ)
        ΔH = H - H′
        DEBUG && @debug "Crossing step" direction H′ ϵ "α = $(min(1, exp(ΔH)))"
        if (direction == 1) && !(ΔH > log(a_cross))
            break
        elseif (direction == -1) && !(ΔH < log(a_cross))
            break
        else
            ϵ = ϵ′
        end
    end
    # Note after the for loop,
    # `ϵ` and `ϵ′` are the two neighbour step sizes across `a_cross`.

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method

    ϵ, ϵ′ = ϵ < ϵ′ ? (ϵ, ϵ′) : (ϵ′, ϵ)  # ensure ϵ < ϵ′;
    # Here we want to use a value between these two given the
    # criteria that this value also gives us a MH ratio between `a_min` and `a_max`.
    # This condition is quite mild and only intended to avoid cases where
    # the middle value of `ϵ` and `ϵ′` is too extreme.
	# return (h, z, AHMC.middle(ϵ, ϵ′))
	i1 = 0
    for _ = 1:max_n_iters
		i1 += 1

        ϵ_mid = AHMC.middle(ϵ, ϵ′)
        z′, H′ = AHMC.A(h, z, ϵ_mid)
        ΔH = H - H′
        DEBUG && @debug "Bisection step" H′ ϵ_mid "α = $(min(1, exp(ΔH)))"
        if (exp(ΔH) > a_max)
            ϵ = ϵ_mid
        elseif (exp(ΔH) < a_min)
            ϵ′ = ϵ_mid
        else
            ϵ = ϵ_mid
            break
        end
    end
    return (ϵ, i0, i1)
end

ϵ = custom_hmc_adaptation(model, spl, vi; max_n_iters = 100)

rng = Random.GLOBAL_RNG
spl2 = local_spl
# Transform the samples to unconstrained space and compute the joint log probability.
link!(vi, spl)
model(rng, vi, spl)

# Extract parameters.
theta = vi[spl2]

# Create a Hamiltonian.
metricT = Turing.Inference.getmetricT(spl2.alg)
metric = metricT(length(theta))
∂logπ∂θ = Turing.Inference.gen_∂logπ∂θ(vi, spl2, model)
logπ = Turing.Inference.gen_logπ(vi, spl2, model)
hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

ϵ = AHMC.find_good_stepsize(hamiltonian, theta)
# Compute phase point z.
# z = AHMC.phasepoint(rng, theta, hamiltonian)


# If no initial parameters are provided, resample until the log probability
# and its gradient are finite.
if init_params === nothing
	while !isfinite(z)
		model(rng, vi, SampleFromUniform())
		link!(vi, spl)
		theta = vi[spl]

		hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
		z = AHMC.phasepoint(rng, theta, hamiltonian)
	end
end

# Cache current log density.
log_density_old = getlogp(vi)

# Find good eps if not provided one
if iszero(spl.alg.ϵ)
	ϵ = AHMC.find_good_stepsize(hamiltonian, theta)
	@info "Found initial step size" ϵ
else
	ϵ = spl.alg.ϵ
end

AHMC.find_good_stepsize(hamiltonian, theta)
@enter find_good_stepsize2(rng, hamiltonian, theta)


