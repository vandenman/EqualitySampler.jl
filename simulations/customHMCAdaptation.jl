import	AdvancedHMC		as AHMC

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
	maxsafety = 100
	ϵ_min = ϵ
	while i1 == max_n_iters && safety < maxsafety
		ϵ, _, i1 = find_good_stepsize2(rng, hamiltonian, theta; max_n_iters = max_n_iters, kwargs...)
		safety += 1
		if ϵ < ϵ_min
			ϵ_min = ϵ
		end
	end
	if safety == maxsafety
		ϵ = ϵ_min
	end
	@info "Found initial step size" ϵ safety i1
	return ϵ / 2
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
