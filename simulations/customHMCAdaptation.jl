import	Random, Turing,
		AdvancedHMC		as AHMC

# Function Piracy to inject a more robust method for finding an initial value
# tab indented lines are modified, space indented lines are not.
# based on Turing v0.19.2
# https://github.com/TuringLang/Turing.jl/blob/5c4b4d5e4715f82c6b53327c051c54e79b8d0079/src/inference/hmc.jl#L143-L221
function AHMC.find_good_stepsize(
	rng::Random.AbstractRNG,
	h::AHMC.Hamiltonian,
	θ::AbstractVector{T};
	max_n_iters::Int=100,
	DEBUG::Bool = false
) where {T<:Real}

	# max_n_iters = 100
	ϵ, _, i1 = find_good_stepsize2_inner(rng, h, θ; max_n_iters = max_n_iters, DEBUG = DEBUG)
	safety = 1
	maxsafety = 100
	ϵ_min = ϵ
	while i1 == max_n_iters && safety < maxsafety
		ϵ, _, i1 = find_good_stepsize2_inner(rng, h, θ; max_n_iters = max_n_iters, DEBUG = DEBUG)
		safety += 1
		if ϵ < ϵ_min
			ϵ_min = ϵ
		end
	end
	if safety == maxsafety
		ϵ = ϵ_min
	end
	@info "Found initial step size" ϵ safety i1
	return ϵ
end

function find_good_stepsize2_inner(
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


# Function Piracy to inject a more robust method for finding an initial value
# tab indented lines are modified, space indented lines are not.
# based on Turing v0.19.2
# https://github.com/TuringLang/Turing.jl/blob/5c4b4d5e4715f82c6b53327c051c54e79b8d0079/src/inference/hmc.jl#L143-L221
# this approach is deprecated since it's safer to patch AHMC than DynamicPPL, but this functions is useful to verify that init_params are actually used
# function DynamicPPL.initialstep(
#     rng::Random.AbstractRNG,
#     model::AbstractMCMC.AbstractModel,
#     spl::DynamicPPL.Sampler{<:Turing.Inference.Hamiltonian},
#     vi::DynamicPPL.AbstractVarInfo;
#     init_params=nothing,
#     nadapts=0,
#     kwargs...
# )

# 	@debug "Pirated DynamicPPL.initialstep"
# 	@debug "constrained θ" vi[spl]

#     # Transform the samples to unconstrained space and compute the joint log probability.
#     DynamicPPL.link!(vi, spl)
#     vi = last(DynamicPPL.evaluate!!(model, rng, vi, spl))

#     # Extract parameters.
#     theta = vi[spl]
# 	@debug "unconstrained θ" theta

#     # Create a Hamiltonian.
#     metricT = Turing.Inference.getmetricT(spl.alg)
#     metric = metricT(length(theta))
#     ∂logπ∂θ = Turing.Inference.gen_∂logπ∂θ(vi, spl, model)
#     logπ = Turing.Inference.gen_logπ(vi, spl, model)
#     hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

#     # Compute phase point z.
#     z = AHMC.phasepoint(rng, theta, hamiltonian)

#     # If no initial parameters are provided, resample until the log probability
#     # and its gradient are finite.
#     if init_params === nothing
#         while !isfinite(z)
#             vi = last(DynamicPPL.evaluate!!(model, rng, vi, SampleFromUniform()))
#             DynamicPPL.link!(vi, spl)
#             theta = vi[spl]

#             hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
#             z = AHMC.phasepoint(rng, theta, hamiltonian)
#         end
#     end

#     # Cache current log density.
#     log_density_old = getlogp(vi)

#     # Find good eps if not provided one
#     if iszero(spl.alg.ϵ)
# 		max_n_iters = 100 # no. steps taken by find_good_stepsize2_inner
# 		maxsafety = 100  # no. times we repeat find_good_stepsize2_inner if it fails to converge
# 		ϵ, _, i1 = find_good_stepsize2_inner(rng, hamiltonian, theta; max_n_iters = max_n_iters)
# 		safety = 1
# 		ϵ_min = ϵ
# 		while i1 == max_n_iters && safety < maxsafety
# 			ϵ, _, i1 = find_good_stepsize2_inner(rng, hamiltonian, theta; max_n_iters = max_n_iters)
# 			safety += 1
# 			if ϵ < ϵ_min
# 				ϵ_min = ϵ
# 			end
# 		end
# 		if safety == maxsafety
# 			ϵ = ϵ_min
# 		end
# 		@info "Found initial step size" ϵ safety i1

#     else
#         ϵ = spl.alg.ϵ
#     end

#     # Generate a kernel.
#     kernel = Turing.Inference.make_ahmc_kernel(spl.alg, ϵ)

#     # Create initial transition and state.
#     # Already perform one step since otherwise we don't get any statistics.
#     t = AHMC.transition(rng, hamiltonian, kernel, z)

#     # Adaptation
#     adaptor = Turing.Inference.AHMCAdaptor(spl.alg, hamiltonian.metric; ϵ=ϵ)
#     if spl.alg isa Turing.Inference.AdaptiveHamiltonian
#         hamiltonian, kernel, _ =
#             AHMC.adapt!(hamiltonian, kernel, adaptor,
#                         1, nadapts, t.z.θ, t.stat.acceptance_rate)
#     end

#     # Update `vi` based on acceptance
#     if t.stat.is_accept
#         vi[spl] = t.z.θ
#         DynamicPPL.setlogp!(vi, t.stat.log_density)
#     else
#         vi[spl] = theta
#         DynamicPPL.setlogp!(vi, log_density_old)
#     end

#     transition = Turing.Inference.HMCTransition(vi, t)
#     state = Turing.Inference.HMCState(vi, 1, kernel, hamiltonian, t.z, adaptor)

#     return transition, state
# end
