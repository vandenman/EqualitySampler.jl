function get_rhats(chn::MCMCChains.Chains)
	MCMCChains.summarystats(chn).nt.rhat
end

function get_ess_per_sec(chn::MCMCChains.Chains)
	MCMCChains.summarystats(chn).nt.ess_per_sec
end


validate_rhat(chn::MCMCChains.Chains, max_rhat = 1.05) = validate_rhat(get_rhats(chn), max_rhat)

function validate_rhat(rhats::Vector{Float64}, max_rhat = 1.05)
	for rhat in rhats
		(isnan(rhat) || rhat > max_rhat) && return false
	end
	return true
end


mutable struct WarningCountingLogger <: Logging.AbstractLogger
	count::Int
	function WarningCountingLogger()
		new(0)
	end
end

Logging.shouldlog(::WarningCountingLogger, args...) = Logging.shouldlog(Logging.global_logger(), args...)
Logging.min_enabled_level(::WarningCountingLogger) = Logging.min_enabled_level(Logging.global_logger())

function Logging.handle_message(logger::WarningCountingLogger, level, message, _module, group, id, filepath, line; kwargs...)
	if level == Logging.Warn && group == :hamiltonian && endswith(filepath, "hamiltonian.jl")#, 47
		# @info "incrementing count"
		logger.count += 1
	else
		Logging.handle_message(Logging.global_logger(), level, message, _module, group, id, filepath, line; kwargs...)
	end
end

function brute_force_ϵ(model, discrete_sampler::Symbol = :custom; dummy_draws = 500, max_attempts = 10, kwargs...)

	wcLogger = WarningCountingLogger()
	# power = 4
	# initial_ϵ = 1 / 2^power

	initial_ϵ = 0.0
	powers = range(-3, stop = -8, length = max_attempts)
	i_found = 0
	rhats_list = Vector{Vector{Float64}}()
	# for _ in 1:max_attempts
	for i in eachindex(powers)
		initial_ϵ = 10 ^ powers[i]
		@debug "brute_force_ϵ" i, initial_ϵ
		samps = Logging.with_logger(wcLogger) do
			AbstractMCMC.sample(model, get_sampler(model, discrete_sampler, initial_ϵ), dummy_draws; progress=false, kwargs...)::MCMCChains.Chains
		end
		# @show wcLogger.count power
		# the maximum rhat doesn't really matter for now (there should be too few samples anyway), as long as there aren't any NaNs, which imply that the sampler got stuck
		rhats = get_rhats(samps)
		push!(rhats_list, rhats)
		if wcLogger.count < 10 && validate_rhat(rhats, 1.3)
			i_found = i
			break
		end
		wcLogger.count = 0

	end
	if iszero(i_found)
		lowest_mean_rhat, i_min_rhats = findmin(mean, rhats_list)
		initial_ϵ = 10 ^ powers[i_min_rhats]
		@info "brute_force_ϵ did not converge, selecting ϵ based on the lowest mean rhat: " ϵ=initial_ϵ powers[i_min_rhats] i_min_rhats lowest_mean_rhat
	else
		@info "brute_force_ϵ converged, using " ϵ=initial_ϵ powers[i_found] i_found
	end
	return initial_ϵ

end
