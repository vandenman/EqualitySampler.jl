import Logging, MCMCChains
function get_rhats(chn::MCMCChains.Chains)
	MCMCChains.summarystats(chn).nt.rhat
end

function validate_rhat(chn::MCMCChains.Chains, max_rhat = 1.05)
	rhats = get_rhats(chn)
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

function brute_force_ϵ(model; dummy_draws = 50, max_attempts = 25, kwargs...)

	wcLogger = WarningCountingLogger()
	power = 4
	initial_ϵ = 1 / 2^power
	for _ in 1:max_attempts
		samps = Logging.with_logger(wcLogger) do
			sample(model, get_sampler(model, initial_ϵ), dummy_draws; progress=false, kwargs...)::MCMCChains.Chains
		end
		# @show wcLogger.count power
		# the maximum rhat doesn't really matter for now (there should be too few samples anyway), as long as there aren't any NaNs, which imply that the sampler got stuck
		wcLogger.count < 10 && validate_rhat(samps, 2) && break
		wcLogger.count = 0
		power += 1
		initial_ϵ = 1 / 2^power
	end
	@info "found an ϵ of " initial_ϵ power

	return initial_ϵ
end
