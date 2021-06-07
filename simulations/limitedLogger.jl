# import Logging

# # TODO: look at LoggingExtras

# struct testLogger <: Logging.AbstractLogger
# 	min_level::Base.CoreLogging.LogLevel
# 	max_warn_count::Int
# 	_current_warn_count::Vector{Int}
# 	function testLogger(max_warn_count::Int) 
# 		new(max_warn_count, [0])
# 	end
# end

# function Logging.handle_message(logger::testLogger, level::Logging.LogLevel, args...; kwargs...)
# 	Logging.handle_message(Logging.global_logger(), level, args...; kwargs...)
# end

# function Logging.shouldlog(logger::testLogger, level, _module, group, id)
# 	if level == Logging.Warn
# 		logger._current_warn_count[1] += 1
#         if logger._current_warn_count[1] == logger.max_warn_count
# 			# TODO: this should always be shown...
#             @warn maximum warning limit reached
#         end
# 	end
# 	return (level == Logging.Warn && logger._current_warn_count[1] <= logger.max_warn_count) || level != Logging.Warn
# end

# Logging.min_enabled_level(logger::testLogger) = Logging.global_logger().min_level

# # only shown three times
# testLogger_inst = testLogger(3)
# Logging.global_logger(testLogger_inst)
# for i in 1:15
# 	@warn "I'm shown?"
# end

# using 	LoggingExtras,
# 		Logging

# function yodawg_filter(log_args)
# 	@show log_args
# 	startswith(log_args.message, "Yo Dawg!")
# end

# filtered_logger = LoggingExtras.ActiveFilteredLogger(yodawg_filter, Logging.global_logger());

# with_logger(filtered_logger) do
#     @info "Boring message"
#     @warn "Yo Dawg! it is bad"
#     @info "Another boring message"
#     @info "Yo Dawg! it is all good"
# end

import 	Logging,
		LoggingExtras

function limited_warning_logger(max_warnings::Int)
    history = Dict{Base.CoreLogging.LogLevel, Int}()
    # We are going to use a closure
    LoggingExtras.EarlyFilteredLogger(Logging.global_logger()) do log
		# @show log
        if !haskey(history, log.level)
            # then we will log it, and update record of when we did
            history[log.level] = 1
            return true
        else
			history[log.level] += 1
			if log.level == Logging.Warn && history[log.level] >= max_warnings
				if history[log.level] == max_warnings 
					@info "maximum warnings reached"
					return true
				else
					return false
				end
			else
				return true
			end
        end
    end
end

# Logging.with_logger(limited_warning_logger(5)) do
#     for ii in 1:10
#         @warn "It happened" ii
#     end
# end