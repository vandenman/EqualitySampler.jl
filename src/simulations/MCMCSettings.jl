"""
	$(TYPEDEF)
	- iterations::T, the number of post warmup MCMC samples.
	- burnin::T, the the number of MCMC samples to discard.
	- chains::T, the the number of MCMC chains.
	- thinning::T, retain only every nth MCMC sample, where n is specified by `thinning`.

	Chains are run in serial by specifying `AbstractMCMC.MCMCSerial` or in parallel by specifying either `AbstractMCMC.MCMCDistributed` or `AbstractMCMC.MCMCThreads`.
"""
struct MCMCSettings{T<:Integer, U<:AbstractMCMC.AbstractMCMCEnsemble}

	iterations::T
	burnin::T
	chains::T
	thinning::T

	"""
	$(TYPEDSIGNATURES)

	Internal constructor.
	"""
	function MCMCSettings(iterations::T, burnin::T, chains::T, thinning::T, ::U; check_args::Bool=true) where {T<:Integer, U<:AbstractMCMC.AbstractMCMCEnsemble}
		if check_args
			iterations <= zero(T)	&& throw(DomainError(iterations, "iterations must be positive!"))
			burnin <= zero(T)		&& throw(DomainError(burnin, "burnin must be positive!"))
			chains <= zero(T)		&& throw(DomainError(chains, "chains must be positive!"))
			thinning <= zero(T)		&& throw(DomainError(thinning, "chains must be positive!"))
		end
		return new{T, U}(iterations, burnin, chains, thinning)
	end
end

"""
$(TYPEDSIGNATURES)
"""
function MCMCSettings(iterations::Integer, burnin::Integer, chains::Integer, thinning::Integer, parallel::AbstractMCMC.AbstractMCMCEnsemble=AbstractMCMC.MCMCSerial())
	iterations, burnin, chains, thinning = promote(iterations, burnin, chains, thinning)
	return MCMCSettings(iterations, burnin, chains, thinning, parallel)
end

"""
$(TYPEDSIGNATURES)
"""
function MCMCSettings(;iterations::Integer=10_000, burnin::Integer=1_000, chains::Integer=3, thinning::Integer=1, parallel::AbstractMCMC.AbstractMCMCEnsemble=AbstractMCMC.MCMCSerial())
	iterations, burnin, chains, thinning = promote(iterations, burnin, chains, thinning)
	return MCMCSettings(iterations, burnin, chains, thinning, parallel)
end

function sample_model(model, spl, settings::MCMCSettings{T, U}; kwargs...) where {T, U<:AbstractMCMC.AbstractMCMCEnsemble}
	AbstractMCMC.sample(model, spl, U(), settings.iterations, settings.chains; discard_initial = settings.burnin, thinning = settings.thinning)
end