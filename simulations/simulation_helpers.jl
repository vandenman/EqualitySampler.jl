# simulation helpers for multipleComparisonPlot (Figure 4).jl and ...

import EqualitySampler, EqualitySampler.Simulations, MCMCChains, ProgressMeter, JLD2, Turing, DataFrames
# stdlib
import Logging, Random, Statistics

#region simulation functions
function instantiate_prior(symbol::Symbol, k::Integer)
	# this works nicely with jld2 but it's not type stable

	symbol == :uniform				&&	return EqualitySampler.UniformMvUrnDistribution(k)
	symbol == :BetaBinomial11		&&	return EqualitySampler.BetaBinomialMvUrnDistribution(k, 1.0, 1.0)
	symbol == :BetaBinomialk1		&&	return EqualitySampler.BetaBinomialMvUrnDistribution(k, k, 1.0)
	symbol == :BetaBinomial1k		&&	return EqualitySampler.BetaBinomialMvUrnDistribution(k, 1.0, k)
	symbol == :BetaBinomial1binomk2	&&	return EqualitySampler.BetaBinomialMvUrnDistribution(k, 1.0, binomial(k, 2))
	symbol == :DirichletProcess0_5	&&	return EqualitySampler.DirichletProcessMvUrnDistribution(k, 0.5)
	symbol == :DirichletProcess1_0	&&	return EqualitySampler.DirichletProcessMvUrnDistribution(k, 1.0)
	symbol == :DirichletProcess2_0	&&	return EqualitySampler.DirichletProcessMvUrnDistribution(k, 2.0)
	# symbol == :DirichletProcessGP	&&
	return EqualitySampler.DirichletProcessMvUrnDistribution(k, :Gopalan_Berry)

end

function get_priors()
	return (
		:uniform,
		:BetaBinomial11,
		:BetaBinomialk1,
		:BetaBinomial1k,
		:BetaBinomial1binomk2,
		:DirichletProcess0_5,
		:DirichletProcess1_0,
		:DirichletProcess2_0,
		:DirichletProcessGP,
		:Westfall,
		:Westfall_uncorrected
	)
end

function get_reference_and_comparison(hypothesis::Symbol, values_are_log_odds::Bool = false)
	if values_are_log_odds
		reference =  0.0
		comparison = hypothesis === :null ? !isless : isless
	else
		reference =  0.5
		comparison = hypothesis === :null ? isless : !isless
	end
	return reference, comparison
end
get_reference_and_comparison(values_are_log_odds::Bool = false) = values_are_log_odds ? (0.0, !isless) : (0.5, isless)

function any_incorrect(x, true_model::Vector{Int}, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(values_are_log_odds)
	for j in 1:size(x, 1)-1, i in j+1:size(x, 1)
		if (true_model[i] == true_model[j] && comparison(x[i, j], reference)) ||
			(true_model[i] != true_model[j] && !comparison(x[i, j], reference))
			# @show i, j, x[i, j], true_model[i] == true_model[j]
			return true
		end
	end
	return false
end


function any_incorrect(x, hypothesis::Symbol, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(hypothesis, values_are_log_odds)
	for j in 1:size(x, 1)-1, i in j+1:size(x, 1)
		if comparison(x[i, j], reference)
			return true
		end
	end
	return false
end

function prop_incorrect(x, hypothesis::Symbol, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(hypothesis, values_are_log_odds)
	count = 0
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		if comparison(x[i, j], reference)
			count += 1
		end
	end
	return count / (n * (n - 1) ÷ 2)
end

function prop_incorrect(x, true_model::Vector{Int}, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(values_are_log_odds)
	count = 0
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		if (true_model[i] == true_model[j] && comparison(x[i, j], reference)) ||
			(true_model[i] != true_model[j] && !comparison(x[i, j], reference))
			count += 1
		end
	end
	return count / (n * (n - 1) ÷ 2)
end

function prop_correct(x, true_model::Vector{Int}, values_are_log_odds::Bool = false)

	reference, comparison = get_reference_and_comparison(values_are_log_odds)
	count = 0
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		if (true_model[i] == true_model[j] && !comparison(x[i, j], reference)) ||
			(true_model[i] != true_model[j] && comparison(x[i, j], reference))
			count += 1
		end
	end
	return count / (n * (n - 1) ÷ 2)
end


function which_incorrect(x, true_model::Vector{Int}, values_are_log_odds::Bool = false)
	reference, comparison = get_reference_and_comparison(values_are_log_odds)
	res = BitArray(undef, size(x))
	n = size(x, 1)
	for j in 1:n-1, i in j+1:n
		res[j, j] = 0
		if (true_model[i] == true_model[j] && comparison(x[i, j], reference)) ||
			(true_model[i] != true_model[j] && !comparison(x[i, j], reference))
			res[i, j] = res[j, i] = 1
		else
			res[i, j] = res[j, i] = 0
		end
	end
	res[n, n] = 0
	return res
end

function get_resultsdir()
	results_dir = joinpath("simulations", "results_multiplecomparisonsplot_200_8")
	!ispath(results_dir) && mkpath(results_dir)
	return results_dir
end

make_filename(results_dir, r, i, hypothesis) = joinpath(results_dir, "repeat_$(r)_groups_$(i)_H_$(hypothesis).jld2")
make_filename(results_dir; kwargs...) = joinpath(results_dir, kwargs_to_filename(kwargs))
make_filename(results_dir, kwargs::NamedTuple) = joinpath(results_dir, kwargs_to_filename(kwargs))

function kwargs_to_filename(kwargs)
	res = ""
	itr = enumerate(zip(keys(kwargs), values(kwargs)))
	for (i, (k, v)) in itr
		res *= string(k) * "=" * string(v)
		if i != length(itr)
			res *= "_"
		end
	end
	res *= ".jld2"
	return res
end


function get_hyperparams_small()
	n_obs_per_group	= 100
	repeats			= 1:200
	groups			= 2:10
	hypothesis		= (:null, :full)
	offset			= 0.2
	priors 			=  get_priors()
	return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors)
end

function get_hyperparams_big()
	n_obs_per_group = (250, 500, 750, 1_000)
	repeats			= 1:100
	groups			= (5, 9)
	hypothesis		= (:p00, :p25, :p50, :p75, :p100)
	offset			= 0.2
	priors			= get_priors()
	return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors)
end

sample_true_model(hypothesis::Symbol, n_groups::Integer) = sample_true_model(Random.GLOBAL_RNG, hypothesis, n_groups)

function sample_true_model(rng::Random.AbstractRNG, hypothesis::Symbol, n_groups::Integer)
	if hypothesis === :null || hypothesis === :p00
		return fill(1, n_groups)
	elseif hypothesis === :full || hypothesis === :p100
		return collect(1:n_groups)
	else

		# foo(n_groups, percentage) = (n_groups-1) * percentage ÷ 100 + 1
		# percentages = 0:25:100
		# [(i, foo.(i, percentages)) for i in 5:15]
		# [(i, foo.(i, percentages)) for i in (5, 9)]

		# foo2(n_groups, percentage) = (n_groups - 1) * percentage ÷ 100 + 1
		# [(i, foo2.(i, percentages)) for i in 5:15]

		percentage = parse.(Int, view(string(hypothesis), 2:3))
		logpdf_idx_one = (n_groups-1) * percentage ÷ 100 + 1
		logpdf = ntuple(i->log(i==logpdf_idx_one), n_groups)

		return rand(rng, EqualitySampler.CustomInclusionMvUrnDistribution(n_groups, logpdf))

	end
end

function validate_r_hat(chn, tolerance = 1.2)
	rhats = MCMCChains.summarystats(chn).nt.rhat
	any(isnan, rhats) && return true, NaN
	any(>(tolerance), rhats) && return true, Statistics.mean(rhats)
	return false, 0.0
end

run_simulation_small(; kwargs...) = run_simulation_small(get_hyperparams_small()...; kwargs...)
run_simulation_big(; kwargs...)   = run_simulation_small(get_hyperparams_big()...;   kwargs...)

const AbstractVecOrSingle{T} = Union{T, AbstractVector{T}, Tuple{T}, NTuple{N, T} where N}
function run_simulation(n_obs_per_group::AbstractVecOrSingle{Int}, repeats::AbstractVecOrSingle{Int}, groups::AbstractVecOrSingle{Int}, hypotheses::AbstractVecOrSingle{Symbol}, offset::AbstractVecOrSingle{Float64}, priors::AbstractVecOrSingle{Symbol};
	results_dir::String,
	# how often to restart a run when the target rhats are not met
	max_retries::Integer = 10,
	# show additional information?
	verbose::Bool=true)

	# results_dir = "simulations/bigsimulation"
	# n_obs_per_group, repeats, groups, hypotheses, offset, priors = get_hyperparams_big()

	!ispath(results_dir) && mkpath(results_dir)
	sim_opts = Iterators.product(n_obs_per_group, repeats, groups, hypotheses, offset, priors)

	nsim = length(sim_opts)
	nsim_without_priors = length(Iterators.product(n_obs_per_group, repeats, groups, hypotheses, offset))
	sim_opts_with_seed = zip(sim_opts, Iterators.repeat(1:nsim_without_priors, length(priors)))
	# collect(sim_opts_with_seed)[range(25, step=4800, length=10)] # shows the seed is the same for the same priors

	@info "Starting simulation" runs=nsim length(priors) threads = Threads.nthreads()

	# for
	# mcmc_settings = Simulations.MCMCSettings(;iterations = 5_000, burnin = 1_000, chains = 1)
	# for SMC
	mcmc_settings = Simulations.MCMCSettings(;iterations = 10_000, burnin = 1, chains = 1)
	# mcmc_settings = MCMCSettings(;iterations = 200, burnin = 100, chains = 1)

	p = ProgressMeter.Progress(nsim)
	Turing.setprogress!(false)
	Logging.disable_logging(Logging.Warn)

	# separate RNGs per thread
	trngs = [Random.MersenneTwister(i) for i in 1:Threads.nthreads()];
	# ((obs_per_group, r, n_groups, hypothesis, offset, prior), seed) = first(sim_opts_with_seed)
	Threads.@threads for ((obs_per_group, r, n_groups, hypothesis, offset, prior), seed) in collect(sim_opts_with_seed)
	# for (iteration, (r, i, hypothesis)) in enumerate(sim_opts)

		filename = make_filename(results_dir, (;obs_per_group, r, n_groups, hypothesis, offset, prior, seed))
		if !isfile(filename)

			rng = trngs[Threads.threadid()]
			Random.seed!(rng, seed)

			true_model = sample_true_model(rng, hypothesis, n_groups)
			true_θ = Simulations.normalize_θ(offset, true_model)

			data_obj = Simulations.simulate_data_one_way_anova(rng, n_groups, obs_per_group, true_θ)
			dat = data_obj.data

			if prior === :Westfall

				result = Simulations.westfall_test(dat)
				post_probs = result.log_posterior_odds_mat
				rhats_retry = zero(max_retries)

			elseif prior === :Westfall_uncorrected

				result = Simulations.westfall_test(dat)
				post_probs = result.logbf_matrix
				rhats_retry = zero(max_retries)

			else

				# otherwise these are scoped to the loop for retry in 1:max_retries
				local post_probs, rhats_retry

				partition_prior = instantiate_prior(prior, n_groups)
				for retry in 0:max_retries

					Random.seed!(rng, seed + retry)
					chain = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, rng = rng, spl = 0.05 * 0.8^retry)
					# chain = Simulations.anova_test(dat, partition_prior; mcmc_settings = mcmc_settings, rng = rng, spl = Turing.SMC())
					any_bad_rhats, mean_rhat_value = validate_r_hat(chain)
					if any_bad_rhats && retry != max_retries
						verbose && @error "This run had a bad r-hat:" settings = (;obs_per_group, r, n_groups, hypothesis, offset, prior, any_bad_rhats, mean_rhat_value, retry)
					else

						# partition_samples = Int.(Array(MCMCChains.group(chain, :partition)))
						# post_probs = Simulations.compute_post_prob_eq(partition_samples)
						partition_samples = MCMCChains.group(chain, :partition).value.data
						post_probs = Simulations.compute_post_prob_eq(partition_samples)

						rhats_retry = retry
						break
					end
				end
			end

			JLD2.jldsave(filename;
				post_probs = post_probs, rhats_retry = rhats_retry,
				true_model = true_model,
				run = (;obs_per_group, r, n_groups, hypothesis, offset, prior, seed)
			)

		end

		ProgressMeter.next!(p)

	end
end

function read_results(results_dir::String)

	filenames = filter(endswith(".jld2"), readdir(results_dir; join=true))
	no_runs = length(filenames)

	df = DataFrames.DataFrame(
		obs_per_group	= Vector{Int}(undef, no_runs),
		repeat			= Vector{Int}(undef, no_runs),
		groups			= Vector{Int}(undef, no_runs),
		hypothesis		= Vector{Symbol}(undef, no_runs),
		offset			= Vector{Float64}(undef, no_runs),
		prior			= Vector{Symbol}(undef, no_runs),
		seed			= Vector{Int}(undef, no_runs),
		true_model		= Vector{Vector{Int}}(undef, no_runs),
		rhats_retry		= Vector{Int}(undef, no_runs),
		post_probs		= Vector{Matrix{Float64}}(undef, no_runs), # could also be some triangular structure
		any_incorrect	= BitArray(undef, no_runs),
		prop_incorrect	= Vector{Float64}(undef, no_runs),
		prop_correct	= Vector{Float64}(undef, no_runs),
	)

	p = ProgressMeter.Progress(no_runs)
	generate_showvalues(filename) = () -> [(:filename, filename)]

	(i, filename) = first(enumerate(filenames))
	for (i, filename) in enumerate(filenames)

		try
			temp = JLD2.jldopen(filename)
			rhats_retry = temp["rhats_retry"]
			true_model  = temp["true_model"]
			obs_per_group, r, n_groups, hypothesis, offset, prior, seed = temp["run"]

			df[i, :obs_per_group]	= obs_per_group
			df[i, :repeat]			= r
			df[i, :groups]			= n_groups
			df[i, :hypothesis]		= hypothesis
			df[i, :offset]			= offset
			df[i, :prior]			= prior
			df[i, :seed]			= seed
			df[i, :true_model]		= true_model
			df[i, :rhats_retry]		= rhats_retry

			post_probs = temp["post_probs"]
			df[i, :post_probs]     = post_probs
			df[i, :any_incorrect]  = any_incorrect( post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)
			df[i, :prop_incorrect] = prop_incorrect(post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)
			df[i, :prop_correct]   =   prop_correct(post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)

		catch e
			@warn "file $filename failed with error $e"
		end
		ProgressMeter.next!(p; showvalues = generate_showvalues(filename))
	end

	return df

end

# n_obs_per_group, repeats, groups, hypotheses, offset, priors = get_hyperparams_big()
# n_obs_per_group = (250,)
# repeats = 1
# groups = (5, )
# run_simulation(n_obs_per_group, repeats, groups, hypotheses, offset, priors; results_dir = "simulations/bigsimulation")

# sim_results = read_results("simulations/bigsimulation")
# @assert (sim_results[!,:prop_incorrect] .> 0.0) == sim_results[!,:any_incorrect]

# ii = 17
# sim_results[ii, :post_probs]
# sim_results[ii, :true_model]
# sim_results[ii, [:any_incorrect, :prop_incorrect]]
# mm = reshape([i == j for i in sim_results[ii, :true_model] for j in sim_results[ii, :true_model]], 5, 5)

# sim_results[ii, :post_probs]
# prop_incorrect(sim_results[ii, :post_probs], sim_results[ii, :true_model], sim_results[ii, :prior] === :Westfall)
# which_incorrect(sim_results[ii, :post_probs], sim_results[ii, :true_model], sim_results[ii, :prior] === :Westfall)

# sim_results[49, :post_probs]
# sim_results[49, :true_model]

# mm = reshape([i == j for i in sim_results[49, :true_model] for j in sim_results[49, :true_model]], 5, 5)

# sim_results[49, [:any_incorrect, :prop_incorrect]]
# prop_incorrect(sim_results[49, :post_probs], sim_results[49, :true_model], true)