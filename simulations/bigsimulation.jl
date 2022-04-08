#=
	to run the simulation, run in a terminal
		julia-1.7.2 --project=simulations -O3 --threads 8 --check-bounds=no simulations/bigsimulation.jl

	TODO: split this file into 2 files?
		- one for running the simulation
			-	rerun simulations when any rhat is NaN or above 1.05?
		- one for creating the figure

=#

include("simulation_helpers.jl")

n_obs_per_group, repeats, groups, hypotheses, offset, priors = get_hyperparams_big()
results_dir = joinpath("simulations", "big_simulation_runs")

run_simulation(
	n_obs_per_group,
	1:10,
	groups,
	hypotheses,
	offset,
	(:uniform, :BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2, :DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcessGP, :Westfall, :Westfall_uncorrected);
	results_dir = results_dir
)
