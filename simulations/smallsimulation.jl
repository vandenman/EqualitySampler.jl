#=
	to run the simulation, run in a terminal
		julia-1.7.2 --project=simulations -O3 --threads 8 --check-bounds=no simulations/smallsimulation.jl
=#

include("simulation_helpers.jl")

n_obs_per_group, repeats, groups, hypotheses, offset, priors = get_hyperparams_small()
results_dir = joinpath("simulations", "small_simulation_runs")

run_simulation(
	n_obs_per_group,
	# repeats,
	1:20,
	groups,
	hypotheses,
	offset,
	priors;
	results_dir = results_dir
)
