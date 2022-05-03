#=
	to run the simulation, run in a terminal
		julia-1.7.2 --project=simulations -O3 --threads 8 --check-bounds=no simulations/bigsimulation.jl

=#

include("simulation_helpers.jl")

n_obs_per_group, repeats, groups, hypotheses, offset, priors = get_hyperparams_big()
results_dir = joinpath("simulations", "big_simulation_runs")

run_simulation(
	n_obs_per_group,
	repeats,
	groups,
	hypotheses,
	offset,
	(:uniform, :BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2, :DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcessGP, :Westfall, :Westfall_uncorrected);
	results_dir = results_dir
)
