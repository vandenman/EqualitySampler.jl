using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots, StatsPlots
import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM,
		CSV

include("simulations/silentGeneratedQuantities.jl")
include("simulations/meansModel_Functions.jl")
include("simulations/helpersTuring.jl")

df = DF.DataFrame(CSV.File(joinpath("demos", "data", "tetris_data.csv")))
df

unique(df[!, :Condition])
describe(df[!, :Days_One_to_Seven_Image_Based_Intrusions_in_Intrusion_Diary])

@df df boxplot( string.(:Condition), :Days_One_to_Seven_Image_Based_Intrusions_in_Intrusion_Diary, fillalpha=0.75, linewidth=2)
@df df dotplot!(string.(:Condition), :Days_One_to_Seven_Image_Based_Intrusions_in_Intrusion_Diary, marker=(:black, stroke(0)))

# I didn't make a formula interface yet
DF.rename!(df, [:g, :y])
DF.combine(DF.groupby(df, :g), :y => mean, :y => median, :y => length)


# frequentist fit
ests, mod = fit_lm(df);
ests

n_groups = length(unique(df[!, :g]))

# fit the model
fitBB11	= fit_model(df, mcmc_iterations = 100_000, mcmc_burnin = 5_000,
					partition_prior = BetaBinomialMvUrnDistribution(n_groups, 1.0, 1.0));

mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fitBB11;
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(ests, mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)

# inspect sampled equality constraints
u = unique(df[!, :g])
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq))

# inspect posterior model probabilities
u = unique(df[!, :g])
mp = sort(compute_model_probs(chain_eq), byvalue=true, rev=true)


