using Turing # Turing v0.14.6

@model function normal_mean_variance(x)
    s ~ Gamma(1, 1) # intentionally not InverseGamma to throw of the sampler
    m ~ Normal(0, 1)
    for i in eachindex(x)
        x[i] ~ Normal(m, s)
    end
    return m, s
end

x = rand(Normal(5, 3), 1_000);
m = normal_mean_variance(x)

# create small example problem for slack!

# in a real case I also have discrete parameters so I need to use Gibbs sampling
spl = NUTS()

# if you run this 2-3 times I'm getting a lot of:
# Warning: The current proposal will be rejected due to numerical error(s).
# │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
f1 = sample(m, spl, 8_000, discard_initial = 2_000)
# note that the results are likely completely wrong (m should be near 5, s near 3)

varinfo = Turing.VarInfo(model);
# model_gamma(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((ρ = obs_rho, τ = mean_obs_tau)));
model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((m = 5, s = 3)));
init_theta = varinfo[Turing.SampleFromPrior()]
init_theta = [5.0, 3.0]
sample(model, spl, 100, init_theta = init_theta)
