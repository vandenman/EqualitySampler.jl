#=

	TODO:

		also sample partitions!

=#

using EqualitySampler, AdvancedHMC, Distributions, ForwardDiff, LinearAlgebra, Bijectors, ForwardDiff
using ProgressMeter
include("simulations/meansModel_Functions.jl")

function loglikelihood_within_model(obs_mean, obs_var, obs_n, μ_grand, σ², θ_cs)
	sum(logpdf(NormalSuffStat(obs_var[j], μ_grand + σ² * θ_cs[j], σ², obs_n[j]), obs_mean[j]) for j in eachindex(obs_mean))
end

function logposterior_within_model(θ)

	μ_grand, σ², g, θ_r = θ[1], invlink(InverseGamma(1, 1), θ[2]), invlink(InverseGamma(0.5, 0.5), θ[3]), view(θ, 4:length(θ))

	θ_s = Q * (sqrt(g) .* θ_r)
	# constrain θ according to the sampled equalities
	# @show partition
	θ_cs = average_equality_constraints(θ_s, partition)

	lpdf =
		logpdf(InverseGamma(1, 1), σ²) +
		logpdf(Normal(0, 1), μ_grand) +
		logpdf(InverseGamma(0.5, 0.5), g) +
		logpdf(MvNormal(n_groups - 1, 1.0), θ_r) +
		loglikelihood_within_model(obs_mean, obs_var, obs_n, μ_grand, σ², θ_cs)
	return lpdf
end

function logposterior_between_model(partition, partition_prior, obs_mean, obs_var, obs_n, μ_grand, σ², θ_s)
	θ_cs = average_equality_constraints(θ_s, partition)
	return loglikelihood_within_model(obs_mean, obs_var, obs_n, μ_grand, σ², θ_cs) + logpdf(partition_prior, partition)
end

function sample_partition!(current_partition, partition_prior, obs_mean, obs_var, obs_n, μ_grand, σ², θ_s)

	# @show current_partition
	probvec = similar(obs_mean)
	for i in eachindex(current_partition)
		for j in eachindex(current_partition)
			current_partition[i] = j
			probvec[j] = logposterior_between_model(current_partition, partition_prior, obs_mean, obs_var, obs_n, μ_grand, σ², θ_s)
		end
		probvec .= exp.(probvec .- EqualitySampler.logsumexp_batch(probvec))
		current_partition[i] = rand(Categorical(probvec))
	end
	# @show current_partition
	return current_partition
end

n_groups = 4
θ_true = [-.6, .2, .2, .2]
partition_true = [1, 2, 2, 2]
_, df, _, true_values = simulate_data_one_way_anova(n_groups, 50000, θ_true, partition_true);
obs_mean, obs_var, obs_n = get_suff_stats(df)

Q = getQ_Stan(n_groups)
partition = collect(1:n_groups)

logposterior_within_model(randn(n_groups + 2))
[reduce_model(sample_partition!(partition, UniformMvUrnDistribution(n_groups), obs_mean, obs_var, obs_n, 0.0, 1.0, θ_true)) for _ in 1:100_000]


n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
D = n_groups + 2; initial_θ = rand(D)
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, logposterior_within_model, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true);

samps = Matrix{Float64}(undef, D, length(samples))
for i in eachindex(samples)

	samps[1, i] = samples[i][1]
	samps[2, i] = invlink(InverseGamma(1, 1), samples[i][2])

	g = invlink(InverseGamma(0.5, 0.5), samples[i][3])
	samps[3:D, i] .= Q * (sqrt(g) .* samples[i][4:D])
end

post_means = mean(samps, dims = 2)
any(<(0.0), samps[2, :])
any(<(0.0), Iterators.map(x->x[2], samples))

true_values
hcat(post_means, [true_values[:μ], true_values[:σ], true_values[:θ]...])

n_samples = 3000

samps_joined = Matrix{Float64}(undef, D+1, n_samples)
partition_samples = Matrix{Int}(undef, n_groups, n_samples)
partition_prior = UniformMvUrnDistribution(n_groups)

@showprogress for i in axes(samps_joined, 2)

	ss, stats = sample(hamiltonian, proposal, initial_θ, 2, adaptor, 1; progress=false, verbose = false);
	sss=ss[2]

	μ_grand	= sss[1]
	σ²		= invlink(InverseGamma(1, 1), sss[2])
	g		= invlink(InverseGamma(0.5, 0.5), sss[3])
	θ_r		= sss[4:D]
	θ_s		= Q * (sqrt(g) .* θ_r)

	sample_partition!(partition, partition_prior, obs_mean, obs_var, obs_n, μ_grand, σ², θ_s)

	samps_joined[1, i] = μ_grand
	samps_joined[2, i] = σ²
	samps_joined[3, i] = g
	samps_joined[4:D+1, i] .= θ_s
	partition_samples[:, i] .= partition

end