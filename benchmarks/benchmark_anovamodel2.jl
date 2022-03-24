using EqualitySampler, EqualitySampler.Simulations, Distributions
using MCMCChains, Random, Statistics, BenchmarkTools
# using AdvancedHMC
import AbstractMCMC, MCMCChains
import Turing, Turing.Essential, ForwardDiff, DynamicPPL, Bijectors, LinearAlgebra

# TODO: why is simulate_data_one_way_anova so memory hungry?
dd = simulate_data_one_way_anova(10, 100)#, Float64[], [1, 1, 1, 1, 2, 2])
data = dd.data

mcmc_settings =  MCMCSettings(4_000, 100, 1, 1, AbstractMCMC.MCMCSerial())
partition_prior = DirichletProcessMvUrnDistribution(length(data.g))

# import Memoization
# Memoization.empty_all_caches!()
# Memoization.caches
# @code_warntype EqualitySampler.Simulations.get_equalizer_matrix_from_partition_memoized([4, 2, 1, 5, 4, 1])

# EqualitySampler.Simulations.get_equalizer_matrix_from_partition([4, 2, 1, 5, 4, 1])
# EqualitySampler.Simulations.get_equalizer_matrix_from_partition_memoized([4, 2, 1, 5, 4, 1])

# Rational.(EqualitySampler.Simulations.get_equalizer_matrix_from_partition([4, 2, 1, 5, 4, 1]))
# LinearAlgebra.factorize(EqualitySampler.Simulations.get_equalizer_matrix_from_partition([4, 2, 1, 4, 4, 1]))


obs_mean, obs_var, obs_n, Q = EqualitySampler.Simulations.prep_model_arguments(data)
model = EqualitySampler.Simulations.one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior)
starting_values = EqualitySampler.Simulations.get_starting_values(data)
init_params = EqualitySampler.Simulations.get_init_params(starting_values...)

parameters = DynamicPPL.syms(DynamicPPL.VarInfo(model))
continuous_parameters = filter(!=(:partition), parameters)


mcmc_sampler0 = Turing.Gibbs(
	Turing.HMC(0.0125, 20, continuous_parameters...),
	Turing.GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(model.args.partition_prior), EqualitySampler.Simulations.get_logπ(model)))
)

chns = (
	old             = EqualitySampler.Simulations.fit_eq_model(data, partition_prior; spl = mcmc_sampler0, mcmc_settings = mcmc_settings, eq_model = :old),
	matrix          = EqualitySampler.Simulations.fit_eq_model(data, partition_prior; spl = mcmc_sampler0, mcmc_settings = mcmc_settings, eq_model = :matrix),
	matrix_memoized = EqualitySampler.Simulations.fit_eq_model(data, partition_prior; spl = mcmc_sampler0, mcmc_settings = mcmc_settings, eq_model = :matrix_memoized)
)

map(MCMCChains.wall_duration, chns)
# (old = 17.893, matrix = 18.07, matrix_memoized = 19.335)
# (old = 8.602, matrix = 8.436, matrix_memoized = 9.213)
# (old = 6.378, matrix = 6.788, matrix_memoized = 7.757)
# (old = 7.824, matrix = 7.146, matrix_memoized = 8.089)
# (old = 5.926, matrix = 5.798, matrix_memoized = 6.685)
map(chns) do chn
	mean(MCMCChains.summarystats(chn).nt.rhat)
end
map(chns) do chn
	mean(MCMCChains.summarystats(chn).nt.ess_per_sec)
end

import Profile
Profile.init(;n=2_000_000, delay = 0.01)
# Profile.clear()
# partition_prior = DirichletProcessMvUrnDistribution(length(data.g))
# partition_prior = UniformMvUrnDistribution(length(data.g))
partition_prior = BetaBinomialMvUrnDistribution(length(data.g))
@profview EqualitySampler.Simulations.fit_eq_model(data, partition_prior; spl = mcmc_sampler0, mcmc_settings = mcmc_settings, eq_model = :old)
view_profile()

function fast_countmap_partition(partition)
	# identical to collect(values(sort(StatsBase.countmap(partition))) but much faster
	res = zero(partition)
	@inbounds for p in partition
		res[p] += 1
	end
	return filter!(!(iszero), res)
end


urns = rand(UniformMvUrnDistribution(20))
sort(StatsBase.countmap(urns))
fast_countmap(urns)
b1 = BenchmarkTools.@benchmark StatsBase.countmap($urns)
b2 = BenchmarkTools.@benchmark fast_countmap_partition($urns)
display.((b1, b2))
judge(median(b2), median(b1))

function no_distinct_groups_in_partition(partition::AbstractVector{<:Integer})
	hash = trues(length(partition))
	no_distinct = 0
	for p in partition
		if hash[p]
			hash[p] = false
			no_distinct += 1
		end
	end
	return no_distinct
end
urns = rand(UniformMvUrnDistribution(20))
(length(Set(urns)), no_distinct_groups_in_partition(urns))
b1 = BenchmarkTools.@benchmark length(Set($urns))
b2 = BenchmarkTools.@benchmark no_distinct_groups_in_partition($urns)
display.((b1, b2))
judge(median(b2), median(b1))

function logpdf_model_distinct2(d::RandomProcessMvUrnDistribution{RPM, T}, partition::AbstractVector{<:Integer})  where {RPM<:Turing.RandomMeasures.DirichletProcess, T<:Integer}

	U = T === BigInt ? BigFloat : Float64
	EqualitySampler.in_eqsupport(d, urns) || return U(-Inf)

	n = length(d)
	M = d.rpm.α
	cc = fast_countmap_partition(partition)

	return length(cc) * log(M) +
		EqualitySampler.logabsgamma(M) -
		EqualitySampler.logabsgamma(M + n) +
		sum(EqualitySampler.logabsgamma, cc)

end

urns = rand(UniformMvUrnDistribution(20))
d = DirichletProcessMvUrnDistribution(length(urns))
logpdf_model_distinct(d, urns) == logpdf_model_distinct2(d, urns)
b1 = @benchmark logpdf_model_distinct(d, urns)
b2 = @benchmark logpdf_model_distinct2(d, urns)
display.((b1, b2))
judge(median(b2), median(b1))


# TODO: what is going wrong with matrix_memoized? ThreadSafeDicts seems like a bad idea after reading
# https://discourse.julialang.org/t/can-dicts-be-threadsafe/27172/16
# although maybe we can preallocate...


no_distinct_groups_in_partition0(partition::AbstractVector{<:Integer}) = length(Set(partition))
function no_distinct_groups_in_partition1(partition::AbstractVector{<:Integer})
	# identical to lenght(Set(partition)) or length(unique(partition))
	hash = trues(length(partition))
	no_distinct = 0
	@inbounds for p in partition
		if hash[p]
			hash[p] = false
			no_distinct += 1
		end
	end
	return no_distinct
end

const powers_2 = 2 .^(0:62)
function no_distinct_groups_in_partition2(partition::AbstractVector{T}) where {T<:Integer}
	count_ones(mapreduce(x->powers_2[x], |, partition))
end
function no_distinct_groups_in_partition3(partition::AbstractVector{<:Integer})
	if length(partition) <= 62
		return no_distinct_groups_in_partition2(partition)
	else
		return no_distinct_groups_in_partition1(partition)
	end
end


function benchmark_no_distinct_groups_in_partition(k)

	partition = rand(UniformMvUrnDistribution(k))

	v1 = no_distinct_groups_in_partition0(partition)
	v2 = no_distinct_groups_in_partition3(partition)
	v3 = no_distinct_groups_in_partition2(partition)
	@assert v1 == v2 == v3

	b1 = @benchmark no_distinct_groups_in_partition0($partition)
	b2 = @benchmark no_distinct_groups_in_partition3($partition)
	b3 = @benchmark no_distinct_groups_in_partition2($partition)
	return b1, b2, b3
end
b1, b2, b3 = benchmark_no_distinct_groups_in_partition(5)
display.((b1, b2, b3, judge(median(b2), median(b1)), judge(median(b3), median(b1)), judge(median(b3), median(b2))))

b1, b2, b3 = benchmark_no_distinct_groups_in_partition(20)
display.((b1, b2, b3, judge(median(b2), median(b1)), judge(median(b3), median(b1)), judge(median(b3), median(b2))))

aa = 2 ^ partition[1]
for i in 2:20
	@show i, aa, bitstring(aa), partition[i]
	aa ⊻= 2^partition[i]
end


function reduce_model0(x::AbstractVector{T}) where T <: Integer
	y = copy(x)
	for i in eachindex(x)
		if x[i] != i
			if !any(==(x[i]), x[1:i - 1])
				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)
				y[idx] .= i
			end
		end
	end
	return y
end

function reduce_model1(x::AbstractVector{T}) where T <: Integer
	y = copy(x)
	k = length(x)
	@inbounds for (i, xi) in enumerate(sort!(Set(x)))
		idx = findall(==(x[i]), view(x, i:k)) .+ (i - 1)
		y[idx] .= i
	end
	return y
end

p = rand(UniformMvUrnDistribution(10))
b1 = BenchmarkTools.@benchmark reduce_model0($p)
b2 = BenchmarkTools.@benchmark reduce_model1($p)
display.((b1, b2))
judge(median(b1), median(b2))


mcmc_sampler1 = Turing.Gibbs(
	Turing.HMCDA(1000, 0.65, 0.3, continuous_parameters...),
	Turing.GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(model.args.partition_prior), EqualitySampler.Simulations.get_logπ(model)))
)

mcmc_sampler2 = Turing.Gibbs(
	Turing.HMC{Turing.TrackerAD}(0.0, 20, continuous_parameters...),
	Turing.GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(model.args.partition_prior), EqualitySampler.Simulations.get_logπ(model)))
)

samplers = (mcmc_sampler0, mcmc_sampler1, mcmc_sampler2)
chains = map(samplers) do spl
	EqualitySampler.Simulations.sample_model(model, spl, mcmc_settings)
end

map(MCMCChains.summarystats, chains)
map(MCMCChains.wall_duration, chains)
map(chains) do chn
	mean(MCMCChains.summarystats(chn).nt.rhat)
end

map(chains) do chn
	mean(MCMCChains.summarystats(chn).nt.ess_per_sec)
end

(MCMCChains.wall_duration(chain0), MCMCChains.wall_duration(chain1))
(mean(MCMCChains.summarystats(chain0).nt.rhat), mean(MCMCChains.summarystats(chain1).nt.rhat))
(mean(MCMCChains.summarystats(chain0).nt.ess_per_sec), mean(MCMCChains.summarystats(chain1).nt.ess_per_sec))

function evaluate_fit(chn, true_values)
	Array(chn, [Symbol("one_way_anova_mv_ss_submodel.μ_grand"), Symbol("one_way_anova_mv_ss_submodel.σ²"), Symbol("one_way_anova_mv_ss_submodel.g"), Symbol("one_way_anova_mv_ss_submodel.θ_r")])

end

dist = InverseGamma(0.5, 0.5; check_args = false)
td = Bijectors.transformed(dist)
x = rand(dist)
y = Bijectors.bijector(dist)(x)
logpdf(td, y)
Bijectors.logpdf_with_trans(dist, x, true)
Bijectors.logpdf_forward(td, x)
Bijectors.invlink(dist, y)
Bijectors.logpdf_forward(Bijectors.transformed(dist), x)

Bijectors.logpdf_forward(td, x.value)

Bijectors.invlink(dist, x)
Bijectors.logpdf_with_trans(dist, Bijectors.invlink(dist, x), true)

Bijectors.logpdf_forward(Bijectors.transformed(InverseGamma(0.5, 0.5; check_args = false)), x)

function transformed_lpdf(dist, x)
	Bijectors.logpdf_with_trans(dist, Bijectors.invlink(dist, x), true)
end

function logpdf_trans(θ, obs_mean, obs_var, obs_n, Q, partition = nothing)

	μ_grand = θ[1]
	σ²		= θ[2]
	g		= θ[3]
	θ_r		= view(θ, 4:lastindex(θ))

	n_groups = length(obs_mean)

	lpdf = zero(eltype(θ))

	# improper priors on grand mean and variance
	# μ_grand 		~ Turing.Flat()
	# σ² 			~ JeffreysPriorVariance()
	# lpdf += Bijectors.logpdf_with_trans(
	# 	EqualitySampler.JeffreysPriorVariance(),
	# 	θ[2],
	# 	true
	# )
	lpdf += transformed_lpdf(EqualitySampler.JeffreysPriorVariance(), σ²)

	# The setup for θ follows Rouder et al., 2012, p. 363
	# g   ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)
	# lpdf += logpdf_with_trans(
	# 	Distributions.InverseGamma(0.5, 0.5; check_args = false),
	# 	Bijectors.invlink(Distributions.InverseGamma(0.5, 0.5; check_args = false), θ[3]),
	# 	true
	# )
	lpdf += transformed_lpdf(InverseGamma(0.5, 0.5; check_args = false), g)

	# θ_r ~ Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1)))
	lpdf += logpdf(Turing.filldist(Normal(), n_groups - 1), θ_r)

	σ²		= Bijectors.invlink(EqualitySampler.JeffreysPriorVariance(), θ[2])
	g		= Bijectors.invlink(InverseGamma(0.5, 0.5; check_args = false), θ[3])

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in eachindex(obs_mean)
		# TODO: how to access this function?
		lpdf += EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end
	return lpdf
end

dd = simulate_data_one_way_anova(6, 5000, Float64[-1.4, -1.4, -0.4, 3, 0.9, 0.9], [1, 1, 2, 3, 4, 4])
data = dd.data

obs_mean, obs_var, obs_n, Q = EqualitySampler.Simulations.prep_model_arguments(data)

ℓπ(θ) = logpdf_trans(θ, obs_mean, obs_var, obs_n, Q)

n_samples, n_adapts = 5_000, 1_000

D = length(obs_mean) + 2

starting_values = EqualitySampler.Simulations.get_starting_values(data)
init_params = EqualitySampler.Simulations.get_init_params(starting_values...)
initial_θ = init_params[length(obs_mean)+1:end]

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

samples_mat = Matrix{Float64}(undef, D, length(samples))
for i in eachindex(samples)
	samples_mat[:, i] .= samples[i]
end
post_means = vec(mean(samples_mat, dims = 2))

post_samples_θ_s = Matrix{Float64}(undef, D - 2, length(samples))
for i in eachindex(samples)
	g_tmp = Bijectors.invlink(InverseGamma(0.5, 0.5; check_args = false), samples[i][3])
	post_samples_θ_s[:, i] .= Q * (sqrt(post_mean_g) .* samples[i][4:end])
end

post_mean_g = mean(map(x->Bijectors.invlink(InverseGamma(0.5, 0.5; check_args = false), x[3]), samples))
post_means_θ_r = post_means[4:end]
post_means_θ_s = mean.(eachrow(post_samples_θ_s))

tv = dd.true_values
tv.θ .- post_means_θ_s # TODO: why is this wrong?


# chn = anova_test(data, UniformMvUrnDistribution(4); mcmc_settings =  MCMCSettings(100, 10, 1, 1, AbstractMCMC.MCMCSerial()))

# function Turing.Essential.gradient_logp(
# 	::Turing.Essential.ForwardDiffAD,
# 	θ::AbstractVector{<:Real},
# 	vi::DynamicPPL.VarInfo,
# 	model::DynamicPPL.Model,
# 	sampler::Turing.Inference.AbstractSampler=DynamicPPL.SampleFromPrior(),
# 	ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
# )
# 	# Define function to compute log joint.
# 	logp_old = DynamicPPL.getlogp(vi)
# 	function f(θ)
# 		new_vi = DynamicPPL.VarInfo(vi, sampler, θ)
# 		new_vi = last(DynamicPPL.evaluate!!(model, new_vi, sampler, ctx))
# 		logp = DynamicPPL.getlogp(new_vi)
# 		# Don't need to capture the resulting `vi` since this is only
# 		# needed if `vi` is mutable.
# 		DynamicPPL.setlogp!!(vi, ForwardDiff.value(logp))
# 		return logp
# 	end

# 	chunk_size = Turing.Essential.getchunksize(typeof(sampler))
# 	# Set chunk size and do ForwardMode.
# 	chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
# 	config = ForwardDiff.GradientConfig(f, θ, chunk)
# 	∂l∂θ = ForwardDiff.gradient!(similar(θ), f, θ, config)
# 	l = DynamicPPL.getlogp(vi)
# 	DynamicPPL.setlogp!!(vi, logp_old)

# 	@show l, θ, ∂l∂θ

# 	return l, ∂l∂θ
# end

# chn = anova_test(data, DirichletProcessMvUrnDistribution(4);
# 	mcmc_settings =  MCMCSettings(5, 1, 1, 1, AbstractMCMC.MCMCSerial()),
# 	spl = 1.0
# )

[-29.41759362639864, -56.161651137526654, -1719.8593712072377, 308.9433234408304, 883.2464508079917, -939.1582066594987]

function manual_lpdf(θ)

	n_groups = length(obs_mean)

	lpdf = zero(eltype(θ))
	# improper priors on grand mean and variance
	μ_grand 		~ Turing.Flat()
	σ² 				~ JeffreysPriorVariance()
	lpdf += logpdf(Turing.Flat(), θ[1]) + logpdf(JeffreysPriorVariance(), θ[2])

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ Distributions.InverseGamma(0.5, 0.5; check_args = false)

	θ_r ~ Distributions.MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1.0, n_groups - 1)))

	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in eachindex(obs_mean)
		# TODO: how to access this function?
		Turing.@addlogprob! EqualitySampler._univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end

	return θ_cs

end


function get_equalizer_matrix_from_partition(partition)
	k = length(partition)
	mmm = Matrix{Float64}(undef, k, k)
	@inbounds for i in axes(mmm, 2)
		v = count(==(partition[i]), partition)
		mmm[:, i] .= (1.0 / v) .* (partition .== partition[i])
	end
	mmm
end

function get_eq_maker_matrix2(partition)
	k = length(partition)
	mmm = Matrix{Float64}(undef, k, k)
	@inbounds for i in axes(mmm, 2)
		idx = partition .== partition[i]
		v = sum(idx)
		mmm[:, i] .= (1 / v) .* idx
	end
	mmm
end


k = 5
Q = EqualitySampler.Simulations.getQ_Rouder(k)
θ_r = randn(k - 1)

partition = [1, 1, 2, 3, 1]
EqualitySampler.Simulations.average_equality_constraints(Q * θ_r, partition)
get_equalizer_matrix_from_partition(partition) * Q * θ_r
get_eq_maker_matrix2(partition) * Q * θ_r

function get_equalizer_matrix_from_partition_with_cache()
	local _cache = Dict{Vector{Int}, Matrix{Float64}}()
	function _inner(partition::Vector{Int})
		if haskey(_cache, partition)
			return _cache[partition]
		else
			res = get_equalizer_matrix_from_partition(partition)
			_cache[partition] = res
			return res
		end
	end
end
foo = get_equalizer_matrix_from_partition_with_cache()
foo._cache
foo(partition)
foo._cache

@code_warntype foo(partition)

with_cached_factorial() = begin
    local _cache = [1] #cache factorial(0)=1
    function _factorial(n)
        if n < length(_cache)
            println("pull out from the cache factorial($n)=$(_cache[n+1])")
            _cache[n+1]
        else
            fres =  n * _factorial(n-1)
            push!(_cache, fres)
            println("put factorial($n)=$fres into the cache of the size=$(sizeof(_cache))") #a
            fres
        end
    end
end


function benchmarker(k)

	θ_r = randn(k - 1)
	Q = EqualitySampler.Simulations.getQ_Rouder(k)
	partition = rand(UniformMvUrnDistribution(k))

	res1 = @benchmark EqualitySampler.Simulations.average_equality_constraints($Q * $θ_r, $partition)
	res2 = @benchmark get_equalizer_matrix_from_partition($partition) * $Q * $θ_r
	res3 = @benchmark get_eq_maker_matrix2($partition) * $Q * $θ_r
	return res1, res2, res3
end

b1, b2, b3 = benchmarker(5)
display.((b1, b2, b3, judge(median(b2), median(b1)), judge(median(b3), median(b1))))

b1, b2, b3 = benchmarker(20)
display.((b1, b2, b3, judge(median(b2), median(b1)), judge(median(b3), median(b1))))











import SparseArrays
function get_eq_maker_matrix_sparse(partition)
	k = length(partition)
	# mmm = Matrix{Float64}(undef, k, k)
	mmm = SparseArrays.spzeros(k, k)
	for i in axes(mmm, 2)
		v = count(==(partition[i]), partition)
		mmm[:, i] .= (1 / v) .* (partition .== partition[i])
	end
	mmm
end

function averager_3(values, partition)
	result = similar(values)
	for u in unique(partition)
		idx_vec = findall(==(u), partition)#view(partition, i:lastindex(partition))) .+ (i - 1)
		result[idx_vec] .= mean(view(values, idx_vec))
	end
	result
end