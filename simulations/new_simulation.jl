using EqualitySampler
import Random
import DataFrames
import JLD2, CodecZlib, ProgressMeter
import Dates
import MLStyle
include("utilities.jl")

MLStyle.@active Startswith{s::String}(x) begin
    x isa AbstractString && startswith(x, s)
end

#region simulation runs
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

# TODO: combine the two functions below to reduce any risk of typos
# Symbol -> Function
# Westfall -> error() # to ensure it is never called
# could be a namedtuple or an ordered dict.
# function get_priors()
#     return (
#         :uniform,

#         :BetaBinomial11,
#         :BetaBinomialk1,
#         :BetaBinomial1k,
#         :BetaBinomial1binomk2,

#         :DirichletProcess0_5,
#         :DirichletProcess1_0,
#         :DirichletProcess2_0,
#         :DirichletProcessGP,

#         :PitmanYorProcess0_25__0_5,
#         :PitmanYorProcess0_50__0_5,
#         :PitmanYorProcess0_75__0_5,
#         :PitmanYorProcess0_25__1_0,
#         :PitmanYorProcess0_50__1_0,
#         :PitmanYorProcess0_75__1_0,

#         :Westfall,
#         :Westfall_uncorrected
#     )
# end


# function instantiate_priors(k::Integer)

#     (
#         EqualitySampler.UniformPartitionDistribution(k),

#         EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, 1.0),
#         EqualitySampler.BetaBinomialPartitionDistribution(k, k, 1.0),
#         EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, k),
#         EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, binomial(k, 2)),

#         EqualitySampler.DirichletProcessPartitionDistribution(k, 0.5),
#         EqualitySampler.DirichletProcessPartitionDistribution(k, 1.0),
#         EqualitySampler.DirichletProcessPartitionDistribution(k, 2.0),
#         EqualitySampler.DirichletProcessPartitionDistribution(k, :Gopalan_Berry),

#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.25, 0.5),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  0.5),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.75, 0.5),

#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.25, 0.5),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  0.5),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.75, 0.5),

#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.25, 1.0),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  1.0),
#         EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.75, 1.0),


#     )

# end


# d = .1:.2:.9
# for i in eachindex(d)
#     θ = -d[i] + .1 : .2 : -d[i] + 1.0
#     for j in eachindex(θ)
#         nm = "PitmanYorProcess$(round(d[i], digits = 2))__$(round(θ[j], digits = 2))"
#         nm = replace(nm, "." => "_")
#         nm = replace(nm, "-" => "m")
#         println("$nm = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, $(round(d[i], digits = 2)), $(round(θ[j], digits = 2)))")
#     end
# end
# for i in eachindex(d)
#     θ = -d[i] + .1 : .2 : -d[i] + 1.0
#     for j in eachindex(θ)
#         println(":PitmanYorProcess$(round(d[i], digits = 2))__$(round(θ[j], digits = 2))")
#     end
# end

function get_priors_obj()
    return (
        uniform = k -> EqualitySampler.UniformPartitionDistribution(k),

        BetaBinomial11       = k -> EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, 1.0),
        BetaBinomialk1       = k -> EqualitySampler.BetaBinomialPartitionDistribution(k, k,   1.0),
        BetaBinomial1k       = k -> EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, k),
        BetaBinomial1binomk2 = k -> EqualitySampler.BetaBinomialPartitionDistribution(k, 1.0, binomial(k, 2)),

        DirichletProcess0_5  = k -> EqualitySampler.DirichletProcessPartitionDistribution(k, 0.5),
        DirichletProcess1_0  = k -> EqualitySampler.DirichletProcessPartitionDistribution(k, 1.0),
        DirichletProcess2_0  = k -> EqualitySampler.DirichletProcessPartitionDistribution(k, 2.0),
        DirichletProcessGP   = k -> EqualitySampler.DirichletProcessPartitionDistribution(k, :Gopalan_Berry),
        DirichletProcessDecr = k -> EqualitySampler.DirichletProcessPartitionDistribution(k, :harmonic),

        PitmanYorProcess0_1__0_0  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.1,  0.0),
        # PitmanYorProcess0_1__0_2  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.1,  0.2),
        # PitmanYorProcess0_1__0_4  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.1,  0.4),
        # PitmanYorProcess0_1__0_6  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.1,  0.6),
        PitmanYorProcess0_1__0_8  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.1,  0.8),
        PitmanYorProcess0_3__m0_2 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.3, -0.2),
        # PitmanYorProcess0_3__0_0  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.3,  0.0),
        # PitmanYorProcess0_3__0_2  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.3,  0.2),
        # PitmanYorProcess0_3__0_4  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.3,  0.4),
        PitmanYorProcess0_3__0_6  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.3,  0.6),
        # PitmanYorProcess0_5__m0_4 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5, -0.4),
        # PitmanYorProcess0_5__m0_2 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5, -0.2),
        # PitmanYorProcess0_5__0_0  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  0.0),
        # PitmanYorProcess0_5__0_2  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  0.2),
        PitmanYorProcess0_5__0_4  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.5,  0.4),
        PitmanYorProcess0_7__m0_6 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.7, -0.6),
        # PitmanYorProcess0_7__m0_4 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.7, -0.4),
        # PitmanYorProcess0_7__m0_2 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.7, -0.2),
        # PitmanYorProcess0_7__0_0  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.7,  0.0),
        PitmanYorProcess0_7__0_2  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.7,  0.2),
        PitmanYorProcess0_9__m0_8 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.9, -0.8),
        # PitmanYorProcess0_9__m0_6 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.9, -0.6),
        # PitmanYorProcess0_9__m0_4 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.9, -0.4),
        # PitmanYorProcess0_9__m0_2 = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.9, -0.2),
        PitmanYorProcess0_9__0_0  = k -> EqualitySampler.PitmanYorProcessPartitionDistribution(k, 0.9,  0.0),

        Westfall                = _ -> nothing,
        Westfall_uncorrected    = _ -> nothing

    )

end

function get_priors_default()

    defaults    = (:uniform, )
    # defaults_bb = (:BetaBinomial11, :BetaBinomialk1, :BetaBinomial1k, :BetaBinomial1binomk2)
    defaults_bb = (:BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2)
    defaults_dp = (:DirichletProcessGP, :DirichletProcessDecr)
    defaults_wf = (:Westfall, :Westfall_uncorrected)

    no_defaults = length(defaults) + length(defaults_bb) + length(defaults_dp) + length(defaults_wf)
    # αs = [.5, 1.0, 2.0]
    αs = [1.0]

    no_elements = no_defaults + length(αs)
    result = Vector{Pair{Symbol, Vector{Float64}}}()
    sizehint!(result, no_elements)
    for s in defaults
        push!(result, s => Float64[])
    end

    for s in defaults_bb
        push!(result, s => [1.0, 1.0])
    end

    for s in defaults_dp
        push!(result, s => Float64[])
    end

    for α in αs
        key = Symbol("DirichletProcess_" * string(α))
        push!(result, key => [α])
    end

    # these must be last
    for s in defaults_wf
        push!(result, s => Float64[])
    end

    return result
end

function get_priors_pitmanyor_dirichlet_comparison()

    αs = logrange(1e-3, 1e1, 25)
    # CM.scatter(eachindex(αs), αs)

    ds = range(1e-5, 1 - 1e-5, 25)
    # CM.scatter(eachindex(ds), ds)

    θs = logrange(1e-3, 1e1, 25)
    # CM.scatter(eachindex(αs), αs)

    # defaults = (
    #     :uniform, :Westfall, :Westfall_uncorrected, :BetaBinomial11,
    #     :BetaBinomialk1, :BetaBinomial1k, :BetaBinomial1binomk2, :DirichletProcessGP, :DirichletProcessDecr
    # )
    defaults = (:DirichletProcessGP, :DirichletProcessDecr)

    no_elements = length(defaults) + length(αs) + length(ds) * length(θs)
    result = Vector{Pair{Symbol, Vector{Float64}}}()
    sizehint!(result, no_elements)
    for s in defaults
        push!(result, s => Float64[])
    end

    for (i, α) in enumerate(αs)
        key = Symbol("DirichletProcess_idx_" * string(i))
        push!(result, key => [α])
    end

    # ee = Set{Tuple{Float64, Float64}}()
    for (i, (d, θ)) in enumerate(Iterators.product(ds, θs))
        key = Symbol("PitmanYorProcess_idx_" * string(i))
        push!(result, key => [d, θ])
        # push!(ee, (d, θ))
    end
    # length(ee) == length(Iterators.product(ds, θs))

    return result

end

function get_family(prior::Symbol)
    prior === :uniform && return :uniform
    str = string(prior)
    MLStyle.@match str begin
        Startswith{"BetaBinomial"}     => :BetaBinomial
        Startswith{"DirichletProcess"} => :DirichletProcess
        Startswith{"PitmanYorProcess"} => :PitmanYorProcess
        Startswith{"Westfall"}         => :Westfall
    end
end

function instantiate_priors(k::Integer, prior::Pair)
    p = prior.first
    p === :uniform && return UniformPartitionDistribution(k)
    str = string(p)
    args = prior.second
    # @show k, prior, str, args
    return MLStyle.@match str begin
        Startswith{"BetaBinomial"}     => instantiate_betabinomial(p, k, args...)
        Startswith{"DirichletProcess"} => instantiate_dirichlet(   p, k, args...)
        Startswith{"PitmanYorProcess"} => instantiate_pitmanyor(   p, k, args...)
    end
end

function instantiate_betabinomial(p::Symbol, k, args...)
    p === :BetaBinomial11       && return BetaBinomialPartitionDistribution(k,     args[1],                  args[2])
    p === :BetaBinomial1k       && return BetaBinomialPartitionDistribution(k,     args[1],              k * args[2])
    p === :BetaBinomialk1       && return BetaBinomialPartitionDistribution(k, k * args[1],                  args[2])
    p === :BetaBinomial1binomk2 && return BetaBinomialPartitionDistribution(k,     args[1], binomial(k, 2) * args[2])
    throw(ArgumentError("Unknown prior: $p. Adjust instantiate_betabinomial()"))
end

function instantiate_dirichlet(p::Symbol, k, args...)
    p === :DirichletProcessGP   && return DirichletProcessPartitionDistribution(k, :Gopalan_Berry)
    p === :DirichletProcessDecr && return DirichletProcessPartitionDistribution(k, :harmonic)
    return DirichletProcessPartitionDistribution(k, args[1])
end

function instantiate_pitmanyor(::Symbol, k, args...)
    PitmanYorProcessPartitionDistribution(k, args...)
end

# function get_args(family, prior, k, prior_settings)

#     MLStyle.@match prior begin
#         :uniform              => Float64[],
#         :BetaBinomial11       => [1.0, 1.0],
#         :BetaBinomialk1       => [k, 1.0],
#         :BetaBinomial1k       => [1.0, k],
#         :BetaBinomial1binomk2 => [1.0, binomial(k, 2)],
#         :DirichletProcess     => prior_settings.dp[k],
#         :DirichletProcessGP   => Float64[],
#         :DirichletProcessDecr => Float64[],
#         :PitmanYorProcess     => prior_settings.py[k],
#         :Westfall             => Float64[],
#         :Westfall_uncorrected => Float64[]

#     end
# end

# function prior_settings()

#     priors = Vector{@NamedTuple{family::Symbol, prior::Symbol, args::Vector{Float64}}}()
#     push!(priors, (family = :uniform,      prior = :uniform,              args = Float64[]))
#     push!(priors, (family = :BetaBinomial, prior = :BetaBinomial11,       args = [1.0, 1.0]))
#     push!(priors, (family = :BetaBinomial, prior = :BetaBinomialk1,       args = [1.0, 1.0]))
#     push!(priors, (family = :BetaBinomial, prior = :BetaBinomial1k,       args = [1.0, 1.0]))
#     push!(priors, (family = :BetaBinomial, prior = :BetaBinomial1binomk2, args = [1.0, 1.0]))

#     dp_args = [.5, 1.0, 2.0]
#     for arg in dp_args
#         push!(priors, (family = :DirichletProcess, prior = Symbol("DirichletProcess_" * string(arg)), args = [arg]))
#     end

#     (family = :DirichletProcess, prior = :DirichletProcess, args = (1.0, 1.0))
#     (family = :DirichletProcess, prior = :DirichletProcess, args = (1.0, 1.0))
#     (family = :DirichletProcess, prior = :DirichletProcess, args = (1.0, 1.0))



#     BetaBinomial11
#     BetaBinomialk1
#     BetaBinomial1k
#     BetaBinomial1binomk2
#     uniform              = nothing
#     betabinomial         = ((1.0, 1.0))
#     betabinomial1k       = (1.0, 1.0)
#     betabinomialbinom1k  = (1.0, 1.0)
#     betabinomialk1       = (1.0, 1.0)

#     DirichletProcess     = (.5, 1.0, 2.0)
#     DirichletProcessGP   = nothing,
#     DirichletProcessDecr = nothing

#     PitmanYorProcess     = Iterators.product(logrange(1e-3, 1e1, 25), logrange(1e-3, 1e1, 25))

#     Westfall             = nothing
#     Westfall_uncorrected = nothing

#     Iterators.flatten(
#         (:uniform, uniform),
#         (:betabinomial, betabinomial),
#         (:betabinomial1k, betabinomial1k),
#         (:betabinomialbinom1k, betabinomialbinom1k),
#         (:betabinomialk1, betabinomialk1),
#         (:DirichletProcess, DirichletProcess),
#         (:DirichletProcessGP, DirichletProcessGP),
#         (:DirichletProcessDecr, DirichletProcessDecr),
#         (:PitmanYorProcess, PitmanYorProcess),
#         (:Westfall, Westfall),
#         (:Westfall_uncorrected, Westfall_uncorrected)
#     )

# end

# get_priors() = collect(keys(get_priors_obj()))
# instantiate_priors(k::Integer) = filter(!isnothing, map(d -> d(k), values(get_priors_obj())))

function instantiate_log_prior_probs_obj(priors_vec, ks)
    log_prior_probs_obj = Dict{Int, Matrix{Float64}}()
    for k in ks

        modelspace = PartitionSpace(k)
        # all_priors = instantiate_priors.(k, priors_vec)
        all_priors = [instantiate_priors(k, prior) for prior in priors_vec if prior.first !== :Westfall && prior.first !== :Westfall_uncorrected]
        log_prior_probs = zeros(length(modelspace), length(all_priors))
        for j in eachindex(all_priors)
            for (i, m) in enumerate(modelspace)
                log_prior_probs[i, j] = logpdf_model_distinct(all_priors[j], m)
            end
        end
        log_prior_probs_obj[k] = log_prior_probs
    end
    return log_prior_probs_obj
end

function get_hyperparameters_big()
    n_obs_per_group = 50:50:500#(50, 100, 250, 500, 750, 1_000)
    repeats			= 1:500
    groups			= (5, 9)
    hypothesis		= (:p00, :p25, :p50, :p75, :p100)
    offset			= 0.2
    priors_vec      = get_priors_default()
    return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors_vec)
end

function get_hyperparameters_test()
    n_obs_per_group = 50:50:200
    repeats			= 1:5
    groups			= (5, 9)
    hypothesis      = (:p00, :p50, :p100)
    offset          = 0.2
    priors_vec      = get_priors_default()
    return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors_vec)
end

function get_hyperparams_small()
    n_obs_per_group = 100
    repeats         = 1:200
    groups          = 2:1:10
    hypothesis      = (:p00, :p100)
    offset          = 0.2
    priors_vec      = get_priors_default()
    return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors_vec)
end

function get_hyperparams_pitmanyor_dirichlet()
    n_obs_per_group = 100
    repeats         = 1:200
    groups          = (5, )
    hypothesis      = (:p00, :p25, :p50, :p75, :p100)
    offset          = 0.2
    priors_vec      = get_priors_pitmanyor_dirichlet_comparison()
    return (; n_obs_per_group, repeats, groups, hypothesis, offset, priors_vec)
end

sample_true_model(hypothesis::Symbol, n_groups::Integer) = sample_true_model(Random.default_rng(), hypothesis, n_groups)
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
        # logpdf = ntuple(i->log(i==logpdf_idx_one), n_groups)
        logpdf = log.((1:n_groups) .== logpdf_idx_one)

        return rand(rng, EqualitySampler.CustomInclusionPartitionDistribution(n_groups, logpdf))

    end
end

function log_model_probs_to_equality_probs(no_groups, log_posterior_probs)
    eq_probs = zeros(Float64, no_groups, no_groups)
    log_model_probs_to_equality_probs!(eq_probs, log_posterior_probs)
end
function log_model_probs_to_equality_probs!(eq_probs, log_posterior_probs)

    @assert size(eq_probs, 1) == size(eq_probs, 2)

    no_groups = size(eq_probs, 1)
    modelspace = PartitionSpace(no_groups, EqualitySampler.DistinctPartitionSpace)
    @assert length(modelspace) == length(log_posterior_probs)

    fill!(eq_probs, zero(eltype(eq_probs)))

    @inbounds for (idx, model) in enumerate(modelspace)

        for j in 1:length(model)-1
            for i in j+1:length(model)
                if model[i] == model[j]
                    eq_probs[i, j] += exp(log_posterior_probs[idx])
                end
            end
        end
    end
    for i in axes(eq_probs, 1)
        eq_probs[i, i] = one(eltype(eq_probs))
    end

    return eq_probs
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

function prop_incorrect_αβ(x, true_model::Vector{Int}, values_are_log_odds::Bool = false)

    reference, comparison = get_reference_and_comparison(values_are_log_odds)
    α_error_count, β_error_count = 0, 0
    α_count, β_count = 0, 0
    n = size(x, 1)
    for j in 1:n-1, i in j+1:n
        if true_model[i] == true_model[j]
            α_count += 1
            if comparison(x[i, j], reference)
                α_error_count += 1
            end
        elseif true_model[i] != true_model[j]
            β_count += 1
            if !comparison(x[i, j], reference)
                β_error_count += 1
            end
        end
    end

    α_error_prop = iszero(α_count) ? 0.0 : α_error_count / α_count
    β_error_prop = iszero(β_count) ? 0.0 : β_error_count / β_count
    total_count = (n * (n - 1) ÷ 2)
    return (; α_error_prop, β_error_prop, α_error_count, β_error_count, α_count, β_count, total_count)
end

function empty_results_df(no_rows)
    DataFrames.DataFrame(
        obs_per_group	= Vector{Int}(            undef, no_rows),
        repeat			= Vector{Int}(            undef, no_rows),
        groups			= Vector{Int}(            undef, no_rows),
        hypothesis		= Vector{Symbol}(         undef, no_rows),
        offset			= Vector{Float64}(        undef, no_rows),
        family			= Vector{Symbol}(         undef, no_rows),
        prior			= Vector{Symbol}(         undef, no_rows),
        prior_args		= Vector{Vector{Float64}}(undef, no_rows),
        seed			= Vector{UInt}(           undef, no_rows),
        true_model		= Vector{Vector{Int}}(    undef, no_rows),
        post_probs		= Vector{Matrix{Float64}}(undef, no_rows),
        any_incorrect	= BitArray(               undef, no_rows),
        prop_incorrect	= Vector{Float64}(        undef, no_rows),
        prop_correct	= Vector{Float64}(        undef, no_rows),
        α_error_prop	= Vector{Float64}(        undef, no_rows),
        β_error_prop	= Vector{Float64}(        undef, no_rows),
        α_error_count	= Vector{Int}(            undef, no_rows),
        β_error_count	= Vector{Int}(            undef, no_rows)
    )
end

function fill_df!(df, true_model, obs_per_group, r, n_groups, hypothesis, offset, seed, all_post_probs, row, priors_vec)

    no_priors = length(priors_vec)
    for j in axes(all_post_probs, 3)

        post_probs = view(all_post_probs, :, :, j)
        prior = priors_vec[j].first
        k = j + (row - 1) * no_priors

        df[k, :obs_per_group] = obs_per_group
        df[k, :repeat]        = r
        df[k, :groups]        = n_groups
        df[k, :hypothesis]    = hypothesis
        df[k, :offset]        = offset
        df[k, :family]        = get_family(prior)
        df[k, :prior]         = prior
        df[k, :prior_args]    = priors_vec[j].second
        df[k, :seed]          = seed
        df[k, :true_model]    = true_model

        df[k, :post_probs]     = copy(post_probs)
        df[k, :any_incorrect]  = any_incorrect( post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)
        df[k, :prop_incorrect] = prop_incorrect(post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)
        df[k, :prop_correct]   =   prop_correct(post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)

        α_error_prop, β_error_prop, α_error_count, β_error_count, α_count, β_count, total_count = prop_incorrect_αβ(post_probs, true_model, prior === :Westfall || prior === :Westfall_uncorrected)
        df[k, :α_error_prop]  = α_error_prop
        df[k, :β_error_prop]  = β_error_prop
        df[k, :α_error_count] = α_error_count
        df[k, :β_error_count] = β_error_count

    end
end

function normalize_θ3(offset::AbstractFloat, true_model::Vector{<:Integer})

    nu = length(unique(true_model))
    if isone(nu)
        x = zero(offset)
    else
        tot = sum((nu - i) * i for i in 1:nu) * 2 / (nu * (nu - 1))
        # tot = (nu + 1) / 3
        x = -offset / tot
    end

    EqualitySampler.normalize_θ(x, true_model)

end

function simulate_data_one_run(rng::Random.AbstractRNG, hypothesis, n_groups, obs_per_group, offset)

    true_model = sample_true_model(rng, hypothesis, n_groups)
    # true_θ = EqualitySampler.normalize_θ(offset, true_model)
    true_θ = normalize_θ3(offset, true_model)

    data_obj = EqualitySampler.simulate_data_one_way_anova(n_groups = n_groups, n_obs_per_group = obs_per_group, θ = true_θ, rng = rng)
    return data_obj.data, data_obj, true_model, true_θ

end

function fit_one_run(dat, log_prior_probs, do_westfall::Bool, do_westfall_uncorrected::Bool)

    y = dat.y::Vector{Float64}
    g = dat.g::Vector{UnitRange{Int}}
    no_groups = length(g)

    method = EqualitySampler.Enumerate()
    results = EqualitySampler.anova_test(y, g, method, UniformPartitionDistribution(no_groups), verbose = false, threaded = true)
    log_bfs              = results.logbfs
    log_posterior_probs  = results.log_posterior_probs
    log_posterior_probs2 = similar(log_posterior_probs)

    extra_cols = do_westfall + do_westfall_uncorrected
    post_probs = Array{Float64}(undef, no_groups, no_groups, size(log_prior_probs, 2) + extra_cols)
    @views log_model_probs_to_equality_probs!(post_probs[:, :, 1], log_posterior_probs)
    for i in 2:size(log_prior_probs, 2)
        EqualitySampler.logbf_to_logposteriorprob!(log_posterior_probs2, log_bfs, view(log_prior_probs, :, i))
        @views log_model_probs_to_equality_probs!(post_probs[:, :, i], log_posterior_probs2)
    end

    if do_westfall || do_westfall_uncorrected
        result_westfall = EqualitySampler.westfall_test(dat)

        if do_westfall && do_westfall_uncorrected
            post_probs[:, :, end - 1] = result_westfall.log_posterior_odds_mat
            post_probs[:, :, end]     = result_westfall.logbf_matrix
        elseif do_westfall
            post_probs[:, :, end]     = result_westfall.logbf_matrix
        else # do_westfall_uncorrected
            post_probs[:, :, end]     = result_westfall.log_posterior_odds_mat
        end
    end

    return post_probs

end

MLStyle.@data SimulationType begin
    Test
    Small
    Big
    DirichletPitmanYor
end

get_postfix(simulation_type::SimulationType) = MLStyle.@match simulation_type begin
    Test               => "_test"
    Small              => "_small"
    Big                => "_big"
    DirichletPitmanYor => "_pitmanyor_dirichlet"
end


function run_simulation_runs(runs_dir, simulation_type::SimulationType)

    n_obs_per_group, repeats, groups, hypotheses, offset, priors_vec = MLStyle.@match simulation_type begin
        Test               => get_hyperparameters_test()
        Small              => get_hyperparams_small()
        Big                => get_hyperparameters_big()
        DirichletPitmanYor => get_hyperparams_pitmanyor_dirichlet()
    end

    do_westfall             = any(x -> x.first === :Westfall, priors_vec)
    do_westfall_uncorrected = any(x -> x.first === :Westfall_uncorrected, priors_vec)

    sim_opts = Iterators.product(n_obs_per_group, repeats, groups, hypotheses, offset)

    nsim = length(sim_opts)
    seeds = vec(Base.hash.(sim_opts))

    sim_opts_with_seed = zip(sim_opts, Iterators.take(Iterators.cycle(seeds), length(sim_opts)))

    collected_opts = collect(sim_opts_with_seed)

    # assert there are no hash collisions, i.e., all seeds are distinct
    @assert allunique(seeds)

    log_prior_probs_obj = instantiate_log_prior_probs_obj(priors_vec, groups)

    # to improve load balancing, shuffle the collected_opts
    Random.seed!(42)
    Random.shuffle!(collected_opts)
    # ((obs_per_group, r, n_groups, hypothesis, offset), seed) = first(collected_opts)

    # customize this as needed. More tasks have more overhead, but better load balancing
    tasks_per_thread = 10

    # note: perhaps not ideal to make this dynamic? resumeability now depends on identical no. threads
    chunk_size = max(5, length(collected_opts) ÷ (tasks_per_thread * Threads.nthreads()))
    # partition data into chunks that individual tasks will deal with
    data_chunks = Iterators.partition(collected_opts, chunk_size)

    prog = ProgressMeter.Progress(nsim)

    log_message("Starting $(nsim) runs with $(Threads.nthreads()) threads with $(length(data_chunks)) chunks.")

    # chunk = first(data_chunks)
    Threads.@threads for chunk in collect(data_chunks)

        seed_for_filename = last(first(chunk))
        filename = make_filename(runs_dir, (seed = seed_for_filename, len = length(chunk)))

        if !isfile(filename)

            df = empty_results_df(length(chunk) * length(priors_vec))

            # also works but is a bit less clear (the parentheses matter!)
            # (i, (((obs_per_group, r, n_groups, hypothesis, offset), seed), )) = first(enumerate(chunk))
            # for (i, ((obs_per_group, r, n_groups, hypothesis, offset), seed)) in enumerate(chunk)

            # x = first(enumerate(chunk))
            # x = collect(enumerate(chunk))[4]
            no_errors = true
            for x in enumerate(chunk)

                i = x[1]
                obs_per_group, r, n_groups, hypothesis, offset = x[2][1]
                seed = x[2][2]
                # log_message("Starting run $i, n_groups=$n_groups, seed=$seed")

                rng = Random.default_rng()
                Random.seed!(rng, seed)

                dat, _, true_model, _ = simulate_data_one_run(rng, hypothesis, n_groups, obs_per_group, offset)

                try
                    all_post_probs = fit_one_run(dat, log_prior_probs_obj[n_groups], do_westfall, do_westfall_uncorrected)

                    fill_df!(df, true_model, obs_per_group, r, n_groups, hypothesis, offset, seed, all_post_probs, i, priors_vec)
                catch e
                    if isa(e, InterruptException)
                        rethrow()
                    else
                        no_errors = false
                        @warn "run $i, n_groups=$n_groups, seed=$seed failed with error $e"
                    end
                end

                ProgressMeter.next!(prog)
            end

            # log_message("saving run to $filename")
            no_errors && JLD2.jldsave(filename, true; results_df = df)
        end
    end
end
#endregion

#region combine simulation runs

function combine_simulation_runs(results_dir, runs_dir, postfix)

    filenames = filter(endswith(".jld2"), readdir(runs_dir; join=true))

    df = JLD2.jldopen(first(filenames))["results_df"]

    prog = ProgressMeter.Progress(length(filenames) - 1)
    generate_showvalues(filename) = () -> [(:filename, filename)]

    filename = first(Iterators.drop(filenames, 1))
    for filename in Iterators.drop(filenames, 1)

        try
            temp = JLD2.jldopen(filename)["results_df"]
            append!(df, temp)

        catch e
            @warn "file $filename failed with error $e"
        end

        ProgressMeter.next!(prog; showvalues = generate_showvalues(filename))
    end

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM")
    filename = joinpath(results_dir, "combined_runs$(postfix)$(timestamp).jld2")
    JLD2.jldsave(filename, true; results_df = df)

    return df

end
#endregion

function main(;
        results_dir::String,
        runs_dir::String,
        test::Bool = false,
        force::Bool = false,
        all_runs = Set((Small, Big, DirichletPitmanYor))
    )

    existing_files = filter(endswith(".jld2"), readdir(results_dir))
    runs = Set{SimulationType}()
    if test
        push!(runs, Test)
    elseif force
        runs = all_runs
    else
        for run in all_runs
            postfix = get_postfix(run)
            !any(startswith("combined_runs_$(postfix)_"), existing_files) && push!(runs, run)
        end
    end

    !ispath(results_dir) && mkpath(results_dir)
    !ispath(runs_dir)    && mkpath(runs_dir)

    run_dirs = Dict(
        Test                => joinpath(runs_dir, "test_simulation_runs"),
        Small               => joinpath(runs_dir, "small_simulation_runs"),
        Big                 => joinpath(runs_dir, "big_simulation_runs"),
        DirichletPitmanYor  => joinpath(runs_dir, "dirichletpitmanyor_simulation_runs")
    )

    for dir in values(run_dirs)
        !ispath(dir) && mkpath(dir)
    end

    for run in runs

        dir = run_dirs[run]
        postfix = get_postfix(run)
        if force || !any(startswith("combined_runs_$(postfix)_"), existing_files)
            simname = string(run)
            log_message("Starting $(simname) simulation")
            run_simulation_runs(dir, run)
            log_message("Finished $(simname) simulation")

            log_message("Combining $(simname) simulation runs into one object")
            combine_simulation_runs(results_dir, dir, postfix)
            log_message("Finished combining simulation runs")
        else
            log_message("Skipping $(simname) simulation")
        end
    end

end

main(;
    results_dir    = joinpath(pwd(), "simulations", "saved_objects"),
    runs_dir       = "/home/don/hdd/postdoc/equalitysampler/runs_dir12",
    test           = false,
    force          = true,
    # all_runs       = Set((DirichletPitmanYor, ))
    # all_runs = Set((Small, Big))
    all_runs = Set((Big, ))
)