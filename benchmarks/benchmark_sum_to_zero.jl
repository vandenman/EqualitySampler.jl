using EqualitySampler
using BenchmarkTools
using StatsBase
using Distributions
include("simulations/meansModel_Functions.jl")

K = 5
qr_s = (getQ_Rouder(K), getQ_Stan(K), getQ_Stan_2(K))
x_raw = randn(K-1)

xs_sum_to_zero = (
	qr_s[1] * x_raw,
	qr_s[2] * x_raw,
	apply_Q_Stan_2(x_raw, qr_s[3])
)
all(isapprox(0.0, atol = 1e-8), sum.(xs_sum_to_zero))

K = 10
qr_s = (getQ_Rouder(K), getQ_Stan(K), getQ_Stan_2(K))
x_raw = randn(K-1)

@btime qr_s[1] * x_raw
@btime qr_s[2] * x_raw
@btime apply_Q_Stan_2(x_raw, qr_s[3])

# TODO: do this also for the other two urns!

D = UniformMvUrnDistribution(k)
index = 2
u0 = rand(UniformMvUrnDistribution(k))
us = Vector{typeof(u0)}(undef, k)
for i in eachindex(us)
	us[i] = copy(u0)
	us[i][index] = i
end
us
pdf.(Ref(D), us)


p0 = pdf.(Ref(D), us)
p1 = pdf_model_distinct.(Ref(D), us)

p0 / sum(p0)
p1 / sum(p1)

pdf_model_distinct.(Ref(D), count_equalities.(us))

oo = pdf_model_distinct.(Ref(D), count_equalities.(us)) .* count_distinct_models_with_incl.(k, count_equalities.(us))
oo ./ sum(oo)

ccc = 1 ./ count_combinations.(us)
ccc ./ sum(ccc)

k = 5


result = zeros(k)

urns = view(complete_urns, findall(!=(index), 1:k))

# This step can be simplified
count = EqualitySampler.get_conditional_counts(k, urns)

idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
result[view(urns, idx_nonzero)] .= count[idx_nonzero]
other = setdiff(1:k, urns)
# this step is possibly incorrect?
result[other] .= count[length(urns) + 1] ./ length(other)
result ./= sum(result)

prediction_rule_sampler

using StatsBase

k = 5
index = 3
indices = rand(UniformMvUrnDistribution(k))

x = indices

abstract type AbstractPredictionRule end
struct UniformPredictionRule <: AbstractPredictionRule end

function prediction_logprobvec(::UniformPredictionRule, index::T, indices::AbstractVector{T}, s::Set{T}) where T<:Integer

	log_probvec = zeros(length(indices))
	for i in eachindex(indices)
		log_probvec[i] = -EqualitySampler.log_count_combinations(length(indices), length(s) + (i ∉ s))
	end
	return log_probvec .- EqualitySampler.logsumexp_batch(log_probvec)

end

prediction_logprobvec(x::AbstractPredictionRule, index::T, indices::AbstractVector{T}) where T<:Integer = prediction_logprobvec(x, index, indices, Set(view(indices, eachindex(indices) .!= index)))
prediction_probvec(x::AbstractPredictionRule, index::T, indices::AbstractVector{T}) where T<:Integer = exp.(prediction_logprobvec(x, index, indices))

function prob_new_value(x::AbstractPredictionRule, index::T, indices::AbstractVector{T}) where T<:Integer

	s = Set(view(indices, eachindex(indices) .!= index))
	logprobvec = prediction_logprobvec(x, index, indices, s)
	res = 0.0
	for i in eachindex(indices)
		if i ∉ s
			res += exp(logprobvec[i])
		end
	end
	res
end

function rand_prediction_rule(indices::AbstractVector{T}, index::T, x::AbstractPredictionRule) where T<:Integer
	return rand(Categorical(prediction_probvec(x, index, indices)))
end

prediction_probvec(UniformPredictionRule(), 1, u0)
prob_new_value(UniformPredictionRule(), 1, u0)
prediction_probvec(UniformPredictionRule(), 2, u0)
prob_new_value(UniformPredictionRule(), 2, u0)

pdf.(Ref(UniformMvUrnDistribution(k)), us) / sum(pdf.(Ref(UniformMvUrnDistribution(k)), us))

struct DirichletPredictionRule <: AbstractPredictionRule
	α::Float64
end

function prediction_logprobvec(x::DirichletPredictionRule, index::T, indices::AbstractVector{T}, s::Set{T}) where T<:Integer

	tb = countmap(view(indices, eachindex(indices) .!= index))
	sum_tb = sum(values(tb))

	n = length(indices) - length(s)
	prob_new_val = log(x.α) - log(n * (x.α + length(indices) - 1))

	log_probvec = zeros(length(indices))
	for i in eachindex(indices)
		if i in s
			log_probvec[i] = log(tb[i] / sum_tb) - EqualitySampler.log_count_combinations(length(indices), length(s) + (i ∉ s))
		else
			log_probvec[i] = prob_new_val - EqualitySampler.log_count_combinations(length(indices), length(s) + (i ∉ s))
		end
	end
	return log_probvec .- EqualitySampler.logsumexp_batch(log_probvec)
end


# DirichletPredictionRule(0.5)
prediction_probvec(DirichletPredictionRule(0.5), 2, u0)
pdf.(Ref(DirichletProcessMvUrnDistribution(k, 0.5)), us) / sum(pdf.(Ref(DirichletProcessMvUrnDistribution(k, 0.5)), us))

function rand_betabinom_mixed(index, indices, d)

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	index_already_sampled = 1:index - 1
	n0 = k

	# no_duplicated = count_equalities(view(urns, index_already_sampled))
	v_known_urns = view(complete_urns, index_already_sampled)
	r = length(Set(v_known_urns))
	n = n0 - (index - r - 1)

	model_probs_by_incl = exp.(log_model_probs_by_incl(d))

	num = r * sum(model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    ) for k in 1:n0)
	den =     sum(model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1) for k in 1:n0)
	probEquality = num / (num + den)

	k = length(indices)

	probvec = zeros(k)
	for i in eachindex(indices)
		if i != index
			probvec[indices[i]] += 1.0
		end
	end
	probvec ./= sum(probvec)


	rand(Categorical(probvec))

end

d = BetaBinomialMvUrnDistribution(5, 1, 1)
result = zeros(length(u0))

index = 3
complete_urns = u0

function probvec_bb_new(d::Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}, index, complete_urns)

	result = zeros(Float64, length(complete_urns))
	probvec_bb_new!(result, d, index, complete_urns)
	return result

end

function probvec_bb_new!(result, d::T, index, complete_urns) where T<:Union{BetaBinomialConditionalUrnDistribution, BetaBinomialMvUrnDistribution}

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	index_already_sampled = 1:index - 1
	n0 = k

	# no_duplicated = count_equalities(view(urns, index_already_sampled))
	v_known_urns = view(complete_urns, index_already_sampled)
	r = length(Set(v_known_urns))
	n = n0 - (index - r - 1)

	model_probs_by_incl = exp.(EqualitySampler.log_model_probs_by_incl(d))

	num = r * sum(model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    ) for k in 1:n0)
	den =     sum(model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1) for k in 1:n0)
	probEquality = num / (num + den)

	# probability of an equality
	current_counts = countmap(v_known_urns)
	total_counts = sum(values(current_counts))
	for (key, val) in current_counts
		result[key] = probEquality * val / total_counts
	end

	# probability of an inequality
	no_inequality_options = (k - length(current_counts)) #length(current_counts) == the number of distinct elements in v_known_urns
	for i in 1:k
		if i ∉ v_known_urns
			result[i] = (1.0 - probEquality) / no_inequality_options
		end
	end
	# inequality_options = setdiff(1:n0, v_known_urns)
	# result[inequality_options] .= (1.0 - probEquality) ./ length(inequality_options)

	return result

end

u0 = [1, 1, 1, 2, 3, 4, 5]
k = length(u0)
d = BetaBinomialMvUrnDistribution(k, 1, 1)
res_new = probvec_bb_new(d, 6, u0)
sum(res_new)
res_old = EqualitySampler._pdf_helper(d, 6, u0)
sum(res_old)

function rand_bb_new(n, d)
	k = length(d)
	result = Matrix{Int}(undef, k, n)
	probvec = zeros(k)
	u = rand(UniformMvUrnDistribution(k))
	for i in axes(result, 2)
		for j in axes(result, 1)
			probvec_bb_new!(probvec, d, j, u)
			u[j] = rand(Categorical(probvec))
			result[j, i] = u[j]
		end
	end
	return result
end

samps = rand_bb_new(300_000, d)
samps_reduced = mapslices(EqualitySampler.reduce_model, samps; dims = 1)
eqs = count_equalities.(eachcol(samps_reduced))
tb = sort(countmap(eqs))
tb_sum = sum(values(tb))
probs = Dict{Int, Float64}()
for (k, v) in tb
	probs[k] = v / tb_sum
end
probs # this looks correct!
