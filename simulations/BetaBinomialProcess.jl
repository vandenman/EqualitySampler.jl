using Distributions, Plots
import StatsBase
import StatsFuns: log1pexp
import LinearAlgebra: dot
include("old/newApproach4.jl")
include("src/helperfunctions.jl")

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		for j in 1:k
			D = updateDistribution(D, urns, j)
			urns[j] = rand(D, 1)[1]
		end
		sampled_models[:, i] .= reduce_model(urns)
	end
	return sampled_models
end

function get_empirical_model_probabilities(sampled_models)
	count_models = countmap(vec(mapslices(x->join(Int.(x)), sampled_models, dims = 1)))
	probs_models = counts2probs(count_models)
	return sort(probs_models, by = x->count_equalities(x))
end

function get_empirical_inclusion_probabilities(sampled_models)
	no_equalities = count_equalities(sampled_models)
	counts_equalities = countmap(no_equalities)
	return counts2probs(counts_equalities)
end

function randBetaBinomialUrn(nosamples, D)

	y = Matrix{Int}(undef, ntrials(D), nosamples)
	randBetaBinomialUrn!(y, D)
	return y
end

function randBetaBinomialUrn!(y::AbstractMatrix, D)
	for i in axes(y, 2)
		randBetaBinomialUrn!(view(y, :, i), D)
	end
end


function randBetaBinomialUrn!(y::AbstractVector, D)

	α = D.α
	β = D.β

	for i in eachindex(y)
		if rand() < (α / (α + β))
			y[i] = 1
			α += 1.0
		else
			y[i] = 0
			β += 1.0
		end
	end
end

α = 2.0
β = 4.0
n = 5

D = BetaBinomial(n, α, β)


nosamples = 100_000
res = randBetaBinomialUrn(nosamples, D)
sums = vec(sum(res, dims = 1))
# histogram(sums)
tb_counts = sort(StatsBase.countmap(sums))
tb_probs = sort(Dict((k => (v / nosamples) for (k, v) in tb_counts)))
sum(values(tb_probs))

scatter(0:n, x->pdf(D, x), markercolor = "gold");
scatter!(tb_probs, markercolor = "red", legend = α >= β ? :topleft : :topright)


function randEqualityBB(nosamples, D, k)

	y = Matrix{Int}(undef, k, nosamples)
	randEqualityBB!(y, D, k)
	return y
end

function randEqualityBB!(y::AbstractMatrix, D, k)
	for i in axes(y, 2)
		randEqualityBB!(view(y, :, i), D)
	end
end

function BN(n::T, r::T) where T <: Integer

	res = zero(T)
	for k in 0:n, i in 0:n
		res +=
			binomial(n, i) *
			Combinatorics.stirlings2(i, k) *
			r^(n - i)
	end
	return res
end

function get_conditional_counts(n::Int, known::AbstractVector{T} = [1]) where T <: Integer

	# TODO: just pass d.partitions and d.index to this function! (or just d itself)
	# or maybe not?
	known = reduce_model(known) # TODO: I don't think this is necessary!
	refinement = length(unique(known))
	n_known = length(known)

	res = zeros(Int, n_known + 1)
	idx = findall(i->known[i] == i, 1:n_known)
	n_idx = length(idx)
	res[idx] .= BN(n - n_known - 1, n_idx)
	res[n_known + 1] = BN(n - n_known - 1, n_idx + 1)
	res
end

function randEqualityBB!(x::AbstractVector, D)

	α = D.α
	β = D.β
	k = length(x)

	# TODO: don't do y for the first round!
	# y = similar(x)
	incl_probs = pdf.(D, 0:k-1)
	no_incl = 0
	no_groups = 
	# y = BitArray(undef, k)
	# x[1] = rand(1:k)
	for i in 1:k

		known = reduce_model(x[1:i])
		counts = get_conditional_counts(k, known)
		for j in eachindex(counts)
			counts[j] *= sum(incl_probs[no_incl +])
		end
		counts[no_incl = 0]


		prob1 = (α / (α + β)) #* (sum(counts[1:i]) / sum(counts))
		if rand() < prob1#(α / (α + β))
			α += 1.0
			y[i] = 1
		else
			β += 1.0
			y[i] = 0
		end
		if isone(i)
			x[i] = rand(1:k)
		else
			if isone(y[1])
				α, β = β, α
			end
			if isone(y[i]) # sample equality
				# x[i] = rand(view(x, 1:i-1))

				# probs = zeros(Float64, k)
				# for xi in view(x, 1:i-1)
				# 	probs[xi] += 1.0
				# end
				# probs[.!iszero.(probs)] .= 1.0 ./ probs[.!iszero.(probs)]
				# probs = probs ./ sum(probs)
				if i == 1 

					x[2] = x[1]

				else

					# known = reduce_model(x[1:i])
					known = reduce_model(x[1:i-1])
					counts = get_conditional_counts(k, known)
					# counts[i+1] = 0 # conditional number of models where x[i] ∉ x[1:i-1]
					counts[i] = 0 # conditional number of models where x[i] ∉ x[1:i-1]
					probs = counts ./ sum(counts)
					idx = rand(Categorical(probs))

					# x[i+1] = x[idx]
					x[i] = x[idx]
				end
			else # sample new value
				# x[i+1] = rand(setdiff(1:k, view(x, 1:i)))
				x[i] = rand(setdiff(1:k, view(x, 1:i-1)))
			end
		end
		# @show y, x
		#i += 1
	end
end

nosamples = 100_000
k = 5
α = 3.0
β = 1.0
n = k - 1

D = BetaBinomial(n, α, β)


res = randEqualityBB(nosamples, D, k)
sampled_models = mapslices(x->reduce_model(x), res, dims = 1)

# nrand = 100_000
urns = collect(1:k)
D2 = BetaBinomialConditionalUrnDistribution(urns, 1, α, β)
# sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D2, empirical_model_probs, empirical_inclusion_probs)



function randEqualityBB2(nosamples, D)

	y = Matrix{Int}(undef, ntrials(D) + 1, nosamples)
	randEqualityBB2!(y, D)
	return y
end

function randEqualityBB2!(y::AbstractMatrix, D)
	for i in axes(y, 2)
		randEqualityBB2!(view(y, :, i), D)
	end
end

function randEqualityBB2!(x::AbstractVector, D)

	# TODO: log scale!
	n = length(x)
	n0 = n
	model_probs_by_incl = exp.(logpdf(D, 0:ntrials(D)) .- log.(expected_inclusion_counts(n)))
	r = 1

	x[1] = rand(1:n0)
	for i in 2:n0

		num = sum(model_probs_by_incl[k] * stirlings2r(n - 1, n0 - k + 1, r    ) for k in 1:n0)
		den = sum(model_probs_by_incl[k] * stirlings2r(n    , n0 - k + 1, r + 1) for k in 1:n0)

		if (rand() < r*num / (r*num + den))
			n -= 1

			known = reduce_model(x[1:i-1])
			counts = get_conditional_counts(k, known)
			counts[i] = 0 # conditional number of models where x[i] ∉ x[1:i-1]
			probs = counts ./ sum(counts)
			idx = rand(Categorical(probs))

			x[i] = x[idx]
		else
			r += 1

			x[i] = rand(setdiff(1:n0, view(x, 1:i-1)))
		end
	end
end

nosamples = 100_000
k = 3
α = 4.0
β = 1.0
ntrial = k - 1

D = BetaBinomial(ntrial, α, β)

res = randEqualityBB2(nosamples, D)
sampled_models = mapslices(x->reduce_model(x), res, dims = 1)

# nrand = 100_000
urns = collect(1:k)
D2 = BetaBinomialConditionalUrnDistribution(urns, 1, α, β)
# sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = get_empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = get_empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D2, empirical_model_probs, empirical_inclusion_probs)


function rstirling2(n::T, k::T, r::T) where T <: Integer

	n < r && return zero(T)
	(k > n || k < r) && return zero(T)
	n == k && return one(T)
	iszero(n) || iszero(k) && return zero(T)
	n == r && return one(T)

	return k * rstirling2(n - 1, k, r) + rstirling2(n - 1, k - 1, r) 

end

rmax = 3
nmax = 10
out = zeros(Int, nmax, nmax, rmax)
for r in 1:rmax, n in r:nmax, k in r:n
	out[n-(r-1), k-(r-1), r] = rstirling2(n, k, r)
end
out

generate_distinct_models(3)

rstirling2.(3, 1:3, 2)
rstirling2.(2, 1:3, 1)

rstirling2(3, 3, 2)

rstirling2.(3, 1:4, 1)
rstirling2.(4, 1:4, 2)


