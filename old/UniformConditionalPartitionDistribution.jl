import Distributions, Random, Combinatorics
import Turing: TArray

struct UniformConditionalPartitionDistribution{T} <: Distributions.DiscreteUnivariateDistribution where T <: Integer
	partitions::Union{Vector{T}, TArray{T,1}}
	index::T
end

function UniformConditionalPartitionDistribution(partitions::Union{Vector{T}, TArray{T,1}}, index::T) where T <: Integer
	n = length(partitions)
	any(x-> 0 < x < n, partitions) || throw(DomainError(partitions, "condition: 0 < partitions[i] < length(partitions) ∀i is violated"))
	0 < index <= n || throw(DomainError(partitions, "condition: 0 < index < length(paritions) is violated"))
	UniformConditionalPartitionDistribution{T}(partitions, index)
end

function Distributions.rand(::Random.AbstractRNG, d::UniformConditionalPartitionDistribution{T}) where T <: Integer

	counts = get_conditional_counts(length(d.partitions), d.partitions[1:d.index-1])
	return rand(Distributions.Categorical(counts ./ sum(counts)), 1)[1]

end

Distributions.logpdf(d::UniformConditionalPartitionDistribution, x::Real) = -Inf
Distributions.pdf(d::UniformConditionalPartitionDistribution{T}, x::Real) where T <: Integer = exp(logpdf(d, x))

function Distributions.logpdf(d::UniformConditionalPartitionDistribution{T}, x::T) where T <: Integer

	one(T) <= x <= length(d.partitions) || return -Inf
	counts = get_conditional_counts(length(d.partitions), d.partitions[1:d.index-1])
	return log(counts[x]) - log(sum(counts))

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

function get_conditional_counts(n::Int, known::Union{Vector{T}, TArray{T,1}} = [1]) where T <: Integer

	# TODO: just pass d.partitions and d.index to this function! (or just d itself)
	refinement = length(unique(known))
	n_known = length(known)

	res = zeros(Int, n_known + 1)
	idx = findall(i->known[i] == i, 1:n_known)
	n_idx = length(idx)
	res[idx] .= BN(n - n_known - 1, n_idx)
	res[n_known + 1] = BN(n - n_known - 1, n_idx + 1)
	res
end

function get_conditional_counts(n::Int, known::Matrix{Int})

	nrows, ncols = size(known)
	res = Matrix{Int}(undef, nrows, ncols + 1)
	for i in axes(known, 1)
		res[i, :] .= get_conditional_counts(n, known[i, :])
	end
	res
end

function sample_all(start::Vector{Int}, n::Int = 1)

	u = copy(start)
	umat = Matrix{Int}(undef, length(u), n)
	for j in 1:n
		for i in 2:length(u)
			u[i] = rand(UniformConditionalPartitionDistribution(u, i), 1)[1]
		end
		umat[:, j] .= copy(u)
	end
	return umat
end

function count_probs_and_models(start::Vector{Int}, nmax::Int = 10_000)

	u = copy(start)
	table = Dict{String, Int}()
	probs = zeros(Float64, length(u), length(u))

	for k in 1:nmax
		for i in 2:length(u)
			u[i] = rand(UniformConditionalPartitionDistribution(u, i), 1)[1]
		end

		for j in eachindex(u)
			idx = j .+ findall(==(u[j]), u[j+1:end])
			probs[idx, j] .+= 1.0
		end

		key = join(string.(u))
		if haskey(table, key)
			table[key] += 1
		else
			table[key] = 1
		end
	end
	return table, probs ./ nmax

end

# sample_all([1, 2, 3], 100)
# p_model, p_equal = count_probs_and_models([1, 2, 3], 100_000)
