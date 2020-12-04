using Distributions, StatsBase, Random, DataStructures, Combinatorics, LinearAlgebra


# struct ConditionalPartitionDistribution{T} <: Distributions.DiscreteUnivariateDistribution where T <: Integer
# 	connections::Vector{T}
# 	index::T
# 	function ConditionalPartitionDistribution(connections::Vector{T}, index::T) where T <: Integer
# 		n = length(connections)
# 		any(x-> 0 < x < n, connections) || throw(DomainError(connections, "condition 0 < connections[i] < length(connections) ∀i is violated")) 
# 		0 < index <= n || throw(DomainError(connections, "condition 0 < index < length(connections) is violated"))
# 		new{T}(connections, index)
# 	end
# end

# function Distributions.rand(::AbstractRNG, d::ConditionalPartitionDistribution{T}) where T <: Integer

# 	vertices = 1:d.index
# 	valid = d.connections[vertices] .== vertices
# 	valid[d.index] = 1
# 	current_view = @view vertices[valid]

# 	# decide to increment or decrement the size


# 	idx = sample(current_view, 1; replace = false)[1]
# 	return idx

# end

# function Distributions.logpdf(d::ConditionalPartitionDistribution{T}, x::Vector{U}) where {T <: Integer, U <: Integer}

# end


function count_connections(ni = 4, nj = 10_000)
	umat = zeros(Int, ni, nj)
	probs = zeros(Float64, ni, ni)
	u = collect(1:ni)
	for j in 1:nj
		for i in 1:ni
			d = ConditionalPartitionDistribution(u, i)
			u[i] = rand(d, 1)[1]
		end
		umat[:, j] .= u
		for j in eachindex(u)
			idx = j .+ findall(==(u[j]), u[j+1:end])
			probs[idx, j] .+= 1.0
		end
	end
	umat, probs ./ nj
end
aa, bb = count_connections(4)
bb
function count_models(umat)

	ss = Vector{String}(undef, size(umat)[2])
	for i in eachindex(ss)
		ss[i] = join(string.(umat[:, i]))
	end
	StatsBase.countmap(ss)

end
cc = count_models(aa)
[o => cc[o] for o in order]
function count_models_size(umat)
	ss = zeros(Int, size(umat, 2))
	for i in eachindex(ss)
		ss[i] = length(unique(umat[:, i]))
	end
	StatsBase.countmap(ss)
end
count_models_size(aa)

function invalid(x::Tuple)
	for i in eachindex(x)
		x[i] != x[x[i]] && return true
	end
	return false
end
function enumerate_models(n::Int)
	no_models = Combinatorics.bellnum(n)
	models = Matrix{Int}(undef, no_models, n)
	count = 1
	for it in Iterators.product(UnitRange.(1, 1:n)...)
		invalid(it) && continue
		models[count, :] .= it
		count += 1
	end
	o = sortperm(collect(join(string.(row)) for row in eachrow(models)))
	models[o, :]
end

function belltriangle(n::T) where T <: Integer
	L = LowerTriangular(zeros(T, n, n))
	L[1] = 1
	for i in 2:n
		L[i, 1] = L[i-1, i-1]
		for j in 2:i
			L[i, j] = L[i, j-1] + L[i - 1, j-1]
		end
	end
	return L
end

function conditional_counts(n::Int, splitby::Tuple = (1, 2, ))

	models = enumerate_models(n);
	tb_all = collect(countmap(x) for x in eachcol(models))

	dd = Dict{String, Vector{Dict{Int, Int}}}()
	splits = Iterators.product(UnitRange.(1, splitby)...)
	nsplits = length(splitby)
	for it in splits
		@show it
		index = findall(i->all(view(models, i, 1:nsplits) .== it), axes(models, 1))
		isempty(index) && continue
		tb = SortedDict.(collect(countmap(x) for x in eachcol(view(models, index, :))))
		key = join(string.(it))
		println(key)
		display(tb)
		dd[key] = tb
	end
	display(belltriangle(n))
	return tb_all, dd
end

function conditional_counts_s(n::Int, splitby::Tuple = (1, 2, ); verbose::Bool = true)

	models = enumerate_models(n);
	tb_all = collect(countmap(x) for x in eachcol(models))

	dd = Dict{String, Vector{SortedDict{Int, Int}}}()
	it = splitby
	index = findall(i->all(view(models, i, 1:length(it)) .== it), axes(models, 1))
	if !isempty(index)
		tb = SortedDict.(collect(countmap(x) for x in eachcol(view(models, index, :))))
		key = join(string.(it))
		if verbose
			println(key)
			display(tb)
		end
		dd[key] = tb
	end
	# verbose && display(belltriangle(n))
	return dd
end

myprint(x) = show(stdout, "text/plain", x)

myprint(enumerate_models(5))

ee = conditional_counts(5, (1, ));
ee = conditional_counts(5);
ee = conditional_counts(5, (1, 2, 3));
ee = conditional_counts.(3:5, Ref((1, )));
ee = conditional_counts.(3:5, Ref((1, 2)));

conditional_counts_s(5, (1,  ));
conditional_counts_s(5, (1, 1));
conditional_counts_s(5, (1, 2));

conditional_counts(5, (1, 2, 3));
conditional_counts_s(5, (1, 2));

conditional_counts_s(5, (1, 1, 3));
conditional_counts_s(5, (1, 2, 1));

conditional_counts_s(6, (1, 1, 3, 3));
conditional_counts_s(6, (1, 2, 1, 2));

function get_unconditional_counts(n::Int)
	bt = belltriangle(n)
	res = Vector{SortedDict{Int, Int}}(undef, n)
	for i in 1:n
		res[i] = SortedDict(1:i .=> [bt[n-1, n-1:-1:n-i+1]..., bt[n, n-i+1]])
	end
	return res
end
get_unconditional_counts(5)
conditional_counts(5, (1, ));


conditional_counts_s(5, (1, ));
get_unconditional_counts(5)
get_conditional_counts(5)


conditional_counts_s(4, (1, ));
get_conditional_counts(4)
conditional_counts_s(4, (1, 1, ));
get_unconditional_counts(3)

conditional_counts_s(4, (1, 2, ));



collect((2^p, 2^(p*(p-1) ÷ 2), bellnum(p)) for p in 2:10)

time1 = median(@benchmark get_unconditional_counts(7);)
time2 = median(@benchmark conditional_counts_s(7, (1, ), verbose = false);)
judge(time1, time2)

get_elem(x) = first(x)[2][5]

function list_counts(n, patterns)

	ddd = SortedDict{String, SortedDict{Int, Int}}()
	k = length(first(patterns)) + 1
	for p in patterns
		local key = join(string.(p))
		ddd[key] = first(conditional_counts_s(n, p, verbose=false))[2][k]
	end
	ddd
end

function get_conditional_counts(n::Int, known::Vector{Int} = [1])

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

get_conditional_counts.(
	2:5,
	Ref([
		1
	])
)

get_conditional_counts.(
	3:5,
	Ref([
		1 1
		1 2
	])
)


get_conditional_counts.(
	4:8,
	Ref([
		1 1 1
		1 1 3
		1 2 1
		1 2 3
	])
)

get_conditional_counts.(4:8, Ref([1, 1, 1]))
get_conditional_counts.(4:8, Ref([1, 1, 3]))
get_conditional_counts.(4:8, Ref([1, 2, 1]))
get_conditional_counts.(4:8, Ref([1, 2, 3]))

count4 = list_counts.(4:8,
	Ref([
		(1, 1, 1),
		(1, 1, 3),
		(1, 2, 1),
		(1, 2, 2),
		(1, 2, 3)
	])
)

count5_2 = get_conditional_counts.(
	5:9,
	Ref([
		1 1 1 1
		1 1 1 4
		1 1 3 1
		1 1 3 3
		1 1 3 4
		1 2 3 4
	])
)

count5 = list_counts.(5:9,
	Ref([
		(1, 1, 1, 1),
		(1, 1, 1, 4),
		(1, 1, 3, 1),
		(1, 1, 3, 3),
		(1, 1, 3, 4),
		(1, 2, 3, 4)
	])
)



belltriangle(9)

bellnum2(n) = sum(k-> k*stirlings2(n, k), 1:n)
bellnum3(n) = sum(i-> BigInt(3.0^(n-i)) * bellnum(i) * binomial(BigInt(n), i), UnitRange{BigInt}(0, n))
bellnum2.(1:10)
bellnum3.(1:10)

bt = belltriangle(10)
diag(bt, 0)
diag(bt, -1)
diag(bt, -2)

egf(x, n) = exp(n*x + exp(x) - 1)
egf.(1:5, 2)


temp(x, r) =x^3+ (3r+ 3)x^2+ (3r^2+ 3r+ 1)x+r^3
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
[BN.(3, 0:mm) temp.(1, 0:mm)]