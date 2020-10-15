#=

	contains functions to brute force the conditional probabilities

=#

using Combinatorics, StatsBase, DataStructures


function invalid(x::Tuple)
	for i in eachindex(x)
		x[i] != x[x[i]] && return true
	end
	return false
end

const visited_models = Dict{Int, Matrix{Int}}()

function enumerate_models(n::Int)

	if haskey(visited_models, n)
		return visited_models[n]
	end

	no_models = Combinatorics.bellnum(n)
	models = Matrix{Int}(undef, no_models, n)
	count = 1
	for it in Iterators.product(UnitRange.(1, 1:n)...)
		invalid(it) && continue
		models[count, :] .= it
		count += 1
	end
	o = sortperm(collect(join(string.(row)) for row in eachrow(models)))
	res = models[o, :]
	visited_models[n] = res
	return res
end

function conditional_counts_s(n::Int, splitby::Vector{Int} = [1, 2])

	models = enumerate_models(n);

	index = findall(i->all(view(models, i, 1:length(splitby)) .== splitby), axes(models, 1))
	nsplit = length(splitby) + 1
	tb = zeros(Int, nsplit)
	if !isempty(index)
		dict = countmap(view(models, index, nsplit))
		for (k, v) in dict
			tb[k] = v
		end
	end
	return tb
end

function list_counts(n, patterns::Vector{Int})

	ddd = SortedDict{String, Vector{Int}}()
	k = length(patterns) + 1
	key = join(string.(patterns))
	ddd[key] = conditional_counts_s(n, patterns)
	ddd
end


function list_counts(n, patterns::Matrix{Int})

	ddd = SortedDict{String, Vector{Int}}()
	k = size(patterns)[2] + 1
	for i in axes(patterns, 1)
		p = patterns[i, :]
		key = join(string.(p))
		ddd[key] = conditional_counts_s(n, p)
	end
	ddd
end
