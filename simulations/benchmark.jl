#=

	for random benchmarks

=#

using BenchmarkTools

function original(known)

	known = reduce_model(known)
	n_known = length(known)
	findall(i->known[i] == i, 1:n_known) # This step fails when not doing reduce_model

end

function try2(known)

	idx = Vector{Int}(undef, length(Set(known)))
	s = Set{Int}()
	count = 1
	for i in eachindex(known)
		if known[i] ∉ s
			idx[count] = i
			count += 1
			push!(s, known[i])
		end
	end
	idx
end

val = [1, 1, 4, 4, 7, 8]
original(val)
try2(val)

k = 5
val = rand(1:k, k)
for _ in 1:100
	val = rand(1:k, k)
	@assert original(val) == try2(val)
end

@btime original($val)
@btime try2($val)