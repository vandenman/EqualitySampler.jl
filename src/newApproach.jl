using Combinatorics, Test

function brute_force(k, f)
	combinations = Iterators.product(fill(1:k, k)...)
	return sum(f, combinations)
end

function brute_force(s::AbstractString)
	s = filter(!isspace, s)
	k = length(s)
	f = make_f_from_pattern(s)
	return brute_force(k, f)
end


count_combinations(k, islands) = factorial(islands) * binomial(k, islands)
function count_combinations(s::AbstractString)
	s = filter(!isspace, s)
	k = length(s)
	islands = length(unique(s))
	return count_combinations(k, islands)
end

count_combinations(x::AbstractVector) = count_combinations(length(x), length(unique(x)))


function make_f_from_pattern(pattern)

	str = "x -> "
	for i in 1:length(pattern) - 1, j in i+1:length(pattern)
		str *= pattern[i] == pattern[j] ? "x[$i] == x[$j]" : "x[$i] != x[$j]"

		if !(i == length(pattern) - 1 && j == length(pattern))
			str *= " && \n"
		end
	end
	f = eval(Meta.parse(str))
	return x -> Base.invokelatest(f, x)
end

function f1(x)
	# 11 222 33
	x[1] == x[2] &&
	x[3] == x[4] && x[3] == x[5] &&
	x[6] == x[7] &&
	x[1] != x[3] &&
	x[1] != x[6] &&
	x[3] != x[6]
end

function f2(x)
	# 11 22 33 4
	x[1] == x[2] &&
	x[3] == x[4] &&
	x[5] == x[6] &&
	x[1] != x[3] &&
	x[1] != x[5] &&
	x[1] != x[7] &&
	x[3] != x[5] &&
	x[3] != x[7] &&
	x[5] != x[7]
end

# brute_force("111 2")
# count_combinations(4, 1)
# count_combinations(4, 2)
# count_combinations(4, 3)
# count_combinations(4, 4)

# @testset "same examples as in R" begin
# 	@testset "210" begin
# 		@test brute_force(7, f1)					== 210
# 		@test count_combinations(7, 3)				== 210
# 		@test brute_force("11 222 33")				== 210
# 		@test count_combinations("11 222 33")		== 210
# 	end
# 	@testset "840" begin
# 		@test brute_force(7, f2)					== 840
# 		@test count_combinations(7, 4)				== 840
# 		@test brute_force("11 22 33 4")				== 840
# 		@test count_combinations("11 22 33 4")		== 840
# 	end
# end
# @testset "random examples" begin
# 	for k in 2:7, _ in 1:10
# 		pattern = join(string.(rand(1:k, k)))
# 		@test brute_force(pattern) == count_combinations(pattern)
# 	end
# end



