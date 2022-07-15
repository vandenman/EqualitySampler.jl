using Test
import Combinatorics

@testset "combinatorialFunctions" begin

	nvals = 12
	kvals = 12
	rvals = 6
	@testset "comparison with Combinatorics" begin

		for strategy in (EqualitySampler.ExplicitStrategy, EqualitySampler.RecursiveStrategy)
			@testset "stirlings2, StirlingStrategy: $strategy" begin
				for n in 1:nvals, k in 1:kvals
					@test Combinatorics.stirlings2(n, k) == stirlings2(n, k, strategy)
				end
			end

			@testset "stirlings1, StirlingStrategy: $strategy" begin
				for n in 1:nvals, k in 1:kvals
					@test Combinatorics.stirlings1(n, k) == unsignedstirlings1(n, k, strategy)
				end
			end
		end

		@testset "bell numbers" begin
			for n in 1:12
				@test Combinatorics.bellnum(n) == bellnumr(n, 0)
			end
		end

		@testset "unsignedstirlings1" begin
			for n in 1:nvals, k in 1:kvals
				@test Combinatorics.stirlings1(n, k) == unsignedstirlings1(n, k)
			end
		end
	end

	for strategy in (EqualitySampler.ExplicitStrategy, EqualitySampler.RecursiveStrategy)
		@testset "compare r-stirlings2 against tables in Broder (1984), StirlingStrategy: $strategy" begin

			# Tables 1-3 of Broder (1984). The $r$-Stirling numbers
			reference_table1 = [
				1       0       0       0       0       0
				1       1       0       0       0       0
				1       3       1       0       0       0
				1       7       6       1       0       0
				1       15      25      10      1       0
				1       31      90      65      15      1
			]
			reference_table2 = [
				1       0       0       0       0       0
				2       1       0       0       0       0
				4       5       1       0       0       0
				8       19      9       1       0       0
				16      65      55      14      1       0
				32      211     285     125     20      1
			]
			reference_table3 = [
				1       0       0       0       0       0
				3       1       0       0       0       0
				9       7       1       0       0       0
				27      37      12      1       0       0
				81      175     97      18      1       0
				243     781     660     205     25      1
			]

			replication_table1 = [stirlings2r(n, k, 1, strategy) for n in 1:6, k in 1:6]
			replication_table2 = [stirlings2r(n, k, 2, strategy) for n in 2:7, k in 2:7]
			replication_table3 = [stirlings2r(n, k, 3, strategy) for n in 3:8, k in 3:8]

			@test reference_table1 == replication_table1
			@test reference_table2 == replication_table2
			@test reference_table3 == replication_table3

		end
	end

	@testset "compare r-Bell numbers against Mezo (2010)" begin

		# Figure 1 of Mezo (2010). The $r$-Bell numbers
		reference = [
			1       1       1       1       1       1       1
			1       2       3       4       5       6       7
			2       5       10      17      26      37      50
			5       15      37      77      141     235     365
			15      52      151     372     799     1540    2727
			52      203     674     1915    4736    10427   20878
			203     877     3263    10481   29371   73013   163967
		]

		replication = [bellnumr(r, n) for r in 0:6, n in 0:6]
		@test reference == replication
	end

	@testset "compare logstirlings2 against log(stirlings2)" begin
		for n in 1:nvals, k in 1:kvals
			@test logstirlings2(n, k) ≈ log(stirlings2(n, k))
		end
	end
	@testset "compare logstirlings2r against log(stirlings2r)" begin
		for n in 1:nvals, k in 1:kvals, r in 1:rvals
			@test logstirlings2r(n, k, r) ≈ log(stirlings2r(n, k, r))
		end
	end
	@testset "compare logunsignedstirlings1 against log(unsignedstirlings1)" begin
		for n in 1:nvals, k in 1:kvals
			@test logunsignedstirlings1(n, k) ≈ log(unsignedstirlings1(n, k))
		end
	end

	@testset "compare unsignedstirlings1 against wikipedia" begin

		reference = [
			1      0       0       0      0      0     0    0   0  0
			0      1       0       0      0      0     0    0   0  0
			0      1       1       0      0      0     0    0   0  0
			0      2       3       1      0      0     0    0   0  0
			0      6      11       6      1      0     0    0   0  0
			0     24      50      35     10      1     0    0   0  0
			0    120     274     225     85     15     1    0   0  0
			0    720    1764    1624    735    175    21    1   0  0
			0   5040   13068   13132   6769   1960   322   28   1  0
			0  40320  109584  118124  67284  22449  4536  546  36  1
		]

		for strategy in (EqualitySampler.ExplicitStrategy, EqualitySampler.RecursiveStrategy)

			replication = [unsignedstirlings1(n, k, strategy) for n in 0:9, k in 0:9]
			@test reference == replication

		end
	end

	@testset "compare count_set_partitions_given_partition_size against OEIS" begin

		# https://oeis.org/A036040
		reference = [1, 1, 1, 1, 3, 1, 1, 4, 3, 6, 1, 1, 5, 10, 10, 15, 10, 1, 1, 6, 15, 10, 15, 60, 15, 20, 45, 15, 1, 1, 7, 21, 35, 21, 105, 70, 105, 35, 210, 105, 35, 105, 21, 1, 1, 8, 28, 56, 35, 28, 168, 280, 210, 280, 56, 420, 280, 840, 105, 70, 560, 420, 56, 210, 28, 1, 1, 9, 36, 84, 126, 36, 252]
		replication = EqualitySampler.count_set_partitions_given_partition_size.(1:9, true)
		# the values listed on the OEIS stop after the first few terms of the 9th row
		replication = vcat(first.(replication)...)[eachindex(reference)]
		@test reference == replication

	end

	@testset "return types are inferred" begin

		n0 = 3
		k0 = 2
		r0 = 1
		for T in (Int, BigInt, Int128)
			n, k, r = T(n0), T(k0), T(r0)


			@inferred stirlings2(n, k)
			@inferred stirlings2r(n, k, r)
			@inferred bellnumr(n, r)
			@inferred unsignedstirlings1(n, k)

		end
	end
end
