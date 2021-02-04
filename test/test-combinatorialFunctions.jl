using Test
import Combinatorics

@testset "combinatorialFunctions" begin

	@testset "comparison with Combinatorics" begin

		nvals = 8
		kvals = 8
		for strategy in (ExplicitStrategy, RecursiveStrategy)
			@testset "stirlings2, StirlingStrategy: $strategy" begin
				for n in 1:nvals, k in 1:kvals
					@test Combinatorics.stirlings2(n, k) == stirlings2(n, k, strategy)
				end
			end
		end

		@testset "bell numbers" begin
			for n in 1:12
				@test Combinatorics.bellnum(n) == bellnumr(n, 0)
			end
		end
	end

	for strategy in (ExplicitStrategy, RecursiveStrategy)
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

	nvals = 8
	kvals = 8
	rvals = 4
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
end

