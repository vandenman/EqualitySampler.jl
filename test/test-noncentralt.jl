using Test, EqualitySampler
import Distributions, Suppressor

@testset verbose = true "noncentral T distribution" begin
    @testset "comparison with R" begin
        # R > dput(dt(seq(-5, 5, .1), 3, 1.5, log = TRUE)) # to get ref
        ref = [
            -8.8480569704851,  -8.77187815307963, -8.69431005312623, -8.61530473450975,
            -8.53481197728393, -8.45277915291473, -8.36915109385416, -8.28386995768451,
            -8.19687508626041, -8.10810286048487, -8.01748655166475, -7.92495617074924,
            -7.83043831723916, -7.73385603014917, -7.63512864417141, -7.53417165515144,
            -7.43089660019664, -7.32521095926271, -7.21701808697562, -7.106217185444,
            -6.99270333466955, -6.87636759196045, -6.75709719181388, -6.63477586738158,
            -6.50928433167534, -6.38050096195239, -6.24830274025473, -6.11256651989266,
            -5.97317069480851, -5.82999737383521, -5.68293517706918, -5.53188279180889,
            -5.37675345227604, -5.21748052188182, -5.05402437795931, -4.88638080009038,
            -4.71459105279181, -4.53875381063014, -4.35903898595045, -4.1757033672766,
            -3.98910774860385, -3.79973489403872, -3.60820724736284, -3.4153027761056,
            -3.22196677678652, -3.02931698747339, -2.83863908698046, -2.65136982569098,
            -2.46906579342874, -2.29335729138328, -2.12588884962351, -1.96825030031008,
            -1.82190445872725, -1.68811877461147, -1.56790834499338, -1.46199628874703,
            -1.37079497207299, -1.29440854978088, -1.23265446716051, -1.18509952430918,
            -1.15110511716057, -1.12987630183653, -1.12051011790924, -1.12203980537674,
            -1.13347283521116, -1.15382181852752, -1.18212824443808, -1.21747959280601,
            -1.25902070525565, -1.30596043232637, -1.35757456959419, -1.41320600606794,
            -1.47226287679807, -1.53421536764092, -1.59859168204178, -1.66497355685235,
            -1.73299161098623, -1.80232072719859, -1.87267560194046, -1.94380654876377,
            -2.01549560399379, -2.0875529571012,  -2.15981370983334, -2.23213495581866,
            -2.30439316463255, -2.3764818495566,  -2.44830949614564, -2.51979772792442,
            -2.5908796859312,  -2.6614985998599,  -2.73160652999195, -2.80116326075528,
            -2.87013532846981, -2.9384951675475,  -3.00622036105513, -3.07329298308631,
            -3.139699021807,   -3.20542787333049, -3.27047189774261, -3.33482602964046,
            -3.39848743647671
        ]
        julia_rep = Suppressor.@suppress Distributions.logpdf.(Distributions.NoncentralT(3, 1.5), -5:.1:5)

        self_rep = EqualitySampler.noncentralt_logpdf.(-5:.1:5, 3, 1.5)

		@test ref ≈ julia_rep
		@test ref ≈ self_rep

    end

	@testset "comparison with Distributions.jl" begin

		δs = -4:.1:4
		νs = [0.1:.1:.9 ; 1.0]# ; 1:.5:10]
		xs = -15:.1:15

		ref = similar(xs)
		rep = similar(xs)
		for δ in δs, ν in νs

			ref .= Suppressor.@suppress Distributions.logpdf.(Distributions.NoncentralT(ν, δ), xs)
			rep .= EqualitySampler.noncentralt_logpdf.(xs, ν, δ)

			# @show δ, ν
			@test ref ≈ rep atol = 1e-1

		end

    end

	# @testset "Integer degrees of freedom" begin

	# 	# NOTE: this comparison is a bit tricky, because the Distributions.jl implementation
	# 	# uses R which uses a pretty inaccurate algorithm. Using big however makes things rather slow.

	# 	δs = range(-25, 25, 11)
	# 	νs = [1:2:9; 10:20:90; 100:200:900; 1000:2000:9000]
	# 	xs = range(-30, 30, 11)

	# 	tol = log(eps(Float64)) / 2

	# 	ν, δ = first(νs), first(δs)
	# 	for ν in νs, δ in δs
	# 		ref  = Suppressor.@suppress Distributions.logpdf.(Distributions.NoncentralT(ν, δ), xs)
	# 		rep = EqualitySampler.noncentralt_logpdf.(xs, ν, δ)

	# 		validref = isfinite.(ref) .& .!isnan.(ref)
	# 		vref = view(ref, validref)
	# 		vrep = view(rep, validref)
	# 		check = vref .≈ vrep .|| (vrep .< tol .&& vrep .< vref .+ 1)
	# 		bad = findall(.!check)
	# 		if iszero(length(bad))# && @show bad, ν, δ
	# 			@test all(check)
	# 		else
	# 			xs2 = xs[findall(validref)[bad]]
	# 			rep2 = EqualitySampler.noncentralt_logpdf.(big.(xs2), big(ν), big(δ))

	# 			check[bad] .= vrep[bad] .≈ rep2 #abs.(vrep[bad] .- rep2) .<= 1
	# 			# !all(check[bad]) && @show ν, δ
	# 			@test all(check)
	# 		end
	# 	end

	# end

	#=
	@testset "limiting behavior" begin

		@testset "fixed ν, large δ" begin

			δs = 100.0:100:10_000
			ν = 1.0
			xs = -1000.:100:1000
			xs_b = BigFloat(-1000):big(100):big(1000)

			ref = similar(xs)
			rep = similar(xs)
			ref_big = similar(xs_b)
			idx = trues(length(xs))
			for δ in δs

				ref .= Suppressor.@suppress Distributions.logpdf.(Distributions.NoncentralT(ν, δ), xs)
				rep .= EqualitySampler.noncentralt_logpdf.(xs, ν, δ)

				ref_big .= EqualitySampler.noncentralt_logpdf.(xs_b, big(ν), big(δ))

				# no positive infinities
				@test all(x-> isfinite(x) || x < zero(x), rep)

				idx .= isfinite.(ref)
				ref_finite     = @view ref[idx]
				rep_finite     = @view rep[idx]
				# ref_big_finite = @view ref_big[idx]

				@test rep_finite ≈ ref_finite
				@test rep ≈ ref_big

				# Distributions.logpdf.(Distributions.NoncentralT(ν, δ), xs[60])
				# EqualitySampler.noncentralt_logpdf(xs[60], ν, δ)
				# EqualitySampler.noncentralt_logpdf(big(xs[2]), big(ν), big(δ))

			end


		end

		δs = -4:.1:4
		νs = [0.1:.1:.9 ; 1.0]# ; 1:.5:10]
		xs = -15:.1:15

		for δ in δs, ν in νs

			ref = Suppressor.@suppress Distributions.logpdf.(Distributions.NoncentralT(ν, δ), xs)
			rep = EqualitySampler.noncentralt_logpdf.(xs, ν, δ)

			# @show δ, ν
			@test ref ≈ rep atol = 1e-1

		end

    end
	=#

end
