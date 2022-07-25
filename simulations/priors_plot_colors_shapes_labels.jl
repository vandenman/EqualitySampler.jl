import ColorSchemes

function instantiate_prior(symbol::Symbol, k::Integer)
	# this works nicely with jld2 but it's not type stable

	symbol == :uniform				&&	return UniformPartitionDistribution(k)
	symbol == :BetaBinomial11		&&	return BetaBinomialPartitionDistribution(k, 1.0, 1.0)
	symbol == :BetaBinomialk1		&&	return BetaBinomialPartitionDistribution(k, k, 1.0)
	symbol == :BetaBinomial1k		&&	return BetaBinomialPartitionDistribution(k, 1.0, k)
	symbol == :BetaBinomial1binomk2	&&	return BetaBinomialPartitionDistribution(k, 1.0, binomial(k, 2))
	symbol == :DirichletProcess0_5	&&	return DirichletProcessPartitionDistribution(k, 0.5)
	symbol == :DirichletProcess1_0	&&	return DirichletProcessPartitionDistribution(k, 1.0)
	symbol == :DirichletProcess2_0	&&	return DirichletProcessPartitionDistribution(k, 2.0)
	# symbol == :DirichletProcessGP	&&
	return DirichletProcessPartitionDistribution(k, :Gopalan_Berry)

end

function get_args_from_prior(symbol::Symbol)
	symbol == :uniform				&&	return ""
	symbol == :BetaBinomial11		&&	return "1.0, 1.0"
	symbol == :BetaBinomialk1		&&	return "k, 1.0"
	symbol == :BetaBinomial1k		&&	return "1.0, k"
	symbol == :BetaBinomial1binomk2	&&	return "1.0, binomial(k, 2)"
	symbol == :DirichletProcess0_5	&&	return "k, 0.5"
	symbol == :DirichletProcess1_0	&&	return "k, 1.0"
	symbol == :DirichletProcess2_0	&&	return "k, 2.0"
	# symbol == :DirichletProcessGP	&&
	return "Gopalan_Berry"

end

function get_labels(priors)
	lookup = Dict(
		:uniform				=> "Uniform",
		:BetaBinomial11			=> "BB α=1, β=1",
		:BetaBinomialk1			=> "BB α=K, β=1",
		:BetaBinomial1k			=> "BB α=1, β=K",
		:BetaBinomial1binomk2	=> "BB α=1, β=binom(K,2)",
		# :BetaBinomial1binomk2	=> L"\mathrm{BB}\,\,\alpha=K, \beta=\binom{K}{2}",
		:DirichletProcess0_5	=> "DPP α=0.5",
		:DirichletProcess1_0	=> "DPP α=1",
		:DirichletProcess2_0	=> "DPP α=2",
		# :DirichletProcessGP		=> "DPP α=Gopalan & Berry",
		:DirichletProcessGP		=> "DPP α=G&B",
		:Westfall				=> "Westfall",
		:Westfall_uncorrected	=> "Pairwise BFs",
	)
	priors_set = sort!(unique(priors))
	# flip order of BB per request Fabian
	# i1 = findfirst(x -> x === :BetaBinomial1k, priors_set)
	# i2 = findfirst(x -> x === :BetaBinomial1binomk2, priors_set)
	# if !isnothing(i1) && !isnothing(i2)
	# 	priors_set[i1], priors_set[i2] = priors_set[i2], priors_set[i1]
	# end

	return reshape([lookup[prior] for prior in priors_set], 1, length(priors_set))
end
get_labels(priors::Symbol) = get_labels((priors, ))

function get_colors(priors, alpha = 0.75)
	colors = ColorSchemes.alphacolor.(ColorSchemes.seaborn_colorblind[1:10], alpha)
	lookup = Dict(
		:uniform              => colors[1],
		:BetaBinomial11       => colors[2],
		:BetaBinomialk1       => colors[3],
		:BetaBinomial1k       => colors[4],
		:BetaBinomial1binomk2 => colors[5],
		:DirichletProcess0_5  => colors[6],
		:DirichletProcess1_0  => colors[7],
		:DirichletProcess2_0  => colors[8],
		:DirichletProcessGP   => colors[9],
		:Westfall             => colors[10],
		:Westfall_uncorrected => colors[3]
	)
	return [lookup[prior] for prior in priors]
end
get_colors(priors::Symbol, alpha = 0.75) = get_colors((priors, ), alpha)

function get_shapes(priors)
	lookup = Dict(
		:uniform              => :rect,
		:BetaBinomial11       => :utriangle,
		:BetaBinomialk1       => :rtriangle,
		:BetaBinomial1k       => :ltriangle,
		:BetaBinomial1binomk2 => :dtriangle,
		:DirichletProcess0_5  => :star4,
		:DirichletProcess1_0  => :star5,
		:DirichletProcess2_0  => :star6,
		:DirichletProcessGP   => :star8,
		:Westfall             => :circle,
		:Westfall_uncorrected => :circle
	)
	return [lookup[prior] for prior in priors]
end
get_shapes(priors::Symbol) = get_shapes((priors, ))