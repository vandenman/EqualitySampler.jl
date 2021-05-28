using EqualitySampler, BenchmarkTools, ProfileView

n_groups = 20
D = BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0)
Base.summarysize(D.urns)
@btime rand($D)
@btime logpdf($D, 1)

VSCodeServer.@profview rand(D, 100000)
VSCodeServer.@profview [logpdf(D, 1) for _ in 1:20]

function _pdf_helper_orig!(result, ::Union{UniformConditionalUrnDistribution, UniformMvUrnDistribution}, index, complete_urns)

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	urns = view(complete_urns, 1:index - 1)

	count = EqualitySampler.get_conditional_counts(k, urns)

	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
	other = setdiff(1:k, urns)
	result[other] .= count[length(urns) + 1] ./ length(other)
	result ./= sum(result)
	return
end

function _pdf_helper_fast_1!(result, ::Union{UniformConditionalUrnDistribution, UniformMvUrnDistribution}, index, complete_urns)

	k = length(result)
	if isone(index)
		fill!(result, 1 / k)
		return
	end

	urns = view(complete_urns, 1:index - 1)

	count = EqualitySampler.get_conditional_counts(k, urns)

	idx_nonzero = findall(!iszero, view(count, 1:length(urns)))
	result[view(urns, idx_nonzero)] .= count[idx_nonzero]
	other = setdiff(1:k, urns)
	result[other] .= count[length(urns) + 1] ./ length(other)
	result ./= sum(result)
	return
end

n_groups = 20
D = BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0)
result = zeros(Float64, length(D))
index = D.index
complete_urns = D.urns

_pdf_helper_orig!(result, D, index, complete_urns)