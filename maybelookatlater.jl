using EqualitySampler, Distributions
import LogExpFunctions

m = Matrix(PartitionSpace(4))
m2 = vcat(m, map(length ∘ unique, eachcol(m))')
m2 = m2[:, sortperm(m2[end, :])]

[count(x->length(unique(x)) == j, eachcol(m)) for j in 1:4]
stirlings2.(4, 1:4)

m
stirlings2r.(3, 1:4, 1)
m2[:, findall(x->x[1] == x[2], eachcol(m2))]

stirlings2r.(4, 1:4, 2)
m2[:, findall(x->x[1] != x[2], eachcol(m2))]

count(x->x[1] != x[2], eachcol(m2))
bellnumr(2, 2)
stirlings2r.(4, 1:4, 2)
[stirlings2r.(2+2, k+2, 2) for k in 0:2]
sum(k->stirlings2r.(2+2, k+2, 2), 0:2)


bellnumr(1, 3)
count(x->x[1] != x[2] && x[1] != x[3] && x[2] != x[3], eachcol(m))

[stirlings2r.(4-r, 2, r) for r in 1:4]


d = DirichletProcessPartitionDistribution(5, .3)

d.α / (d.α + (k - 1))
prediction_rule(d, j)

d = DirichletProcessPartitionDistribution(2, .3)
d.α / (d.α + (length(d) - 1))
prediction_rule(d, j)

d = DirichletProcessPartitionDistribution(2, .3)
x = rand(d, 100_000)
EqualitySampler.reduce_model
for i in axes(x, 2)
    x[:, i] = EqualitySampler.reduce_model_2(x[:, i])
end


d.α / (d.α + (length(d) - 1))
sum(x[1, ] .!= x[2, :]) / size(x, 2)



du = UniformPartitionDistribution(3)
x = rand(du, 100_000)
for i in axes(x, 2)
    x[:, i] = EqualitySampler.reduce_model_2(x[:, i])
end
eq_tb = zeros(length(du), length(du))
for i in 1:3
    for j in 1:3
        eq_tb[i, j] = sum(x[i, :] .== x[j, :]) / size(x, 2)
    end
end


prediction_rule(du, 1)
prediction_rule(du, 2)

m = Matrix(PartitionSpace(3))
map(x->pdf(du, x), eachcol(m))

prediction_rule(UniformPartitionDistribution(2), 0)
prediction_rule(UniformPartitionDistribution(3), 1)
(1 - prediction_rule(UniformPartitionDistribution(3), 2)) * prediction_rule(UniformPartitionDistribution(3), 1)
prediction_rule(UniformPartitionDistribution(3), 2)

@edit prediction_rule(UniformPartitionDistribution(3), 1)

k = length(du)
r = 1
partition = ones(typeof(k), k)
    partition[k-r+1:k-1] .= 2:r
    probvec = zeros(k)
EqualitySampler._pdf_helper!(probvec, d, k, partition, zeros(Int, k))
sum(probvec[eachindex(probvec) .∉ Ref(1:r)])

prediction_rule(UniformPartitionDistribution(2), 1)
prediction_rule(UniformPartitionDistribution(3), 1)
prediction_rule(UniformPartitionDistribution(3), 2)

function conditional_probs(d::UniformPartitionDistribution, i::Int, partition::Vector{Int})
    k = length(d)
    probvec = zeros(k)
    EqualitySampler._pdf_helper!(probvec, d, i, partition, zeros(Int, k))
    return probvec
    # return sum(probvec[partition .== u] for u in unique(partition[1:i-1]))
end

partition = [1, 0, 0]
e10 = conditional_probs(du, 2, [1, 0, 0])
e1 = [e10[1]; e10[2] + e10[3]]
e20 = conditional_probs(du, 3, [1, 1])
e2 = [e20[1]; e20[2] + e20[3]]
e3 = conditional_probs(du, 3, [1, 2])

p_du(k, j, r)  = bellnumr(k - j, r + one(r)) / (bellnumr(k - j, r + one(r)) + r * bellnumr(k - j, r))
p_du2(k, j, r) = bellnumr(k - j, r + one(r)) / bellnumr(k - j + 1, r)

p_du(5, 2, 1)
bellnumr(5 - 2 + 1, 1)
1 / bellnum(5)
bellnumr(5 - 5, 2)
bellnumr(5 - 5, 2)
bellnumr.(0, 2:5)

bellnumr(5 - 2, 2)
bellnumr(5 - 2 + 1, 1)

bellnumr(5 - 3, 2)
bellnumr(5 - 3 + 1, 2)



p_du(3, 2, 1) ≈ p_du2(3, 2, 1)
p_du(3, 3, 1) ≈ p_du2(3, 3, 1)
p_du(3, 3, 2) ≈ p_du2(3, 3, 2)

(1 - p_du2(k, j, r)) ≈ r * bellnumr(k - j, r) / bellnumr(k - j + 1, r)

prob_tie

d = BetaBinomialPartitionDistribution(4, 1.5, 1)
prediction_rule(d, 2)

k = length(d)
r = 2
index = 3
# r = sum(!iszero, tb)
n = k - (index - r - one(r))

r = oftype(n, r)
k = oftype(n, k)

log_incl_probs = EqualitySampler.log_model_probs_by_incl(d)
log_num = LogExpFunctions.logsumexp(
    log_incl_probs[i] + logstirlings2r(n - one(n), i, r    )
    for i in 1:k
)
log_den = LogExpFunctions.logsumexp(
    log_incl_probs[i] + logstirlings2r(n    , i, r + one(r))
    for i in 1:k
)

num = r * exp(log_num)
den = 	  exp(log_den)

prob_new_label = den / (den + num)

incl_probs = exp.(log_incl_probs)
num_stir = [stirlings2r(n - one(n), i, r    )      for i in 1:k]
den_stir = [stirlings2r(n         , i, r + one(r)) for i in 1:k]
den_stir ≈ [stirlings2r(k - (index - r - one(r)), i, r + one(r)) for i in 1:k]

sum(incl_probs .* den_stir) / (r * sum(incl_probs .* num_stir) + sum(incl_probs .* den_stir))
sum(incl_probs .* den_stir) / sum(r * incl_probs .* num_stir + incl_probs .* den_stir)
sum(incl_probs .* den_stir) / sum(incl_probs .* (r .* num_stir .+ den_stir))

n_min_r_p1 =
(r .* num_stir .+ den_stir) ≈
    [r * stirlings2r(n - one(n), i, r    ) + stirlings2r(n, i, r + one(r)) for i in 1:k] ≈
    [r * stirlings2r(k - index + r, i, r    ) + stirlings2r(k - index + r + 1, i, r + one(r)) for i in 1:k] ≈
    [r * stirlings2r(k - index + r, i, r    ) + stirlings2r(k - index + r + 1, i, r + one(r)) for i in 1:k] ≈
    [stirlings2r(k - index + r + 1, i, r) for i in 1:k]

get_num_stir(k, index, r)      = [stirlings2r(k - index + r,     i, r)     for i in 1:k]
get_den_stir(k, index, r)      = [stirlings2r(k - index + r + 1, i, r + 1) for i in 1:k]
get_den_stir_simp(k, index, r) = [stirlings2r(k - index + r + 1, i, r)     for i in 1:k]
den_stir_simp = [stirlings2r(k - index + r + 1, i, r) for i in 1:k]

1 - sum(incl_probs .* den_stir) / sum(incl_probs .* (r .* num_stir .+ den_stir))
r * sum(incl_probs .* get_num_stir(k, index, r)) / sum(incl_probs .* get_den_stir_simp(k, index, r))


sum(incl_probs .* den_stir) / sum(incl_probs .* (r .* num_stir .+ den_stir))
sum(incl_probs .* den_stir) / sum(incl_probs .* den_stir_simp)
sum(incl_probs .* get_den_stir(k, index, r)) / sum(incl_probs .* get_den_stir_simp(k, index, r))

# {{}} -> r = 0
get_den_stir(4, 1, 0), get_den_stir_simp(4, 1, 0)
# {{1,}} -> r = 1
get_den_stir(4, 2, 1), get_den_stir_simp(4, 2, 1)
# {{1}, {2}} -> r = 2
get_den_stir(4, 3, 2), get_den_stir_simp(4, 3, 2)
# {{1, 3}, {2}} or {{1}, {2, 3}} r -> 2
get_num_stir(4, 4, 2), get_den_stir_simp(4, 4, 2)

get_den_stir(4, 4, 2), get_den_stir_simp(4, 4, 2)
# {{1}, {2}, {3]}} r -> 3
get_den_stir(4, 4, 3), get_den_stir_simp(4, 4, 3)

last(incl_probs) / (incl_probs' * get_den_stir_simp(4, 1, 0))
pdf(d, [1, 2, 3, 4])

pdf(d, [1, 2, 3, 3])
exp.(EqualitySampler.log_model_probs_by_incl(d))
dot(incl_probs, get_den_stir(4, 1, 0)) / dot(incl_probs, get_den_stir_simp(4, 1, 0)) *
    dot(incl_probs, get_den_stir(4, 2, 1)) / dot(incl_probs, get_den_stir_simp(4, 2, 1)) *
    dot(incl_probs, get_den_stir(4, 3, 2)) / dot(incl_probs, get_den_stir_simp(4, 3, 2)) *
    dot(incl_probs, get_num_stir(4, 4, 3)) / dot(incl_probs, get_den_stir_simp(4, 4, 3)) ≈
    dot(incl_probs, get_num_stir(4, 4, 3)) / dot(incl_probs, get_den_stir_simp(4, 2, 1))


[stirlings2r(k - index + r + 1, i, r) for i in 1:k]
[stirlings2( k - index + r + 1, i) for i in 1:k]

[stirlings2r(r,     i, r)     for i in 1:k]
[stirlings2r(r + 1, i, r + 1) for i in 1:k]

[stirlings2r(k - k + r + 1, i, r) for i in 1:k]
[stirlings2( k - k + r + 1, i) for i in 1:k]

12
5
[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

j2 = 2
jk = k
get_den_stir_simp(k, j2, 1)
get_num_stir(k, jk, 4)

dot(incl_probs, get_num_stir(4, 4, 3)) / dot(incl_probs, get_den_stir_simp(4, 2, 1))

[stirlings2r(k - index + r + 1, i, r + one(r)) for i in 1:k]
[stirlings2r(k - index + r + 1, i, r         ) for i in 1:k]
Matrix(PartitionSpace(k))


stirlings2r(n    , 2, r + one(r))
stirlings2r(n    , 2, r) - (r - 1) * stirlings2r(n - one(n), 2, r)


ds = [
    DirichletProcessPartitionDistribution(5, :harmonic),
    DirichletProcessPartitionDistribution(15, :harmonic),
    DirichletProcessPartitionDistribution(25, :harmonic)
]
modelspace = Matrix(PartitionSpace(5))
sizes = map(EqualitySampler.count_parameters, eachcol(modelspace))
lpdfs = map(x->logpdf(DirichletProcessPartitionDistribution(5, :harmonic), x), eachcol(modelspace))

max_lpdfs_by_size = [maximum(lpdfs[sizes .== i]) for i in unique(sizes)]
max_lpdfs_by_size[2:end] .< max_lpdfs_by_size[1:end - 1]

pdf_incl(ds[1], 3)
sum(exp, lpdfs[sizes .== 3])

k = 100
issorted(logpdf_incl.(Ref(DirichletProcessPartitionDistribution(k, :harmonic)), 1:k), rev = true)

import SpecialFunctions

[3, 3, 2, 2, 2]
prod(SpecialFunctions.gamma, [3, 3, 2, 2, 2])
function best_partition(n::Int, K::Int)
    giant_component = n - (K - 1)  # One large cluster
    return vcat([giant_component], fill(1, K - 1))
end
function worst_partition(n::Int, K::Int)
    q, r = divrem(n, K)  # q = floor(n / K), r = remainder
    return vcat(fill(q + 1, r), fill(q, K - r))
end
best_partition(12, 5)
worst_partition(12, 5)

best_shortcut(n, K) = Int(SpecialFunctions.gamma(n - (K - 1)))
function worst_shortcut(n, K)
    q, r = divrem(n, K)
    Int(SpecialFunctions.gamma(q + 1))^(r) * Int(SpecialFunctions.gamma(q)) ^(K - r)
end

get_val(x) = Int(prod(SpecialFunctions.gamma, x))
# get_val(x) = Int(prod(factorial, x .- 1))  # Use factorial(n_k - 1)
for n in 2:10
    for k in 1:n
        w0 = get_val(worst_partition(n, k))
        wr = worst_shortcut(n, k)
        wc = w0 == wr
        b0 = get_val(best_partition(n, k))
        br = best_shortcut(n, k)
        bc = b0 == br
        @assert wc == bc == true
        println("n = $n, k = $k, worst: ", w, " ", wr, " ", wc, " best:", b0, " ", br, " ", bc, " ratio: ", b0 / w0)
    end
end

ns = 20
ratios = zeros(Int, ns, ns)
for n in 1:ns
    for k in 1:n
        br = best_shortcut(n, k)
        wr = worst_shortcut(n, k)
        ratios[n, k] = br / wr
    end
end
i=2
join(ratios[i:min(ns, i+7), i], " ")
i=3
join(ratios[i:min(ns, i+7), i], " ")
i=4
join(ratios[i:min(ns, i+7), i], " ")
i=5
join(ratios[i:min(ns, i+7), i], " ")

import Combinatorics
Combinatorics.multinomial([4; (4 .+ (1:4)) ./ 4]...)

n!/ ([n/4]!, [(n+1)/4]!, [(n+2)/4]!, [(n+3)/4]!)

fn(n) = SpecialFunctions.gamma(n+1) / prod(i->SpecialFunctions.gamma(floor(Int, (i + n) / n + 1)), 1:n)
fn.(1:7)

function generalized_multinomial(n::Int, k::Int)
    groups = [floor(Int, (n + i) / k) for i in 0:(k-1)]
    return Combinatorics.multinomial(groups...)
end

all(generalized_multinomial.(0:ns - i, i) == ratios[i:ns, i] for i in axes(ratios, 2))

ns = 20
ratios = zeros(Int, ns, ns)
for n in 1:ns
    for k in 1:n
        br = best_shortcut(n, k-1)
        wr = worst_shortcut(n, k)
        ratios[n, k] = br / wr
    end
end

factorial(3)



best_partition(5, 3)
worst_partition(5, 3)


get_val(worst_partition(3, 3))
get_val(worst_partition(3, 2))
get_val(worst_partition(3, 1))
get_val(worst_partition(4, 4))
get_val(worst_partition(4, 3))
get_val(worst_partition(4, 2))
get_val(worst_partition(4, 1))


pdf(ds[1], [1, 1, 2, 2, 3])
pdf(ds[1], [1, 1, 1, 1, 2])

pdf(ds[1], [1, 1, 1, 1, 2]) / pdf(ds[1], [1, 1, 1, 2, 3]) ≈
(SpecialFunctions.gamma(α) / SpecialFunctions.gamma(α + 5) * α^2 * prod(SpecialFunctions.gamma, [4, 1])) /
(SpecialFunctions.gamma(α) / SpecialFunctions.gamma(α + 5) * α^3 * prod(SpecialFunctions.gamma, [3, 1, 1])) ≈
(prod(SpecialFunctions.gamma, [4, 1])) / (α * prod(SpecialFunctions.gamma, [3, 1, 1]))
1 / α * prod(SpecialFunctions.gamma, [4, 1]) / prod(SpecialFunctions.gamma, [3, 1, 1])

1 / α * prod(SpecialFunctions.gamma, [5])       / prod(SpecialFunctions.gamma, [4, 1])
1 / α * prod(SpecialFunctions.gamma, [4, 1])    / prod(SpecialFunctions.gamma, [3, 1, 1])
1 / α * prod(SpecialFunctions.gamma, [3, 1, 1]) / prod(SpecialFunctions.gamma, [2, 1, 1, 1])
1 / α * prod(SpecialFunctions.gamma, [2, 1, 1, 1]) / prod(SpecialFunctions.gamma, [1, 1, 1, 1, 1])

nc = 1
k - nc, SpecialFunctions.gamma(k - nc + 1) / SpecialFunctions.gamma(k - (nc + 1) + 1)

1 / α * SpecialFunctions.gamma(k - nc + 1) / (SpecialFunctions.gamma(k - (nc + 1) + 1) * SpecialFunctions.gamma(1)^((nc + 1) - 1))
1 / α * (k - nc)
collect(1 / α .* (k .- (1:5)))

dp = DirichletProcessPartitionDistribution(6, :harmonic)
dp = DirichletProcessPartitionDistribution(6, 1.0)
logpdf(dp, [1, 1, 1, 2, 2, 2]) > logpdf(dp, [1, 1, 1, 1, 2, 3])

pdf(dp, [1, 1, 1, 2, 2, 2]) / pdf(dp, [1, 1, 1, 1, 2, 3])
1 / dp.α * prod(SpecialFunctions.gamma, [3, 3]) / prod(SpecialFunctions.gamma, [4, 1, 1])

logpdf(DirichletProcessPartitionDistribution(12, 1.), [ones(Int, 7); 2:(12 - 6)])
logpdf(DirichletProcessPartitionDistribution(12, 1.), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

pdf(DirichletProcessPartitionDistribution(12, 1.), [ones(Int, 7); 2:(12 - 6)]) /
pdf(DirichletProcessPartitionDistribution(12, 1.), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

1 / α * prod(SpecialFunctions.gamma, [7; ones(Int, 12-7)]) / prod(SpecialFunctions.gamma, fill(2, 6))

k = 12; n = 7; m = 6
function create_balanced_array(k::Int, m::Int)::Vector{Int}
    # Ensure k > m
    if k <= m
        throw(ArgumentError("k must be greater than m"))
    end

    # Calculate the base number of occurrences for each value
    base_count = div(k, m)
    remainder = k % m

    # Initialize the array with the base count for each value
    array = vcat([fill(i, base_count) for i in 1:m]...)

    # Distribute the remainder to the first 'remainder' values
    for i in 1:remainder
        push!(array, i)
    end

    return array
end
function the_ratio(k, n, m)
    log_num = SpecialFunctions.loggamma(m)

    partition = create_balanced_array(k, n)
    den_config = EqualitySampler.fast_countmap_partition(partition)
    @assert sum(den_config) == k

    log_den = sum(SpecialFunctions.loggamma, den_config)
    exp(log_num - log_den)
end
exp(
    logpdf(DirichletProcessPartitionDistribution(12, 1.), [ones(Int, 7); 2:(12 - 6)]) -
    logpdf(DirichletProcessPartitionDistribution(12, 1.), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
)
SpecialFunctions.gamma(7) / prod(SpecialFunctions.gamma, fill(2, 6))

exp(
    logpdf(DirichletProcessPartitionDistribution(12, 1.), ones(Int, k)) -
    logpdf(DirichletProcessPartitionDistribution(12, 1.), [ones(Int, k - 1); 2])
)
exp(
    logpdf(DirichletProcessPartitionDistribution(12, 1.), [fill(1, k ÷ 2); fill(2, k ÷ 2)]) -
    logpdf(DirichletProcessPartitionDistribution(12, 1.), ones(Int, k))
)


the_ratio(12, 7, 6)
the_ratio(12, 3, 2)
[(i, the_ratio(12, i+1, i)) for i in 1:10]
argmax(the_ratio(12, i+1, i) for i in 1:10)

k = 12; m = 11; n = m + 1
base_count, rem = divrem(k, m)
den_config = fill(base_count, m)
!iszero(rem) && push!(den_config, rem)
@assert sum(den_config) == k
SpecialFunctions.gamma(n) / prod(SpecialFunctions.gamma, den_config)

den_part = [1:m-1; fill(2, k-m+1)]
exp(
    logpdf(DirichletProcessPartitionDistribution(k, 1.), [ones(Int, n); 2:(k - n + 1)]) -
    logpdf(DirichletProcessPartitionDistribution(k, 1.), [ones(Int, n); 2:(k - n + 1)])
)





logpdf(DirichletProcessPartitionDistribution(6, 1.), [1, 1, 1, 1, 1, 2])
logpdf(DirichletProcessPartitionDistribution(6, 1.), [1, 1, 1, 2, 2, 2]) > logpdf(DirichletProcessPartitionDistribution(6, 1.), [1, 1, 1, 1, 2, 3])
logpdf(DirichletProcessPartitionDistribution(6, 1.), [1, 1, 2, 2, 3, 3])



α = ds[1].α
SpecialFunctions.gamma(α) /SpecialFunctions.gamma(α + 5) * α^3 * prod(
    SpecialFunctions.gamma, [2, 2, 1]
)
SpecialFunctions.gamma(α) / SpecialFunctions.gamma(α + 5) * α^3 * prod(SpecialFunctions.gamma, [3, 1, 1])

SpecialFunctions.gamma(α) / SpecialFunctions.gamma(α + 5) * α^3 * prod(SpecialFunctions.gamma, [3, 1, 1])
prod(SpecialFunctions.gamma, [3, 1, 1])
prod(SpecialFunctions.gamma, [4, 1])




using BenchmarkTools
probvec = randn(10)
r = 4
f1(probvec, r) = sum(probvec[eachindex(probvec) .∉ Ref(1:r)])
f2(probvec, r) = sum(view(probvec, r+1:lastindex(probvec)))
f1(probvec, r)
f2(probvec, r)
@benchmark f1($probvec, $r)
@benchmark f2($probvec, $r)



# p({1, 1} | {1}) x p({1, 1, 1} | {1, 1})
(1 - p_du2(3, 2, 1)) * (1 - p_du2(3, 3, 1)) ≈
    (1 * bellnumr(3 - 2, 1) / bellnumr(3 - 2 + 1, 1)) *
    (1 * bellnumr(3 - 3, 1) / bellnumr(3 - 3 + 1, 1)) ≈
    (bellnumr(1, 1) / bellnumr(2, 1)) *
    (bellnumr(0, 1) / bellnumr(1, 1)) ≈ 1 / bellnumr(2, 1) ≈ 1 / bellnum(3)

    # simplification to bell numbers follows from
    # bellnumr(n, r) ≈ bellnumr(n-1, r+1) + r * bellnumr(n-1, r)
    # bellnumr(n, 0) ≈ bellnumr(n-1, 1) + 0 * bellnumr(n-1, 0)
    # bellnumr(3, 0) ≈ bellnumr(3-1, 1) + 0 * bellnumr(3-1, 0)

# p({1, 1} | {1}) x p({1, 1, 2} | {1, 1})
(1 - p_du2(3, 2, 1)) * p_du2(3, 3, 1) ≈
    (1 * bellnumr(3 - 2, 1) / bellnumr(3 - 2 + 1, 1)) *
    (bellnumr(3 - 3, 1) / bellnumr(3 - 3 + 1, 1)) ≈
    (bellnumr(1, 1) / bellnumr(2, 1)) *
    (bellnumr(0, 1) / bellnumr(1, 1)) ≈ 1 / bellnumr(2, 1) ≈ 1 / bellnum(3)

k, j, r = 4, 2, 1
p_du2(k, j, r) ≈ bellnumr(k - j, r + 1) / bellnumr(k - j + 1, r) ≈

1 - p_du2(k, j, r)
r * bellnumr(k - j, r) / bellnumr(k - j + 1, r)

k, j, r = 4, 3, 1
k, j, r = 4, 3, 2
bellnumr(k - j, r + 1) / bellnumr(k - j + 1, r)

p_du2(4, 2, 1) ≈ bellnumr(k - 2, 2) / bellnumr(k - 1, 1)
p_du2(4, 3, 2) ≈ bellnumr(k - 3, 3) / bellnumr(k - 2, 2)
p_du2(4, 4, 3) ≈ bellnumr(k - 4, 4) / bellnumr(k - 3, 3) ≈ 1 / bellnumr(k - 3, 3)
prod(p_du2(4, r+1, r) for r in 1:3) ≈ 1 / bellnumr(k - 1, 1) ≈ 1 / bellnumr(k, 0)

1 - p_du2(4, 2, 1) ≈ 1 * bellnumr(k - 2, 1) / bellnumr(k - 1, 1)
    p_du2(4, 3, 1) ≈     bellnumr(k - 3, 2) / bellnumr(k - 2, 1)
1 - p_du2(4, 4, 2) ≈ 2 * bellnumr(k - 4, 2) / bellnumr(k - 3, 2)
(1 - p_du2(4, 2, 1)) * p_du2(4, 3, 1) * (1 - p_du2(4, 4, 2)) / 2





tie_probability(DirichletProcessPartitionDistribution(4, .3))
1 - (.3 / (1 + .3))
(1 + .3) / (1 + .3) - (.3 / (1 + .3))
1  / (1 + .3)

1 / (1 -.3 - 1)

x = rand(DirichletProcessPartitionDistribution(4, .3), 100_000)
sum(x[1, :] .== x[2, :]) / size(x, 2)



bellnumr(k - 2, 2) / bellnumr(k - 1, 1)






m = Matrix(PartitionSpace(4))
map(x->pdf(UniformPartitionDistribution(4), x), eachcol(m))
inv(bellnum(4))
(1 - p_du2(4, 2, 1)) * (1 - p_du2(4, 3, 1)) * (1 - p_du2(4, 4, 1))

p_du(4, 2, 1)
bellnumr(4 - 2, 2) / (bellnumr(4 - 2, 2) + 1 * bellnumr(4 - 2, 1))
(bellnumr(4 - 1, 1) + 1 * bellnumr(4 - 2, 1))



p_du(3, 2, 1), 1 - p_du(3, 2, 1)
p_du(3, 3, 1), 1 - p_du(3, 3, 1)
p_du(3, 3, 2), 1 - p_du(3, 3, 2)

(1 - p_du(3, 2, 1)) * (1 - p_du(3, 3, 1))
(1 - p_du(3, 2, 1)) * p_du(3, 3, 1)

(1 * bellnumr(3 - 1, 1)) / (bellnumr(3 - 1, 1 + one(1)) + 1 * bellnumr(3 - 1, 1)) *
    ((bellnumr(3 - 2, 1 + one(1)) / (bellnumr(3 - 2, 1 + one(1)) + 1 * bellnumr(3 - 2, 1))))

bellnumr(2, 1) * bellnumr(1, 2)
bellnumr(3 - 1, 1) * bellnumr(3 - 2, 1 + one(1)) / (
    (bellnumr(3 - 1, 1 + one(1)) + 1 * bellnumr(3 - 1, 1)) *
    (bellnumr(3 - 2, 1 + one(1)) + 1 * bellnumr(3 - 2, 1))
)

bellnumr(3 - 1, 1) * bellnumr(3 - 2, 1 + one(1)) / (
    (bellnumr(3 - 1, 1 + one(1)) + 1 * bellnumr(3 - 1, 1)) *
    (bellnumr(3 - 2, 1 + one(1)) + 1 * bellnumr(3 - 2, 1))
)

(bellnumr(k - j, r + one(r)) + r * bellnumr(k - j, r))
bellnumr(k - j + 1, r)

bellnumr(2, 1) * bellnumr(1, 2)
bellnumr(2, 1) ≈ bellnumr(1, 2) + 1 * bellnumr(1, 1)

bellnumr(6, 3) ≈ bellnumr(6 - 1, 3 + 1) + 3 * bellnumr(6 - 1, 3)

bellnumr(2, 1) * bellnumr(1, 2) ≈ (bellnumr(1, 2) + 1 * bellnumr(1, 1)) * bellnumr(1, 2)




p_du(3, 2, 1) * # p({1, 2} | {1})
    (1 - p_du(3, 3, 2)) *
    1/2


p_du(3, 2, 1) * # p({1, 2} | {1})
    (1 - p_du(3, 3, 2)) *
    1/2 # divide by 2 because moving to {1, 2, 1} or {1, 2, 2} are equally likely


bellnumr(k - j, r + one(r)) / (bellnumr(k - j, r + one(r)) + r * bellnumr(k - j, r))


k, j, r = length(du), 3, 1 # [1, 1]
bellnumr(k - j, r + one(r)) / (bellnumr(k - j, r + one(r)) + r * bellnumr(k - j, r))

e1[1] * e2[1] # p({1, 1} | {1}) x p({1, 1, 1} | {1, 1})
e1[1] * e2[2] # p({1, 1} | {1}) x p({1, 1, 2} | {1, 1})

e1[2] * e2[1] # p({1, 2} | {1}) x p({1, 2, 1} | {1, 2})
e1[2] * e2[2] # p({1, 2} | {1}) x p({1, 2, 2} | {1, 2})
e1[2] * e2[3] # p({1, 2} | {1}) x p({1, 2, 3} | {1, 2})


.4
.6


probvec = zeros(length(du))
k = length(probvec)

complete_urns = [1, 0, 0]
index = 2
k = length(complete_urns)

partition_sizes = similar(complete_urns)

v_known_urns = view(complete_urns, 1:index - 1)

tb = EqualitySampler.fast_countmap_partition_incl_zero!(partition_sizes, v_known_urns)
r = sum(!iszero, tb)

lognum = logbellnumr(k - index, r + one(r))
logden = log(r) + logbellnumr(k - index, r)
prob_new_label = exp(lognum - LogExpFunctions.logsumexp([lognum, logden]))

lognum = logbellnumr(k - index, r + one(r))
logden = log(r) + logbellnumr(k - index, r)
# prob_new_label = LogExpFunctions.logistic(lognum - logden)
prob_new_label = exp(lognum - LogExpFunctions.logsumexp([lognum, logden]))

for i in eachindex(probvec)
    # if i in urns_set
    if !iszero(tb[i])
        probvec[i] = (1 - prob_new_label) / r
    else
        probvec[i] = prob_new_label / (k - r)
    end
end




j = 1
d.α / (d.α + j)

r = j
k = length(d)


partition = ones(typeof(k), k)
partition[k-r+1:k-1] .= 2:r
probvec = zeros(k)
EqualitySampler._pdf_helper!(probvec, d, k, partition, zeros(Int, k))

sum(probvec[eachindex(probvec) .∉ Ref(1:r)])