using Documenter, EqualitySampler

makedocs(
	sitename="EqualitySampler.jl",
	modules  = [EqualitySampler],
	format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
	warnonly = true
)

deploydocs(;
	repo = "github.com/vandenman/EqualitySampler.git"
)