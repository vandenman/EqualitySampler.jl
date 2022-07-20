using Documenter, EqualitySampler

makedocs(
	sitename="My Documentation",
	modules  = [EqualitySampler],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
)