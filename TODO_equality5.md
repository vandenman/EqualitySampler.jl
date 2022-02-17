## Chores
- [x] Upload ISBA slides to GitHub
- [x] Email Han

## Julia

### simulations/visualizePriors.jl
- [x] Legend in  in the topright of the top row

### simulations/multipleComparisonPlot.jl
- [ ] axes lines should not exceed the limits (y: (0, 1), x: (2, 10))
- [x] odd numbers on the x-axis
- [x] try no space between prior family and parameters
- [x] hashtag symbol for no. groups in legend
- [x] legend slighlty left
- [x] x-axis title should be Number of groups

### simulations/meansModel_simulation_convergence.jl
- [x] More repetitions, at least 50! After reparametrizing the model maybe?

### simulations/meansModel_analyze_convergence_results.jl
- [x] One big figure for the appendix
- [x] Labelling went wrong! BetaBinomial should be DPP in right columns

### demos/proportions.jl
- [ ] same issue where axes lines should not exceed the limits (y: (0, 1), x: (2, 10))



### General speed improvements:

- [ ] intterupt stopped at:
logpdf_model_distinct at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:192
logpdf_model at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:48 [inlined]
logpdf at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:52 [inlined]
logposterior at /home/dvdb/github/EqualitySampler/simulations/meansModel_Functions.jl:294

examine whether DPP prior is not horribly slow!

- [ ] Figure out why DPP and BB (mainly DPP) are so much slower than Uniform!

- [ ] play with JETTest.jl

- [ ] benchmark logpdf of models somehow?

- [ ] detect whether a run contains many errors and store this in the results?

- [ ] consider looping over the priors within each run
  - [ ] possibly loading the old file to see if every prior is already there


### Figure 2
- [x] Reorder DPP, Uniform, BetaBinomial
- [x] No box around the legend
- [x] Left panels need y-axis labels
- [x] Bigger font size
- [x] Switch color for DPP so orange line is decreasing
- [x] Add "Prior" to the column titles
- [x] Column title "BetaBinomial" -> "Beta-Binomial"
- [x] Drop partition numbers (x-axis tick labels)

### Figure 3
- [x] "BetaBinomial" -> "Beta-Binomial"
- [x] "Gopalan Berry" -> "Gopalan & Berry" (ampersand)

### Figure 4 + Figure 5
- [?] check zero vs non-zero? What was this about?
- [x] adjust y-axis limits to (0, .5)
- [x] use mean instead of median
- [ ] For manuscript figure, put No. parameters left and don't repeat prior in the title (basically mimic facet_grid)
- [ ] create nicer figures for true inequalities (power) and all other options

### Scott & Berger plot
- [x] \alpha = 0.5
- [x] make figure and add it to the overleaf

### Add proportion demo
- [ ] Review all the comments in the manuscript

### Appendix
- [x] "BetaBinomial" -> "Beta-Binomial"


## 11-08-2021
- [ ] For all plots
	- [ ] Change order to DPP, beta-binomial, uniform
	- [ ] Increase line width in all plots
	- [ ] Ensure digits have 2 decimals, so 0.50 and not 0.5, 1.00 and not 1.0
- [ ] Figure 2 (visualizePriors)
	- [x] change order
	- [ ] Legend: increase font size and decrease the symbol size
	- [x] Legend: "Gopalan Berry" -> "Gopalan & Berry"
	- [x] double check colors in bottom plot vs top plot and if they match, mainly for DPP
- [ ] Figure 3 (Scott & Berger)
	- [ ] Look at most ??? what did I write here?
	- [x] change order
	- [x] y-axis label should be odds! Copy text from Scott & Berger
	- [x] do Beta-binomial (\alpha = k, \beta = 1)
	- [x] legend should be "1 inequality added", "2 inequalities added", "5 ...", "10 ..."
	- [x] put legend in Beta-binomial plot (after reordering)
	- [x] double check if DPP is not flipped!
	- [x] leave top row, let it go to K = 20.
	- [x] rescale y-axis for Uniform in both bottom and top row.
	- [x] do A-B thingy like in Scott & Berger
	- [x] see if the DPP alpha value determines the asymptote.
- [ ] Figure 5
	- [x] Legend title
	- [x] Legend labels
	- [x] Order of columns is incorrect
	- [x] rename parameters to $K$ = 5
	- [x] x-axis titles and y-axis titles are missing
	- [x] rotate x-axis tick labels by 30 degrees
	- [x] Bonus: make y-axis go to 0.2 where possible


# 16-09-2021

- [ ] Paper
	- [ ] Figure out whether the prediction rule for a Beta-binomial is always a uniform,
		when drawing the last value conditional on the others.
	- [x] Try out the Beta-binomial with the prediction rule from the DPP
	- [ ] Rewrite section 3.3 to be more clear
	- [ ] Cut Figure 7

- [ ] Julia
	- [x] Look at simplifying the Turing model
		- [x] Beta-binomial with weighted categorical jump
		- [x] Remove pdf of Urn models from the Turing model (since we already use it in the Gibbs step)
	- [x] Remake figure 2 with new Beta-binomial with weighted categorical jump
		- [x] did not work out
	- [x] Figure 3
		- [x] the x-axis of the top figures don't start at 0.
		- [x] double check the y-axis of the DPP
		- [x] Beta-binomial 5.00 should be 30?
		- [x] Why is the DPP alpha not equal to 0.5?
		- [x] Increase line thickness and font size
		- [x] Legend do p(#9) / p(#10) instead of "10 inequalities added"
		- [x] DPP: double check if the prior odds are 1 / \alpha.
			- They are \alpha, see last bit of code in ...(Figure 3).jl
		- [x] x-axis label: "Number of groups"
	- [x] Figure 4
		- [x] rerun with more iterations (200)
		- [x] side by side
			- [x] Left: current figure
			- [x] Right: Rate of errors #errors / #total possible errors
			- [x] Legend: Change # groups for K
	- [ ] Figure 5
		- [ ] k in \alpha = k should be capitalized
		- [ ] increase font size
		- [ ] also compute rate of errors
	- [ ] Big simulation study
		- [ ] Add 0 % as a category (e.g., H0)
		- [ ] Change K = 10 => K = 9
		- [ ] Remove from the simulation:
			- BB(1, 1)
			- DPP(1)
			- DPP(Gopalan & Berry)


# 17 - 11 - 2021

- [ ] Make individual plots and combine these with minipage in LaTeX ?
- [x] Figure 3
  - [x] K = 30 for both rows.
  - [x] same x-axis (title + breaks) for both panels
- [x] Figure 4
  - [x] proportion of errors.
- [ ] make DPP prior_comparison for different K (5, 10, 15, 20, 25, 30)
- [ ] add a row where we don't use lopdf_distinct_model but logpdf_model
	- this made no sense at all.


# 01 - 12 - 2021

- [ ] Figure 2, don't use distinct models but with re
- [ ] Figure 3
  - [ ] do both rows use the same distinct/ non-distinct models?
  - [ ] really double check bottom left panel!
  - [ ] top middle panel should be monotonically decreasing?
  - [ ] double check BetaBinomial with
  ```julia
  log_model_probs_by_incl = Distributions.logpdf.(Distributions.BetaBinomial(k - 1, α, β), 0:k - 1) .- log_expected_equality_counts(k)
  ```

A:

- [x] BB(k, 1) is definitely not monotonically decreasing on the prior model inclusion space,
  so sum(P(M_i) for i in 1:... if no_equalities(M_i) == j)

- [x] Somewhere I do logstirlings2(k, k:-1:1) should this not be logstirlings2(k, 1:k)?
  - given k in 0:n-1, stirlings(n, n - k) counts the number of models given k equality constraints necessary. Meanwhile, stirlings(n, k) counts the number of free parameters in the model.


- [x] Flip stirling numbers + interpretation of betabinomial
  -> x-axis becomes no. free parameters
  -> Beta-binomial parametrization becomes (1, f(k)) rather than (f(k), 1)

- [x] Recreate Figure 2
- [x] Recreate Figure 3
- [x] Recreate Figure 4
	priors:
		- BB(1, 1), BB(1, k), BB(1, binomial(k, 2)),
		- DPP(.5), DPP(1.0), DPP(Gopalan Berry),
		- Uniform

- use DocStringUtils

- [ ] Update figure 4 with Plots.jl

- [ ] Proportion example -- more figures (check only after uploading to overleaf!)
	- [ ] Two panel plot
		- [ ] Left: for estimation
			- [ ] Show posterior density of proportions for each journal
			- [ ] Ridgeline plot, possibly on one line with different colors
		- [ ] Right: for testing
			- [ ] Create posterior model probabilities table, see if some models dominate
			- [ ] Try out some stuff
			- [ ] Heatmap with posterior propabilties of equality (8x8 table) with values in there up to two digits

- [ ] Variances example -- more figures (check only after uploading to overleaf!)
	- [ ] Two panel plot
		- [ ] Left: for estimation
			- [ ] Show posterior density of proportions for each journal
			- [ ] Ridgeline plot, possibly on one line with different colors
		- [ ] Right: for testing
			- [ ] Create posterior model probabilities table, see if some models dominate
			- [ ] Try out some stuff
			- [ ] Heatmap with posterior propabilties of equality (10x10 table) with values in there up to two digits
			- [ ] Check if we can use the upper triangle for something interesting (e.g., the prior)

- [ ] Package changes
	- [ ] move all the functionality inside anovaFunctions.jl into the package.
		- [ ] remove `include("anovaFunctions.jl")` in the demos/ simulations.
	- [ ] Add generic functions for user friendliness
		- [ ] anova_test		(formula, data, prior, ...)
		- [ ] proportion_test	(formula, data, prior, ...)
		- [ ] variance_test		(formula, data, prior, ...)
	- [ ] look at https://cran.r-project.org/package=JuliaCall
	- [ ] stick to one naming convention for filenames!