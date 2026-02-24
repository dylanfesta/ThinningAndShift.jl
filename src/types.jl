
"""
Jittering selects the type of perturbation applied to the event times.

NoJitter: 
  no jittering

JitterDistribution(Distributions.UnivariateDistribution):
  Sample i.i.d. and add to event times
JitterIncremental(Distributions.UnivariateDistribution): 
  apply a distribution to each spike time, cumulatively
JitterPaired(Distributions.UnivariateDistribution):
  apply a distribution to each pair of spike times
"""
abstract type Jittering end

# no jittering
struct NoJitter <: Jittering end
@inline function jitter!(v::Vector{Float64}, ::NoJitter)
  return v
end
# apply a distribution
struct JitterDistribution{D<:Distribution} <: Jittering
  d::D
end
# increment by given distribution
struct JitterIncremental{D<:UnivariateDistribution} <: Jittering
  d::D
end
# only for sets of two, shifts one spike time w.r.t. the other
struct JitterPaired{D<:UnivariateDistribution} <: Jittering
  d::D
end


"""
Generalized Thinning & Shift (GTAS) generative model for spike-trains.

Parameters
-------------
- n::I: number of emitting units
- rate_parent::R: rate of the parent Poisson process, upper bound for the instantaneous rate of all units
- markings::Vector{Vector{I}}: markings of the parent Poisson process, each element is a vector of unit indices that are marked by the parent Poisson process
- marking_probs::Vector{R}: probabilities of each marking
- jitters::Vector{J}: jitters for each marking

"""
struct GTAS{R,I,J<:Jittering}
  n::I
  rate_parent::R
  markings::Vector{Vector{I}}
  marking_selector::Categorical{R}
  jitters::Vector{J}
  function GTAS(rate_parent::R, markings::Vector{Vector{I}},
    markings_probs::Vector{R}, jitters::Vector{J}) where {R<:Real,I<:Integer,J<:Jittering}
    @assert sum(markings_probs) ≈ 1
    @assert length(markings_probs) == length(markings)
    @assert length(markings_probs) == length(jitters)
    marking_select = Categorical(markings_probs)
    n = maximum(maximum.(markings))
    return new{R,I,J}(n, rate_parent, markings, marking_select, jitters)
  end
end