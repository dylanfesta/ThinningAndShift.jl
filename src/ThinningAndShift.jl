module ThinningAndShift

using Distributions
using Random

import LinearAlgebra: dot
import Statistics: mean,var,cov
import StatsBase: midpoints

abstract type Jittering end

#####
# no jittering
struct NoJitter  <: Jittering end
@inline function jitter!(v::Vector{Float64},::NoJitter)
  return v
end

####
# apply a distribution
struct JitterDistribution{D<:Distribution} <: Jittering
  d::D
end
@inline function jitter!(v::Vector{Float64},j::JitterDistribution{D}) where D<:UnivariateDistribution
  jits = rand(j.d,length(v))
  v .+= jits
  return v
end
@inline function jitter!(v::Vector{Float64},j::JitterDistribution)
  jits = rand(j.d)
  v .+= jits
  return v
end

####
## only for sets of two, shifts one spike time w.r.t. the other
struct JitterPaired{D<:UnivariateDistribution} <: Jittering
  d::D
end
@inline function jitter!(v::Vector{Float64},j::JitterPaired)
  ji = rand(j.d)
  v[2] .+= ji
  return v
end



struct GTAS{R,I,J<:Jittering}
  n::I
  rate_ancestor::R
  markings::Vector{Vector{I}}
  marking_selector::Categorical{R}
  jitters::Vector{J}
  function GTAS(rate_ancestor::R,markings::Vector{Vector{I}},
      markings_probs::Vector{R},jitters::Vector{J}) where {R<:Real,I<:Integer,J<:Jittering}
    @assert sum(markings_probs) ≈ 1  
    @assert length(markings_probs) == length(markings)
    @assert length(markings_probs) == length(jitters)
    @assert all(issorted.(markings)) "Markings should be sorted"
    marking_select=Categorical(markings_probs)
    n = maximum(maximum.(markings))
    return new{R,I,J}(n,rate_ancestor,markings,marking_select,jitters)
  end
end

jitter!(v::Vector{R},k::Integer,g::GTAS) where R = jitter!(v,g.jitters[k])

function make_samples_with_ancestor(g::GTAS{R,I},t_tot::R) where {R,I}
  ts_ancestor = make_poisson_samples(g.rate_ancestor,t_tot) 
  nt = length(ts_ancestor)
  attributions = rand(g.marking_selector,nt)
  trains = [Vector{R}(undef,0) for _ in 1:g.n] 
  for (t_k,k) in zip(ts_ancestor,attributions)
    mark_k = g.markings[k]
    n_k = length(mark_k)
    spikes_k = jitter!(fill(t_k,n_k),k,g)
    for (tkk,kk) in zip(spikes_k,mark_k)
      push!(trains[kk],tkk)
    end
  end
  return trains,ts_ancestor,attributions
end
  
function make_poisson_samples(rate::R,t_tot::R) where R
  ret = Vector{R}(undef,round(Integer,1.3*rate*t_tot+10)) # preallocate
  t_curr = zero(R)
  k_curr = 1
  while t_curr <= t_tot
    Δt = -log(rand())/rate
    t_curr += Δt
    ret[k_curr] = t_curr
    k_curr += 1
  end
  return keepat!(ret,1:k_curr-2)
end

####
# Cumulants

function get_expected_rates(g::GTAS{R,I}) where {R,I}
  mks = g.markings
  probs = g.marking_selector.p
  rat_an = g.rate_ancestor
  ret = fill(zero(R),g.n)
  for (marks,p) in zip(mks,probs)
    for neu in marks
      ret[neu] += p*rat_an
    end
  end
  return ret
end


#########
# numerical analysis

function bin_spikes(Y::Vector{R},dt::R,Tend::R;
    Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  ret = fill(0,length(times)-1)
  for y in Y
    if Tstart < y <= last(times)
      k = searchsortedfirst(times,y)-1
      ret[k] += 1
    end
  end
  return ret
end

function bin_and_rates(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  return midpoints(times),bin_spikes(Y,dt,Tend;Tstart=Tstart) ./ dt
end

function bin_and_rates(Ys::Vector{Vector{R}},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)
  ret = [bin_spikes(Y,dt,Tend;Tstart=Tstart) ./ dt for Y in Ys]
  return midpoints(times),vcat(transpose.(ret)...)
end


@inline function get_times_strict(dt::R,Tend::R;Tstart::R=0.0) where R<:Real
  return range(Tstart,Tend-dt;step=dt)
end

# the first element (zero lag) is always rate/dτ
function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tend::Union{R,Nothing}=nothing) where R
  τtimes,ret = covariance_density_numerical([Y,],dτ,τmax;verbose=false,Tend=Tend)
  return  τtimes, ret[:,1,1]
end

function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R;
   Tend::Union{R,Nothing}=nothing,verbose::Bool=false) where R
  Tend = something(Tend, maximum(last,Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,ndt,n,n)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tend/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tend)
    fmi = length(Ys[i]) / Tend # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tend)
      fmj = length(Ys[j]) / Tend # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = (_ret_alloc / (ndt_tot*dτ^2)) - fmi*fmj
      ret[:,i,j] = _ret_alloc
    end
  end
  return get_times_strict(dτ,τmax), ret
end


end # of Module
