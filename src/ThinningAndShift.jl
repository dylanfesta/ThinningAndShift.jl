module ThinningAndShift

using Distributions
using Random

import LinearAlgebra: dot,diagind
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
  n = length(v)
  if n > 1
    jits = rand(j.d,n)
    v .+= jits
  end
  return v
end
@inline function jitter!(v::Vector{Float64},j::JitterDistribution)
  jits = rand(j.d)
  v .+= jits
  return v
end

# increment by given distribution
struct JitterIncremental{D<:UnivariateDistribution} <: Jittering
  d::D
end

@inline function jitter!(v::Vector{Float64},j::JitterIncremental)
  n = length(v)
  if n>1
    jits = cumsum(rand(j.d,n-1))
    v[2:end] .+= jits
  end
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
  rate_parent::R
  markings::Vector{Vector{I}}
  marking_selector::Categorical{R}
  jitters::Vector{J}
  function GTAS(rate_parent::R,markings::Vector{Vector{I}},
      markings_probs::Vector{R},jitters::Vector{J}) where {R<:Real,I<:Integer,J<:Jittering}
    @assert sum(markings_probs) ≈ 1  
    @assert length(markings_probs) == length(markings)
    @assert length(markings_probs) == length(jitters)
    marking_select=Categorical(markings_probs)
    n = maximum(maximum.(markings))
    return new{R,I,J}(n,rate_parent,markings,marking_select,jitters)
  end
end

jitter!(v::Vector{R},k::Integer,g::GTAS) where R = jitter!(v,g.jitters[k])

function make_samples_with_parent(g::GTAS{R,I},t_tot::R) where {R,I}
  ts_ancestor = make_poisson_samples(g.rate_parent,t_tot) 
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
  # clean up trains... sorted, and > 0
  sort!.(trains)
  tmin = minimum(first.(trains))
  if tmin < 0
    for tr in trains
      tr .-= tmin
    end
    ts_ancestor .-= tmin
  end
  return trains,ts_ancestor,attributions
end

function make_samples(gtas::GTAS,t_tot::Real)
  return make_samples_with_parent(gtas,t_tot)[1]
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


function get_probability_weight(neu::I,markings::Vector{Vector{I}},
      probs::Vector{<:Real}) where I 
  probs_all = map(zip(markings,probs)) do (marks,p)
    fill(p,length(marks))
  end
  probs_all = vcat(probs_all...)
  marks_all = vcat(markings...)
  idxs = findall(==(neu),marks_all)
  return sum(probs_all[idxs])
end


function get_parent_rate(neu::I,rate_target::Real,markings::Vector{Vector{I}},
     probs::Vector{<:Real}) where I<:Integer
  ptot = get_probability_weight(neu,markings,probs)
  return rate_target / ptot
end
function get_parent_rate_target_mean(rate_target::Real,markings::Vector{Vector{I}},
  probs::Vector{<:Real}) where I<:Integer
n = maximum(maximum.(markings))
p_avg = mapreduce(neu->get_probability_weight(neu,markings,probs),+,1:n) / n
return rate_target / p_avg
end


function get_expected_rates(g::GTAS{R,I}) where {R,I}
  mks = g.markings
  probs = g.marking_selector.p
  rat_par = g.rate_parent
  ret = rat_par .* [get_probability_weight(i,mks,probs) for i in 1:g.n]
  return ret
end

# UNDERESTIMATE ! Must account for oder zero ! Chance of events being in the same
# time interval considered. This one converges to numeric only for very small time bins!
function get_expected_Pearson(i::Integer,j::Integer,g::GTAS{R,I}) where {R,I}
  marks = g.markings
  probs = g.marking_selector.p
  uiall = 0.0
  ujall = 0.0
  uijall = 0.0
  for (mark,prob) in zip(marks,probs)
    hasi = i in mark
    hasj = j in mark
    if hasi && hasj
      uijall += prob
    elseif hasi
      uiall += prob
    elseif hasj
      ujall += prob
    end
  end
  return uijall / (uijall+uiall+ujall)
end

function get_expected_Pearson(g::GTAS{R,I}) where {R,I}
  ret = Matrix{R}(undef,g.n,g.n)
  for i in 1:g.n, j in i+1:g.n
    ret[i,j] = get_expected_Pearson(i,j,g)
    ret[j,i] = ret[i,j]
  end
  ret[diagind(ret)] .= one(R)
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

function covariance_density_ij(Ys::Vector{Vector{R}},i::Integer,j::Integer,dτ::R,τmax::R;
   Tend::Union{R,Nothing}=nothing) where R
  return covariance_density_ij(Ys[i],Ys[j],dτ,τmax;Tend=Tend)
end

function covariance_density_ij(X::Vector{R},Y::Vector{R},dτ::Real,τmax::R;
    Tend::Union{R,Nothing}=nothing) where R
  times = get_times_strict(dτ,τmax)
  ndt = length(times)
  times_ret = vcat(-reverse(times[2:end]),times)
  Tend = something(Tend, max(X[end],Y[end])- dτ)
  ret = Vector{Float64}(undef,2*ndt-1)
  binnedx = bin_spikes(X,dτ,Tend)
  binnedy = bin_spikes(Y,dτ,Tend)
  fx = length(X) / Tend # mean frequency
  fy = length(Y) / Tend # mean frequency
  ndt_tot = length(binnedx)
  binned_sh = similar(binnedx)
  # 0 and forward
  @simd for k in 0:ndt-1
    circshift!(binned_sh,binnedy,-k)
    ret[ndt-1+k+1] = dot(binnedx,binned_sh)
  end
  # backward
  @simd for k in 1:ndt-1
    circshift!(binned_sh,binnedy,k)
    ret[ndt-k] = dot(binnedx,binned_sh)
  end
  @. ret = (ret / (ndt_tot*dτ^2)) - fx*fy
  return times_ret, ret
end


## include the part with anti-spikes and negative correlations 
include("A-GTAS.jl")

end # of Module
