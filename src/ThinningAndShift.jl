module ThinningAndShift

using Distributions
using Statistics
using Random

struct GTAS{R,I,U<:Union{Distribution,Nothing}}
  n::I
  rate_ancestor::R
  markings::Vector{Vector{I}}
  marking_selector::Categorical{R}
  jitters::Vector{U}
  function GTAS(rate_ancestor::R,markings::Vector{Vector{I}},
      markings_probs::Vector{R},jitters::Vector{U}) where {R<:Real,I<:Integer,U<:Union{Distribution,Nothing}}
    @assert sum(markings_probs) ≈ 1  
    @assert length(markings_probs) == length(markings)
    @assert all(issorted.(markings)) "Markings should be sorted"
    marking_select=Categorical(markings_probs)
    n = maximum(maximum.(markings))
    return new{R,I,U}(n,rate_ancestor,markings,marking_select,jitters)
  end
end


function make_samples_with_ancestor(g::GTAS{R,I},t_tot::R) where {R,I}
  ts_ancestor = make_poisson_samples(g.rate_ancestor,t_tot) 
  nt = length(ts_ancestor)
  attributions = rand(g.marking_selector,nt)
  trains = [Vector{R}(undef,0) for _ in 1:g.n] 
  for (t_anc,attrib) in zip(ts_ancestor,attributions)
    for k in g.markings[attrib]
      jitt = g.jitters[k]
      tk = t_anc + (isnothing(jitt) ? 0.0 : rand(jitt))
      push!(trains[k],tk)
    end
  end
  trains,ts_ancestor,attributions
end
  
function make_poisson_samples(rate::R,t_tot::R) where R
  ret = Vector{R}(undef,round(Integer,1.3*rate*t_tot)) # preallocate
  t_curr = zero(R)
  k_curr = 1
  while t_curr <= t_tot
    ret[k_curr] = t_curr
    Δt = -log(rand())/rate
    t_curr += Δt
    k_curr += 1
  end
  return keepat!(ret,1:k_curr-1)
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


function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tmax::Union{R,Nothing}=nothing) where R
  ret = covariance_density_numerical([Y,],dτ,τmax,Tmax;verbose=false)
  return ret[:,1,1]
end

function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R,
   Tmax::Union{R,Nothing}=nothing ; verbose::Bool=false) where R
  Tmax = something(Tmax, maximum(x->x[end],Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,ndt,n,n)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tmax/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tmax)
    fmi = length(Ys[i]) / Tmax # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tmax)
      fmj = length(Ys[j]) / Tmax # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = _ret_alloc / (ndt_tot*dτ^2) - fmi*fmj
      ret[:,i,j] = _ret_alloc
    end
  end
  return ret
end


end # of Module
