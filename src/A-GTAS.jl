# GTAS but with negative spikes ! 

abstract type AntiKernel end


# negative correlations for pairs only!

struct pAGTAS{R,I,J<:Jittering,AK<:AntiKernel}
  gtas::GTAS{R,I,J}
  antimarkings::Vector{Tuple{I,I}}
  antiprobabilities::Vector{R}
  antikernels::Vector{AK}
end

function pAGTAS(rate_parent::R,markings::Vector{Vector{I}},
    marking_probs::Vector{R},jitters::Vector{J},
    antimarkings::Vector{Tuple{I,I}},
    antiprobabilities::Vector{R},
    antikernels::Vector{AK}) where {R<:Real,I<:Integer,J<:Jittering,AK<:AntiKernel}
  @assert length(antimarkings) == length(antiprobabilities)
  @assert length(antikernels) == length(antiprobabilities)
  @assert all(<=(1.0),antiprobabilities)
  gtas = GTAS(rate_parent,markings,marking_probs,jitters)
  return pAGTAS{R,I,J,AK}(gtas,antimarkings,antiprobabilities,antikernels)
end



n_units(agtas::pAGTAS) = agtas.gtas.n
n_units(gtas::GTAS) = gtas.n


struct AntiExponential <: AntiKernel
  τ::Float64
end


function compute_forward_cutprob(t_now::R,antiker::AntiExponential,
    train::Vector{R},Nmax::Integer,Cmax::R;dowarn::Bool=true) where R
  idxstart = searchsortedfirst(train,t_now+eps())
  idx = idxstart  
  ret = Vector{Float64}(undef,Nmax)
  d = Exponential(antiker.τ)
  for k in 1:Nmax
    _cdf = cdf(d,train[idx]-t_now)
    ret[k] = _cdf
    idx+=1
    if _cdf >= Cmax 
      break
    end
    if ! checkbounds(Bool,train,idx)
      @error "Not enough spikes left ! More spikes needed!"
      break
    end
  end
  nkeep = idx - idxstart
  keepat!(ret,1:nkeep)
  if (nkeep == Nmax) && dowarn
    @warn "the cdf is $(ret[end]) and not $Cmax !"
  end
  return (idxstart,ret)
end


function apply_antispike!(idx_start::Integer,cdfs::Vector{R},train::Vector{R}) where R
  umax = cdfs[end]
  u = rand()*umax
  idx_kill = idx_start
  for cdf in cdfs
    if u<=cdf
      break 
    else
      idx_kill +=1
    end
  end
  deleteat!(train,idx_kill)
  return nothing
end

function apply_antispike!(t_now::R,antiker::AntiExponential,
      train::Vector{R};Nmax::Integer=200,Cmax::Float64=0.999) where R
  idx_start,cdfs = compute_forward_cutprob(t_now,antiker,train,Nmax,Cmax)
  apply_antispike!(idx_start,cdfs,train)
  return nothing
end


function make_samples(g::pAGTAS,t_tot::Real)
  trains = make_samples(g.gtas,1.5*t_tot)
  for ((pre,post),p,antiker) in zip(g.antimarkings,g.antiprobabilities,g.antikernels)
    trainpre = trains[pre]
    trainpost = trains[post]
    for t_now in trainpre
      if t_now > t_tot
        break
      elseif rand() < p
        apply_antispike2!(t_now,antiker,trainpost)
      end
    end
  end
  # remove excess time
  for train in trains
    idx = searchsortedfirst(train,t_tot)
    keepat!(train,1:idx-1)
  end
  return trains
end

function compute_forward_cutprobabilities(t_now::R,antiker::AntiKernel,
    train::Vector{R},eps_prob::R) where R
  idxstart = searchsortedfirst(train,t_now+eps())
  t_last = t_now + antikernel_horizon(antiker,eps_prob)
  idxend = searchsortedfirst(train,t_last)-1
  if idxend == length(train)-1
    @error "Not enough spikes left ! More spikes needed!"
  end
  ret = compute_forward_cutprobabilities(t_now,view(train,idxstart:idxend),antiker)
  return (idxstart,ret)
end

@inline function antikernel_horizon(antiker::AntiExponential,eps_prob::Float64)
  τ = antiker.τ
  return - τ * (log(τ)+log(eps_prob))
end

@inline function compute_forward_cutprobabilities(t_now::Real,train::SubArray{Float64},antiker::AntiExponential)
  d = Exponential(antiker.τ)
  ret = map(t -> pdf(d,t-t_now),train)
  ret ./= sum(ret)
  return ret
end

function apply_antispike2!(t_now::R,antiker::AntiExponential,
      train::Vector{R};eps_prob::Float64=1E-4) where R
  idx_start,cutprobs = compute_forward_cutprobabilities(t_now,antiker,train,eps_prob)
  idx_cut = rand(Categorical(cutprobs))
  deleteat!(train,idx_start+idx_cut-1)
  return nothing
end


##


# sequential antispikes 

abstract type AbstractMarking end
abstract type AbstractAntiJittering <: Jittering end

struct Marking <: AbstractMarking
  vals::Vector{Int64}
end
struct AntiMarking <: AbstractMarking
  vals::Vector{Int64}
end



struct AntiJitterExp <: AbstractAntiJittering
  τ::Float64
end
struct AntiJitterExpSequential <: AbstractAntiJittering
  τ::Float64
end
struct AntiJitterStepSequential <: AbstractAntiJittering
  τ::Float64
end


struct sAGTAS{N,NTR<:NTuple{N,Float64},
        NTM<:NTuple{N,AbstractMarking},
        NTJ<:NTuple{N,Jittering}}
  n::Int64
  rate_markings::NTR
  markings::NTM
  jitters::NTJ
end
nmarkings(agta::sAGTAS) = length(agta.markings)

function sAGTAS(n::Int64,
    rate_markings::Vector{Float64},
    markings::Vector{M},
    jitters::Vector{J}) where {M<:AbstractMarking,J<:Jittering}
  @assert length(markings) == length(jitters)
  @assert length(rate_markings) == length(jitters)
  return sAGTAS(n,Tuple(rate_markings),Tuple(markings),Tuple(jitters))
end


function make_samples(g::sAGTAS,t_tot::R) where R<:Real
  # initialize trains
  trains = [Vector{Float64}(undef,0) for _ in 1:g.n] 
  # one train for each marking
  # (saved to keep antimarkings for backwards move)
  t_forw = t_tot*1.1+10.0
  trains_markings = map(g.rate_markings) do rat
    make_poisson_samples(rat,t_forw)
  end
  # forward move
  for (train_marking,marking,jitt) in zip(trains_markings,g.markings,g.jitters)
    add_marking_to_trains!(trains,train_marking,marking,jitt)
  end
  # clean up trains: sorted, and > 0
  sort!.(trains)
  tmin = minimum(first.(trains))
  if tmin < 0
    for tr in trains
      tr .-= tmin
    end
  end
  # backwards now!
  for (train_marking,marking,jitt) in zip(trains_markings,g.markings,g.jitters)
    remove_antimarking_from_trains!(trains,train_marking,marking,jitt,t_tot)
  end
  # stop time right before t_tot
  for train in trains
    # filter!(<=(t_tot),train)
    k = searchsortedfirst(train,t_tot)
    keepat!(train,1:(k-1))
  end
  return trains
end

# do nothing for antimarking
function add_marking_to_trains!(::Vector,
    ::Vector,::AntiMarking,jitter::Jittering)
  return nothing
end

# do the usual for markings 
function add_marking_to_trains!(trains::Vector{Vector{R}},
    trainmark::Vector{R},marking::Marking,jitter::Jittering) where R<:Real
  mark = marking.vals
  n = length(mark)
  for t_k in trainmark
    spikes_k = jitter!(fill(t_k,n),jitter)
    for (tkk,kk) in zip(spikes_k,mark)
     push!(trains[kk],tkk)
    end
  end
  return nothing
end


# do nothing for markings
function remove_antimarking_from_trains!(::Vector,
    ::Vector,::Marking,::Jittering,::Real)
  return nothing
end


@inline function antijitter_horizon(
    antijitt::Union{AntiJitterExp,AntiJitterExpSequential},
    eps_prob::Float64)
  τ = antijitt.τ
  return - τ * (log(τ)+log(eps_prob))
end

@inline function antijitter_horizon(antijitt::AntiJitterStepSequential,::Float64)
  return antijitter_horizon(antijitt)
end
@inline function antijitter_horizon(antijitt::AntiJitterStepSequential)
  return antijitt.τ
end


function remove_antimarking_from_trains!(trains::Vector{Vector{R}},
    trainmark::Vector{R},anti::AntiMarking,
    antijitter::Union{AntiJitterExpSequential,AntiJitterStepSequential},
    t_tot::R) where R<:Real
  mark = anti.vals
  # remove excess time from mark train
  kmax = searchsortedfirst(trainmark,t_tot)
  keepat!(trainmark,1:(kmax-1))
  for t_k in trainmark
    # find and remove next spike in first train 
    train1 = trains[mark[1]]
    idx_cut = searchsortedfirst(train1,t_k)
    deleteat!(train1,idx_cut)
    if length(mark) > 1
      tnow = train1[idx_cut]
      t_incr =  antijitter_horizon(antijitter,1E-5)
      # in following trains (if any) remove with exp probability
      for ms in mark[2:end]
        trainh = trains[ms]
        tmax = tnow + t_incr
        idx_start,cutprobs = 
          compute_forward_killprobabilities(tnow,tmax,trainh,antijitter)
        idx_cut = idx_start + rand(Categorical(cutprobs)) - 1
        tnow = trainh[idx_cut]
        deleteat!(trainh,idx_cut)
      end
    end
  end
  return nothing
end


function remove_antimarking_from_trains!(trains::Vector{Vector{R}},
    trainmark::Vector{R},anti::AntiMarking,
    antijitter::AntiJitterExp,t_tot::R) where R<:Real
  mark = anti.vals
  # remove excess time from mark train
  kmax = searchsortedfirst(trainmark,t_tot)
  keepat!(trainmark,1:(kmax-1))
  # time horizon
  t_incr =  antijitter_horizon(antijitter,1E-5)
  for t_k in trainmark
    tnow = t_k
    tmax = t_k + t_incr
    for ms in mark
      trainh = trains[ms]
      idx_start,cutprobs = 
          compute_forward_killprobabilities(tnow,tmax,trainh,antijitter)
      idx_cut = idx_start + rand(Categorical(cutprobs)) - 1
      deleteat!(trainh,idx_cut)
    end
  end
  return nothing
end


# returns idx of closest time >= t_start and probabilities of killing spike 
function compute_forward_killprobabilities(t_start::R,t_end::R,train::Vector{R},
     antijitter::Union{AntiJitterExpSequential,AntiJitterExp}) where R<:Real
  idx_start =  searchsortedfirst(train,t_start)
  idx_end = searchsortedfirst(train,t_end)
  ret = map(t -> exp(-(t-t_start)/antijitter.τ), view(train,idx_start:idx_end) )
  ret ./= sum(ret)
  return idx_start,ret
end

function compute_forward_killprobabilities(t_start::R,t_end::R,train::Vector{R},
     ::AntiJitterStepSequential) where R<:Real
  idx_start =  searchsortedfirst(train,t_start)
  idx_end = searchsortedfirst(train,t_end)-1 # smaller than t_end
  n_spikes = max(idx_end-idx_start+1,1) # if no spikes in interval, just erase next one :-(
  return idx_start,fill(inv(n_spikes),n_spikes)
end
