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