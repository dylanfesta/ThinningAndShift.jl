# GTAS but with negative spikes ! 

abstract type AntiKernel end


# negative correlations for pairs only!

struct pAGTAS{R,I,J<:Jittering,AK<:AntiKernel}
  gtas::GTAS{R,I,J}
  antimarkings::Vector{Vector{I}}
  antiprobabilities::Vector{R}
  antikernels::Vector{AK}
  function AGTAS(rate_parent::R,markings::Vector{Vector{I}},
      markings_probs::Vector{R},jitters::Vector{J},
      antimarkings::Vector{Vector{I}},
      antiprobabilities::Vector{R},
      antikernels::Vector{AK}) where {R<:Real,I<:Integer,J<:Jittering,AK<:AntiKernel}
    @assert sum(markings_probs) ≈ 1  
    @assert length(markings_probs) == length(markings)
    @assert length(markings_probs) == length(jitters)
    @assert length(antimarkings) == length(antiprobabilities)
    @assert length(antikernels) == length(antiprobabilities)
    marking_select = Categorical(markings_probs)
    n = maximum(maximum.(markings))
    gtas = GTAS(n,rate_parent,markings,marking_select,jitters)
    return new{R,I,J,AK}(gtas,antimarkings,antiprobabilities,antikernels)
  end
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
  end
  nkeep = idx - idxstart
  keepat!(ret,1:nkeep)
  if (nkeep == Nmax) && dowarn
    @warn "the cdf is $(ret[end]) and not $Cmax !"
  end
  return (ret,idxstart)
end


function apply_antispike!(idx_start::R,cdfs::Vector{R},train::Vector{R}) where R
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
      train::Vector{R};Nmax::Integer=50,Cmax::Float64=0.9999) where R
  cdfs,idx_start = compute_forward_cutprob(t_now,antiker,train,Nmax,Cmax)
  apply_antispike!(idx_start,cdfs,train)
  return nothing
end

