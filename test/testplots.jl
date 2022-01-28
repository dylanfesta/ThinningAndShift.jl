
push!(LOAD_PATH, abspath(@__DIR__,".."))

using ThinningAndShift ; global const T = ThinningAndShift

using Random
Random.seed!(0)
using Test
using Plots; using NamedColors; theme(:dark)
using LinearAlgebra
using Distributions



##
myrate = 100.0
markings = [[1],[2]]
probs = [1.,1.8]
jitters = [T.NoJitter(),T.NoJitter()]
probs ./= sum(probs)
antiprobs = [0.8]
antimarkings = [(1,2),]
antikernels = [T.AntiExponential(0.2),]

rate_parent = T.get_parent_rate(1,myrate,markings,probs)
gen = T.pAGTAS(rate_parent,markings,probs,jitters,
      antimarkings,antiprobs,antikernels)



##
const Ttot = 5000.0
@time trains = T.make_samples(gen,Ttot);

const rates_num = [ length(tr)/Ttot for tr in trains ]

##

function crosscov_an(t)
  t < 0 && return 0.0
  τ = antikernels[1].τ
  p = probs[1]*antiprobs[1]
  return - rate_parent*p*exp(-t/τ)/τ#*0.8
end

_ = let dtcov = 20E-3,
  Tcov = 1.0
  (covtimes,covs) =  T.covariance_density_ij(trains...,dtcov,Tcov)
  plt = plot()
  scatter!(plt,covtimes,covs;label="numerical",leg=false,xlabel="time (s)",
    ylabel="cross covariance density")
  ts = range(-Tcov,Tcov;length=200)
  plot!(plt,ts,crosscov_an.(ts);leg=false, linewidth=2 )
  plt
end

##


##

_ = let plt = plot(),
  tlims = (0,2.0),
  msize = 34,
  trplot1 = filter(t-> tlims[1]< t<tlims[2],trains[1])
  trplot2 = filter(t-> tlims[1]< t<tlims[2],trains[2])
  scatter!(plt,trplot1,fill(1.0,length(trplot1));
    marker=:vline,markersize=msize)
  scatter!(plt,trplot2,fill(1.05,length(trplot2));marker=:vline,
    markersize=msize)
  plot!(plt;ylims=(0.95,1.1),leg=false)
end

##
_ = let plt=plot()
end

##

_ = let r1 = length()

_ = let x=range(0,10.;length=150)
  d = Exponential(inv(3.0)) 
  plot(x,cdf.(d,x);ylims=(0,1.1),xlims=(0,10),leg=false,linewidth=2)
end


_ = let d = Exponential(0.1) 
  cdf(d, 0.8725210739840403 - 0.17 )
end


##

markings = [ [1,2],]
markings_probs = [1.0]
const mySigma = let σ1 = 0.17,σ2 = 0.3, ρ = 0.8
    [ σ1^2 ρ*σ1*σ2   ; ρ*σ1*σ2  σ2^2 ]
end
jitters = [T.JitterDistribution( MultivariateNormal(mySigma)) ,]

const rate_parent = 50.0
const Tend = 20_000.0

gtas_test = T.GTAS(rate_parent,markings,markings_probs,jitters)
trains,t_parent,attr = T.make_samples_with_parent(gtas_test,20_000.0)

##
const dtcov = 0.25
const Tcov = 5.0
timescov,covboth = T.covariance_density_numerical(trains,dtcov,Tcov)

@show rate_parent/dtcov;
@show covboth[1,1,1] ;

std_diff = sqrt(mySigma[1,1]+mySigma[2,2] - 2*mySigma[1,2])
density_an = Normal(0.0,std_diff)
density_an_vals = rate_parent .* pdf.(density_an,timescov)
density_num =  @. 0.5(covboth[:,1,2]+covboth[:,2,1])
plot(timescov,[density_an_vals density_num])

##

const dT = 25.0
times,rates  = T.bin_and_rates(trains,dT,Tend)
@show mean(rates;dims=2)
covnum = cov(rates;dims=2) * dT

rate_parent

##