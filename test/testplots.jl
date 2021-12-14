
using Distributions
push!(LOAD_PATH, abspath(@__DIR__,".."))

using ThinningAndShift ; global const T = ThinningAndShift

using Random
Random.seed!(0)
using Plots; using NamedColors; theme(:dark)
##

markings = [ [1,2],]
markings_probs = [1.0]
const mySigma = let σ1sq = 0.17,σ2sq = 0.2, ρ = 0.7
    [ σ1sq ρ*σ1sq*σ2sq   ; ρ*σ1sq*σ2sq  σ2sq ]
end
jitters = [T.JitterDistribution( MultivariateNormal(mySigma)) ,]

lambda_ancestor = 100.0

gtas_test = T.GTAS(lambda_ancestor,markings,markings_probs,jitters)

T.get_expected_rates(gtas_test)
##

trains,t_ancestor,attr = T.make_samples_with_ancestor(gtas_test,10_000.0)

timescov,covboth = T.covariance_density_numerical(trains,0.15,6.0)

var_sum = sum(mySigma) 
std_sum = sqrt(var_sum)

density_an = Normal(0.0,std_sum)
density_an_vals = lambda_ancestor .* pdf.(density_an,timescov)

plot(timescov,[ covboth[:,1,2] covboth[:,2,1] density_an_vals ])


density_num =  @. 0.5(covboth[:,1,2]+covboth[:,2,1])

plot(timescov,[ density_num density_an_vals ])
all(isapprox.(density_an_vals,covboth[:,1,2];atol=20.,rtol=0.3))