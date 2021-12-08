


using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using ThinningAndShift ; global const T = ThinningAndShift

using Distributions

##

markings = [ [1,2],[3]]
markings_probs = [0.7,0.3]
# jitters_d = [ Normal(0.0,0.01) for _ in 1:3 ]
jitters_d = fill(nothing,3)

gtas_test = T.GTAS(100.0,markings,markings_probs,jitters_d)

T.get_expected_rates(gtas_test)
##

trains,t_ancestor,attr = T.make_samples_with_ancestor(gtas_test,50.0)

##
t_tot = 567.8
rate = 123.45
trains_poisson = T.make_poisson_samples(rate,t_tot)
rate_num = length(trains_poisson)/t_tot

mean(diff(trains_poisson))
std(diff(trains_poisson))
