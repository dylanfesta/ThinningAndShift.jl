#=
Try one neuron with negative feedback on itself
=#

docs_path = abspath(@__DIR__, "..", "docs") |> normpath
using Pkg;
Pkg.activate(docs_path);

using ThinningAndShift;
const global T = ThinningAndShift;
using Makie, CairoMakie
using SpikeTrainUtilities;
const global U = SpikeTrainUtilities;
using Distributions
using Random;
Random.seed!(0);

##

myrate = 100.0
killratio = 0.59
rates = [myrate, killratio * myrate]

markings = [T.Marking([1]), T.AntiMarking([1])]
jitters = [T.NoJitter(), T.AntiJitterExpSequential(1000.000)]

gen = T.sAGTAS(1, rates, markings, jitters)

Ttot = 1E3
trains = T.make_samples(gen, Ttot);

rate_num = length(trains[1]) / Ttot
rate_expected = myrate * (1 - killratio)

@test isapprox(rate_num, rate_expected; rtol=0.2)

# test FF 
ff_expected = sum(rates) / (rates[1] - rates[2])
ff_num = get_ff_num(trains[1], Ttot)
@test isapprox(ff_expected, ff_num; rtol=0.2)

# repeat with the non-sequential one
jitters = [T.NoJitter(), T.AntiJitterExp(30E-3)]
gen = T.sAGTAS(1, rates, markings, jitters)
Ttot = 1E3
trains = T.make_samples(gen, Ttot);
rate_num = length(trains[1]) / Ttot
rate_expected = myrate * (1 - killratio)
@test isapprox(rate_num, rate_expected; rtol=0.2)

# test FF 
ff_expected = sum(rates) / (rates[1] - rates[2])
ff_num = get_ff_num(trains[1], Ttot)
@test isapprox(ff_expected, ff_num; rtol=0.2)

# repeat with interval
jitters = [T.NoJitter(), T.AntiJitterStepSequential(1000.000)]
gen = T.sAGTAS(1, rates, markings, jitters)
Ttot = 1E3
trains = T.make_samples(gen, Ttot);
rate_num = length(trains[1]) / Ttot
rate_expected = myrate * (1 - killratio)
@test isapprox(rate_num, rate_expected; rtol=0.2)

# test FF 
ff_expected = sum(rates) / (rates[1] - rates[2])
ff_num = get_ff_num(trains[1], Ttot)
@test isapprox(ff_expected, ff_num; rtol=0.2)



# # Simple correlated firing, but for more neurons 


#=
Here I consder k neurons. A group of kas of them fires together p% of the time, with specific delays between one another.
The other half is just uncorrelated. All neurons have the same rate.
=#

const k = 20
const kas = 8
const pkas = 0.5
const therate = 30.0
const markings = vcat([collect(1:kas)], [[i,] for i in 1:k])

markings_probs = vcat([pkas,], fill((1 - pkas), kas), fill(1.0, k - kas))
markings_probs ./= sum(markings_probs)
