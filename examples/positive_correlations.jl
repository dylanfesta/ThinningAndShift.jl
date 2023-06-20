push!(LOAD_PATH, abspath(@__DIR__,"..")) #src
using ThinningAndShift ; global const T = ThinningAndShift


# # Simple correlated firing

#=
Neuron A and B fire at 10 Hz each, half of the time they fire together.
=#

markings = [ [1,],[2,],[1,2]]
markings_probs = fill(0.33333,3)
markings_probs ./= sum(markings_probs)

parent_rate  = 15.0
jitters_d = fill(T.NoJitter(),3)

gtas_test = T.GTAS(parent_rate,markings,markings_probs,jitters_d)

t_end = 500.0
(t1,t2),_ = T.make_samples_with_parent(gtas_test,t_end)

r1_num = length(t1)/t_end
r2_num = length(t2)/t_end






## publish in documentation #src
thisfile = joinpath(splitpath(@__FILE__)[end-1:end]...) #src
using Literate; Literate.markdown(thisfile,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src