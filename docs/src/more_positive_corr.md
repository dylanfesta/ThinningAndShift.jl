```@meta
EditURL = "../../examples/more_positive_corr.jl"
```

````@example more_positive_corr
using ThinningAndShift ; global const T = ThinningAndShift
using Makie, CairoMakie
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Distributions
using Random ; Random.seed!(0)
````

# Simple correlated firing, but for more neurons

Here I consder k neurons. A group of kas of them fires together p% of the time, with specific delays between one another.
The other half is just uncorrelated. All neurons have the same rate.

````@example more_positive_corr
const k = 20
const kas = 8
const pkas = 0.5
const therate = 30.0
const markings = vcat([collect(1:kas)],[ [i,] for i in 1:k])

markings_probs =  vcat([pkas,],fill((1-pkas),kas),fill(1.0,k-kas))
markings_probs ./= sum(markings_probs)

const parent_rate  = therate/markings_probs[end]

const true_lags = vcat(0.0,rand(Uniform(-0.3,0.3),kas-1))

jit_for_assembly = MultivariateNormal(true_lags,5E-3*ones(kas))


jitters_d = vcat(T.JitterDistribution(jit_for_assembly), fill(T.NoJitter(),length(markings)-1))

gtas_test = T.GTAS(parent_rate,markings,markings_probs,jitters_d)

t_end = 10_000.0
thetrains,train_parent,attrib = T.make_samples_with_parent(gtas_test,t_end)

therates =  @. length(thetrains)/t_end
````

## Export to HDF5

````@example more_positive_corr
using HDF5
const savefile = "/tmp/test_train_single_assembly.h5"
````

save true lags

````@example more_positive_corr
using HDF5

h5open(savefile, "w") do file
    write(file, "true_lags", true_lags)
    write(file,"t_end",t_end)
    for neu in 1:k
        write(file, "train_$(neu)", thetrains[neu])
    end
    write(file,"train_parent",train_parent)
    write(file,"attrib",attrib)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

