```@meta
EditURL = "https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/positive_correlations.jl"
```

````@example positive_correlations
using ThinningAndShift ; global const T = ThinningAndShift
using Makie, CairoMakie
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
````

# Simple correlated firing

Neuron A and B fire at 10 Hz each, half of the time they fire together.
in perfect sync

````@example positive_correlations
markings = [ [1,],[2,],[1,2]]
markings_probs = fill(0.33333,3)
markings_probs ./= sum(markings_probs)

parent_rate  = 15.0
jitters_d = fill(T.NoJitter(),3)

gtas_test = T.GTAS(parent_rate,markings,markings_probs,jitters_d)

t_end = 500.0
(train1,train2),_ = T.make_samples_with_parent(gtas_test,t_end)

r1_num = length(train1)/t_end
r2_num = length(train2)/t_end

"""
Rate should be around 10.0
rate 1 : $(r1_num)
rate 2 : $(r2_num)"""
````

Now show the raster

````@example positive_correlations
raster_img = U.draw_spike_raster([train1,train2],1E-2,10.0;
  spike_size=50,spike_separator=10)
````

and the plot!

````@example positive_correlations
raster_plot = U.plot_spike_raster([train1,train2],1E-2,10.0;
  spike_size=50,spike_separator=10)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

