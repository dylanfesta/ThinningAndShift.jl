var documenterSearchIndex = {"docs":
[{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"EditURL = \"https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/positive_correlations.jl\"","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"using ThinningAndShift ; global const T = ThinningAndShift\nusing Makie, CairoMakie\nusing SpikeTrainUtilities ; global const U = SpikeTrainUtilities","category":"page"},{"location":"positive_correlations/#Simple-correlated-firing","page":"Simple correlated firing","title":"Simple correlated firing","text":"","category":"section"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"Neuron A and B fire at 10 Hz each, half of the time they fire together. in perfect sync","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"markings = [ [1,],[2,],[1,2]]\nmarkings_probs = fill(0.33333,3)\nmarkings_probs ./= sum(markings_probs)\n\nparent_rate  = 15.0\njitters_d = fill(T.NoJitter(),3)\n\ngtas_test = T.GTAS(parent_rate,markings,markings_probs,jitters_d)\n\nt_end = 500.0\n(train1,train2),_ = T.make_samples_with_parent(gtas_test,t_end)\n\nr1_num = length(train1)/t_end\nr2_num = length(train2)/t_end\n\n\"\"\"\nRate should be around 10.0\nrate 1 : $(r1_num)\nrate 2 : $(r2_num)\"\"\"","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"Now show the raster","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"raster_img = U.draw_spike_raster([train1,train2],1E-2,10.0;\n  spike_size=50,spike_separator=10)","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"and the plot!","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"raster_plot = U.plot_spike_raster([train1,train2],1E-2,10.0;\n  spike_size=50,spike_separator=10)","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"","category":"page"},{"location":"positive_correlations/","page":"Simple correlated firing","title":"Simple correlated firing","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ThinningAndShift","category":"page"},{"location":"#ThinningAndShift","page":"Home","title":"ThinningAndShift","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Warning\nThe documentation is still missing. Please see the \"examples\" section for usage.","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"First example","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ThinningAndShift]","category":"page"}]
}
