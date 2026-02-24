#= julia script

This example shows how to generate a 1D self-interacting Poisson process.

=#

docs_path = abspath(@__DIR__, "..", "docs") |> normpath
using Pkg;
Pkg.activate(docs_path);

using ThinningAndShift
const global T = ThinningAndShift
using Makie, CairoMakie
using SpikeTrainUtilities
const global U = SpikeTrainUtilities
using Random
Random.seed!(0)

## Define the kernel
const τker = 200E-3
const wker = 0.8

function my_kernel(t::Real)
  if t < 0
    return 0.0
  end
  return wker * (exp(-t / τker) / τker)
end

const Ttot = 300.0
const rstart = 3.0

const r_expected = rstart / (1 - wker)

@info "Generating self-interacting Poisson process with expected rate $(r_expected)"

##

train_self = T.self_interacting_1D_train(my_kernel, rstart, Ttot;
  rate_max=10 * r_expected, verbose=true)

##
true_rate = length(train_self) / Ttot
@info "Expected rate: $r_expected"
@info "True rate: $true_rate"

##
raster_plot = U.plot_spike_raster([train_self,], 1E-2, 80.0; spike_size=500)
##
# test that Poisson train can be empty!
# T.make_poisson_samples(0.01, 5.0)
# T.modulated_event_train(t -> 0.5, 100.0, 1.0)

