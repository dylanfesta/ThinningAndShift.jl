#= 
examples/modulated_train.jl

Generated simple spike raster with time modulation
=#
##

docs_path = abspath(@__DIR__, "..", "docs") |> normpath
using Pkg;
Pkg.activate(docs_path);

using ThinningAndShift;
const global T = ThinningAndShift;
using Makie, CairoMakie
using SpikeTrainUtilities;
const global U = SpikeTrainUtilities;
using Random;
Random.seed!(0);

using Statistics: mean

##
# rate modulated as sinusoidal curve, between 5 and 30 Hz
t_tot = 100.0
const θ = 10.0
const r_min = 5.0
const r_max = 30.0
rate_func(t) = r_min + (r_max - r_min) * 0.5 * (1 + sin(-π / 2 + (2π * t) / θ))

train_modulated = T.modulated_event_train(rate_func, r_max + 0.1, t_tot)


raster_plot = U.plot_spike_raster([train_modulated,], 1E-2, 30.0;
  spike_size=50, spike_separator=10)

## okay, better test now: generate 10_000 trains, and plot population rate

N = 10_000
trains_modulated = [T.modulated_event_train(rate_func, r_max + 0.1, t_tot) for _ in 1:N]

spiketrains_modulated = U.SpikeTrains(trains_modulated; t_end=t_tot)

##


spiketrains_rates = U.discretize(spiketrains_modulated, 100E-3, U.BinRate())
spiketrains_avg_rate = mean(spiketrains_rates.ys, dims=1)[:]
##

plot_x = U.get_t_midpoints(spiketrains_rates)
plot_y_num = spiketrains_avg_rate
plot_y_th = [rate_func(t) for t in plot_x]


fig = Figure()
ax = Axis(fig[1, 1], xlabel="time (s)", ylabel="rate (Hz)")
lines!(ax, plot_x, plot_y_th, label="Theoretical", linewidth=2, color=:red)
scatter!(ax, plot_x, plot_y_num, label="Numerical", markersize=5, color=:black)
axislegend(ax)
display(fig)

##

