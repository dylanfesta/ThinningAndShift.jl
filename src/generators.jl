



"""
make_poisson_samples(rate::Real, t_tot::Real)

Sample from a uniform Poisson process with rate `rate` on the interval `[0, t_tot]`.
Returns a vector of event times of variable size.
"""
function make_poisson_samples(rate::Real, t_tot::Real)
  ret = Vector{Float64}(undef, round(Integer, 1.3 * rate * t_tot + 10)) # preallocate
  t_curr = 0.0
  k_curr = 1
  while t_curr <= t_tot
    Δt = -log(rand()) / rate
    t_curr += Δt
    ret[k_curr] = t_curr
    k_curr += 1
  end
  return keepat!(ret, 1:k_curr-2)
end


"""
  modulated_event_train(rate_func::Function, rate_max::Real, t_tot::Real) -> modulated_train::Vector{Float64}

Generates a Poisson train with instantaneous rate matching `rate_function(t)` for t in [0,t_tot].

Based on simple thinning: samples from a Poisson process with rate `rate_max` and
keeps each sample with probability `rate_func(t) / rate_max`.

`rate_max` must be strictly greater than `rate_function(t)` for all t in [0,t_tot].
"""
function modulated_event_train(rate_func::Function, rate_max::Real, t_tot::Real)
  parent_train = make_poisson_samples(rate_max, t_tot)
  keep_spike = trues(length(parent_train))
  for (k, t) in enumerate(parent_train)
    rate_here = rate_func(t)
    @assert 0 <= rate_here <= rate_max "rate_here must be between 0 and rate_max, but instead got $rate_here at time $t"
    if rand() > rate_func(t) / rate_max
      keep_spike[k] = false
    end
  end
  return parent_train[keep_spike]
end



function make_samples_with_parent(g::GTAS{R,I}, t_tot::R) where {R,I}
  ts_ancestor = make_poisson_samples(g.rate_parent, t_tot)
  nt = length(ts_ancestor)
  attributions = rand(g.marking_selector, nt)
  trains = [Vector{R}(undef, 0) for _ in 1:g.n]
  # can be optimized by preallocating trains, based on expected rate!
  for (t_k, k) in zip(ts_ancestor, attributions)
    mark_k = g.markings[k]
    n_k = length(mark_k)
    spikes_k = jitter!(fill(t_k, n_k), k, g)
    for (tkk, kk) in zip(spikes_k, mark_k)
      push!(trains[kk], tkk)
    end
  end
  # clean up trains... sorted, and > 0
  sort!.(trains)
  tmin = minimum(first.(trains))
  if tmin < 0
    for tr in trains
      tr .-= tmin
    end
    ts_ancestor .-= tmin
  end
  return trains, ts_ancestor, attributions
end

function make_samples(gtas::GTAS, t_tot::Real)
  return make_samples_with_parent(gtas, t_tot)[1]
end


"""
Generates a 1D self-interacting Poisson process following an arbitrary interaction kernel function.

IMPORTANT: the kernel function cannot have area larger than 1. And `rate_max` must be strictly greater than `rate_function(t)` for all t in [0,t_tot].
this is non-trivial, as repeated nearby spikes result in multiple staked kernels.

Final rate will be rate_start/(1-kernel_area)

As a rule of thumb, consider the typical timescale of the kernel, and how many spikes you expect in that timescale. 
"""
function self_interacting_1D_train(kernel_func::Function, rate_start::Real, t_tot::Real;
  rate_max::Union{Real,Nothing}=nothing,
  verbose::Bool=false)
  # TODO: implement automatic calculation of rate_max
  if rate_max === nothing
    error("rate_max calculation not implemented yet, please provide rate_max")
  end

  # convolution of kernel with all (meaningful) spiketimes
  # TO-DO , consider adding a time-horizon argument to further reduce this iteration
  function kernel_superfunction(spiketimes::Vector{Float64}, t_curr::Real)
    # pick only spikes that are in the past to save some steps
    spiketimes_past = @view spiketimes[spiketimes.<t_curr]
    return mapreduce(t_spk -> kernel_func(t_curr - t_spk), +, spiketimes_past; init=0.0)
  end
  k_generation = 0
  current_generation = make_poisson_samples(rate_start, t_tot)
  ret_spikes = Vector{Float64}[]
  history_lengths = Int[]
  n_window = 20

  while length(current_generation) > 0
    k_generation += 1
    push!(history_lengths, length(current_generation))
    if verbose
      @info "Generation $k_generation: $(length(current_generation)) spikes"
    end

    if k_generation >= 2 * n_window && k_generation % n_window == 0
      avg_prev = sum(@view history_lengths[end-2*n_window+1:end-n_window]) / n_window
      avg_curr = sum(@view history_lengths[end-n_window+1:end]) / n_window
      if avg_curr >= avg_prev
        error("No net reduction in generation size across $n_window iterations (prev avg: $avg_prev, curr avg: $avg_curr). Check the area under the kernel curve.")
      end
    end

    push!(ret_spikes, current_generation)
    current_generation = modulated_event_train(t -> kernel_superfunction(current_generation, t), rate_max, t_tot)
  end
  return sort!(vcat(ret_spikes...))
end