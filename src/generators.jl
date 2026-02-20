



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
modulated_event_train(rate_func::Function, rate_max::Real, t_tot::Real)

Sample from a Poisson process with rate `rate_func(t)` on the interval `[0, t_tot]`.

Uses simple thinning: samples from a Poisson process with rate `rate_max` and
keeps each sample with probability `rate_func(t) / rate_max`.
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
