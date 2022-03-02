using ThinningAndShift ; global const T=ThinningAndShift
using Test
using Distributions
using Random ; Random.seed!(0)

@testset "Poisson generator" begin
  t_tot = 567.8
  rate = 123.45
  trains_poisson = T.make_poisson_samples(rate,t_tot)
  rate_num = length(trains_poisson)/t_tot
  @test isapprox(rate,rate_num;rtol=0.1)
  isis = diff(trains_poisson)
  # ISI is exponential. mean same as variance same as 1/rate
  @test isapprox(mean(isis),inv(rate);rtol=0.1)
  @test isapprox(std(isis),mean(isis);rtol=0.2)
end

@testset "Indepdentend processes" begin
  markings = [ [1,],[2,],[3,]]
  p1,p2,p3 = 0.6,0.3,0.1
  motherrat  = 234.5
  markings_probs = [p1,p2,p3]
  jitters_d = fill(T.NoJitter(),3)
  gtas_test = T.GTAS(motherrat,markings,markings_probs,jitters_d)
  t_tot = 500.0
  (t1,t2,t3),_ = T.make_samples_with_parent(gtas_test,t_tot)
  r1_num = length(t1)/t_tot
  r2_num = length(t2)/t_tot
  r3_num = length(t3)/t_tot
  @test isapprox(r1_num,p1*motherrat;rtol=0.1)
  @test isapprox(r3_num,p3*motherrat;rtol=0.1)
  isis1 = diff(t1)
  isis3 = diff(t3)
  @test isapprox(std(isis1),mean(isis1);rtol=0.2)
  @test isapprox(std(isis3),mean(isis3);rtol=0.2)
  rat_an = T.get_expected_rates(gtas_test) 
  @test all(isapprox.([r1_num,r2_num,r3_num],rat_an,;rtol=0.1))
end

@testset "cross-correlations, 2D, Gaussian jitter" begin
  markings = [ [1,2],]
  markings_probs = [1.0]
  mySigma = let σ1 = 0.17,σ2 = 0.2, ρ = 0.7
      [ σ1^2 ρ*σ1*σ2 ; ρ*σ1*σ2  σ2^2 ]
  end
  jitters = [T.JitterDistribution( MultivariateNormal(mySigma)) ,]
  lambda_parent = 100.0
  gtas_test = T.GTAS(lambda_parent,markings,markings_probs,jitters)
  trains,t_parent,attr = T.make_samples_with_parent(gtas_test,10_000.0)
  timescov,covboth = T.covariance_density_numerical(trains,0.15,6.0)
  std_diff = sqrt(mySigma[1,1]+mySigma[2,2] - 2*mySigma[1,2])
  density_an = Normal(0.0,std_diff)
  density_an_vals = lambda_parent .* pdf.(density_an,timescov)
  density_num =  @. 0.5(covboth[:,1,2]+covboth[:,2,1])
  @test all(isapprox.(density_an_vals,density_num;atol=10.,rtol=0.2))
end

@testset "some internal stuff for negative corr" begin
  t_tot = 10.0
  rate = 25.0
  train = T.make_poisson_samples(rate,t_tot)
  antiker = T.AntiExponential(0.1)
  d = Exponential(0.1) 
  t_now = 0.5 + rand()*3.0
  Cmax = 0.9999
  testidx,testcdf= T.compute_forward_cutprob(t_now,antiker,train,
    50,Cmax)
  @test train[testidx] > t_now
  @test train[testidx-1] < t_now
  idx_end = testidx + length(testcdf) - 1
  @test cdf(d, train[idx_end] - t_now ) > Cmax 
  @test cdf(d, train[idx_end-1] - t_now ) < Cmax 

  antiker = T.AntiExponential(0.1)
  timehor = T.antikernel_horizon(antiker,1E-5)

  @test isapprox(pdf(Exponential(0.1),timehor),1E-5)
  @test pdf(Exponential(0.1),timehor*0.999) > 1E-5
end



function get_ff_num(_train,Ttot)
  dtbin = 8.0
  rates = T.bin_spikes(_train,dtbin,Ttot)
  var(rates)/mean(rates)
end

@testset "negative corr 1D" begin

  myrate = 100.0
  killratio = 0.59
  rates = [myrate,killratio*myrate]

  markings = [T.Marking([1]),T.AntiMarking([1])]
  jitters = [T.NoJitter(),T.AntiJitterExpSequential(1000.000)]

  gen = T.sAGTAS(1,rates,markings,jitters)

  Ttot = 1E3
  trains = T.make_samples(gen,Ttot);

  rate_num = length(trains[1])/Ttot 
  rate_expected = myrate * (1-killratio)

  @test isapprox(rate_num,rate_expected;rtol=0.2)

  # test FF 
  ff_expected = sum(rates) / (rates[1]-rates[2])
  ff_num = get_ff_num(trains[1],Ttot)
  @test isapprox(ff_expected,ff_num;rtol=0.2)

  # repeat with the non-sequential one
  jitters = [T.NoJitter(),T.AntiJitterExp(30E-3)]
  gen = T.sAGTAS(1,rates,markings,jitters)
  Ttot = 1E3
  trains = T.make_samples(gen,Ttot);
  rate_num = length(trains[1])/Ttot 
  rate_expected = myrate * (1-killratio)
  @test isapprox(rate_num,rate_expected;rtol=0.2)

  # test FF 
  ff_expected = sum(rates) / (rates[1]-rates[2])
  ff_num = get_ff_num(trains[1],Ttot)
  @test isapprox(ff_expected,ff_num;rtol=0.2)

  # repeat with interval
  jitters = [T.NoJitter(),T.AntiJitterStepSequential(1000.000)]
  gen = T.sAGTAS(1,rates,markings,jitters)
  Ttot = 1E3
  trains = T.make_samples(gen,Ttot);
  rate_num = length(trains[1])/Ttot 
  rate_expected = myrate * (1-killratio)
  @test isapprox(rate_num,rate_expected;rtol=0.2)

  # test FF 
  ff_expected = sum(rates) / (rates[1]-rates[2])
  ff_num = get_ff_num(trains[1],Ttot)
  @test isapprox(ff_expected,ff_num;rtol=0.2)

end
