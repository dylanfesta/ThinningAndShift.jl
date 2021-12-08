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
  jitters_d = fill(nothing,3)
  gtas_test = T.GTAS(motherrat,markings,markings_probs,jitters_d)
  t_tot = 500.0
  (t1,t2,t3),_ = T.make_samples_with_ancestor(gtas_test,t_tot)
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