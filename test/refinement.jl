function test_refinement()
    M1 = AN.ann_refine(5)
    M2 = AN.ann_refine_2d(5)
    Z = M([0.7; 0.6])
    Z1 = M1([0.7])
    Z2 = M1([0.6])
    @test norm(Z[1:2:end]-Z1) < 1e-10
    @test norm(Z[2:2:end]-Z2) < 1e-10
end
