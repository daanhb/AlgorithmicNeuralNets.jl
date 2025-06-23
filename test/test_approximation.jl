@testset "Approximation" begin
    @testset "x^2" begin
        M = ann_square(10; init=:taylor)
        x1 = sqrt(2)/2
        @test abs(M(x1) - AN.square_approx(x1; L=10)) < 1e-15
        @test abs(M(x1) - x1^2) < 1e-6
        @test abs(ann_square(10; init=:interpolation)(x1) - AN.square_approx(x1; L=10, init=:interpolation)) < 1e-15
        @test abs(ann_square(10; init=:midvalue)(x1) - AN.square_approx(x1; L=10, init=:midvalue)) < 1e-15
        x_dyadic = 117/2^10
        @test abs(AN.square_approx(x_dyadic; L=10, init=:interpolation)-x_dyadic^2) < 1e-15
    end
    
    @testset "exp(x)" begin
        M = ann_exp(10; init=:taylor)
        x1 = sqrt(2)/2
        @test norm(M([x1]) - AN.exp_approx(x1; L=10)) < 1e-15
        @test norm(M([x1]) - AN.exp_approx_relu(x1; L=10)) < 1e-15
        @test abs(M([x1])[1] - exp(x1)) < 1e-6
        @test abs(M([x1])[2] - exp(-x1)) < 1e-6
        @test norm(ann_exp(10; init=:interpolation)([x1]) - AN.exp_approx(x1; L=10, init=:interpolation)) < 1e-15
        x_dyadic = 117/2^10
        @test norm(AN.exp_approx(x_dyadic; L=10, init=:interpolation) - [exp(x_dyadic); exp(-x_dyadic)]) < 1e-15
    end

    @testset "trigonometric functions" begin
        M = ann_sincos(10; init=:taylor)
        x1 = sqrt(2)/2
        @test norm(M([x1]) - AN.sincos_approx(x1; L=10)) < 1e-15
        @test norm(M([x1]) - AN.sincos_approx_relu(x1; L=10)) < 1e-15
        @test abs(M([x1])[1] - cos(x1)) < 1e-6
        @test abs(M([x1])[2] - sin(x1)) < 1e-6
        @test norm(ann_sincos(10; init=:interpolation)([x1]) - AN.sincos_approx(x1; L=10, init=:interpolation)) < 1e-15
        x_dyadic = pi * 117/2^10
        @test norm(AN.sincos_approx(x_dyadic; L=10, init=:interpolation) - [cos(x_dyadic); sin(x_dyadic)]) < 1e-15
    end

    @testset "monomials" begin
        degree = 5
        M = ann_monomials(10, degree)
        x1 = sqrt(2)/2
        @test norm(M([x1]) - [x1^k for k in 2:degree]) < 1e-6
        @test norm(M([x1]) - AN.monomials_approx_relu(x1, degree; L=10)) < 1e-15
        @test norm(ann_monomials(10, degree; init=:interpolation)([x1]) - AN.monomials_approx(x1, degree; L=10, init=:interpolation)) < 1e-15
    end
end
