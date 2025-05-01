@testset "Basic nets" begin
    id1 = ann_id()
    @test size(id1) == (1,1)
    @test id1(0.5) == 0.5
    @test id1(-0.5) == -0.5

    zero1 = AN.ann_zero()
    @test size(zero1) == (1,1)
    @test zero1(0.5) == 0

    ct1 = AN.ann_constant(1.0)
    @test size(ct1) == (1,1)
    @test ct1(0.5) == 1

    sel1 = AN.ann_select(2, 5)
    @test size(sel1) == (1,5)
    @test sel1([1,2,3,4,5]) == [2]

    inp1 = AN.ann_input(2, 5)
    @test size(inp1) == (5,1)
    @test inp1([2]) == [0,2,0,0,0]

    id2 = AN.ann_id(3)
    @test size(id2) == (3,3)
    @test depth(id2) == 3
    @test id2([1,2,3]) == [1,2,3]

    shi1 = AN.ann_shift(1.0)
    @test size(shi1) == (1,1)
    @test shi1(1) == 2

    abs1 = ann_abs()
    @test size(abs1) == (1,1)
    @test abs1(0.5) == 0.5
    @test abs1(-0.5) == 0.5

    min2 = ann_min2()
    @test size(min2) == (1,2)
    @test min2([1;2]) == [1]
    @test min2([2;-1]) == [-1]

    max2 = ann_max2()
    @test size(max2) == (1,2)
    @test max2([1;2]) == [2]
    @test max2([2;-1]) == [2]

    sort2 = ann_sort2()
    @test size(sort2) == (2,2)
    @test sort2([1;2]) == [1; 2]
    @test sort2([2;-1]) == [-1; 2]

    perm2 = AN.ann_permute2()
    @test size(perm2) == (2,2)
    @test perm2([1;2]) == [2; 1]
end
