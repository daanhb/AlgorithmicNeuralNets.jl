@testset "Concatenation" begin
    @test size(hcat_model(ann_sort2(), ann_sort2())) == (2,2)
    @test hcat_model(ann_sort2(), ann_sort2())([1,2]) == [1,2]
    @test hcat_model(ann_sort2(), ann_max2())([1,2]) == [2]

    @test size(vcat_model(ann_sort2(), ann_sort2())) == (4,4)
    @test vcat_model(ann_sort2(), ann_sort2())([1,2,4,1]) == [1,2,1,4]
    @test vcat_model(ann_sort2(), ann_max2())([2,1,4,1]) == [1,2,4]
end

