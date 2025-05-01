using AlgorithmicNeuralNets
using Documenter

DocMeta.setdocmeta!(AlgorithmicNeuralNets, :DocTestSetup, :(using AlgorithmicNeuralNets); recursive=true)

makedocs(;
    modules=[AlgorithmicNeuralNets],
    authors="Daan Huybrechs <daan.huybrechs@kuleuven.be> and contributors",
    sitename="AlgorithmicNeuralNets.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
