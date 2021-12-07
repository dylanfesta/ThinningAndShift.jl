using ThinningAndShift
using Documenter

DocMeta.setdocmeta!(ThinningAndShift, :DocTestSetup, :(using ThinningAndShift); recursive=true)

makedocs(;
    modules=[ThinningAndShift],
    authors="Dylan Festa",
    repo="https://github.com/dylanfesta/ThinningAndShift.jl/blob/{commit}{path}#{line}",
    sitename="ThinningAndShift.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/ThinningAndShift.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/ThinningAndShift.jl",
    devbranch="main",
)
