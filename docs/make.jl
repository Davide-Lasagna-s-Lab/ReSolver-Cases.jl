using Documenter
using ReSolverTensorProductGrids

DocMeta.setdocmeta!(
    ReSolverTensorProductGrids,
    :DocTestSetup,
    :(using LinearAlgebra, ReSolverFlowsBase, ReSolverTensorProductGrids);
    recursive=true,
)

makedocs(
    sitename="ReSolverTensorProductGrids.jl",
    modules=[ReSolverTensorProductGrids],
    authors="Davide Lasagna, Thomas Burton, and contributors",
    doctest=true,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Davide-Lasagna-s-Lab.github.io/ReSolver-TensorProductGrids.jl/dev/",
        edit_link="main",
        repolink="https://github.com/Davide-Lasagna-s-Lab/ReSolver-TensorProductGrids.jl",
    ),
    remotes=Dict(normpath(joinpath(@__DIR__, "..")) =>
                 (Documenter.Remotes.GitHub("Davide-Lasagna-s-Lab", "ReSolver-TensorProductGrids.jl"), "main")),
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "Installation" => "installation.md",
            "Quick start" => "quickstart.md",
        ],
        "Concepts" => [
            "Coordinates and conventions" => "conventions.md",
            "Tensor-product grid design" => "tensorproduct.md",
        ],
        "Grids" => [
            "Channel" => "grids/channel.md",
            "2D square lid-driven cavity" => "grids/lid_driven_cavity_2d.md",
            "3D cubic lid-driven cavity" => "grids/lid_driven_cavity_3d.md",
            "Square duct" => "grids/square_duct.md",
        ],
        "Cases" => [
            "Reusable forcings" => "forcings.md",
            "Channel flows" => "cases/channel.md",
            "2D square lid-driven cavity" => "cases/lid_driven_cavity_2d.md",
            "3D cubic lid-driven cavity" => "cases/lid_driven_cavity_3d.md",
            "Square-duct flow" => "cases/square_duct.md",
        ],
        "Worked examples" => "examples.md",
        "Extending the package" => "extending.md",
        "API reference" => "api.md",
    ],
    checkdocs=:exports,
    linkcheck=get(ENV, "DOCUMENTER_LINKCHECK", "false") == "true",
    warnonly=false,
)

deploydocs(repo="github.com/Davide-Lasagna-s-Lab/ReSolver-TensorProductGrids.jl.git", devbranch="main")
