"""

 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

File:       PlanarQuadMPC.jl
Author:     Gabriel Barsi Haberfeld, 2020. gbh2@illinois.edu
Function:   This program implemetns an MPC controller for trajectory tracking of
            a planar quadrotor.

Instructions:   Run this file in juno with Julia 1.5.1 or later.
Requirements:   JuMP, Ipopt, Plots, LinearAlgebra, BenchmarkTools.

"""

using JuMP, Ipopt
using Plots, LinearAlgebra
using BenchmarkTools
using Random
using RecipesBase
using SparseArrays
using Statistics
using Printf
using LaTeXStrings
using Measures
import Distributions: MvNormal
import Random.seed!
g = 9.81

function dynamics(x=0.0.*zeros(6), u=0.0.*zeros(2), dt=1.0)
    #x = [px pz θ vx vz θ̇]
    #     1  2  3 4  5  6
    px = x[1]
    pz = x[2]
    θ  = x[3]
    vx = x[4]
    vz = x[5]
    θ̇  = x[6]
    x0 = copy(x);
    x[1] = vx.*cos.(θ) .- vz.*sin.(θ)
    x[2] = vx.*sin.(θ) .+ vz.*cos.(θ)
    x[3] = θ̇
    x[4] = vz.*θ̇ .- g.*cos.(θ)
    x[5] = .-vx.*θ̇ .- g.*sin.(θ) .+ u[1]
    x[6] = 0.0 .+ u[2]
    x = x.*dt .+ x0
end

function test_dynamics()
    dt = 0.1
    t = Array(0:0.1:10)
    N = length(t)
    x = 0.0.*zeros(6)
    u = 0.0.*zeros(2)
    xv = 0.0.*zeros(6,N)
    clearconsole()
    for i=1:N
        u[1] = 10.0
        u[2] = 0.0*(cos(t[i]/10.0)-0.5)
        x = dynamics(x,u,dt)
        @show u, x
    end
end
