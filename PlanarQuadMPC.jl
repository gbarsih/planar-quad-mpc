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
m = 0.483
g = 9.81
I = 0.01532

function dynamics(x = 0.0 .* zeros(6), u = 0.0 .* zeros(2), dt = 1.0)
    #x = [px pz θ vx vz θ̇]
    #     1  2  3 4  5  6
    px = x[1]
    pz = x[2]
    θ = x[3]
    vx = x[4]
    vz = x[5]
    θ̇ = x[6]
    x0 = copy(x)
    x[1] = vx * cos(θ) - vz * sin(θ)
    x[2] = vx * sin(θ) + vz * cos(θ)
    x[3] = θ̇
    x[4] = vz * θ̇ - g * sin(θ)
    x[5] = -vx * θ̇ - g * cos(θ) + (u[1] + u[2]) / m
    x[6] = 0.0 + (u[1] - u[2]) / I
    x = x * dt + x0
end

function angelaDynamics(x = 0.0 .* zeros(6), u = 0.0 .* zeros(2), dt = 1.0)
    #x = [px pz θ vx vz θ̇]
    #     1  2  3 4  5  6
    px = x[1]
    pz = x[2]
    θ = x[3]
    vx = x[4]
    vz = x[5]
    θ̇ = x[6]
    x0 = copy(x)
    x[1] = vx
    x[2] = vz
    x[3] = θ̇
    x[4] = -1 / m * sin(θ)
    x[5] = 1 / m * cos(θ) * u[1] - g
    x[6] = 0.0 + u[2] / I
    x = x * dt + x0
end

function test_dynamics()
    dt = 0.1
    t = Array(0:0.1:1)
    N = length(t)
    x = 0.0 .* zeros(6)
    x[3] = 0.0
    u = 0.0 .* zeros(2)
    xv = 0.0 .* zeros(6, N)
    for i = 1:N
        u[1] = g
        u[2] = 0.1
        x = dynamics(x, u, dt)
        @show u, x
    end
end

function SimpleMPC(
    x0,
    xref = 0.0 .* zeros(6),
    θlim = pi / 4,
    θ̇lim = pi / 3,
    vxlim = 2,
    vzlim = 1,
    dt = 0.1,
    N = 20,
)
    MPC = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "max_iter" => 10,
        "print_level" => 0,
    ))
    @variable(MPC, px[i = 0:N])
    @variable(MPC, pz[i = 0:N])
    @variable(MPC, -θlim <= θ[i = 0:N] <= θlim)
    @variable(MPC, -vxlim <= vx[i = 0:N] <= vxlim)
    @variable(MPC, -vzlim <= vz[i = 0:N] <= vxlim)
    @variable(MPC, -θ̇lim <= θ̇[i = 0:N] <= θ̇lim)
    @variable(MPC, 0 <= u1[i = 0:N])
    @variable(MPC, 0 <= u2[i = 0:N])

    @constraint(MPC, px[0] == x0[1])
    @constraint(MPC, pz[0] == x0[2])
    @constraint(MPC, θ[0] == x0[3])
    @constraint(MPC, vx[0] == x0[4])
    @constraint(MPC, vz[0] == x0[5])
    @constraint(MPC, θ̇[0] == x0[6])

    #x = [px pz θ vx vz θ̇]
    #     1  2  3 4  5  6
    for k = 0:N-1
        @NLconstraint(MPC, px[k+1] == px[k] + (vx[k] * cos(θ[k]) - vz[k] * sin(θ[k])) * dt)
        @NLconstraint(MPC, pz[k+1] == pz[k] + (vx[k] * sin(θ[k]) + vz[k] * cos(θ[k])) * dt)
        @constraint(MPC, θ[k+1] == θ[k] + θ̇[k] * dt)
        @NLconstraint(MPC, vx[k+1] == vx[k] + (vz[k] * θ̇[k] - g * sin(θ[k])) * dt)
        @NLconstraint(
            MPC,
            vz[k+1] == vz[k] + (-vx[k] * θ̇[k] - g * cos(θ[k]) + (u1[k] + u2[k]) / m) * dt
        )
        @constraint(MPC, θ̇[k+1] == θ̇[k] + (u1[k] - u2[k]) / I * dt)
    end

    @objective(MPC, Min, sum(pz[i]^2 + px[i]^2 + 1e-6*u1[i]^2 + 1e-6*u2[i]^2 for i = 1:N))
    optimize!(MPC)
    return value.(u1), value.(u2), value.(px), value.(pz)
end

function runSimpleMPC()
    x = [1.0 1.0 0.0 0.0 0.0 0.0]
    dt = 0.1
    N = 50
    xv = zeros(6,N)
    tv = Array(0:dt:dt*(N-1))
    for t = 1:N
        u1, u2, px, pz = SimpleMPC(x, 0.0*zeros(6), pi / 4, pi / 3, 2, 1, dt)
        u = [u1[0] u2[0]]
        x = dynamics(x, u, dt)
        xv[:,t] = x
        @show u, x
    end
    plot(tv,xv[1,:])
    plot!(tv,xv[2,:])
end

function doubleIntMPC(x0,dt,T=10,n=2)
    MPC = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "max_iter" => 5000,
        "print_level" => 0,
    ))
    T = 10
    n = 2
    dt = 0.1
    @variable(MPC, x[i=1:n,t=0:T])
    @variable(MPC, -1 <= u[t=0:T] <= 1)
    @objective(MPC, Min, sum((x[i,T]).^2 for i=1:n))
    for i=1:n
        @constraint(MPC, x[i,0] == x0[i])
    end
    for t=0:T-1
        @constraint(MPC, x[1,t+1] == x[1,t] + x[2,t]*dt)
        @constraint(MPC, x[2,t+1] == x[2,t] + u[t]*dt)
    end
    optimize!(MPC)
    return value.(u), value.(x)
end

function doubleIntDynamics(x,u,dt)
    x[1] = x[1] + x[2]*dt
    x[2] = x[2] + u*dt
    return x
end

function runDoubleIntMPC()
    clearconsole()
    x = [1.0 1.0]
    dt = 0.1
    for t = 1:60
        u,xout = doubleIntMPC(x,dt,10)
        u = u[0]
        x = doubleIntDynamics(x,u,dt)
        @show u, x
    end
end
