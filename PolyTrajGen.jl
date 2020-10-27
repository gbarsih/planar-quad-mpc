"""

 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

File:       PolyTrajGen.jl
Author:     Gabriel Barsi Haberfeld, 2020. gbh2@illinois.edu
Function:   This program implements minSnapTrajec_matlab by Clark-Youngdong-Son
            in Julia.

Instructions:   Run this file in juno with Julia 1.5.1 or later.
Requirements:   JuMP, COSMO, Plots, LinearAlgebra, Polynomials, ControlSystems,
                and BenchmarkTools

"""

using JuMP, COSMO
using Plots, LinearAlgebra
using Polynomials
using ControlSystems
using BenchmarkTools

function computeTraj()
    m_q = 4                                #mass of a quadrotor
    I_q = diag([0.3 0.3 0.3])              #moment of inertia of a quadrotor
    g = 9.81                               #gravitational acceleration

    #Trajectory
    #keyframes, column = index, rows = xyz yaw
    keyframe = [
        0 7.1324 14.3516 7.9345
        0 8.6237 0.7840 -5.3136
        0 1.0192 1.3676 0.9495
        0 0 0 0
    ]
    m = size(keyframe, 2)
    n = 4                                  #number of flat outputs (x, y, z, psi)
    t_f = 15                               #final time of the trajectory

    order = 6                              #order of polynomial functions

    time_interval_selection_flag = true    #true : fixed time interval, false : optimal time interval
    if (time_interval_selection_flag)
        t = collect(range(0, stop = t_f, length = m + 1))
    end

    n_intermediate = 5
    corridor_width = 0.05
    corridor_position = [3 4]
    c = zeros(4 * (order + 1) * m)
    mu_r = 1
    mu_psi = 1
    k_r = 4
    k_psi = 2
    A = computeCostMat(order, m, mu_r, mu_psi, k_r, k_psi, t)
    C, b, Cin, bin = computeConstraint(
        order,
        m,
        3,
        2,
        t,
        keyframe,
        corridor_position,
        n_intermediate,
        corridor_width,
    )
    solution = optimizeTraj(A, C, b)
    PlotTraj(solution, m, t, keyframe, n)
end

mutable struct uavstate
    position
    angle
    speed
end

mutable struct waypoints
    position
    heading
    speed
    time
end

function a2bPoly(aState, bState, tfinal=10.0)
    waypoints1 = waypoints(aState.position, aState.angle[1], eul2rotmZYX(aState.angle)*aState.speed, 0.0)
    waypoints2 = waypoints(bState.position, bState.angle[1], eul2rotmZYX(bState.angle)*bState.speed, tfinal)
    r_xyz = 1; r_yaw = 1;
    H = zeros(18,18);    # 5 * 3 + 3 (4th order for xyz & 2nd order for yaw)
    H[1,1] = r_xyz;
    H[6,6] = r_xyz;
    H[11,11] = 10000*r_xyz;
    H[16,16] = r_yaw;
    # start & final constraints (equality)
    Aeq = [waypoints1.time^4 waypoints1.time^3 waypoints1.time^2 waypoints1.time 1 zeros(1,13)];            # x
    Aeq = [Aeq;[zeros(1,5) waypoints1.time^4 waypoints1.time^3 waypoints1.time^2 waypoints1.time 1 zeros(1,8)]];  # y
    Aeq = [Aeq;[zeros(1,10) waypoints1.time^4 waypoints1.time^3 waypoints1.time^2 waypoints1.time 1 zeros(1,3)]]; # z
    Aeq = [Aeq;[zeros(1,15) waypoints1.time^2 waypoints1.time 1]]; # yaw
    beq = [waypoints1.position[1];waypoints1.position[2];waypoints1.position[3];waypoints1.heading];

    Aeq = [Aeq;[waypoints2.time^4 waypoints2.time^3 waypoints2.time^2 waypoints2.time 1 zeros(1,13)]];            # x
    Aeq = [Aeq;[zeros(1,5) waypoints2.time^4 waypoints2.time^3 waypoints2.time^2 waypoints2.time 1 zeros(1,8)]];  # y
    Aeq = [Aeq;[zeros(1,10) waypoints2.time^4 waypoints2.time^3 waypoints2.time^2 waypoints2.time 1 zeros(1,3)]]; # z
    Aeq = [Aeq;[zeros(1,15) waypoints2.time^2 waypoints2.time 1]];                                                    # yaw
    beq = [beq;waypoints2.position[1];waypoints2.position[2];waypoints2.position[3];waypoints2.heading];

    # velocity of the start point
    Aeq = [Aeq;[4*waypoints1.time^3 3*waypoints1.time^2 2*waypoints1.time 1 0 zeros(1,13)]];                              # x'
    Aeq = [Aeq;[zeros(1,5) 4*waypoints1.time^3 3*waypoints1.time^2 2*waypoints1.time 1 0 zeros(1,8)]];                    # y'
    Aeq = [Aeq;[zeros(1,10) 4*waypoints1.time^3 3*waypoints1.time^2 2*waypoints1.time 1 0 zeros(1,3)]];                   # z'
    beq = [beq;waypoints1.speed[1];waypoints1.speed[2];waypoints1.speed[3]];
    # velocity of the final point
    Aeq = [Aeq;[cos(waypoints2.heading)*[4*waypoints2.time^3 3*waypoints2.time^2 2*waypoints2.time 1 0] sin(waypoints2.heading)*[4*waypoints2.time^3 3*waypoints2.time^2 2*waypoints2.time 1 0] zeros(1,8)]];
    Aeq = [Aeq;[sin(waypoints2.heading)*[4*waypoints2.time^3 3*waypoints2.time^2 2*waypoints2.time 1 0] -cos(waypoints2.heading)*[4*waypoints2.time^3 3*waypoints2.time^2 2*waypoints2.time 1 0] zeros(1,8)]];
    Aeq = [Aeq;[zeros(1,10) 4*waypoints2.time^3 3*waypoints2.time^2 2*waypoints2.time 1 0 zeros(1,3)]];                   # z'
    beq = [beq;waypoints2.speed[1];waypoints2.speed[2];waypoints2.speed[3]];
    # solving quadratic problem
   sol = optimizeTraj(H,Aeq,beq)
   plota2bPoly(sol, tfinal, aState, bState)
   return sol
end

function computeCostMat(order, m, mu_r, mu_psi, k_r, k_psi, t)
    polynomial_r = Polynomial(ones(order + 1))
    for i = 1:k_r
        polynomial_r = derivative(polynomial_r)       #Differentiation up to k
    end
    polynomial_r = reverse(polynomial_r.coeffs)
    polynomial_psi = Polynomial(ones(order + 1))
    for i = 1:k_psi
        polynomial_psi = derivative(polynomial_psi)   #Differentiation up to k
    end
    polynomial_psi = reverse(polynomial_psi.coeffs)
    A = []
    for i = 1:m
        A_x = zeros(order + 1, order + 1)
        A_y = zeros(order + 1, order + 1)
        A_z = zeros(order + 1, order + 1)
        A_psi = zeros(order + 1, order + 1)
        for j = 1:order+1
            for k = j:order+1

                #Position
                if (j <= length(polynomial_r) && (k <= length(polynomial_r)))
                    order_t_r = ((order - k_r - j + 1) + (order - k_r - k + 1))
                    if (j == k)
                        A_x[j, k] =
                            polynomial_r[j]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_y[j, k] =
                            polynomial_r[j]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_z[j, k] =
                            polynomial_r[j]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                    else
                        A_x[j, k] =
                            2 * polynomial_r[j] * polynomial_r[k] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_y[j, k] =
                            2 * polynomial_r[j] * polynomial_r[k] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_z[j, k] =
                            2 * polynomial_r[j] * polynomial_r[k] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                    end
                end

                #Yaw
                if (j <= length(polynomial_psi) && (k <= length(polynomial_psi)))
                    order_t_psi = ((order - k_psi - j + 1) + (order - k_psi - k + 1))
                    if (j == k)
                        A_psi[j, k] =
                            polynomial_psi[j]^2 / (order_t_psi + 1) *
                            (t[i+1]^(order_t_psi + 1) - t[i]^(order_t_psi + 1))
                    else
                        A_psi[j, k] =
                            2 * polynomial_psi[j] * polynomial_psi[k] / (order_t_psi + 1) *
                            (t[i+1]^(order_t_psi + 1) - t[i]^(order_t_psi + 1))
                    end
                end

            end
        end
        if i == 1
            blocks = [mu_r * A_x, mu_r * A_y, mu_r * A_z, mu_psi * A_psi]
        else
            blocks = [A, mu_r * A_x, mu_r * A_y, mu_r * A_z, mu_psi * A_psi]
        end
        A = ControlSystems.blockdiag(blocks...)
    end
    A = 0.5 * (A + A') #Make it symmetric
end


function computeConstraint(
    order,
    m,
    k_r,
    k_psi,
    t,
    keyframe,
    corridor_position,
    n_intermediate,
    corridor_width,
)

    n = 4                              #State number

    #Waypoint constraints
    C1 = zeros(2 * m * n, n * (order + 1) * m)
    b1 = zeros(2 * m * n)
    C = []
    b = []
    computeMat = diagm(ones(order + 1))          #Required for computation of polynomials
    for i = 1:m
        waypoint = keyframe[:, i]       #Waypoint at t(i)

        if (i == 1)                        #Initial and Final Position
            #Initial
            values = zeros(1, order + 1)
            for j = 1:order+1
                poly = Polynomial(computeMat[j, :])
                values[j] = poly(t[i])
            end
            values = reverse(values, dims = 2)
            for k = 1:n
                c = zeros(1, n * (order + 1) * m)
                c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                    values
                C1[k, :] = c
            end
            b1[1:n] = waypoint

            #Final
            for j = 1:order+1
                poly = Polynomial(computeMat[j, :])
                values[j] = poly(t[m+1])
            end
            values = reverse(values, dims = 2)
            for k = 1:n
                c = zeros(1, n * (order + 1) * m)
                c[((m-1)*(order+1)*n+(k-1)*(order+1)+1):((m-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                    values
                C1[k+n, :] = c
            end
            b1[n+1:2*n] = waypoint

        else
            #Elsewhere
            values = zeros(1, order + 1)
            for j = 1:order+1
                poly = Polynomial(computeMat[j, :])
                values[j] = poly(t[i])
            end
            values = reverse(values, dims = 2)
            for k = 1:n
                c = zeros(1, n * (order + 1) * m)
                c[((i-2)*(order+1)*n+(k-1)*(order+1)+1):((i-2)*(order+1)*n+(k-1)*(order+1))+order+1] =
                    values
                C1[k+2*n*(i-1), :] = c
            end
            b1[2*n*(i-1)+1:2*n*(i-1)+n] = waypoint

            for k = 1:n
                c = zeros(1, n * (order + 1) * m)
                c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                    values
                C1[k+2*n*(i-1)+n, :] = c
            end
            b1[2*n*(i-1)+n+1:2*n*(i-1)+2*n] = waypoint

        end

    end


    # Derivative constraints

    # Position
    C2 = zeros(2 * m * (n - 1) * k_r, n * (order + 1) * m)                    #(n-1) : yaw excluded here
    b2 = ones(2 * m * (n - 1) * k_r, 1) * eps(1.0)

    #Only for the quadrotor system
    #Position constraints
    constraintData_r = zeros(m, k_r, 3)
    #velocity
    if (k_r >= 1)
        constraintData_r[1, 1, 1:3] .= 0          #At starting position
        constraintData_r[2:m, 1, 1:2] .= eps(1.0)      #x,y velocities
        constraintData_r[2:m, 1, 3] .= eps(1.0)        #z velocity
    end
    #acceleration
    if (k_r >= 2)
        constraintData_r[1, 2, 3] = 0            #At starting position
        constraintData_r[2:m, 2, 1:2] .= eps(1.0)      #x,y accelerations
        constraintData_r[2:m, 2, 3] .= eps(1.0)        #z acceleration
    end
    #jerk
    if (k_r >= 3)
        #all zeros
    end
    #snap
    if (k_r >= 4)
        #all zeros
    end
    #Yaw constraints
    constraintData_psi = zeros(m, k_psi, 1)
    #velocity
    if (k_psi >= 1)
        constraintData_psi[1, 1, 1] = 0          #At starting position
    end
    #acceleration
    if (k_psi >= 2)
        #all zeros
    end
    #jerk
    if (k_psi >= 3)
        #all zeros
    end
    #snap
    if (k_psi >= 4)
        #all zeros
    end

    #constraintData_r = zeros(m,k_r,3);
    for i = 1:m
        for h = 1:k_r
            if (i == 1)
                #Initial
                values = zeros(1, order + 1)
                for j = 1:order+1
                    tempCoeffs = computeMat[j, :]
                    for k = 1:h
                        tempCoeffs = derivative(Polynomial(tempCoeffs)).coeffs
                    end
                    tempPoly = Polynomial(tempCoeffs)
                    values[j] = tempPoly(t[i])
                end
                values = reverse(values, dims = 2)
                continuity = zeros(1, n - 1)
                for k = 1:n-1
                    if (constraintData_r[i, h, k] == eps(1.0))
                        #Continuity
                        continuity[k] = true
                    end

                    c = zeros(1, n * (order + 1) * m)
                    if (continuity[k] == true)
                        c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        c[((m-1)*(order+1)*n+(k-1)*(order+1)+1):((m-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            -values
                        C2[k+(h-1)*(n-1), :] = c
                        b2[k+(h-1)*(n-1)] = 0
                    else
                        c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        C2[k+(h-1)*(n-1), :] = c
                        b2[k+(h-1)*(n-1)] = constraintData_r[i, h, k]
                    end
                end

                #Final
                values = zeros(1, order + 1)
                for j = 1:order+1
                    tempCoeffs = computeMat[j, :]
                    for k = 1:h
                        tempCoeffs = derivative(Polynomial(tempCoeffs)).coeffs
                    end
                    tempPoly = Polynomial(tempCoeffs)
                    values[j] = tempPoly(t[i])
                end
                values = reverse(values, dims = 2)
                for k = 1:n-1
                    if (constraintData_r[i, h, k] == eps(1.0))
                        #Continuity
                    end
                    c = zeros(1, n * (order + 1) * m)
                    if (continuity[k] == 0)
                        c[((m-1)*(order+1)*n+(k-1)*(order+1)+1):((m-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        C2[k+(h-1)*(n-1)+(n-1)*k_r, :] = c
                        b2[k+(h-1)*(n-1)+(n-1)*k_r] = constraintData_r[i, h, k]
                    end
                end

            else

                #Elsewhere
                values = zeros(1, order + 1)
                for j = 1:order+1
                    tempCoeffs = computeMat[j, :]
                    for k = 1:h
                        tempCoeffs = derivative(Polynomial(tempCoeffs)).coeffs
                    end
                    tempPoly = Polynomial(tempCoeffs)
                    values[j] = tempPoly(t[i])
                end
                values = reverse(values, dims = 2)
                continuity = zeros(1, n - 1)
                for k = 1:n-1
                    if (constraintData_r[i, h, k] == eps(1.0))
                        #Continuity
                        continuity[k] = true
                    end

                    c = zeros(1, n * (order + 1) * m)
                    if (continuity[k] == true)
                        c[((i-2)*(order+1)*n+(k-1)*(order+1)+1):((i-2)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            -values
                        C2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r, :] = c
                        b2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r] = 0
                    else
                        c[((i-2)*(order+1)*n+(k-1)*(order+1)+1):((i-2)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        C2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r, :] = c
                        b2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r] = constraintData_r[i, h, k]
                    end
                end

                continuity = zeros(1, n - 1)
                for k = 1:n-1
                    if (constraintData_r[i, h, k] == eps(1.0))
                        #Continuity
                        continuity[k] = true
                    end
                    c = zeros(1, n * (order + 1) * m)

                    if (continuity[k] == 0)
                        c[((i-1)*(order+1)*n+(k-1)*(order+1)+1):((i-1)*(order+1)*n+(k-1)*(order+1))+order+1] =
                            values
                        C2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r+(n-1)*k_r, :] = c
                        b2[k+(h-1)*(n-1)+2*(i-1)*(n-1)*k_r+(n-1)*k_r] =
                            constraintData_r[i, h, k]
                    end

                end

            end
        end
    end

    #Corridor constraints

    C3 = []

    b3 = []
    t_vector =
        (keyframe[1:3, corridor_position[2]] - keyframe[1:3, corridor_position[1]]) /
        norm(keyframe[1:3, corridor_position[2]] - keyframe[1:3, corridor_position[1]])
    #unit vector of direction of the corridor

    t_intermediate = collect(range(
        t[corridor_position[1]],
        stop = t[corridor_position[2]],
        length = n_intermediate + 2,
    ))
    t_intermediate = t_intermediate[2:end-1]
    #intermediate time stamps

    computeMat = diagm(ones(order + 1))#Required for computation of polynomials
    for i = 1:n_intermediate
        values = zeros(1, order + 1)
        for j = 1:order+1
            poly = Polynomial(computeMat[j, :])
            values[j] = poly(t_intermediate[i])
        end

        c = zeros(6, n * (order + 1) * m)       #Absolute value constraint : two inequality constraints
        b = zeros(6, 1)

        rix = keyframe[1, corridor_position[1]]
        riy = keyframe[2, corridor_position[1]]
        riz = keyframe[3, corridor_position[1]]
        #x
        c[
            1,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            [values zeros(1, 2 * (order + 1))] -
            t_vector[1] * [t_vector[1] * values t_vector[2] * values t_vector[3] * values]
        b[1] =
            corridor_width +
            rix +
            t_vector[1] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])
        c[
            2,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            -c[
                1,
                (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
            ]
        b[2] =
            corridor_width - rix -
            t_vector[1] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])
        #y
        c[
            3,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            [zeros(1, order + 1) values zeros(1, order + 1)] -
            t_vector[2] * [t_vector[1] * values t_vector[2] * values t_vector[3] * values]
        b[3] =
            corridor_width +
            riy +
            t_vector[2] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])
        c[
            4,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            -c[
                3,
                (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
            ]
        b[4] =
            corridor_width - riy -
            t_vector[2] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])
        #z
        c[
            5,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            [zeros(1, 2 * (order + 1)) values] -
            t_vector[3] * [t_vector[1] * values t_vector[2] * values t_vector[3] * values]
        b[5] =
            corridor_width +
            riz +
            t_vector[3] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])
        c[
            6,
            (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
        ] =
            -c[
                5,
                (corridor_position[1]-1)*n*(order+1)+0*(order+1)+1:(corridor_position[1]-1)*n*(order+1)+3*(order+1),
            ]
        b[6] =
            corridor_width - riz -
            t_vector[3] * (-rix * t_vector[1] - riy * t_vector[2] - riz * t_vector[3])

        if (i == 1)
            C3 = c
            b3 = b
        else
            C3 = [C3; c]
            b3 = [b3; b]
        end

        C = [C1; C2]
        b = [b1; b2]
    end
    return C, b, C3, b3
end

function optimizeTraj(A, C, b)
    A = 2 .* A
    m = JuMP.Model(with_optimizer(COSMO.Optimizer, max_iter = 5000, verbose = false))
    @variable(m, x[1:size(A, 1)])
    @objective(m, Min, 1 / 2 * x' * A * x)
    @constraint(m, C * x .== b)
    JuMP.optimize!(m)
    return value.(x)
end

function PlotTraj(solution, m, t, keyframe, n)

    dt = 0.01
    l = 0
    lm = length(t[i]:dt:t[i+1])
    l = m * lm
    x_trajec = zeros(l)
    y_trajec = zeros(l)
    z_trajec = zeros(l)
    psi_trajec = zeros(l)
    tvec = zeros(l)

    for i = 1:m
        idx1 = (i - 1) * lm + 1
        idx2 = i * lm
        x_trajec[idx1:idx2] = polyval(
            solution[(i-1)*n*(order+1)+1+0*(order+1):(i-1)*n*(order+1)+(order+1)+0*(order+1)],
            t[i]:dt:t[i+1],
        )
        y_trajec[idx1:idx2] = polyval(
            solution[(i-1)*n*(order+1)+1+1*(order+1):(i-1)*n*(order+1)+(order+1)+1*(order+1)],
            t[i]:dt:t[i+1],
        )
        z_trajec[idx1:idx2] = polyval(
            solution[(i-1)*n*(order+1)+1+2*(order+1):(i-1)*n*(order+1)+(order+1)+2*(order+1)],
            t[i]:dt:t[i+1],
        )
        psi_trajec[idx1:idx2] = polyval(
            solution[(i-1)*n*(order+1)+1+3*(order+1):(i-1)*n*(order+1)+(order+1)+3*(order+1)],
            t[i]:dt:t[i+1],
        )
        tvec[idx1:idx2] = Array(t[i]:dt:t[i+1])
    end

    l = @layout [a; b; c]
    default(dpi = 300)
    default(thickness_scaling = 2)
    default(size = [1200, 800])
    plot(tvec, x_trajec, lw = 3, ylabel = "x", label = nothing)
    p1 = plot!(t, keyframe[1, :], seriestype = :scatter, label = nothing)
    plot(tvec, y_trajec, lw = 3, ylabel = "y", label = nothing)
    p2 = plot!(t, keyframe[2, :], seriestype = :scatter, label = nothing)
    plot(tvec, z_trajec, lw = 3, xlabel = "Time [s]", ylabel = "z", label = nothing)
    p3 = plot!(t, keyframe[3, :], seriestype = :scatter, label = nothing)
    p = plot(p1, p2, p3, layout = l)
    display(p)
end

function plota2bPoly(path_c,tfinal,aState,bState)
    tvec = 0:0.01:tfinal;
    x_trajec = path_c[1] * tvec.^4 .+ path_c[2] * tvec.^3 .+ path_c[3] * tvec.^2 .+ path_c[4] * tvec .+ path_c[5];
    y_trajec = path_c[6] * tvec.^4 .+ path_c[7] * tvec.^3 .+ path_c[8] * tvec.^2 .+ path_c[9] * tvec .+ path_c[10];
    z_trajec = path_c[11] * tvec.^4 .+ path_c[12] * tvec.^3 .+ path_c[13] * tvec.^2 .+ path_c[14] * tvec .+ path_c[15];
    yaw = path_c[16] * tvec.^2 .+ path_c[17] * tvec .+ path_c[18];
    default(dpi = 300)
    default(thickness_scaling = 2)
    default(size = [1200, 800])
    l = @layout [a; b; c]
    t = [0.0, tfinal]
    pos = [aState.position bState.position]
    p1 = plot(tvec, x_trajec, lw = 3, ylabel = "x", label = nothing)
    p1 = plot!(t, pos[1,:], seriestype = :scatter, label = nothing)
    p2 = plot(tvec, y_trajec, lw = 3, ylabel = "y", label = nothing)
    p2 = plot!(t, pos[2,:], seriestype = :scatter, label = nothing)
    p3 = plot(tvec, z_trajec, lw = 3, xlabel = "Time [s]", ylabel = "z", label = nothing)
    p3 = plot!(t, pos[3,:], seriestype = :scatter, label = nothing)
    p = plot(p1, p2, p3, layout = l)
    display(p)
    return nothing
end

function polyval(matlabPoly, tvec)
    #mimics matlabs polyval function
    poly = Polynomial(reverse(matlabPoly))
    poly.(tvec)
end

function eul2rotmZYX(eul)
    #mimics matlab
    ct = cos.(eul)
    st = sin.(eul)
    R = zeros(3, 3)
    R[1, 1] = ct[2] .* ct[1]
    R[1, 2] = st[3] .* st[2] .* ct[1] - ct[3] .* st[1]
    R[1, 3] = ct[3] .* st[2] .* ct[1] + st[3] .* st[1]
    R[2, 1] = ct[2] .* st[1]
    R[2, 2] = st[3] .* st[2] .* st[1] + ct[3] .* ct[1]
    R[2, 3] = ct[3] .* st[2] .* st[1] - st[3] .* ct[1]
    R[3, 1] = -st[2]
    R[3, 2] = st[3] .* ct[2]
    R[3, 3] = ct[3] .* ct[2]
    return R
end
