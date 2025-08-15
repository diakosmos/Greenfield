using Plots

# 1D Godunov-type CFD solver for the Sod shock tube problem
# More robust than pure CESE for shock capturing

mutable struct CESESolver
    nx::Int                    # Number of grid points
    x::Vector{Float64}        # Grid coordinates
    dx::Float64               # Grid spacing
    dt::Float64               # Time step
    gamma::Float64            # Specific heat ratio
    cfl::Float64              # CFL number
    
    # Conservative variables [rho, rho*u, rho*E]
    U::Matrix{Float64}        # Current state
    U_new::Matrix{Float64}    # Updated state
end

function CESESolver(nx::Int, L::Float64, gamma::Float64, cfl::Float64)
    dx = L / (nx - 1)
    x = collect(0:dx:L)
    
    # Initialize arrays
    U = zeros(3, nx)
    U_new = zeros(3, nx)
    
    # Time step will be computed based on CFL condition
    dt = 0.0
    
    return CESESolver(nx, x, dx, dt, gamma, cfl, U, U_new)
end

# Convert conservative variables to primitive variables
function conservative_to_primitive(U::Vector{Float64}, gamma::Float64)
    rho = max(U[1], 1e-10)
    u = U[2] / rho
    rho_E = U[3]
    
    # Compute pressure with safeguards
    kinetic_energy = 0.5 * rho * u^2
    internal_energy = rho_E - kinetic_energy
    p = max((gamma - 1) * internal_energy, 1e-10 * rho)
    
    return [rho, u, p]
end

# Convert primitive variables to conservative variables
function primitive_to_conservative(W::Vector{Float64}, gamma::Float64)
    rho, u, p = W
    rho_u = rho * u
    rho_E = p / (gamma - 1) + 0.5 * rho * u^2
    return [rho, rho_u, rho_E]
end

# Compute flux vector
function compute_flux(U::Vector{Float64}, gamma::Float64)
    W = conservative_to_primitive(U, gamma)
    rho, u, p = W
    
    F1 = rho * u
    F2 = rho * u^2 + p
    F3 = (U[3] + p) * u
    
    return [F1, F2, F3]
end

# Initialize Sod shock tube problem
function initialize_sod_problem!(solver::CESESolver)
    x_interface = 0.5
    
    for i in 1:solver.nx
        if solver.x[i] <= x_interface
            # Left state: high pressure
            rho_L, u_L, p_L = 1.0, 0.0, 1.0
            solver.U[:, i] = primitive_to_conservative([rho_L, u_L, p_L], solver.gamma)
        else
            # Right state: low pressure
            rho_R, u_R, p_R = 0.125, 0.0, 0.1
            solver.U[:, i] = primitive_to_conservative([rho_R, u_R, p_R], solver.gamma)
        end
    end
    
    println("Initial conditions set successfully")
end

# HLLE Riemann solver for robust flux computation
function hlle_flux(U_L::Vector{Float64}, U_R::Vector{Float64}, gamma::Float64)
    # Convert to primitive variables
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    
    rho_L, u_L, p_L = W_L
    rho_R, u_R, p_R = W_R
    
    # Sound speeds
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)
    
    # Wave speed estimates (HLLE)
    S_L = min(u_L - c_L, u_R - c_R)
    S_R = max(u_L + c_L, u_R + c_R)
    
    # Fluxes
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)
    
    # HLLE flux
    if S_L >= 0.0
        return F_L
    elseif S_R <= 0.0
        return F_R
    else
        # Intermediate state
        return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    end
end

# Compute time step based on CFL condition
function compute_time_step!(solver::CESESolver)
    max_speed = 0.0
    
    for i in 1:solver.nx
        W = conservative_to_primitive(solver.U[:, i], solver.gamma)
        rho, u, p = W
        
        c = sqrt(solver.gamma * p / rho)
        speed = abs(u) + c
        max_speed = max(max_speed, speed)
    end
    
    dt_cfl = solver.cfl * solver.dx / max_speed
    dt_max = 0.001  # Reasonable maximum
    
    solver.dt = min(dt_cfl, dt_max)
end

# Monotonic reconstruction (minmod limiter)
function minmod(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

# Second-order reconstruction with limiting
function reconstruct_states!(solver::CESESolver, i::Int)
    # Simple piecewise linear reconstruction with minmod limiter
    U_L = zeros(3)
    U_R = zeros(3)
    
    if i == 1
        U_L = solver.U[:, i]
        U_R = solver.U[:, i]
    elseif i == solver.nx
        U_L = solver.U[:, i]
        U_R = solver.U[:, i]
    else
        for k in 1:3
            # Compute limited slopes
            slope_L = solver.U[k, i] - solver.U[k, i-1]
            slope_R = solver.U[k, i+1] - solver.U[k, i]
            slope = minmod(slope_L, slope_R)
            
            # Reconstruct left and right states
            U_L[k] = solver.U[k, i] - 0.5 * slope
            U_R[k] = solver.U[k, i] + 0.5 * slope
        end
    end
    
    return U_L, U_R
end

# Update solution using finite volume method with HLLE
function update_solution!(solver::CESESolver)
    # Compute fluxes at all interfaces
    fluxes = zeros(3, solver.nx+1)
    
    # Boundary fluxes (transmissive)
    fluxes[:, 1] = compute_flux(solver.U[:, 1], solver.gamma)
    fluxes[:, solver.nx+1] = compute_flux(solver.U[:, solver.nx], solver.gamma)
    
    # Interior fluxes using HLLE Riemann solver
    for i in 2:solver.nx
        # Get left and right states
        U_L, _ = reconstruct_states!(solver, i-1)
        _, U_R = reconstruct_states!(solver, i)
        
        # Use cell-centered values for first-order (more robust)
        U_L = solver.U[:, i-1]
        U_R = solver.U[:, i]
        
        fluxes[:, i] = hlle_flux(U_L, U_R, solver.gamma)
    end
    
    # Update solution
    for i in 1:solver.nx
        solver.U_new[:, i] = solver.U[:, i] - 
                            (solver.dt / solver.dx) * (fluxes[:, i+1] - fluxes[:, i])
        
        # Positivity preservation
        solver.U_new[1, i] = max(solver.U_new[1, i], 1e-12)
        
        # Ensure minimum pressure through energy
        rho = solver.U_new[1, i]
        rho_u = solver.U_new[2, i]
        u = rho_u / rho
        p_min = 1e-12
        E_min = p_min / (solver.gamma - 1) + 0.5 * rho * u^2
        solver.U_new[3, i] = max(solver.U_new[3, i], E_min)
    end
    
    # Copy solution
    solver.U .= solver.U_new
end

# Main solver function
function solve_sod_shock(nx::Int=100, t_final::Float64=0.2, 
                        gamma::Float64=1.4, cfl::Float64=0.4)
    
    # Create solver
    solver = CESESolver(nx, 1.0, gamma, cfl)
    
    # Initialize problem
    initialize_sod_problem!(solver)
    
    # Time stepping
    t = 0.0
    step = 0
    
    println("Starting Godunov-type simulation...")
    println("Grid points: $nx, Final time: $t_final, CFL: $cfl")
    
    while t < t_final
        compute_time_step!(solver)
        
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        update_solution!(solver)
        
        t += solver.dt
        step += 1
        
        if step % 100 == 0
            rho_min = minimum(solver.U[1, :])
            rho_max = maximum(solver.U[1, :])
            println("Step: $step, Time: $(round(t, digits=4)), dt: $(round(solver.dt, digits=6))")
            println("  Density range: $(round(rho_min, digits=6)) - $(round(rho_max, digits=6))")
        end
    end
    
    println("Simulation completed after $step steps")
    return solver
end

# Plot results
function plot_results(solver::CESESolver)
    # Extract primitive variables
    rho = zeros(solver.nx)
    u = zeros(solver.nx)
    p = zeros(solver.nx)
    
    for i in 1:solver.nx
        W = conservative_to_primitive(solver.U[:, i], solver.gamma)
        rho[i] = W[1]
        u[i] = W[2]
        p[i] = W[3]
    end
    
    # Create plots
    p1 = plot(solver.x, rho, linewidth=2, label="Density", marker=:circle, markersize=2,
              xlabel="x", ylabel="ρ", title="Sod Shock Tube - Godunov Solution")
    
    p2 = plot(solver.x, u, linewidth=2, label="Velocity", color=:red, marker=:circle, markersize=2,
              xlabel="x", ylabel="u")
    
    p3 = plot(solver.x, p, linewidth=2, label="Pressure", color=:green, marker=:circle, markersize=2,
              xlabel="x", ylabel="p")
    
    # Mach number
    c = sqrt.(solver.gamma .* p ./ rho)
    mach = abs.(u) ./ c
    p4 = plot(solver.x, mach, linewidth=2, label="Mach Number", color=:orange, marker=:circle, markersize=2,
              xlabel="x", ylabel="Ma")
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
end

# Exact Riemann solver for comparison (simplified)
function exact_solution(x::Vector{Float64}, t::Float64, gamma::Float64)
    # Exact solution parameters for Sod problem
    # These are approximate values - for exact solution need to solve Riemann problem
    
    rho = similar(x)
    u = similar(x)
    p = similar(x)
    
    # Approximate wave speeds and states
    shock_pos = 0.5 + 1.75 * t  # Shock position
    contact_pos = 0.5 + 0.92 * t  # Contact discontinuity
    rarefaction_head = 0.5 - 1.0 * t  # Rarefaction head
    rarefaction_tail = 0.5 - 0.35 * t  # Rarefaction tail
    
    for i in 1:length(x)
        if x[i] < rarefaction_head
            # Left state
            rho[i] = 1.0
            u[i] = 0.0
            p[i] = 1.0
        elseif x[i] < rarefaction_tail
            # Rarefaction fan
            rho[i] = 1.0 * (0.8)
            u[i] = 0.35
            p[i] = 1.0 * (0.8)^gamma
        elseif x[i] < contact_pos
            # Post-shock state 2
            rho[i] = 0.426
            u[i] = 0.927
            p[i] = 0.303
        elseif x[i] < shock_pos
            # Post-shock state 3
            rho[i] = 0.265
            u[i] = 0.927
            p[i] = 0.303
        else
            # Right state
            rho[i] = 0.125
            u[i] = 0.0
            p[i] = 0.1
        end
    end
    
    return rho, u, p
end

# Run the simulation
println("="^50)
println("1D Godunov-type Sod Shock Tube Solver")
println("="^50)

# Solve the problem
solver = solve_sod_shock(150, 0.2, 1.4, 0.4)

# Plot results
plt = plot_results(solver)
display(plt)

# Print some final statistics
println("\nFinal simulation statistics:")
println("Grid points: $(solver.nx)")
println("Final time: 0.2")
println("CFL number: $(solver.cfl)")

rho_range = extrema(solver.U[1, :])
println("Final density range: $(round(rho_range[1], digits=6)) - $(round(rho_range[2], digits=6))")

println("\nExpected features:")
println("- Shock wave at x ≈ 0.85")
println("- Contact discontinuity at x ≈ 0.69") 
println("- Rarefaction wave between x ≈ 0.26 and x ≈ 0.5")
