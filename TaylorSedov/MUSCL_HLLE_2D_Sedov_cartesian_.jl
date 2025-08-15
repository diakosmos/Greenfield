using Plots
using LinearAlgebra

# 2D Cartesian MUSCL+HLLE Euler Solver
# Much simpler with structured grid

# EOS interface (same as before)
abstract type EquationOfState end

struct IdealGas <: EquationOfState
    gamma::Float64
end

function pressure(eos::IdealGas, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
    u = rho_u / rho
    v = rho_v / rho
    kinetic = 0.5 * rho * (u^2 + v^2)
    return (eos.gamma - 1) * (rho_E - kinetic)
end

function sound_speed(eos::IdealGas, rho::Float64, p::Float64)
    return sqrt(eos.gamma * p / rho)
end

function total_energy(eos::IdealGas, rho::Float64, u::Float64, v::Float64, p::Float64)
    return p / (eos.gamma - 1) + 0.5 * rho * (u^2 + v^2)
end

# Simple 2D Cartesian solver
mutable struct CartesianSolver
    nx::Int                 # Number of cells in x
    ny::Int                 # Number of cells in y
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    dx::Float64             # Grid spacing in x
    dy::Float64             # Grid spacing in y
    eos::EquationOfState
    cfl::Float64
    dt::Float64
    
    # Conservative variables: [rho, rho*u, rho*v, rho*E]
    # Stored as 4D array: (variable, i, j, time_level)
    U::Array{Float64, 4}    # (4, nx, ny, 2) - ping-pong buffers
    current::Int            # Current time level (1 or 2)
    
    # Cell centers
    x::Vector{Float64}
    y::Vector{Float64}
end

function CartesianSolver(nx::Int, ny::Int, x_min::Float64, x_max::Float64,
                        y_min::Float64, y_max::Float64, eos::EquationOfState, cfl::Float64)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    # Cell centers
    x = [x_min + (i-0.5)*dx for i in 1:nx]
    y = [y_min + (j-0.5)*dy for j in 1:ny]
    
    U = zeros(4, nx, ny, 2)
    
    return CartesianSolver(nx, ny, x_min, x_max, y_min, y_max, dx, dy, eos, cfl, 0.0, 
                          U, 1, x, y)
end

# Conservative to primitive conversion
function conservative_to_primitive_2d(U::Vector{Float64}, eos::EquationOfState)
    rho = max(U[1], 1e-12)
    u = U[2] / rho
    v = U[3] / rho
    p = max(pressure(eos, rho, U[4], U[2], U[3]), 1e-12)
    return [rho, u, v, p]
end

# Primitive to conservative conversion
function primitive_to_conservative_2d(W::Vector{Float64}, eos::EquationOfState)
    rho, u, v, p = W
    return [rho, rho*u, rho*v, total_energy(eos, rho, u, v, p)]
end

# HLLE Riemann solver for 2D
function hlle_flux_2d(U_L::Vector{Float64}, U_R::Vector{Float64}, direction::Int, eos::EquationOfState)
    # direction: 1 = x-direction, 2 = y-direction
    
    W_L = conservative_to_primitive_2d(U_L, eos)
    W_R = conservative_to_primitive_2d(U_R, eos)
    
    rho_L, u_L, v_L, p_L = W_L
    rho_R, u_R, v_R, p_R = W_R
    
    c_L = sound_speed(eos, rho_L, p_L)
    c_R = sound_speed(eos, rho_R, p_R)
    
    if direction == 1  # x-direction
        # Wave speeds
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        # x-direction fluxes
        F_L = [rho_L * u_L,
               rho_L * u_L^2 + p_L,
               rho_L * u_L * v_L,
               (U_L[4] + p_L) * u_L]
        
        F_R = [rho_R * u_R,
               rho_R * u_R^2 + p_R,
               rho_R * u_R * v_R,
               (U_R[4] + p_R) * u_R]
               
    else  # y-direction
        # Wave speeds  
        S_L = min(v_L - c_L, v_R - c_R)
        S_R = max(v_L + c_L, v_R + c_R)
        
        # y-direction fluxes
        F_L = [rho_L * v_L,
               rho_L * u_L * v_L,
               rho_L * v_L^2 + p_L,
               (U_L[4] + p_L) * v_L]
        
        F_R = [rho_R * v_R,
               rho_R * u_R * v_R,
               rho_R * v_R^2 + p_R,
               (U_R[4] + p_R) * v_R]
    end
    
    # HLLE flux
    if S_L >= 0.0
        return F_L
    elseif S_R <= 0.0
        return F_R
    else
        return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    end
end

# Minmod limiter
function minmod(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

# MUSCL reconstruction with limiting
function muscl_reconstruct(U_left::Vector{Float64}, U_center::Vector{Float64}, U_right::Vector{Float64})
    # Compute limited slopes
    U_L = zeros(4)
    U_R = zeros(4)
    
    for k in 1:4
        slope = minmod(U_center[k] - U_left[k], U_right[k] - U_center[k])
        U_L[k] = U_center[k] - 0.5 * slope
        U_R[k] = U_center[k] + 0.5 * slope
    end
    
    return U_L, U_R
end

# Compute time step
function compute_time_step!(solver::CartesianSolver)
    max_speed = 0.0
    current = solver.current
    
    for j in 1:solver.ny
        for i in 1:solver.nx
            U = solver.U[:, i, j, current]
            W = conservative_to_primitive_2d(U, solver.eos)
            rho, u, v, p = W
            
            c = sound_speed(solver.eos, rho, p)
            speed_x = abs(u) + c
            speed_y = abs(v) + c
            
            # CFL condition for 2D
            dt_x = solver.dx / speed_x
            dt_y = solver.dy / speed_y
            dt_cell = min(dt_x, dt_y)
            
            max_speed = max(max_speed, 1.0 / dt_cell)
        end
    end
    
    solver.dt = solver.cfl / max_speed
    
    # Safety limits
    solver.dt = min(solver.dt, 1e-4)
    solver.dt = max(solver.dt, 1e-8)
end

# Main update step with proper symmetry preservation
function update_solution!(solver::CartesianSolver)
    current = solver.current
    next = 3 - current  # Ping-pong between 1 and 2
    
    # Copy current solution to next
    solver.U[:, :, :, next] .= solver.U[:, :, :, current]
    
    # Compute ALL fluxes first (preserves symmetry)
    flux_x = zeros(4, solver.nx+1, solver.ny)  # x-direction fluxes
    flux_y = zeros(4, solver.nx, solver.ny+1)  # y-direction fluxes
    
    # x-direction fluxes at all i+1/2 interfaces
    for j in 2:solver.ny-1
        for i in 1:solver.nx-1  # Interface i+1/2 between cells i and i+1
            if i == 1
                # Left boundary: use cell values directly
                U_L = solver.U[:, 1, j, current]
                U_R = solver.U[:, 2, j, current]
            elseif i == solver.nx-1
                # Right boundary: use cell values directly  
                U_L = solver.U[:, solver.nx-1, j, current]
                U_R = solver.U[:, solver.nx, j, current]
            else
                # Interior: MUSCL reconstruction
                U_im1 = solver.U[:, i-1, j, current]
                U_i   = solver.U[:, i,   j, current]
                U_ip1 = solver.U[:, i+1, j, current]
                U_ip2 = solver.U[:, i+2, j, current]
                
                # Left state (from cell i)
                U_L, _ = muscl_reconstruct(U_im1, U_i, U_ip1)
                # Right state (from cell i+1)  
                _, U_R = muscl_reconstruct(U_i, U_ip1, U_ip2)
            end
            
            # Ensure physical states
            U_L[1] = max(U_L[1], 1e-12)
            U_R[1] = max(U_R[1], 1e-12)
            
            # HLLE flux in x-direction
            flux_x[:, i+1, j] = hlle_flux_2d(U_L, U_R, 1, solver.eos)
        end
    end
    
    # y-direction fluxes at all j+1/2 interfaces  
    for j in 1:solver.ny-1  # Interface j+1/2 between cells j and j+1
        for i in 2:solver.nx-1
            if j == 1
                # Bottom boundary: use cell values directly
                U_L = solver.U[:, i, 1, current]
                U_R = solver.U[:, i, 2, current]
            elseif j == solver.ny-1
                # Top boundary: use cell values directly
                U_L = solver.U[:, i, solver.ny-1, current]
                U_R = solver.U[:, i, solver.ny, current]
            else
                # Interior: MUSCL reconstruction
                U_jm1 = solver.U[:, i, j-1, current]
                U_j   = solver.U[:, i, j,   current]
                U_jp1 = solver.U[:, i, j+1, current]
                U_jp2 = solver.U[:, i, j+2, current]
                
                # Left state (from cell j)
                U_L, _ = muscl_reconstruct(U_jm1, U_j, U_jp1)
                # Right state (from cell j+1)
                _, U_R = muscl_reconstruct(U_j, U_jp1, U_jp2)
            end
            
            # Ensure physical states
            U_L[1] = max(U_L[1], 1e-12)
            U_R[1] = max(U_R[1], 1e-12)
            
            # HLLE flux in y-direction
            flux_y[:, i, j+1] = hlle_flux_2d(U_L, U_R, 2, solver.eos)
        end
    end
    
    # Now apply ALL flux updates simultaneously
    for j in 2:solver.ny-1
        for i in 2:solver.nx-1
            # x-direction flux contributions
            flux_left  = flux_x[:, i,   j]
            flux_right = flux_x[:, i+1, j]
            
            # y-direction flux contributions  
            flux_bottom = flux_y[:, i, j]
            flux_top    = flux_y[:, i, j+1]
            
            # Update cell (i,j) with all flux contributions
            solver.U[:, i, j, next] -= (solver.dt / solver.dx) * (flux_right - flux_left)
            solver.U[:, i, j, next] -= (solver.dt / solver.dy) * (flux_top - flux_bottom)
        end
    end
    
    # Apply boundary conditions (outflow)
    apply_boundary_conditions!(solver, next)
    
    # Ensure physical states
    for j in 1:solver.ny
        for i in 1:solver.nx
            solver.U[1, i, j, next] = max(solver.U[1, i, j, next], 1e-12)  # density
            
            # Check pressure and fix energy if needed
            U = solver.U[:, i, j, next]
            W = conservative_to_primitive_2d(U, solver.eos)
            if W[4] <= 1e-12  # pressure
                rho, u, v = W[1], W[2], W[3]
                p_min = 1e-12
                solver.U[4, i, j, next] = total_energy(solver.eos, rho, u, v, p_min)
            end
        end
    end
    
    # Switch time levels
    solver.current = next
end

# Apply outflow boundary conditions
function apply_boundary_conditions!(solver::CartesianSolver, time_level::Int)
    nx, ny = solver.nx, solver.ny
    
    # Left and right boundaries (x-direction)
    for j in 1:ny
        solver.U[:, 1,  j, time_level] = solver.U[:, 2,    j, time_level]  # Left
        solver.U[:, nx, j, time_level] = solver.U[:, nx-1, j, time_level]  # Right
    end
    
    # Bottom and top boundaries (y-direction)
    for i in 1:nx
        solver.U[:, i, 1,  time_level] = solver.U[:, i, 2,    time_level]  # Bottom
        solver.U[:, i, ny, time_level] = solver.U[:, i, ny-1, time_level]  # Top
    end
end

# Initialize Sedov blast wave with reasonable conditions
function initialize_sedov_blast!(solver::CartesianSolver, E_blast::Float64, r_blast::Float64)
    current = solver.current
    
    # Much more reasonable ambient conditions
    rho_ambient = 1.0
    p_ambient = 0.1          # Changed from 1e-6 to 0.1 - much more reasonable!
    u_ambient = 0.0
    v_ambient = 0.0
    
    # Find blast center (middle of domain)
    x_center = (solver.x_min + solver.x_max) / 2
    y_center = (solver.y_min + solver.y_max) / 2
    
    blast_volume = 0.0
    blast_cells = 0
    
    # First pass: calculate total blast volume
    for j in 1:solver.ny
        for i in 1:solver.nx
            x = solver.x[i]
            y = solver.y[j]
            r = sqrt((x - x_center)^2 + (y - y_center)^2)
            
            if r <= r_blast
                blast_volume += solver.dx * solver.dy
                blast_cells += 1
            end
        end
    end
    
    println("Blast initialization:")
    println("  Blast cells: $blast_cells")
    println("  Blast volume: $blast_volume")
    
    # Calculate blast pressure with reasonable bounds
    p_blast_raw = E_blast * (solver.eos.gamma - 1) / blast_volume
    
    # Limit the pressure ratio to prevent stiffness
    max_pressure_ratio = 100.0  # Reasonable ratio instead of 1e6+
    p_blast = min(p_blast_raw, p_ambient * max_pressure_ratio)
    
    println("  Raw blast pressure: $p_blast_raw")
    println("  Limited blast pressure: $p_blast")
    println("  Pressure ratio: $(p_blast / p_ambient)")
    
    # Smooth the initial condition to avoid sharp discontinuities
    for j in 1:solver.ny
        for i in 1:solver.nx
            x = solver.x[i]
            y = solver.y[j]
            r = sqrt((x - x_center)^2 + (y - y_center)^2)
            
            if r <= r_blast
                # Smooth transition using tanh profile
                transition_width = r_blast * 0.2  # 20% of blast radius
                if r < r_blast - transition_width
                    # Core: full blast pressure
                    p = p_blast
                else
                    # Transition region: smooth blend
                    xi = (r - (r_blast - transition_width)) / transition_width
                    weight = 0.5 * (1 + tanh(2 * (1 - xi)))  # Smooth step
                    p = p_ambient + weight * (p_blast - p_ambient)
                end
            else
                # Outside blast: ambient conditions
                p = p_ambient
            end
            
            W = [rho_ambient, u_ambient, v_ambient, p]
            solver.U[:, i, j, current] = primitive_to_conservative_2d(W, solver.eos)
        end
    end
    
    # Verify initialization
    all_finite = true
    for j in 1:solver.ny
        for i in 1:solver.nx
            if !all(isfinite.(solver.U[:, i, j, current]))
                all_finite = false
                println("Non-finite at ($i, $j): $(solver.U[:, i, j, current])")
            end
        end
    end
    
    println("All values finite after initialization: $all_finite")
    
    if all_finite
        # Sample some values
        center_i, center_j = solver.nx÷2, solver.ny÷2
        edge_i, edge_j = solver.nx÷4, solver.ny÷2
        
        U_center = solver.U[:, center_i, center_j, current]
        U_edge = solver.U[:, edge_i, edge_j, current]
        
        W_center = conservative_to_primitive_2d(U_center, solver.eos)
        W_edge = conservative_to_primitive_2d(U_edge, solver.eos)
        
        println("Sample values:")
        println("  Center: ρ=$(W_center[1]), p=$(W_center[4])")
        println("  Edge: ρ=$(W_edge[1]), p=$(W_edge[4])")
        
        println("Sedov blast initialized successfully")
    else
        error("Initialization failed - non-finite values detected")
    end
end

# Visualization with robust error handling
function plot_solution(solver::CartesianSolver, t::Float64)
    current = solver.current
    
    # Extract primitive variables
    rho = zeros(solver.nx, solver.ny)
    pressure_vals = zeros(solver.nx, solver.ny)
    velocity_mag = zeros(solver.nx, solver.ny)
    
    for j in 1:solver.ny
        for i in 1:solver.nx
            U = solver.U[:, i, j, current]
            W = conservative_to_primitive_2d(U, solver.eos)
            rho[i, j] = W[1]
            u, v = W[2], W[3]
            pressure_vals[i, j] = W[4]
            velocity_mag[i, j] = sqrt(u^2 + v^2)
        end
    end
    
    # Check for valid data
    rho_finite = all(isfinite.(rho))
    p_finite = all(isfinite.(pressure_vals))
    v_finite = all(isfinite.(velocity_mag))
    
    println("Solution diagnostics:")
    println("  Density range: $(extrema(rho))")
    println("  Pressure range: $(extrema(pressure_vals))")
    println("  Velocity range: $(extrema(velocity_mag))")
    println("  All finite: ρ=$rho_finite, p=$p_finite, v=$v_finite")
    
    if !rho_finite || !p_finite || !v_finite
        println("ERROR: Non-finite values detected in solution!")
        # Create error plot
        return plot(title="Error: Non-finite values in solution", 
                   xlabel="Simulation failed", ylabel="Check console output")
    end
    
    # Clean data for plotting
    rho_clean = clamp.(rho, 1e-8, 1e8)
    p_clean = clamp.(pressure_vals, 1e-12, 1e8)
    v_clean = clamp.(velocity_mag, 0.0, 1e6)
    
    try
        # Create contour plots
        x_plot = solver.x
        y_plot = solver.y
        
        p1 = contourf(x_plot, y_plot, rho_clean', 
                     title="Density at t = $(round(t, digits=4))", 
                     xlabel="x", ylabel="y", aspect_ratio=:equal)
        
        p2 = contourf(x_plot, y_plot, log10.(p_clean' .+ 1e-12), 
                     title="log₁₀(Pressure)", 
                     xlabel="x", ylabel="y", aspect_ratio=:equal)
        
        p3 = contourf(x_plot, y_plot, v_clean', 
                     title="Velocity Magnitude", 
                     xlabel="x", ylabel="y", aspect_ratio=:equal)
        
        # Radial profile along x-axis
        j_center = solver.ny ÷ 2
        x_profile = solver.x
        rho_profile = rho_clean[:, j_center]
        
        p4 = plot(x_profile, rho_profile, linewidth=2, 
                 xlabel="x", ylabel="Density", 
                 title="Density Profile (y=center)", legend=false)
        
        return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
        
    catch e
        println("Plotting error: $e")
        # Fallback to simple line plots
        try
            x_flat = repeat(solver.x, solver.ny)
            y_flat = repeat(solver.y', solver.nx)[:]
            rho_flat = rho_clean[:]
            
            p1 = scatter(x_flat, rho_flat, markersize=1, alpha=0.7,
                        title="Density (fallback)", xlabel="x", ylabel="ρ")
            
            p2 = plot(solver.x, rho_clean[:, solver.ny÷2], 
                     title="Density Profile", xlabel="x", ylabel="ρ")
            
            return plot(p1, p2, layout=(1,2), size=(800, 300))
        catch e2
            println("Fallback plotting also failed: $e2")
            return plot(title="Plotting failed: $e2")
        end
    end
end

# Main simulation function
function solve_cartesian_blast(nx::Int=50, ny::Int=50, t_final::Float64=0.2, 
                              E_blast::Float64=1.0, r_blast::Float64=0.1)
    
    println("="^60)
    println("2D Cartesian Sedov Blast Wave Simulation")
    println("="^60)
    
    # Create solver
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    eos = IdealGas(1.4)
    cfl = 0.4
    
    solver = CartesianSolver(nx, ny, x_min, x_max, y_min, y_max, eos, cfl)
    
    println("Grid: $nx × $ny")
    println("Domain: [$x_min, $x_max] × [$y_min, $y_max]")
    println("dx = $(solver.dx), dy = $(solver.dy)")
    
    # Initialize blast
    initialize_sedov_blast!(solver, E_blast, r_blast)
    
    # Time integration
    t = 0.0
    step = 0
    
    println("\nStarting simulation...")
    
    while t < t_final
        compute_time_step!(solver)
        
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        update_solution!(solver)
        
        t += solver.dt
        step += 1
        
        # Check solution health every 20 steps
        if step % 20 == 0
            current = solver.current
            
            # Sample some values to check for problems
            rho_vals = solver.U[1, :, :, current]
            p_vals = zeros(solver.nx, solver.ny)
            
            for j in 1:solver.ny
                for i in 1:solver.nx
                    U = solver.U[:, i, j, current]
                    W = conservative_to_primitive_2d(U, solver.eos)
                    p_vals[i, j] = W[4]
                end
            end
            
            rho_min, rho_max = extrema(rho_vals)
            p_min, p_max = extrema(p_vals)
            
            all_finite = all(isfinite.(rho_vals)) && all(isfinite.(p_vals))
            
            println("Step: $step, Time: $(round(t, digits=4)), dt: $(round(solver.dt, digits=6))")
            println("  ρ: [$(round(rho_min, digits=4)), $(round(rho_max, digits=4))]")
            println("  p: [$(round(p_min, digits=8)), $(round(p_max, digits=4))]")
            println("  All finite: $all_finite")
            
            if !all_finite
                println("ERROR: Non-finite values detected at step $step!")
                break
            end
        end
    end
    
    println("Simulation completed after $step steps")
    return solver, t
end

# Run the simulation with much more reasonable parameters
solver, final_time = solve_cartesian_blast(30, 30, 0.05, 0.1, 0.2)  # Smaller energy, shorter time

# Plot results
plt = plot_solution(solver, final_time)
display(plt)

println("\n2D Cartesian MUSCL+HLLE Simulation Complete!")
println("✓ Structured Cartesian grid")
println("✓ Much simpler and cleaner code")
println("✓ MUSCL reconstruction with minmod limiting")
println("✓ HLLE Riemann solver")
println("✓ Modular EOS interface")
println("✓ Sedov blast wave test case")

global solver
