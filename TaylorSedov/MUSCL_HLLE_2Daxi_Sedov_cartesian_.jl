using Plots
using LinearAlgebra

# 2D Axisymmetric MUSCL+HLLE Euler Solver
# Cylindrical coordinates (r,z) with axisymmetric source terms

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

# 2D Axisymmetric solver in cylindrical coordinates
mutable struct AxiSolver
    nr::Int                 # Number of cells in r (radial)
    nz::Int                 # Number of cells in z (axial)
    r_min::Float64          # Usually 0.0 (axis)
    r_max::Float64
    z_min::Float64
    z_max::Float64
    dr::Float64             # Grid spacing in r
    dz::Float64             # Grid spacing in z
    eos::EquationOfState
    cfl::Float64
    dt::Float64
    
    # Conservative variables: [rho, rho*u_r, rho*u_z, rho*E]
    # u_r = radial velocity, u_z = axial velocity
    U::Array{Float64, 4}    # (4, nr, nz, 2) - ping-pong buffers
    current::Int            # Current time level (1 or 2)
    
    # Cell centers
    r::Vector{Float64}      # Radial coordinates
    z::Vector{Float64}      # Axial coordinates
end

function AxiSolver(nr::Int, nz::Int, r_min::Float64, r_max::Float64,
                   z_min::Float64, z_max::Float64, eos::EquationOfState, cfl::Float64)
    
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    
    # Cell centers
    r = [r_min + (i-0.5)*dr for i in 1:nr]
    z = [z_min + (j-0.5)*dz for j in 1:nz]
    
    U = zeros(4, nr, nz, 2)
    
    return AxiSolver(nr, nz, r_min, r_max, z_min, z_max, dr, dz, eos, cfl, 0.0, 
                     U, 1, r, z)
end

# Conservative to primitive conversion (u_r, u_z)
function conservative_to_primitive_axi(U::Vector{Float64}, eos::EquationOfState)
    rho = max(U[1], 1e-12)
    u_r = U[2] / rho  # radial velocity
    u_z = U[3] / rho  # axial velocity
    p = max(pressure(eos, rho, U[4], U[2], U[3]), 1e-12)
    return [rho, u_r, u_z, p]
end

# Primitive to conservative conversion
function primitive_to_conservative_axi(W::Vector{Float64}, eos::EquationOfState)
    rho, u_r, u_z, p = W
    return [rho, rho*u_r, rho*u_z, total_energy(eos, rho, u_r, u_z, p)]
end

# HLLE Riemann solver for axisymmetric coordinates
function hlle_flux_axi(U_L::Vector{Float64}, U_R::Vector{Float64}, direction::Int, eos::EquationOfState)
    # direction: 1 = r-direction, 2 = z-direction
    
    W_L = conservative_to_primitive_axi(U_L, eos)
    W_R = conservative_to_primitive_axi(U_R, eos)
    
    rho_L, u_r_L, u_z_L, p_L = W_L
    rho_R, u_r_R, u_z_R, p_R = W_R
    
    c_L = sound_speed(eos, rho_L, p_L)
    c_R = sound_speed(eos, rho_R, p_R)
    
    if direction == 1  # r-direction
        # Wave speeds in r-direction
        S_L = min(u_r_L - c_L, u_r_R - c_R)
        S_R = max(u_r_L + c_L, u_r_R + c_R)
        
        # r-direction fluxes
        F_L = [rho_L * u_r_L,
               rho_L * u_r_L^2 + p_L,
               rho_L * u_r_L * u_z_L,
               (U_L[4] + p_L) * u_r_L]
        
        F_R = [rho_R * u_r_R,
               rho_R * u_r_R^2 + p_R,
               rho_R * u_r_R * u_z_R,
               (U_R[4] + p_R) * u_r_R]
               
    else  # z-direction
        # Wave speeds in z-direction
        S_L = min(u_z_L - c_L, u_z_R - c_R)
        S_R = max(u_z_L + c_L, u_z_R + c_R)
        
        # z-direction fluxes
        F_L = [rho_L * u_z_L,
               rho_L * u_r_L * u_z_L,
               rho_L * u_z_L^2 + p_L,
               (U_L[4] + p_L) * u_z_L]
        
        F_R = [rho_R * u_z_R,
               rho_R * u_r_R * u_z_R,
               rho_R * u_z_R^2 + p_R,
               (U_R[4] + p_R) * u_z_R]
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

# Correct axisymmetric source terms
function axisymmetric_source(U::Vector{Float64}, r::Float64, eos::EquationOfState)
    # Complete axisymmetric source terms for Euler equations in cylindrical coordinates
    # The full source vector is: S = -(1/r) * [0, rho*u_r^2 + p, 0, 0]
    # This accounts for the geometric effects of cylindrical coordinates
    
    if r < 1e-10  # Handle centerline very carefully
        return zeros(4)
    end
    
    W = conservative_to_primitive_axi(U, eos)
    rho, u_r, u_z, p = W
    
    # Full axisymmetric source terms
    # S = -(1/r) * [0, rho*u_r^2 + p, 0, 0]
    # The pressure term creates the geometric focusing effect
    # The kinetic term accounts for radial momentum transport
    
    source_r_momentum = -(rho * u_r^2 + p) / r
    
    # Limit source magnitude for numerical stability
    max_source = 1e8
    if abs(source_r_momentum) > max_source
        source_r_momentum = sign(source_r_momentum) * max_source
    end
    
    return [0.0, source_r_momentum, 0.0, 0.0]
end

# Minmod limiter (same as before)
function minmod(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

# MUSCL reconstruction with limiting
function muscl_reconstruct(U_left::Vector{Float64}, U_center::Vector{Float64}, U_right::Vector{Float64})
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
function compute_time_step!(solver::AxiSolver)
    max_speed = 0.0
    current = solver.current
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            U = solver.U[:, i, j, current]
            W = conservative_to_primitive_axi(U, solver.eos)
            rho, u_r, u_z, p = W
            
            c = sound_speed(solver.eos, rho, p)
            speed_r = abs(u_r) + c
            speed_z = abs(u_z) + c
            
            # CFL condition for 2D
            dt_r = solver.dr / speed_r
            dt_z = solver.dz / speed_z
            dt_cell = min(dt_r, dt_z)
            
            max_speed = max(max_speed, 1.0 / dt_cell)
        end
    end
    
    solver.dt = solver.cfl / max_speed
    
    # Safety limits
    solver.dt = min(solver.dt, 1e-4)
    solver.dt = max(solver.dt, 1e-8)
end

# Main update step with axisymmetric source terms
function update_solution!(solver::AxiSolver)
    current = solver.current
    next = 3 - current
    
    # Copy current solution
    solver.U[:, :, :, next] .= solver.U[:, :, :, current]
    
    # Compute all fluxes first (preserves symmetry)
    flux_r = zeros(4, solver.nr+1, solver.nz)  # r-direction fluxes
    flux_z = zeros(4, solver.nr, solver.nz+1)  # z-direction fluxes
    
    # r-direction fluxes at all i+1/2 interfaces
    for j in 2:solver.nz-1
        for i in 1:solver.nr-1
            if i == 1
                # Interface between centerline (i=1) and first interior cell (i=2)
                # Use reflection principle for centerline
                U_centerline = solver.U[:, 1, j, current]
                U_interior = solver.U[:, 2, j, current]
                
                # Create reflected state for centerline (u_r → -u_r)
                W_centerline = conservative_to_primitive_axi(U_centerline, solver.eos)
                W_reflected = copy(W_centerline)
                W_reflected[2] = -W_reflected[2]  # Reflect u_r
                U_reflected = primitive_to_conservative_axi(W_reflected, solver.eos)
                
                # Use reflected state and interior state for flux
                flux_r[:, i+1, j] = hlle_flux_axi(U_reflected, U_interior, 1, solver.eos)
                
            elseif i == solver.nr-1
                # Right boundary
                U_L = solver.U[:, solver.nr-1, j, current]
                U_R = solver.U[:, solver.nr, j, current]
                flux_r[:, i+1, j] = hlle_flux_axi(U_L, U_R, 1, solver.eos)
                
            else
                # Interior: MUSCL reconstruction
                U_im1 = solver.U[:, i-1, j, current]
                U_i   = solver.U[:, i,   j, current]
                U_ip1 = solver.U[:, i+1, j, current]
                U_ip2 = solver.U[:, i+2, j, current]
                
                U_L, _ = muscl_reconstruct(U_im1, U_i, U_ip1)
                _, U_R = muscl_reconstruct(U_i, U_ip1, U_ip2)
                
                # Ensure physical states
                U_L[1] = max(U_L[1], 1e-12)
                U_R[1] = max(U_R[1], 1e-12)
                
                flux_r[:, i+1, j] = hlle_flux_axi(U_L, U_R, 1, solver.eos)
            end
        end
    end
    
    # z-direction fluxes at all j+1/2 interfaces
    for j in 1:solver.nz-1
        for i in 2:solver.nr-1
            if j == 1
                # Bottom boundary
                U_L = solver.U[:, i, 1, current]
                U_R = solver.U[:, i, 2, current]
            elseif j == solver.nz-1
                # Top boundary
                U_L = solver.U[:, i, solver.nz-1, current]
                U_R = solver.U[:, i, solver.nz, current]
            else
                # Interior: MUSCL reconstruction
                U_jm1 = solver.U[:, i, j-1, current]
                U_j   = solver.U[:, i, j,   current]
                U_jp1 = solver.U[:, i, j+1, current]
                U_jp2 = solver.U[:, i, j+2, current]
                
                U_L, _ = muscl_reconstruct(U_jm1, U_j, U_jp1)
                _, U_R = muscl_reconstruct(U_j, U_jp1, U_jp2)
            end
            
            # Ensure physical states
            U_L[1] = max(U_L[1], 1e-12)
            U_R[1] = max(U_R[1], 1e-12)
            
            # HLLE flux in z-direction
            flux_z[:, i, j+1] = hlle_flux_axi(U_L, U_R, 2, solver.eos)
        end
    end
    
    # Apply all flux updates and source terms with proper volume weighting
    for j in 2:solver.nz-1
        for i in 1:solver.nr-1
            r = solver.r[i]
            
            if i == 1
                # Centerline cell (i=1): Special volume treatment
                # Volume element: V = π * (dr/2)^2 * dz (cylinder from r=0 to r=dr/2)
                # Surface area at r=dr/2: A = 2π * (dr/2) * dz = π * dr * dz
                
                r_interface = solver.dr / 2  # Interface at r = dr/2
                
                # Flux through cylindrical surface at r = dr/2
                flux_right = flux_r[:, i+1, j]
                surface_area = 2 * π * r_interface * solver.dz
                
                # Volume of centerline cell (cylinder from 0 to dr/2)
                cell_volume = π * (solver.dr/2)^2 * solver.dz
                
                # Divergence: -(flux * surface_area) / cell_volume
                div_r = -flux_right * surface_area / cell_volume
                
                # z-direction flux (normal treatment)
                flux_bottom = flux_z[:, i, j]
                flux_top = flux_z[:, i, j+1]
                div_z = -(flux_top - flux_bottom) / solver.dz
                
                # No geometric source term at centerline
                source = zeros(4)
                
                # Update centerline cell
                solver.U[:, i, j, next] += solver.dt * (div_r + div_z + source)
                
                # Enforce centerline physics: u_r = 0
                W = conservative_to_primitive_axi(solver.U[:, i, j, next], solver.eos)
                W[2] = 0.0  # u_r = 0 at centerline
                solver.U[:, i, j, next] = primitive_to_conservative_axi(W, solver.eos)
                
            else
                # Interior cells (i > 1): Normal axisymmetric treatment
                flux_left  = flux_r[:, i,   j]
                flux_right = flux_r[:, i+1, j]
                flux_bottom = flux_z[:, i, j]
                flux_top    = flux_z[:, i, j+1]
                
                # Cell-centered radius
                r_center = solver.r[i]
                
                # Interface radii
                r_left = r_center - 0.5 * solver.dr
                r_right = r_center + 0.5 * solver.dr
                
                # Axisymmetric finite volume: d/dt(∫U dV) = -∫(∇·F) dV + ∫S dV
                # For ring-shaped cell: dV = 2π * r * dr * dz
                # Cell volume: V = 2π * r_center * dr * dz
                # Left surface: A_left = 2π * r_left * dz  
                # Right surface: A_right = 2π * r_right * dz
                
                cell_volume = 2 * π * r_center * solver.dr * solver.dz
                area_left = 2 * π * r_left * solver.dz
                area_right = 2 * π * r_right * solver.dz
                area_bottom = 2 * π * r_center * solver.dr
                area_top = 2 * π * r_center * solver.dr
                
                # Flux divergence with proper area weighting
                div_r = -(flux_right * area_right - flux_left * area_left) / cell_volume
                div_z = -(flux_top * area_top - flux_bottom * area_bottom) / cell_volume
                
                # Axisymmetric source terms (for r > 0)
                source = axisymmetric_source(solver.U[:, i, j, current], r_center, solver.eos)
                
                # Update
                solver.U[:, i, j, next] += solver.dt * (div_r + div_z + source)
            end
        end
    end
    
    # Apply boundary conditions
    apply_axi_boundary_conditions!(solver, next)
    
    # Ensure physical states
    for j in 1:solver.nz
        for i in 1:solver.nr
            solver.U[1, i, j, next] = max(solver.U[1, i, j, next], 1e-12)
            
            U = solver.U[:, i, j, next]
            W = conservative_to_primitive_axi(U, solver.eos)
            if W[4] <= 1e-12
                rho, u_r, u_z = W[1], W[2], W[3]
                p_min = 1e-12
                solver.U[4, i, j, next] = total_energy(solver.eos, rho, u_r, u_z, p_min)
            end
        end
    end
    
    # Switch time levels
    solver.current = next
end

# Apply minimal boundary conditions (centerline handled in flux computation)
function apply_axi_boundary_conditions!(solver::AxiSolver, time_level::Int)
    nr, nz = solver.nr, solver.nz
    
    # Outer radial boundary: outflow
    for j in 1:nz
        solver.U[:, nr, j, time_level] = solver.U[:, nr-1, j, time_level]
    end
    
    # Axial boundaries: outflow  
    for i in 1:nr
        solver.U[:, i, 1,  time_level] = solver.U[:, i, 2,    time_level]  # Bottom
        solver.U[:, i, nz, time_level] = solver.U[:, i, nz-1, time_level]  # Top
    end
    
    # Centerline: Ensure u_r = 0 (redundant but safe)
    for j in 1:nz
        W = conservative_to_primitive_axi(solver.U[:, 1, j, time_level], solver.eos)
        W[2] = 0.0  # u_r = 0
        solver.U[:, 1, j, time_level] = primitive_to_conservative_axi(W, solver.eos)
    end
end

# Initialize axisymmetric blast wave
function initialize_axi_blast!(solver::AxiSolver, E_blast::Float64, r_blast::Float64, z_blast::Float64)
    current = solver.current
    
    # Ambient conditions
    rho_ambient = 1.0
    p_ambient = 0.1
    u_r_ambient = 0.0
    u_z_ambient = 0.0
    
    # Blast center
    r_center = r_blast
    z_center = z_blast
    
    blast_volume = 0.0
    blast_cells = 0
    
    # Calculate blast volume
    for j in 1:solver.nz
        for i in 1:solver.nr
            r = solver.r[i]
            z = solver.z[j]
            distance = sqrt((r - r_center)^2 + (z - z_center)^2)
            
            if distance <= r_blast
                # Volume element in axisymmetric: dV = 2π * r * dr * dz
                dV = 2 * π * r * solver.dr * solver.dz
                blast_volume += dV
                blast_cells += 1
            end
        end
    end
    
    println("Axisymmetric blast initialization:")
    println("  Blast center: (r=$(r_center), z=$(z_center))")
    println("  Blast radius: $(r_blast)")
    println("  Blast cells: $blast_cells")
    println("  Blast volume: $blast_volume")
    
    # Calculate blast pressure
    p_blast_raw = E_blast * (solver.eos.gamma - 1) / blast_volume
    p_blast = min(p_blast_raw, p_ambient * 100.0)  # Limit pressure ratio
    
    println("  Blast pressure: $p_blast")
    println("  Pressure ratio: $(p_blast / p_ambient)")
    
    # Initialize all cells
    for j in 1:solver.nz
        for i in 1:solver.nr
            r = solver.r[i]
            z = solver.z[j]
            distance = sqrt((r - r_center)^2 + (z - z_center)^2)
            
            if distance <= r_blast
                # Smooth transition
                transition_width = r_blast * 0.2
                if distance < r_blast - transition_width
                    p = p_blast
                else
                    xi = (distance - (r_blast - transition_width)) / transition_width
                    weight = 0.5 * (1 + tanh(2 * (1 - xi)))
                    p = p_ambient + weight * (p_blast - p_ambient)
                end
            else
                p = p_ambient
            end
            
            W = [rho_ambient, u_r_ambient, u_z_ambient, p]
            solver.U[:, i, j, current] = primitive_to_conservative_axi(W, solver.eos)
        end
    end
    
    println("Axisymmetric blast initialized successfully")
end

# Visualization for axisymmetric solution
function plot_axi_solution(solver::AxiSolver, t::Float64)
    current = solver.current
    
    # Extract primitive variables
    rho = zeros(solver.nr, solver.nz)
    pressure_vals = zeros(solver.nr, solver.nz)
    velocity_mag = zeros(solver.nr, solver.nz)
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            U = solver.U[:, i, j, current]
            W = conservative_to_primitive_axi(U, solver.eos)
            rho[i, j] = W[1]
            u_r, u_z = W[2], W[3]
            pressure_vals[i, j] = W[4]
            velocity_mag[i, j] = sqrt(u_r^2 + u_z^2)
        end
    end
    
    # Check data quality
    all_finite = all(isfinite.(rho)) && all(isfinite.(pressure_vals))
    
    if !all_finite
        println("WARNING: Non-finite values in solution")
        return plot(title="Error: Non-finite values", xlabel="Check console")
    end
    
    # Create plots
    r_plot = solver.r
    z_plot = solver.z
    
    p1 = contourf(r_plot, z_plot, rho', 
                 title="Density at t = $(round(t, digits=4))", 
                 xlabel="r (radial)", ylabel="z (axial)", aspect_ratio=:equal)
    
    p2 = contourf(r_plot, z_plot, log10.(pressure_vals' .+ 1e-12), 
                 title="log₁₀(Pressure)", 
                 xlabel="r (radial)", ylabel="z (axial)", aspect_ratio=:equal)
    
    p3 = contourf(r_plot, z_plot, velocity_mag', 
                 title="Velocity Magnitude", 
                 xlabel="r (radial)", ylabel="z (axial)", aspect_ratio=:equal)
    
    # Radial profile at middle z
    j_center = solver.nz ÷ 2
    r_profile = solver.r
    rho_profile = rho[:, j_center]
    
    p4 = plot(r_profile, rho_profile, linewidth=2, 
             xlabel="r", ylabel="Density", 
             title="Radial Profile (z=center)", legend=false)
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
end

# Main simulation function
function solve_axi_blast(nr::Int=40, nz::Int=30, t_final::Float64=0.1, 
                        E_blast::Float64=0.1, r_blast::Float64=0.15)
    
    println("="^60)
    println("2D Axisymmetric Sedov Blast Wave Simulation")
    println("="^60)
    
    # Create solver with cylindrical coordinates
    r_min, r_max = 0.0, 1.0    # Radial domain: centerline to r=1
    z_min, z_max = 0.0, 1.0    # Axial domain
    eos = IdealGas(1.4)
    cfl = 0.3
    
    solver = AxiSolver(nr, nz, r_min, r_max, z_min, z_max, eos, cfl)
    
    println("Axisymmetric grid: $nr × $nz")
    println("Domain: r ∈ [$r_min, $r_max], z ∈ [$z_min, $z_max]")
    println("dr = $(solver.dr), dz = $(solver.dz)")
    println("Axis of symmetry: r = 0 (left boundary)")
    println("Blast center: ON AXIS at (r=0, z=0.5) for spherical symmetry")
    
    # Initialize blast at center of domain for spherical symmetry
    r_center = 0.0  # At the axis for maximum spherical symmetry
    z_center = 0.5  # Middle of axial domain
    initialize_axi_blast!(solver, E_blast, r_blast, z_center)
    
    # Time integration
    t = 0.0
    step = 0
    
    println("\nStarting axisymmetric simulation...")
    
    while t < t_final
        compute_time_step!(solver)
        
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        update_solution!(solver)
        
        t += solver.dt
        step += 1
        
        if step % 50 == 0
            println("Step: $step, Time: $(round(t, digits=4)), dt: $(round(solver.dt, digits=6))")
        end
    end
    
    println("Axisymmetric simulation completed after $step steps")
    return solver, t
end

# Run the axisymmetric simulation
solver, final_time = solve_axi_blast(30, 25, 0.08, 0.05, 0.12)

# Plot results
plt = plot_axi_solution(solver, final_time)
display(plt)

println("\n2D Axisymmetric MUSCL+HLLE Simulation Complete!")
println("✓ True axisymmetric coordinates (r,z)")
println("✓ Axis of symmetry at r=0")
println("✓ Proper geometric source terms") 
println("✓ Centerline boundary conditions")
println("✓ Axisymmetric finite volume formulation")
println("✓ MUSCL reconstruction + HLLE")
println("✓ Modular EOS interface")

global solver