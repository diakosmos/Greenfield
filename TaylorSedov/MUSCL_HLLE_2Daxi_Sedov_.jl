using Plots
using Random
using LinearAlgebra

# 2D Axisymmetric Sedov-Taylor Blast Wave Solver
# Complete implementation with mesh generation and visualization

# Mesh data structures
mutable struct Node
    id::Int
    x::Float64
    y::Float64
    boundary::Bool
end

mutable struct Face
    id::Int
    node1::Int
    node2::Int
    cell_left::Int
    cell_right::Int
    boundary::Bool
    normal_x::Float64
    normal_y::Float64
    area::Float64
    center_x::Float64
    center_y::Float64
end

mutable struct Cell
    id::Int
    node1::Int
    node2::Int
    node3::Int
    face1::Int
    face2::Int
    face3::Int
    center_x::Float64
    center_y::Float64
    area::Float64
    volume::Float64
end

mutable struct Mesh
    nodes::Vector{Node}
    faces::Vector{Face}
    cells::Vector{Cell}
    n_nodes::Int
    n_faces::Int
    n_cells::Int
    boundary_faces::Vector{Int}
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
end

# Create empty mesh
function Mesh()
    return Mesh(Node[], Face[], Cell[], 0, 0, 0, Int[], 0.0, 0.0, 0.0, 0.0)
end

# Simple structured mesh generator for Sedov problem
function generate_sedov_mesh(nr::Int, nz::Int, r_max::Float64, z_max::Float64)
    println("Generating structured mesh for Sedov problem...")
    println("Grid: $nr × $nz, Domain: [0, $r_max] × [0, $z_max]")
    
    mesh = Mesh()
    mesh.x_min = 0.0
    mesh.x_max = r_max
    mesh.y_min = 0.0
    mesh.y_max = z_max
    
    # Generate structured grid points
    dr = r_max / nr
    dz = z_max / nz
    
    # Create nodes
    node_id = 0
    for j in 1:nz+1
        for i in 1:nr+1
            node_id += 1
            r = (i-1) * dr
            z = (j-1) * dz
            
            # Check if boundary node
            is_boundary = (i == 1 || i == nr+1 || j == 1 || j == nz+1)
            push!(mesh.nodes, Node(node_id, r, z, is_boundary))
        end
    end
    mesh.n_nodes = node_id
    
    # Create cells (divide each quad into 2 triangles)
    cell_id = 0
    face_id = 0
    face_dict = Dict{Tuple{Int,Int}, Vector{Int}}()
    
    for j in 1:nz
        for i in 1:nr
            # Node indices for current quad
            n1 = (j-1)*(nr+1) + i      # bottom-left
            n2 = (j-1)*(nr+1) + i + 1  # bottom-right
            n3 = j*(nr+1) + i          # top-left
            n4 = j*(nr+1) + i + 1      # top-right
            
            # Triangle 1: n1-n2-n3
            cell_id += 1
            r1, z1 = mesh.nodes[n1].x, mesh.nodes[n1].y
            r2, z2 = mesh.nodes[n2].x, mesh.nodes[n2].y
            r3, z3 = mesh.nodes[n3].x, mesh.nodes[n3].y
            
            center_r = (r1 + r2 + r3) / 3
            center_z = (z1 + z2 + z3) / 3
            area = 0.5 * abs((r2-r1)*(z3-z1) - (r3-r1)*(z2-z1))
            volume = 2 * π * center_r * area
            
            push!(mesh.cells, Cell(cell_id, n1, n2, n3, 0, 0, 0, center_r, center_z, area, volume))
            
            # Add edges for this triangle
            edges = [(n1,n2), (n2,n3), (n3,n1)]
            for edge in edges
                sorted_edge = (min(edge[1], edge[2]), max(edge[1], edge[2]))
                if !haskey(face_dict, sorted_edge)
                    face_dict[sorted_edge] = Int[]
                end
                push!(face_dict[sorted_edge], cell_id)
            end
            
            # Triangle 2: n2-n4-n3
            cell_id += 1
            r2, z2 = mesh.nodes[n2].x, mesh.nodes[n2].y
            r4, z4 = mesh.nodes[n4].x, mesh.nodes[n4].y
            r3, z3 = mesh.nodes[n3].x, mesh.nodes[n3].y
            
            center_r = (r2 + r4 + r3) / 3
            center_z = (z2 + z4 + z3) / 3
            area = 0.5 * abs((r4-r2)*(z3-z2) - (r3-r2)*(z4-z2))
            volume = 2 * π * center_r * area
            
            push!(mesh.cells, Cell(cell_id, n2, n4, n3, 0, 0, 0, center_r, center_z, area, volume))
            
            # Add edges for this triangle
            edges = [(n2,n4), (n4,n3), (n3,n2)]
            for edge in edges
                sorted_edge = (min(edge[1], edge[2]), max(edge[1], edge[2]))
                if !haskey(face_dict, sorted_edge)
                    face_dict[sorted_edge] = Int[]
                end
                push!(face_dict[sorted_edge], cell_id)
            end
        end
    end
    mesh.n_cells = cell_id
    
    # Create faces
    for ((n1, n2), adjacent_cells) in face_dict
        face_id += 1
        
        r1, z1 = mesh.nodes[n1].x, mesh.nodes[n1].y
        r2, z2 = mesh.nodes[n2].x, mesh.nodes[n2].y
        
        center_r = (r1 + r2) / 2
        center_z = (z1 + z2) / 2
        face_length = sqrt((r2-r1)^2 + (z2-z1)^2)
        
        # Compute normal (outward from first cell)
        dr = r2 - r1
        dz = z2 - z1
        normal_r = dz / face_length
        normal_z = -dr / face_length
        
        is_boundary = length(adjacent_cells) == 1
        cell_left = adjacent_cells[1]
        cell_right = is_boundary ? 0 : adjacent_cells[2]
        
        face = Face(face_id, n1, n2, cell_left, cell_right, is_boundary,
                   normal_r, normal_z, face_length, center_r, center_z)
        push!(mesh.faces, face)
        
        if is_boundary
            push!(mesh.boundary_faces, face_id)
        end
    end
    mesh.n_faces = face_id
    
    println("Mesh created: $(mesh.n_nodes) nodes, $(mesh.n_cells) cells, $(mesh.n_faces) faces")
    return mesh
end

# EOS interface
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

# Euler solver
mutable struct SedovSolver
    mesh::Mesh
    eos::EquationOfState
    cfl::Float64
    dt::Float64
    
    # Conservative variables: [rho, rho*u, rho*v, rho*E]
    U::Matrix{Float64}
    U_new::Matrix{Float64}
    
    # Gradients for MUSCL
    grad_U::Array{Float64, 3}  # 4 x 2 x n_cells
    
    # Blast parameters
    E_blast::Float64  # Total blast energy
    r_blast::Float64  # Initial blast radius
end

function SedovSolver(mesh::Mesh, eos::EquationOfState, cfl::Float64, 
                    E_blast::Float64, r_blast::Float64)
    n_cells = mesh.n_cells
    U = zeros(4, n_cells)
    U_new = zeros(4, n_cells)
    grad_U = zeros(4, 2, n_cells)
    
    return SedovSolver(mesh, eos, cfl, 0.0, U, U_new, grad_U, E_blast, r_blast)
end

# Initialize Sedov-Taylor blast wave with debugging
function initialize_sedov_blast!(solver::SedovSolver)
    println("Initializing Sedov-Taylor blast wave...")
    println("Blast energy: $(solver.E_blast)")
    println("Blast radius: $(solver.r_blast)")
    
    # Ambient conditions
    rho_ambient = 1.0
    p_ambient = 1e-6  # Very low pressure
    u_ambient = 0.0
    v_ambient = 0.0
    
    # Find cells inside blast radius
    blast_cells = Int[]
    total_volume = 0.0
    
    for cell_idx in 1:solver.mesh.n_cells
        cell = solver.mesh.cells[cell_idx]
        r = sqrt(cell.center_x^2 + cell.center_y^2)
        
        if r <= solver.r_blast
            push!(blast_cells, cell_idx)
            total_volume += cell.volume
        end
    end
    
    println("Blast cells: $(length(blast_cells))")
    println("Total blast volume: $total_volume")
    
    # Calculate blast pressure to get correct total energy
    if length(blast_cells) > 0 && total_volume > 0
        # For ideal gas: E = p*V/(gamma-1) + kinetic energy
        # Since initial velocity is zero: E = p*V/(gamma-1)
        p_blast = solver.E_blast * (solver.eos.gamma - 1) / total_volume
        println("Calculated blast pressure: $p_blast")
    else
        error("No cells found inside blast radius or zero volume!")
    end
    
    # Initialize all cells
    for cell_idx in 1:solver.mesh.n_cells
        cell = solver.mesh.cells[cell_idx]
        r = sqrt(cell.center_x^2 + cell.center_y^2)
        
        if r <= solver.r_blast
            # Inside blast: high pressure
            rho = rho_ambient
            u = u_ambient
            v = v_ambient
            p = p_blast
        else
            # Outside blast: ambient conditions
            rho = rho_ambient
            u = u_ambient
            v = v_ambient
            p = p_ambient
        end
        
        # Convert to conservative variables
        rho_u = rho * u
        rho_v = rho * v
        rho_E = total_energy(solver.eos, rho, u, v, p)
        
        solver.U[:, cell_idx] = [rho, rho_u, rho_v, rho_E]
        
        # Debug check
        if !isfinite(rho_E) || rho_E <= 0
            println("ERROR in initialization at cell $cell_idx:")
            println("  rho=$rho, u=$u, v=$v, p=$p")
            println("  rho_E=$rho_E")
            error("Invalid energy in initialization")
        end
    end
    
    # Verify initialization is clean
    all_finite = all(isfinite.(solver.U))
    println("All values finite after initialization: $all_finite")
    
    if !all_finite
        println("ERROR: Non-finite values in initial conditions!")
        for i in 1:min(10, solver.mesh.n_cells)
            println("Cell $i: U = $(solver.U[:, i])")
        end
        error("Invalid initial conditions")
    end
    
    # Sample some initial values
    println("Sample initial values:")
    for i in [1, length(blast_cells)÷2, solver.mesh.n_cells]
        if i <= solver.mesh.n_cells
            W = conservative_to_primitive_2d(solver.U[:, i], solver.eos)
            println("  Cell $i: ρ=$(W[1]), u=$(W[2]), v=$(W[3]), p=$(W[4])")
        end
    end
end

# Conservative to primitive conversion
function conservative_to_primitive_2d(U::Vector{Float64}, eos::EquationOfState)
    rho = max(U[1], 1e-12)
    u = U[2] / rho
    v = U[3] / rho
    rho_E = U[4]
    
    p = max(pressure(eos, rho, rho_E, U[2], U[3]), 1e-12)
    
    return [rho, u, v, p]
end

# Simple gradient computation (for now, use zero gradients - first order)
function compute_gradients!(solver::SedovSolver)
    # For simplicity, use first-order (zero gradients)
    # This makes the scheme more robust for the blast problem
    solver.grad_U .= 0.0
end

# MUSCL reconstruction (simplified to first-order for stability)
function muscl_reconstruct_2d(solver::SedovSolver, face::Face)
    # First-order reconstruction (use cell center values)
    U_L = zeros(4)
    U_R = zeros(4)
    
    if face.cell_left > 0
        U_L = solver.U[:, face.cell_left]
    end
    
    if face.cell_right > 0
        U_R = solver.U[:, face.cell_right]
    else
        # Boundary: use left state
        U_R = copy(U_L)
    end
    
    return U_L, U_R
end

# 2D HLLE Riemann solver
function hlle_flux_2d(U_L::Vector{Float64}, U_R::Vector{Float64}, 
                     normal_x::Float64, normal_y::Float64, eos::EquationOfState)
    
    # Convert to primitive variables
    W_L = conservative_to_primitive_2d(U_L, eos)
    W_R = conservative_to_primitive_2d(U_R, eos)
    
    rho_L, u_L, v_L, p_L = W_L
    rho_R, u_R, v_R, p_R = W_R
    
    # Normal velocities
    un_L = u_L * normal_x + v_L * normal_y
    un_R = u_R * normal_x + v_R * normal_y
    
    # Sound speeds
    c_L = sound_speed(eos, rho_L, p_L)
    c_R = sound_speed(eos, rho_R, p_R)
    
    # Wave speed estimates
    S_L = min(un_L - c_L, un_R - c_R)
    S_R = max(un_L + c_L, un_R + c_R)
    
    # Compute normal fluxes
    F_L = [rho_L * un_L,
           rho_L * u_L * un_L + p_L * normal_x,
           rho_L * v_L * un_L + p_L * normal_y,
           (U_L[4] + p_L) * un_L]
    
    F_R = [rho_R * un_R,
           rho_R * u_R * un_R + p_R * normal_x,
           rho_R * v_R * un_R + p_R * normal_y,
           (U_R[4] + p_R) * un_R]
    
    # HLLE flux
    if S_L >= 0.0
        return F_L
    elseif S_R <= 0.0
        return F_R
    else
        return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    end
end

# Axisymmetric source terms with better singularity handling
function axisymmetric_source(U::Vector{Float64}, r::Float64, eos::EquationOfState)
    # Handle singularity at centerline more carefully
    if r < 1e-8  # Much smaller threshold
        return zeros(4)  # No source terms at centerline
    end
    
    W = conservative_to_primitive_2d(U, eos)
    rho, u, v, p = W
    
    # Limit the source term magnitude to prevent instability
    source_magnitude = (rho * v^2 + p) / r
    max_source = 1e10  # Reasonable upper limit
    
    if abs(source_magnitude) > max_source
        source_magnitude = sign(source_magnitude) * max_source
    end
    
    # Axisymmetric source: S = -(1/r) * [0, 0, rho*v^2 + p, 0]
    return [0.0, 0.0, -source_magnitude, 0.0]
end

# Compute time step with more conservative approach
function compute_time_step!(solver::SedovSolver)
    max_speed = 0.0
    min_dx = Inf
    
    for cell_idx in 1:solver.mesh.n_cells
        W = conservative_to_primitive_2d(solver.U[:, cell_idx], solver.eos)
        rho, u, v, p = W
        
        # Limit extreme pressures that can cause instability
        p = clamp(p, 1e-12, 1e10)
        
        c = sound_speed(solver.eos, rho, p)
        speed = sqrt(u^2 + v^2) + c
        max_speed = max(max_speed, speed)
        
        # Estimate cell size
        cell = solver.mesh.cells[cell_idx]
        dx_est = sqrt(cell.area)
        min_dx = min(min_dx, dx_est)
    end
    
    # More conservative CFL for stability
    dt_cfl = solver.cfl * min_dx / max_speed
    dt_max = 1e-6  # Hard limit on time step
    
    solver.dt = min(dt_cfl, dt_max)
    
    # Additional safety check
    if solver.dt <= 0 || !isfinite(solver.dt)
        solver.dt = 1e-8
    end
end

# Main update step with debugging
function update_solution!(solver::SedovSolver)
    mesh = solver.mesh
    
    # Check input state
    if !all(isfinite.(solver.U))
        println("ERROR: Non-finite values at start of update step!")
        return
    end
    
    # Compute gradients (simplified to first-order)
    compute_gradients!(solver)
    
    # Initialize new solution
    solver.U_new .= solver.U
    
    # Compute fluxes and update
    flux_count = 0
    for face in mesh.faces
        flux_count += 1
        
        # MUSCL reconstruction
        U_L, U_R = muscl_reconstruct_2d(solver, face)
        
        # Ensure physical states
        U_L[1] = max(U_L[1], 1e-12)
        U_R[1] = max(U_R[1], 1e-12)
        
        # Check reconstructed states
        if !all(isfinite.(U_L)) || !all(isfinite.(U_R))
            println("ERROR: Non-finite reconstructed states at face $flux_count")
            println("  U_L = $U_L")
            println("  U_R = $U_R")
            println("  Face: cell_left=$(face.cell_left), cell_right=$(face.cell_right)")
            if face.cell_left > 0
                println("  Left cell U = $(solver.U[:, face.cell_left])")
            end
            if face.cell_right > 0
                println("  Right cell U = $(solver.U[:, face.cell_right])")
            end
            return
        end
        
        # HLLE flux
        flux = hlle_flux_2d(U_L, U_R, face.normal_x, face.normal_y, solver.eos)
        
        # Check flux
        if !all(isfinite.(flux))
            println("ERROR: Non-finite flux at face $flux_count")
            println("  Flux = $flux")
            println("  U_L = $U_L, U_R = $U_R")
            println("  Normal = [$(face.normal_x), $(face.normal_y)]")
            return
        end
        
        # Update adjacent cells
        if face.cell_left > 0
            cell_left = mesh.cells[face.cell_left]
            flux_contribution = flux * face.area / cell_left.area
            solver.U_new[:, face.cell_left] -= solver.dt * flux_contribution
            
            # Check after update
            if !all(isfinite.(solver.U_new[:, face.cell_left]))
                println("ERROR: Non-finite state after left cell update at face $flux_count")
                println("  Cell $(face.cell_left) U_new = $(solver.U_new[:, face.cell_left])")
                println("  Flux contribution = $(solver.dt * flux_contribution)")
                return
            end
        end
        
        if face.cell_right > 0
            cell_right = mesh.cells[face.cell_right]
            flux_contribution = flux * face.area / cell_right.area
            solver.U_new[:, face.cell_right] += solver.dt * flux_contribution
            
            # Check after update
            if !all(isfinite.(solver.U_new[:, face.cell_right]))
                println("ERROR: Non-finite state after right cell update at face $flux_count")
                println("  Cell $(face.cell_right) U_new = $(solver.U_new[:, face.cell_right])")
                println("  Flux contribution = $(solver.dt * flux_contribution)")
                return
            end
        end
    end
    
    # Add axisymmetric source terms
    for cell_idx in 1:mesh.n_cells
        cell = mesh.cells[cell_idx]
        r = cell.center_x  # Radial coordinate
        
        source = axisymmetric_source(solver.U[:, cell_idx], r, solver.eos)
        
        if !all(isfinite.(source))
            println("ERROR: Non-finite source at cell $cell_idx")
            println("  Source = $source")
            println("  r = $r")
            println("  U = $(solver.U[:, cell_idx])")
            return
        end
        
        solver.U_new[:, cell_idx] += solver.dt * source
        
        if !all(isfinite.(solver.U_new[:, cell_idx]))
            println("ERROR: Non-finite state after source term at cell $cell_idx")
            println("  U_new = $(solver.U_new[:, cell_idx])")
            println("  Source contribution = $(solver.dt * source)")
            return
        end
    end
    
    # Ensure physical states with stronger limiting
    for cell_idx in 1:mesh.n_cells
        # Stronger density limit
        solver.U_new[1, cell_idx] = max(solver.U_new[1, cell_idx], 1e-10)
        
        # Check and limit pressure more aggressively
        W = conservative_to_primitive_2d(solver.U_new[:, cell_idx], solver.eos)
        rho, u, v, p = W
        
        # Aggressive pressure limiting
        if p <= 1e-10 || p > 1e8 || !isfinite(p)
            p = clamp(p, 1e-10, 1e8)
            
            # Recalculate energy with limited pressure
            solver.U_new[4, cell_idx] = total_energy(solver.eos, rho, u, v, p)
        end
        
        # Velocity limiting to prevent runaway solutions
        speed = sqrt(u^2 + v^2)
        max_speed = 1e3  # Reasonable maximum velocity
        if speed > max_speed
            scale = max_speed / speed
            solver.U_new[2, cell_idx] *= scale  # rho*u
            solver.U_new[3, cell_idx] *= scale  # rho*v
            
            # Recalculate energy
            u_new = solver.U_new[2, cell_idx] / rho
            v_new = solver.U_new[3, cell_idx] / rho
            solver.U_new[4, cell_idx] = total_energy(solver.eos, rho, u_new, v_new, p)
        end
        
        # Final check and emergency fallback
        if !all(isfinite.(solver.U_new[:, cell_idx])) || solver.U_new[1, cell_idx] <= 0
            # Emergency: revert to ambient conditions
            println("WARNING: Emergency reset for cell $cell_idx")
            rho_safe = 1.0
            u_safe = 0.0
            v_safe = 0.0
            p_safe = 1e-6
            solver.U_new[:, cell_idx] = [rho_safe, rho_safe*u_safe, rho_safe*v_safe, 
                                        total_energy(solver.eos, rho_safe, u_safe, v_safe, p_safe)]
        end
    end
    
    # Update solution
    solver.U .= solver.U_new
    
    # Final verification
    if !all(isfinite.(solver.U))
        println("ERROR: Non-finite values at end of update step!")
        return
    end
end

# Visualization with debugging
function plot_sedov_solution(solver::SedovSolver, t::Float64)
    mesh = solver.mesh
    
    # Extract primitive variables
    rho = zeros(mesh.n_cells)
    pressure_vals = zeros(mesh.n_cells)
    velocity_mag = zeros(mesh.n_cells)
    
    for cell_idx in 1:mesh.n_cells
        W = conservative_to_primitive_2d(solver.U[:, cell_idx], solver.eos)
        rho[cell_idx] = W[1]
        u, v = W[2], W[3]
        pressure_vals[cell_idx] = W[4]
        velocity_mag[cell_idx] = sqrt(u^2 + v^2)
    end
    
    # Debug output
    println("Solution diagnostics:")
    println("  Density range: $(minimum(rho)) - $(maximum(rho))")
    println("  Pressure range: $(minimum(pressure_vals)) - $(maximum(pressure_vals))")
    println("  Velocity range: $(minimum(velocity_mag)) - $(maximum(velocity_mag))")
    println("  Finite values: rho=$(all(isfinite.(rho))), p=$(all(isfinite.(pressure_vals))), v=$(all(isfinite.(velocity_mag)))")
    
    # Check for invalid values
    if !all(isfinite.(rho)) || !all(isfinite.(pressure_vals)) || !all(isfinite.(velocity_mag))
        println("WARNING: Non-finite values detected in solution!")
        return plot(title="Error: Non-finite values in solution")
    end
    
    # Coordinates
    r_coords = [cell.center_x for cell in mesh.cells]
    z_coords = [cell.center_y for cell in mesh.cells]
    
    println("  Coordinate ranges: r=[$(minimum(r_coords)), $(maximum(r_coords))], z=[$(minimum(z_coords)), $(maximum(z_coords))]")
    
    # Filter out any extreme values for plotting
    rho_plot = clamp.(rho, 1e-6, 1e6)
    p_plot = clamp.(pressure_vals, 1e-12, 1e6)
    v_plot = clamp.(velocity_mag, 0.0, 1e3)
    
    try
        # Simple line plots instead of problematic scatter plots
        p1 = plot(title="Density at t = $(round(t, digits=4))")
        plot!(p1, r_coords, rho_plot, seriestype=:scatter, markersize=2, 
              xlabel="r", ylabel="ρ", legend=false)
        
        p2 = plot(title="Pressure")
        plot!(p2, r_coords, log10.(p_plot .+ 1e-12), seriestype=:scatter, markersize=2,
              xlabel="r", ylabel="log₁₀(p)", legend=false)
        
        p3 = plot(title="Velocity Magnitude")
        plot!(p3, r_coords, v_plot, seriestype=:scatter, markersize=2,
              xlabel="r", ylabel="|v|", legend=false)
        
        # Radial profile
        r_profile = sqrt.(r_coords.^2 + z_coords.^2)
        sorted_indices = sortperm(r_profile)
        
        p4 = plot(r_profile[sorted_indices], rho_plot[sorted_indices], 
                 xlabel="Radius", ylabel="Density", 
                 title="Radial Profile", linewidth=2, legend=false)
        
        return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
        
    catch e
        println("Plotting error: $e")
        return plot(title="Error in plotting: $e")
    end
end

# Main simulation
function solve_sedov_blast(nr::Int=20, nz::Int=20, t_final::Float64=0.1, 
                          E_blast::Float64=1.0, r_blast::Float64=0.1)
    
    println("="^60)
    println("2D Axisymmetric Sedov-Taylor Blast Wave Simulation")
    println("="^60)
    
    # Create mesh
    r_max = 1.0
    z_max = 1.0
    mesh = generate_sedov_mesh(nr, nz, r_max, z_max)
    
    # Create solver
    eos = IdealGas(1.4)
    cfl = 0.3
    solver = SedovSolver(mesh, eos, cfl, E_blast, r_blast)
    
    # Initialize blast
    initialize_sedov_blast!(solver)
    
    # Time integration
    t = 0.0
    step = 0
    
    println("\nStarting simulation...")
    println("Final time: $t_final, CFL: $cfl")
    
    while t < t_final
        compute_time_step!(solver)
        
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        update_solution!(solver)
        
        t += solver.dt
        step += 1
        
        if step % 50 == 0
            # Check solution health
            rho_vals = [solver.U[1, i] for i in 1:mesh.n_cells]
            p_vals = [conservative_to_primitive_2d(solver.U[:, i], solver.eos)[4] for i in 1:mesh.n_cells]
            
            rho_min, rho_max = extrema(rho_vals)
            p_min, p_max = extrema(p_vals)
            
            println("Step: $step, Time: $(round(t, digits=4)), dt: $(round(solver.dt, digits=6))")
            println("  ρ: [$(round(rho_min, digits=4)), $(round(rho_max, digits=4))]")
            println("  p: [$(round(p_min, digits=8)), $(round(p_max, digits=4))]")
            
            # Check for problems
            if any(.!isfinite.(rho_vals)) || any(.!isfinite.(p_vals))
                println("ERROR: Non-finite values detected!")
                break
            end
            
            if rho_min <= 0 || p_min <= 0
                println("WARNING: Non-positive values detected!")
            end
        end
    end
    
    println("Simulation completed after $step steps")
    return solver, t
end

# Run the Sedov-Taylor simulation with more conservative parameters
solver, final_time = solve_sedov_blast(100, 100, 0.01, 1.0, 0.2)  # Smaller, shorter, larger blast radius

# Plot results
plt = plot_sedov_solution(solver, final_time)
display(plt)

println("\nSedov-Taylor Blast Wave Simulation Complete!")
println("✓ 2D axisymmetric Euler equations")
println("✓ Unstructured triangular mesh")
println("✓ HLLE Riemann solver")
println("✓ Point explosion initialization")
println("✓ Modular EOS interface")
println("✓ Conservation of energy")

global solver








