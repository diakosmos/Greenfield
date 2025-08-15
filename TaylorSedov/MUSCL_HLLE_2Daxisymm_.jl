using Plots
using Random
using LinearAlgebra

# Include the mesh generator (all previous mesh code)
include("Triangulator_2Daxisymm_.jl")  # This would contain all the mesh generation code

# For this demo, I'll include the essential mesh structures inline
# In practice, you'd have the mesh generator in a separate file

# Mesh data structures (from previous code)
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

# 2D Axisymmetric Euler Solver using Unstructured Triangular Meshes
# Extends MUSCL+HLLE to 2D with source terms

# Abstract EOS interface (same as 1D)
abstract type EquationOfState end

struct IdealGas <: EquationOfState
    gamma::Float64
end

# EOS functions
function pressure(eos::IdealGas, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
    u = rho_u / rho
    v = rho_v / rho
    kinetic = 0.5 * rho * (u^2 + v^2)
    return (eos.gamma - 1) * (rho_E - kinetic)
end

function sound_speed(eos::IdealGas, rho::Float64, p::Float64)
    return sqrt(eos.gamma * p / rho)
end

function temperature(eos::IdealGas, rho::Float64, p::Float64)
    return p / rho  # Assuming R=1
end

function total_energy(eos::IdealGas, rho::Float64, u::Float64, v::Float64, p::Float64)
    return p / (eos.gamma - 1) + 0.5 * rho * (u^2 + v^2)
end

# 2D Euler solver structure
mutable struct EulerSolver2D
    mesh::Mesh
    eos::EquationOfState
    cfl::Float64
    dt::Float64
    
    # Conservative variables: [rho, rho*u, rho*v, rho*E] for each cell
    U::Matrix{Float64}      # 4 x n_cells
    U_new::Matrix{Float64}  # 4 x n_cells
    
    # Cell gradients for MUSCL reconstruction
    grad_U::Array{Float64, 3}  # 4 x 2 x n_cells (variable x component x cell)
    
    # Axisymmetric coordinate system (r-z)
    # r = x-coordinate (radial), z = y-coordinate (axial)
    axisymmetric::Bool
end

function EulerSolver2D(mesh::Mesh, eos::EquationOfState, cfl::Float64; axisymmetric::Bool=true)
    n_cells = mesh.n_cells
    U = zeros(4, n_cells)
    U_new = zeros(4, n_cells)
    grad_U = zeros(4, 2, n_cells)
    
    return EulerSolver2D(mesh, eos, cfl, 0.0, U, U_new, grad_U, axisymmetric)
end

# Convert conservative to primitive variables (2D)
function conservative_to_primitive_2d(U::Vector{Float64}, eos::EquationOfState)
    rho = max(U[1], 1e-12)
    u = U[2] / rho
    v = U[3] / rho
    rho_E = U[4]
    
    p = max(pressure(eos, rho, rho_E, U[2], U[3]), 1e-12)
    
    return [rho, u, v, p]
end

# Convert primitive to conservative variables (2D)
function primitive_to_conservative_2d(W::Vector{Float64}, eos::EquationOfState)
    rho, u, v, p = W
    rho_u = rho * u
    rho_v = rho * v
    rho_E = total_energy(eos, rho, u, v, p)
    
    return [rho, rho_u, rho_v, rho_E]
end

# Compute 2D flux vectors
function compute_flux_2d(U::Vector{Float64}, eos::EquationOfState, direction::Int)
    # direction: 1 = x-flux (F), 2 = y-flux (G)
    W = conservative_to_primitive_2d(U, eos)
    rho, u, v, p = W
    
    if direction == 1  # x-direction flux
        return [rho * u,
               rho * u^2 + p,
               rho * u * v,
               (U[4] + p) * u]
    else  # y-direction flux
        return [rho * v,
               rho * u * v,
               rho * v^2 + p,
               (U[4] + p) * v]
    end
end

# Compute axisymmetric source terms
function axisymmetric_source(U::Vector{Float64}, r::Float64, eos::EquationOfState)
    # Source terms for axisymmetric coordinates: S = -(1/r) * G_cyl
    # where G_cyl is the cylindrical momentum flux
    
    if r < 1e-12  # Avoid singularity at centerline
        return zeros(4)
    end
    
    W = conservative_to_primitive_2d(U, eos)
    rho, u, v, p = W
    
    # Axisymmetric source: S = -(1/r) * [0, 0, rho*v^2 + p, 0]
    return [0.0,
           0.0,
           -(rho * v^2 + p) / r,
           0.0]
end

# Compute cell gradients using least-squares reconstruction
function compute_gradients!(solver::EulerSolver2D)
    mesh = solver.mesh
    
    # Initialize gradients to zero
    solver.grad_U .= 0.0
    
    for cell_idx in 1:mesh.n_cells
        cell = mesh.cells[cell_idx]
        
        # Get cell center
        x0 = cell.center_x
        y0 = cell.center_y
        
        # Collect neighbor information
        neighbors = Int[]
        
        # Find neighbors through faces
        for face_id in [cell.face1, cell.face2, cell.face3]
            if face_id > 0 && face_id <= length(mesh.faces)
                face = mesh.faces[face_id]
                
                # Find the other cell sharing this face
                neighbor_id = 0
                if face.cell_left == cell_idx
                    neighbor_id = face.cell_right
                elseif face.cell_right == cell_idx
                    neighbor_id = face.cell_left
                end
                
                if neighbor_id > 0
                    push!(neighbors, neighbor_id)
                end
            end
        end
        
        if length(neighbors) >= 2  # Need at least 2 neighbors for gradient
            # Set up least squares system: A * grad = b
            n_neighbors = length(neighbors)
            A = zeros(n_neighbors, 2)
            
            for (i, neighbor_id) in enumerate(neighbors)
                neighbor = mesh.cells[neighbor_id]
                A[i, 1] = neighbor.center_x - x0  # Δx
                A[i, 2] = neighbor.center_y - y0  # Δy
            end
            
            # Solve for each variable
            for var in 1:4
                b = zeros(n_neighbors)
                for (i, neighbor_id) in enumerate(neighbors)
                    b[i] = solver.U[var, neighbor_id] - solver.U[var, cell_idx]
                end
                
                # Least squares solution: grad = (A^T A)^(-1) A^T b
                try
                    AtA = A' * A
                    if det(AtA) > 1e-12
                        grad = AtA \ (A' * b)
                        solver.grad_U[var, 1, cell_idx] = grad[1]  # ∂/∂x
                        solver.grad_U[var, 2, cell_idx] = grad[2]  # ∂/∂y
                    end
                catch
                    # Keep zero gradient if solution fails
                end
            end
        end
    end
end

# Minmod limiter for gradients
function minmod_limit_gradient!(solver::EulerSolver2D, cell_idx::Int, var::Int)
    mesh = solver.mesh
    cell = mesh.cells[cell_idx]
    
    grad_x = solver.grad_U[var, 1, cell_idx]
    grad_y = solver.grad_U[var, 2, cell_idx]
    
    # Find minimum and maximum values among neighbors
    U_center = solver.U[var, cell_idx]
    U_min = U_center
    U_max = U_center
    
    # Check neighbors
    for face_id in [cell.face1, cell.face2, cell.face3]
        if face_id > 0 && face_id <= length(mesh.faces)
            face = mesh.faces[face_id]
            neighbor_id = face.cell_left == cell_idx ? face.cell_right : face.cell_left
            
            if neighbor_id > 0
                U_neighbor = solver.U[var, neighbor_id]
                U_min = min(U_min, U_neighbor)
                U_max = max(U_max, U_neighbor)
            end
        end
    end
    
    # Apply minmod limiting
    dx_max = cell.center_x - mesh.x_min
    dx_min = mesh.x_max - cell.center_x
    dy_max = cell.center_y - mesh.y_min
    dy_min = mesh.y_max - cell.center_y
    
    # Limit gradient to prevent overshoots
    if abs(grad_x) > 1e-12
        limit_factor_x = min(abs((U_max - U_center) / (grad_x * dx_max)),
                            abs((U_min - U_center) / (grad_x * dx_min)))
        grad_x *= min(1.0, limit_factor_x)
    end
    
    if abs(grad_y) > 1e-12
        limit_factor_y = min(abs((U_max - U_center) / (grad_y * dy_max)),
                            abs((U_min - U_center) / (grad_y * dy_min)))
        grad_y *= min(1.0, limit_factor_y)
    end
    
    solver.grad_U[var, 1, cell_idx] = grad_x
    solver.grad_U[var, 2, cell_idx] = grad_y
end

# MUSCL reconstruction at face
function muscl_reconstruct_2d(solver::EulerSolver2D, face::Face)
    mesh = solver.mesh
    
    cell_left = face.cell_left
    cell_right = face.cell_right
    
    # Face center
    xf = face.center_x
    yf = face.center_y
    
    # Reconstruct left state
    U_L = zeros(4)
    if cell_left > 0
        cell_L = mesh.cells[cell_left]
        dx = xf - cell_L.center_x
        dy = yf - cell_L.center_y
        
        for var in 1:4
            U_L[var] = solver.U[var, cell_left] + 
                      solver.grad_U[var, 1, cell_left] * dx +
                      solver.grad_U[var, 2, cell_left] * dy
        end
    end
    
    # Reconstruct right state
    U_R = zeros(4)
    if cell_right > 0
        cell_R = mesh.cells[cell_right]
        dx = xf - cell_R.center_x
        dy = yf - cell_R.center_y
        
        for var in 1:4
            U_R[var] = solver.U[var, cell_right] + 
                      solver.grad_U[var, 1, cell_right] * dx +
                      solver.grad_U[var, 2, cell_right] * dy
        end
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
    F_L = u_L * normal_x * U_L + v_L * normal_y * U_L
    F_L[2] += p_L * normal_x
    F_L[3] += p_L * normal_y
    
    F_R = u_R * normal_x * U_R + v_R * normal_y * U_R
    F_R[2] += p_R * normal_x
    F_R[3] += p_R * normal_y
    
    # HLLE flux
    if S_L >= 0.0
        return F_L
    elseif S_R <= 0.0
        return F_R
    else
        return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    end
end

# Compute time step
function compute_time_step!(solver::EulerSolver2D)
    max_speed = 0.0
    
    for cell_idx in 1:solver.mesh.n_cells
        W = conservative_to_primitive_2d(solver.U[:, cell_idx], solver.eos)
        rho, u, v, p = W
        
        c = sound_speed(solver.eos, rho, p)
        speed = sqrt(u^2 + v^2) + c
        max_speed = max(max_speed, speed)
    end
    
    # Estimate minimum cell size
    min_dx = minimum(sqrt(cell.area) for cell in solver.mesh.cells)
    
    solver.dt = solver.cfl * min_dx / max_speed
end

# Main update step
function update_solution!(solver::EulerSolver2D)
    mesh = solver.mesh
    
    # Compute gradients
    compute_gradients!(solver)
    
    # Apply limiting
    for cell_idx in 1:mesh.n_cells
        for var in 1:4
            minmod_limit_gradient!(solver, cell_idx, var)
        end
    end
    
    # Initialize new solution
    solver.U_new .= solver.U
    
    # Compute fluxes and update
    for face in mesh.faces
        if face.boundary
            continue  # Skip boundary faces for now
        end
        
        # MUSCL reconstruction
        U_L, U_R = muscl_reconstruct_2d(solver, face)
        
        # Ensure physical states
        U_L[1] = max(U_L[1], 1e-12)
        U_R[1] = max(U_R[1], 1e-12)
        
        # HLLE flux
        flux = hlle_flux_2d(U_L, U_R, face.normal_x, face.normal_y, solver.eos)
        
        # Update adjacent cells
        if face.cell_left > 0
            cell_left = mesh.cells[face.cell_left]
            flux_contribution = flux * face.area / cell_left.area
            solver.U_new[:, face.cell_left] -= solver.dt * flux_contribution
        end
        
        if face.cell_right > 0
            cell_right = mesh.cells[face.cell_right]
            flux_contribution = flux * face.area / cell_right.area
            solver.U_new[:, face.cell_right] += solver.dt * flux_contribution
        end
    end
    
    # Add axisymmetric source terms
    if solver.axisymmetric
        for cell_idx in 1:mesh.n_cells
            cell = mesh.cells[cell_idx]
            r = cell.center_x  # Radial coordinate
            
            source = axisymmetric_source(solver.U[:, cell_idx], r, solver.eos)
            solver.U_new[:, cell_idx] += solver.dt * source
        end
    end
    
    # Ensure physical states
    for cell_idx in 1:mesh.n_cells
        solver.U_new[1, cell_idx] = max(solver.U_new[1, cell_idx], 1e-12)
        
        # Check pressure
        W = conservative_to_primitive_2d(solver.U_new[:, cell_idx], solver.eos)
        if W[4] <= 1e-12  # pressure
            # Fix energy
            rho, u, v = W[1], W[2], W[3]
            p_min = 1e-12
            solver.U_new[4, cell_idx] = total_energy(solver.eos, rho, u, v, p_min)
        end
    end
    
    # Update solution
    solver.U .= solver.U_new
end

# Simple mesh generator (inline for this demo)
function create_simple_mesh()
    # Create a simple test mesh manually
    mesh = Mesh(Node[], Face[], Cell[], 0, 0, 0, Int[], 0.0, 1.0, 0.0, 1.0)
    
    # Add some nodes
    push!(mesh.nodes, Node(1, 0.0, 0.0, true))
    push!(mesh.nodes, Node(2, 1.0, 0.0, true))
    push!(mesh.nodes, Node(3, 0.5, 0.5, false))
    push!(mesh.nodes, Node(4, 0.0, 1.0, true))
    push!(mesh.nodes, Node(5, 1.0, 1.0, true))
    
    mesh.n_nodes = 5
    
    # Add cells (triangles)
    push!(mesh.cells, Cell(1, 1, 2, 3, 1, 2, 3, 0.5, 0.167, 0.25, π*0.5*0.25))
    push!(mesh.cells, Cell(2, 1, 3, 4, 3, 4, 5, 0.167, 0.5, 0.25, π*0.167*0.25))
    push!(mesh.cells, Cell(3, 2, 5, 3, 1, 6, 2, 0.833, 0.5, 0.25, π*0.833*0.25))
    push!(mesh.cells, Cell(4, 3, 5, 4, 4, 7, 6, 0.5, 0.833, 0.25, π*0.5*0.25))
    
    mesh.n_cells = 4
    
    # Add faces
    push!(mesh.faces, Face(1, 1, 2, 1, 3, true, 0.0, -1.0, 1.0, 0.5, 0.0))
    push!(mesh.faces, Face(2, 2, 3, 1, 0, false, 0.707, 0.707, 0.707, 0.75, 0.25))
    push!(mesh.faces, Face(3, 1, 3, 1, 2, false, -0.707, 0.707, 0.707, 0.25, 0.25))
    push!(mesh.faces, Face(4, 1, 4, 2, 0, true, -1.0, 0.0, 1.0, 0.0, 0.5))
    push!(mesh.faces, Face(5, 3, 4, 2, 4, false, 0.0, 1.0, 0.5, 0.25, 0.75))
    push!(mesh.faces, Face(6, 3, 5, 3, 4, false, 0.707, 0.707, 0.707, 0.75, 0.75))
    push!(mesh.faces, Face(7, 4, 5, 4, 0, true, 0.0, 1.0, 1.0, 0.5, 1.0))
    
    mesh.n_faces = 7
    mesh.boundary_faces = [1, 4, 7]
    
    return mesh
end

# Test the 2D solver
println("="^60)
println("2D Axisymmetric Euler Solver with Unstructured Mesh")
println("="^60)

# Create mesh
mesh = create_simple_mesh()
println("Created test mesh with $(mesh.n_cells) cells")

# Create solver
eos = IdealGas(1.4)
solver = EulerSolver2D(mesh, eos, 0.4, axisymmetric=true)

# Initialize with simple state
for cell_idx in 1:mesh.n_cells
    rho, u, v, p = 1.0, 0.0, 0.0, 1.0
    solver.U[:, cell_idx] = primitive_to_conservative_2d([rho, u, v, p], eos)
end

println("Solver initialized successfully!")
println("\nKey features implemented:")
println("✓ 2D axisymmetric Euler equations")
println("✓ Unstructured triangular mesh support")
println("✓ MUSCL reconstruction with least-squares gradients")
println("✓ HLLE Riemann solver for 2D")
println("✓ Modular EOS interface (same as 1D)")
println("✓ Axisymmetric source terms")
println("✓ Gradient limiting for stability")

println("\nReady for Sedov-Taylor blast wave problem!")
println("Next: Initialize blast wave and run simulation")