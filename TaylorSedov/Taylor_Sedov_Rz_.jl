using LinearAlgebra
using Plots

# Structure to hold 2D grid and solution data
struct Grid2D
    r::Vector{Float64}    # radial coordinates
    z::Vector{Float64}    # axial coordinates
    Δr::Float64          # radial step
    Δz::Float64          # axial step
    Δt::Float64          # time step
    nr::Int64            # number of radial points
    nz::Int64            # number of axial points
end

# Structure for 2D conservation variables
struct ConservationVars2D
    ρ::Matrix{Float64}    # density
    ρu::Matrix{Float64}   # radial momentum
    ρw::Matrix{Float64}   # axial momentum
    E::Matrix{Float64}    # total energy
end

# Initialize 2D grid
function create_grid_2d(rmax::Float64, zmax::Float64, nr::Int64, nz::Int64, cfl::Float64)
    r = range(0, rmax, length=nr) |> collect
    z = range(0, zmax, length=nz) |> collect
    Δr = r[2] - r[1]
    Δz = z[2] - z[1]
    # Estimate time step based on CFL condition
    Δt = cfl * min(Δr, Δz)
    return Grid2D(r, z, Δr, Δz, Δt, nr, nz)
end

# Initialize solution with point explosion conditions
function initialize_solution_2d(grid::Grid2D, E0::Float64)
    ρ = ones(grid.nr, grid.nz)     # ambient density
    ρu = zeros(grid.nr, grid.nz)   # initial radial velocity
    ρw = zeros(grid.nr, grid.nz)   # initial axial velocity
    E = zeros(grid.nr, grid.nz)
    
    # Set initial energy near origin
    E[1:2, 1:2] .= E0 / 4.0  # Distribute energy over a few cells
    
    return ConservationVars2D(ρ, ρu, ρw, E)
end

# Calculate fluxes for CESE method in 2D
function calculate_fluxes_2d(U::ConservationVars2D, γ::Float64)
    # Preallocate flux arrays
    F_mass_r = similar(U.ρ)
    F_mass_z = similar(U.ρ)
    F_rmom_r = similar(U.ρ)
    F_rmom_z = similar(U.ρ)
    F_zmom_r = similar(U.ρ)
    F_zmom_z = similar(U.ρ)
    F_energy_r = similar(U.ρ)
    F_energy_z = similar(U.ρ)
    
    # Calculate pressure from conservation variables
    p = (γ - 1.0) * (U.E - 0.5 * (U.ρu.^2 + U.ρw.^2) ./ U.ρ)
    
    # Calculate fluxes
    for j in axes(U.ρ, 2), i in axes(U.ρ, 1)
        u = U.ρu[i,j] / U.ρ[i,j]
        w = U.ρw[i,j] / U.ρ[i,j]
        
        # r-direction fluxes
        F_mass_r[i,j] = U.ρu[i,j]
        F_rmom_r[i,j] = U.ρu[i,j] * u + p[i,j]
        F_zmom_r[i,j] = U.ρw[i,j] * u
        F_energy_r[i,j] = (U.E[i,j] + p[i,j]) * u
        
        # z-direction fluxes
        F_mass_z[i,j] = U.ρw[i,j]
        F_rmom_z[i,j] = U.ρu[i,j] * w
        F_zmom_z[i,j] = U.ρw[i,j] * w + p[i,j]
        F_energy_z[i,j] = (U.E[i,j] + p[i,j]) * w
    end
    
    return (F_mass_r, F_mass_z), (F_rmom_r, F_rmom_z), 
           (F_zmom_r, F_zmom_z), (F_energy_r, F_energy_z)
end

# CESE time integration step in 2D
function cese_step_2d!(U::ConservationVars2D, grid::Grid2D, γ::Float64)
    # Calculate fluxes at current time
    F_mass, F_rmom, F_zmom, F_energy = calculate_fluxes_2d(U, γ)
    
    # Temporary arrays for new solution
    ρ_new = similar(U.ρ)
    ρu_new = similar(U.ρu)
    ρw_new = similar(U.ρw)
    E_new = similar(U.E)
    
    # CESE integration
    for j in 2:(grid.nz-1), i in 2:(grid.nr-1)
        r = grid.r[i]
        
        # Source terms for axisymmetric geometry
        S_mass = -U.ρu[i,j] / r
        S_rmom = -U.ρu[i,j]^2 / (r * U.ρ[i,j])
        S_zmom = 0.0  # No geometric source term for axial momentum
        S_energy = -U.ρu[i,j] * U.E[i,j] / (r * U.ρ[i,j])
        
        # Conservation element integration
        ρ_new[i,j] = U.ρ[i,j] - 
                     grid.Δt / grid.Δr * (F_mass[1][i+1,j] - F_mass[1][i-1,j]) / 2 -
                     grid.Δt / grid.Δz * (F_mass[2][i,j+1] - F_mass[2][i,j-1]) / 2 +
                     grid.Δt * S_mass
        
        ρu_new[i,j] = U.ρu[i,j] - 
                      grid.Δt / grid.Δr * (F_rmom[1][i+1,j] - F_rmom[1][i-1,j]) / 2 -
                      grid.Δt / grid.Δz * (F_rmom[2][i,j+1] - F_rmom[2][i,j-1]) / 2 +
                      grid.Δt * S_rmom
        
        ρw_new[i,j] = U.ρw[i,j] - 
                      grid.Δt / grid.Δr * (F_zmom[1][i+1,j] - F_zmom[1][i-1,j]) / 2 -
                      grid.Δt / grid.Δz * (F_zmom[2][i,j+1] - F_zmom[2][i,j-1]) / 2 +
                      grid.Δt * S_zmom
        
        E_new[i,j] = U.E[i,j] - 
                     grid.Δt / grid.Δr * (F_energy[1][i+1,j] - F_energy[1][i-1,j]) / 2 -
                     grid.Δt / grid.Δz * (F_energy[2][i,j+1] - F_energy[2][i,j-1]) / 2 +
                     grid.Δt * S_energy
    end
    
    # Apply boundary conditions
    # Axis (r=0)
    ρ_new[1,:] = ρ_new[2,:]
    ρu_new[1,:] = -ρu_new[2,:]  # Reflective
    ρw_new[1,:] = ρw_new[2,:]
    E_new[1,:] = E_new[2,:]
    
    # Outer boundaries (zero gradient)
    ρ_new[end,:] = ρ_new[end-1,:]
    ρu_new[end,:] = ρu_new[end-1,:]
    ρw_new[end,:] = ρw_new[end-1,:]
    E_new[end,:] = E_new[end-1,:]
    
    ρ_new[:,end] = ρ_new[:,end-1]
    ρu_new[:,end] = ρu_new[:,end-1]
    ρw_new[:,end] = ρw_new[:,end-1]
    E_new[:,end] = E_new[:,end-1]
    
    # Update solution
    U.ρ .= ρ_new
    U.ρu .= ρu_new
    U.ρw .= ρw_new
    U.E .= E_new
end

# Visualization function
function visualize_solution(grid::Grid2D, U::ConservationVars2D, γ::Float64)
    # Calculate pressure and velocity magnitude
    p = (γ - 1.0) * (U.E - 0.5 * (U.ρu.^2 + U.ρw.^2) ./ U.ρ)
    vel_mag = sqrt.((U.ρu ./ U.ρ).^2 + (U.ρw ./ U.ρ).^2)
    
    # Create subplots
    p1 = heatmap(grid.r, grid.z, transpose(U.ρ),
                 title="Density",
                 xlabel="r", ylabel="z",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    p2 = heatmap(grid.r, grid.z, transpose(p),
                 title="Pressure",
                 xlabel="r", ylabel="z",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    p3 = heatmap(grid.r, grid.z, transpose(vel_mag),
                 title="Velocity Magnitude",
                 xlabel="r", ylabel="z",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    # Combine plots
    plot(p1, p2, p3, layout=(1,3), size=(1200,400))
end

# Main solving function
function solve_taylor_sedov_2d(rmax::Float64, zmax::Float64, nr::Int64, nz::Int64, 
                             tfinal::Float64, γ::Float64, E0::Float64)
    # Create grid
    grid = create_grid_2d(rmax, zmax, nr, nz, 0.3)
    
    # Initialize solution
    U = initialize_solution_2d(grid, E0)
    
    # Time stepping
    t = 0.0
    while t < tfinal
        cese_step_2d!(U, grid, γ)
        t += grid.Δt
        
        # Visualize every few steps (adjust as needed)
        if mod(round(t/grid.Δt), 20) == 0
            visualize_solution(grid, U, γ)
            display(plot!(title="Time = $(round(t, digits=3))"))
        end
    end
    
    return grid, U
end

# Example usage
rmax = 1.0
zmax = 1.0
nr = 100
nz = 100
tfinal = 0.1
γ = 1.4  # ratio of specific heats
E0 = 1.0 # initial energy

grid, solution = solve_taylor_sedov_2d(rmax, zmax, nr, nz, tfinal, γ, E0)
visualize_solution(grid, solution, γ)