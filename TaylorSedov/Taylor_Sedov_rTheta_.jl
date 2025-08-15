using LinearAlgebra
using Plots

# Structure to hold spherical grid data
struct SphericalGrid
    r::Vector{Float64}    # radial coordinates
    θ::Vector{Float64}    # polar angle coordinates
    Δr::Float64          # radial step
    Δθ::Float64          # angular step
    Δt::Float64          # time step
    nr::Int64            # number of radial points
    nθ::Int64            # number of angular points
end

# Structure for conservation variables in spherical coordinates
struct SphericalVars
    ρ::Matrix{Float64}    # density
    ρu::Matrix{Float64}   # radial momentum
    ρv::Matrix{Float64}   # theta momentum
    E::Matrix{Float64}    # total energy
end

# Initialize spherical grid
function create_spherical_grid(rmax::Float64, nr::Int64, nθ::Int64, cfl::Float64)
    r = range(0, rmax, length=nr) |> collect
    θ = range(0, π, length=nθ) |> collect  # Full range of θ
    Δr = r[2] - r[1]
    Δθ = θ[2] - θ[1]
    # Estimate time step based on CFL condition
    Δt = cfl * min(Δr, r[end]*Δθ)
    return SphericalGrid(r, θ, Δr, Δθ, Δt, nr, nθ)
end

# Initialize solution with point explosion conditions
function initialize_solution_spherical(grid::SphericalGrid, E0::Float64)
    ρ = ones(grid.nr, grid.nθ)     # ambient density
    ρu = zeros(grid.nr, grid.nθ)   # initial radial velocity
    ρv = zeros(grid.nr, grid.nθ)   # initial theta velocity
    E = zeros(grid.nr, grid.nθ)
    
    # Set initial energy near origin
    E[1:2, :] .= E0 / (2π * sum(sin.(grid.θ)) * grid.Δθ)
    
    return SphericalVars(ρ, ρu, ρv, E)
end

# Calculate fluxes for CESE method in spherical coordinates
function calculate_fluxes_spherical(U::SphericalVars, grid::SphericalGrid, γ::Float64)
    # Preallocate flux arrays
    F_mass_r = similar(U.ρ)
    F_mass_θ = similar(U.ρ)
    F_rmom_r = similar(U.ρ)
    F_rmom_θ = similar(U.ρ)
    F_θmom_r = similar(U.ρ)
    F_θmom_θ = similar(U.ρ)
    F_energy_r = similar(U.ρ)
    F_energy_θ = similar(U.ρ)
    
    # Calculate pressure from conservation variables
    p = (γ - 1.0) * (U.E - 0.5 * (U.ρu.^2 + U.ρv.^2) ./ U.ρ)
    
    # Calculate fluxes
    for j in axes(U.ρ, 2), i in axes(U.ρ, 1)
        u = U.ρu[i,j] / U.ρ[i,j]  # radial velocity
        v = U.ρv[i,j] / U.ρ[i,j]  # theta velocity
        
        # r-direction fluxes
        F_mass_r[i,j] = U.ρu[i,j]
        F_rmom_r[i,j] = U.ρu[i,j] * u + p[i,j]
        F_θmom_r[i,j] = U.ρv[i,j] * u
        F_energy_r[i,j] = (U.E[i,j] + p[i,j]) * u
        
        # θ-direction fluxes
        F_mass_θ[i,j] = U.ρv[i,j]
        F_rmom_θ[i,j] = U.ρu[i,j] * v
        F_θmom_θ[i,j] = U.ρv[i,j] * v + p[i,j]
        F_energy_θ[i,j] = (U.E[i,j] + p[i,j]) * v
    end
    
    return (F_mass_r, F_mass_θ), (F_rmom_r, F_rmom_θ), 
           (F_θmom_r, F_θmom_θ), (F_energy_r, F_energy_θ)
end

# CESE time integration step in spherical coordinates
function cese_step_spherical!(U::SphericalVars, grid::SphericalGrid, γ::Float64)
    # Calculate fluxes at current time
    F_mass, F_rmom, F_θmom, F_energy = calculate_fluxes_spherical(U, grid, γ)
    
    # Temporary arrays for new solution
    ρ_new = similar(U.ρ)
    ρu_new = similar(U.ρu)
    ρv_new = similar(U.ρv)
    E_new = similar(U.E)
    
    # CESE integration
    for j in 2:(grid.nθ-1), i in 2:(grid.nr-1)
        r = grid.r[i]
        θ = grid.θ[j]
        sin_θ = sin(θ)
        
        # Geometric source terms for spherical coordinates
        S_mass = -2U.ρu[i,j]/r - U.ρv[i,j]*cos(θ)/(r*sin_θ)
        S_rmom = -(U.ρu[i,j]^2)/(r*U.ρ[i,j]) + 
                 (U.ρv[i,j]^2)/(r*U.ρ[i,j])
        S_θmom = -U.ρu[i,j]*U.ρv[i,j]/(r*U.ρ[i,j]) -
                 (U.ρv[i,j]^2)*cos(θ)/(r*sin_θ*U.ρ[i,j])
        S_energy = -(2U.ρu[i,j]*U.E[i,j])/(r*U.ρ[i,j]) -
                   (U.ρv[i,j]*U.E[i,j]*cos(θ))/(r*sin_θ*U.ρ[i,j])
        
        # Conservation element integration
        ρ_new[i,j] = U.ρ[i,j] - 
                     grid.Δt/grid.Δr * (F_mass[1][i+1,j] - F_mass[1][i-1,j])/2 -
                     grid.Δt/(r*sin_θ*grid.Δθ) * (sin_θ*F_mass[2][i,j+1] - sin_θ*F_mass[2][i,j-1])/2 +
                     grid.Δt * S_mass
        
        ρu_new[i,j] = U.ρu[i,j] - 
                      grid.Δt/grid.Δr * (F_rmom[1][i+1,j] - F_rmom[1][i-1,j])/2 -
                      grid.Δt/(r*sin_θ*grid.Δθ) * (sin_θ*F_rmom[2][i,j+1] - sin_θ*F_rmom[2][i,j-1])/2 +
                      grid.Δt * S_rmom
        
        ρv_new[i,j] = U.ρv[i,j] - 
                      grid.Δt/grid.Δr * (F_θmom[1][i+1,j] - F_θmom[1][i-1,j])/2 -
                      grid.Δt/(r*sin_θ*grid.Δθ) * (sin_θ*F_θmom[2][i,j+1] - sin_θ*F_θmom[2][i,j-1])/2 +
                      grid.Δt * S_θmom
        
        E_new[i,j] = U.E[i,j] - 
                     grid.Δt/grid.Δr * (F_energy[1][i+1,j] - F_energy[1][i-1,j])/2 -
                     grid.Δt/(r*sin_θ*grid.Δθ) * (sin_θ*F_energy[2][i,j+1] - sin_θ*F_energy[2][i,j-1])/2 +
                     grid.Δt * S_energy
    end
    
    # Apply boundary conditions
    # r = 0 (origin)
    ρ_new[1,:] = ρ_new[2,:]
    ρu_new[1,:] = -ρu_new[2,:]  # Reflective
    ρv_new[1,:] = ρv_new[2,:]
    E_new[1,:] = E_new[2,:]
    
    # θ = 0 and θ = π (poles)
    ρ_new[:,1] = ρ_new[:,2]
    ρu_new[:,1] = ρu_new[:,2]
    ρv_new[:,1] .= 0.0  # No theta velocity at poles
    E_new[:,1] = E_new[:,2]
    
    ρ_new[:,end] = ρ_new[:,end-1]
    ρu_new[:,end] = ρu_new[:,end-1]
    ρv_new[:,end] .= 0.0  # No theta velocity at poles
    E_new[:,end] = E_new[:,end-1]
    
    # Outer boundary (zero gradient)
    ρ_new[end,:] = ρ_new[end-1,:]
    ρu_new[end,:] = ρu_new[end-1,:]
    ρv_new[end,:] = ρv_new[end-1,:]
    E_new[end,:] = E_new[end-1,:]
    
    # Update solution
    U.ρ .= ρ_new
    U.ρu .= ρu_new
    U.ρv .= ρv_new
    U.E .= E_new
end

# Convert spherical coordinates to Cartesian for visualization
function spherical_to_cartesian(r::Float64, θ::Float64)
    x = r * sin(θ)
    y = r * cos(θ)
    return x, y
end

# Visualization function
function visualize_spherical_solution(grid::SphericalGrid, U::SphericalVars, γ::Float64)
    # Create meshgrid for plotting
    X = zeros(grid.nr, grid.nθ)
    Y = zeros(grid.nr, grid.nθ)
    for j in 1:grid.nθ, i in 1:grid.nr
        X[i,j], Y[i,j] = spherical_to_cartesian(grid.r[i], grid.θ[j])
    end
    
    # Calculate derived quantities
    p = (γ - 1.0) * (U.E - 0.5 * (U.ρu.^2 + U.ρv.^2) ./ U.ρ)
    vel_mag = sqrt.((U.ρu ./ U.ρ).^2 + (U.ρv ./ U.ρ).^2)
    
    # Create subplots
    p1 = contourf(X, Y, U.ρ,
                 title="Density",
                 xlabel="x", ylabel="y",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    p2 = contourf(X, Y, p,
                 title="Pressure",
                 xlabel="x", ylabel="y",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    p3 = contourf(X, Y, vel_mag,
                 title="Velocity Magnitude",
                 xlabel="x", ylabel="y",
                 aspect_ratio=:equal,
                 color=:viridis)
    
    # Combine plots
    plot(p1, p2, p3, layout=(1,3), size=(1200,400))
end

# Main solving function
function solve_taylor_sedov_spherical(rmax::Float64, nr::Int64, nθ::Int64,
                                    tfinal::Float64, γ::Float64, E0::Float64)
    # Create grid
    grid = create_spherical_grid(rmax, nr, nθ, 0.3)
    
    # Initialize solution
    U = initialize_solution_spherical(grid, E0)
    
    # Time stepping
    t = 0.0
    while t < tfinal
        cese_step_spherical!(U, grid, γ)
        t += grid.Δt
        
        # Visualize every few steps
        if mod(round(t/grid.Δt), 20) == 0
            visualize_spherical_solution(grid, U, γ)
            display(plot!(title="Time = $(round(t, digits=3))"))
        end
    end
    
    return grid, U
end

# Example usage
rmax = 1.0
nr = 100
nθ = 50  # Fewer θ points needed due to symmetry
tfinal = 0.1
γ = 1.4  # ratio of specific heats
E0 = 1.0 # initial energy

grid, solution = solve_taylor_sedov_spherical(rmax, nr, nθ, tfinal, γ, E0)
visualize_spherical_solution(grid, solution, γ)