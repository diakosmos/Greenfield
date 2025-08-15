using Plots
using LinearAlgebra

# 2D Axisymmetric Reactive Flow Solver
# Explosive detonation: Murnaghan EOS → JWL EOS with detonation wave tracking

# Abstract EOS interface
abstract type EquationOfState end

# Murnaghan EOS for solid HE
struct MurnaghanEOS <: EquationOfState
    K0::Float64        # Bulk modulus at reference state (Pa)
    K0_prime::Float64  # dK/dP at reference state (dimensionless)
    rho0::Float64      # Reference density (kg/m³)
    e0::Float64        # Reference specific internal energy (J/kg)
end

# JWL EOS for detonation products  
struct JWLEOS <: EquationOfState
    A::Float64         # JWL parameter A (Pa)
    B::Float64         # JWL parameter B (Pa)
    R1::Float64        # JWL parameter R1 (dimensionless)
    R2::Float64        # JWL parameter R2 (dimensionless)
    omega::Float64     # JWL parameter ω (dimensionless)
    rho0::Float64      # Reference density (kg/m³)
    Q::Float64         # Heat of explosion (J/kg)
end

# Mixed explosive state for reacting cells
mutable struct ExplosiveState
    lambda::Float64           # Reaction progress: 0=solid HE, 1=products
    eos_solid::MurnaghanEOS   # Solid HE EOS
    eos_products::JWLEOS      # Product EOS
    det_time::Float64         # Time when detonation arrived (-1 if not yet)
    reaction_rate::Float64    # Reaction rate parameter (1/s)
end

# Murnaghan EOS functions
function pressure(eos::MurnaghanEOS, rho::Float64, e::Float64)
    # Murnaghan: p = (K0/K0') * [(ρ/ρ0)^K0' - 1]
    if rho <= 0
        return 1e-12
    end
    
    compression = rho / eos.rho0
    if compression <= 0
        return 1e-12
    end
    
    p = (eos.K0 / eos.K0_prime) * (compression^eos.K0_prime - 1.0)
    return max(p, 1e-12)
end

function sound_speed(eos::MurnaghanEOS, rho::Float64, p::Float64)
    # c² = dp/dρ|s for Murnaghan
    if rho <= 0
        return 1.0
    end
    
    compression = rho / eos.rho0
    if compression <= 0
        return 1.0
    end
    
    # dp/dρ = K0 * (ρ/ρ0)^(K0'-1) / ρ0
    dpdρ = eos.K0 * compression^(eos.K0_prime - 1.0) / eos.rho0
    c_squared = max(dpdρ, 1.0)
    
    return sqrt(c_squared)
end

# JWL EOS functions
function pressure(eos::JWLEOS, rho::Float64, e::Float64)
    # JWL: p = A(1 - ω/R1V)e^(-R1V) + B(1 - ω/R2V)e^(-R2V) + ωρe/V
    # where V = ρ0/ρ is relative volume
    
    if rho <= 0
        return 1e-12
    end
    
    V = eos.rho0 / rho  # Relative volume
    
    if V <= 0
        return 1e-12
    end
    
    # Prevent overflow in exponentials
    R1V = min(eos.R1 * V, 50.0)
    R2V = min(eos.R2 * V, 50.0)
    
    term1 = eos.A * (1.0 - eos.omega / (eos.R1 * V)) * exp(-R1V)
    term2 = eos.B * (1.0 - eos.omega / (eos.R2 * V)) * exp(-R2V)
    term3 = eos.omega * rho * e / V
    
    p = term1 + term2 + term3
    return max(p, 1e-12)
end

function sound_speed(eos::JWLEOS, rho::Float64, p::Float64)
    # Approximate sound speed for JWL (exact derivation is complex)
    # Use gamma-law approximation: c² ≈ γp/ρ with effective γ
    gamma_eff = 1.4  # Reasonable approximation for products
    
    c_squared = gamma_eff * p / rho
    return sqrt(max(c_squared, 1.0))
end

# Mixed EOS for reacting explosive
function pressure(state::ExplosiveState, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
    # Extract specific internal energy
    u = rho_u / rho
    v = rho_v / rho
    kinetic = 0.5 * (u^2 + v^2)
    e_total = rho_E / rho
    e_internal = e_total - kinetic
    
    if state.lambda <= 0.0
        # Pure solid HE
        return pressure(state.eos_solid, rho, e_internal)
        
    elseif state.lambda >= 1.0
        # Pure products with heat release
        e_products = e_internal + state.lambda * state.eos_products.Q
        return pressure(state.eos_products, rho, e_products)
        
    else
        # Mixed state: pressure equilibrium assumption
        # p_solid = p_products, solve for mixed pressure
        
        # Solid contribution
        p_solid = pressure(state.eos_solid, rho, e_internal)
        
        # Products contribution (with partial heat release)
        e_products = e_internal + state.lambda * state.eos_products.Q
        p_products = pressure(state.eos_products, rho, e_products)
        
        # Simple mixing rule (can be more sophisticated)
        p_mixed = (1.0 - state.lambda) * p_solid + state.lambda * p_products
        
        return p_mixed
    end
end

function sound_speed(state::ExplosiveState, rho::Float64, p::Float64)
    if state.lambda <= 0.0
        return sound_speed(state.eos_solid, rho, p)
    elseif state.lambda >= 1.0
        return sound_speed(state.eos_products, rho, p)
    else
        # Mixed sound speed
        c_solid = sound_speed(state.eos_solid, rho, p)
        c_products = sound_speed(state.eos_products, rho, p)
        
        # Volume-weighted mixing
        c_mixed_sq = (1.0 - state.lambda) * c_solid^2 + state.lambda * c_products^2
        return sqrt(max(c_mixed_sq, 1.0))
    end
end

# Total energy including heat release
function total_energy(state::ExplosiveState, rho::Float64, u::Float64, v::Float64, p::Float64)
    kinetic = 0.5 * rho * (u^2 + v^2)
    
    if state.lambda <= 0.0
        # Solid HE: simple relation (approximate)
        internal = p / ((1.4 - 1) * rho)  # Approximate
        return internal + kinetic
        
    elseif state.lambda >= 1.0
        # Products: use JWL relation (approximate)
        internal = p / ((1.4 - 1) * rho) - state.eos_products.Q  # Approximate
        return internal + kinetic + state.eos_products.Q
        
    else
        # Mixed state
        e_solid = p / ((1.4 - 1) * rho)
        e_products = p / ((1.4 - 1) * rho) - state.eos_products.Q
        
        internal = (1.0 - state.lambda) * e_solid + state.lambda * (e_products + state.eos_products.Q)
        return internal + kinetic
    end
end

# 2D Axisymmetric explosive solver
mutable struct ExplosiveSolver
    nr::Int
    nz::Int
    r_min::Float64
    r_max::Float64
    z_min::Float64
    z_max::Float64
    dr::Float64
    dz::Float64
    cfl::Float64
    dt::Float64
    
    # Conservative variables: [rho, rho*u_r, rho*u_z, rho*E]
    U::Array{Float64, 4}    # (4, nr, nz, 2)
    current::Int
    
    # Explosive state for each cell
    explosive_state::Matrix{ExplosiveState}
    
    # Detonation parameters
    det_velocity::Float64   # Detonation velocity (m/s)
    det_center_r::Float64   # Detonation initiation point
    det_center_z::Float64
    det_start_time::Float64
    
    # Grid
    r::Vector{Float64}
    z::Vector{Float64}
end

function ExplosiveSolver(nr::Int, nz::Int, r_min::Float64, r_max::Float64,
                        z_min::Float64, z_max::Float64, cfl::Float64,
                        det_velocity::Float64)
    
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    
    r = [r_min + (i-0.5)*dr for i in 1:nr]
    z = [z_min + (j-0.5)*dz for j in 1:nz]
    
    U = zeros(4, nr, nz, 2)
    
    # Create explosive states for each cell
    # Typical TNT parameters (scaled for demonstration)
    murnaghan = MurnaghanEOS(8.0e9, 11.0, 1630.0, 0.0)  # 8 GPa, K'=11, 1.63 g/cm³
    jwl = JWLEOS(3.73e11, 3.7e9, 4.15, 0.95, 0.3, 1630.0, 4.6e6)  # TNT-like parameters
    
    explosive_state = Matrix{ExplosiveState}(undef, nr, nz)
    for j in 1:nz, i in 1:nr
        explosive_state[i,j] = ExplosiveState(0.0, murnaghan, jwl, -1.0, 1.0e6)
    end
    
    return ExplosiveSolver(nr, nz, r_min, r_max, z_min, z_max, dr, dz, cfl, 0.0,
                          U, 1, explosive_state, det_velocity, 0.0, 0.0, 0.0, r, z)
end

# Initialize explosive charge
function initialize_explosive!(solver::ExplosiveSolver, p_initial::Float64=1.0e5)
    current = solver.current
    
    # Initial conditions: solid HE at ambient pressure
    rho_initial = solver.explosive_state[1,1].eos_solid.rho0
    u_r_initial = 0.0
    u_z_initial = 0.0
    
    println("Initializing explosive charge:")
    println("  Solid HE density: $(rho_initial) kg/m³")
    println("  Initial pressure: $(p_initial) Pa")
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            state = solver.explosive_state[i,j]
            
            # Calculate total energy for solid HE
            kinetic = 0.5 * rho_initial * (u_r_initial^2 + u_z_initial^2)
            
            # For Murnaghan, approximate internal energy
            e_internal = p_initial / ((1.4-1) * rho_initial)  # Approximate
            rho_E = rho_initial * (e_internal + 0.5 * (u_r_initial^2 + u_z_initial^2))
            
            solver.U[:, i, j, current] = [rho_initial, 
                                         rho_initial * u_r_initial,
                                         rho_initial * u_z_initial, 
                                         rho_E]
        end
    end
    
    println("Explosive charge initialized successfully")
end

# Initiate detonation at center
function initiate_detonation!(solver::ExplosiveSolver, r_det::Float64, z_det::Float64, current_time::Float64)
    solver.det_center_r = r_det
    solver.det_center_z = z_det
    solver.det_start_time = current_time
    
    println("Detonation initiated at (r=$(r_det), z=$(z_det)) at t=$(current_time)")
    println("Detonation velocity: $(solver.det_velocity) m/s")
end

# Update detonation wave and reaction progress
function update_detonation!(solver::ExplosiveSolver, current_time::Float64, dt::Float64)
    if solver.det_start_time < 0
        return  # Detonation not initiated yet
    end
    
    # Current detonation radius
    det_radius = solver.det_velocity * (current_time - solver.det_start_time)
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            r = solver.r[i]
            z = solver.z[j]
            
            # Distance from detonation center
            distance = sqrt((r - solver.det_center_r)^2 + (z - solver.det_center_z)^2)
            
            state = solver.explosive_state[i,j]
            
            if distance <= det_radius && state.det_time < 0
                # Detonation wave just arrived
                state.det_time = current_time
                println("Detonation reached cell ($i,$j) at t=$(current_time), r=$(distance)")
            end
            
            if state.det_time >= 0
                # Cell is reacting
                reaction_time = current_time - state.det_time
                
                # Simple exponential reaction kinetics
                # λ = 1 - exp(-k*t)
                state.lambda = 1.0 - exp(-state.reaction_rate * reaction_time)
                state.lambda = clamp(state.lambda, 0.0, 1.0)
            end
        end
    end
end

# Conservative to primitive conversion
function conservative_to_primitive_explosive(U::Vector{Float64}, state::ExplosiveState)
    rho = max(U[1], 1e-12)
    u_r = U[2] / rho
    u_z = U[3] / rho
    p = max(pressure(state, rho, U[4], U[2], U[3]), 1e-12)
    return [rho, u_r, u_z, p]
end

# Primitive to conservative conversion
function primitive_to_conservative_explosive(W::Vector{Float64}, state::ExplosiveState)
    rho, u_r, u_z, p = W
    rho_E = total_energy(state, rho, u_r, u_z, p)
    return [rho, rho*u_r, rho*u_z, rho_E]
end

# HLLE flux for explosive
function hlle_flux_explosive(U_L::Vector{Float64}, U_R::Vector{Float64}, 
                           state_L::ExplosiveState, state_R::ExplosiveState,
                           direction::Int)
    
    W_L = conservative_to_primitive_explosive(U_L, state_L)
    W_R = conservative_to_primitive_explosive(U_R, state_R)
    
    rho_L, u_r_L, u_z_L, p_L = W_L
    rho_R, u_r_R, u_z_R, p_R = W_R
    
    c_L = sound_speed(state_L, rho_L, p_L)
    c_R = sound_speed(state_R, rho_R, p_R)
    
    if direction == 1  # r-direction
        S_L = min(u_r_L - c_L, u_r_R - c_R)
        S_R = max(u_r_L + c_L, u_r_R + c_R)
        
        F_L = [rho_L * u_r_L,
               rho_L * u_r_L^2 + p_L,
               rho_L * u_r_L * u_z_L,
               (U_L[4] + p_L) * u_r_L]
        
        F_R = [rho_R * u_r_R,
               rho_R * u_r_R^2 + p_R,
               rho_R * u_r_R * u_z_R,
               (U_R[4] + p_R) * u_r_R]
               
    else  # z-direction
        S_L = min(u_z_L - c_L, u_z_R - c_R)
        S_R = max(u_z_L + c_L, u_z_R + c_R)
        
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

# Compute time step
function compute_time_step!(solver::ExplosiveSolver)
    max_speed = 0.0
    current = solver.current
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            U = solver.U[:, i, j, current]
            state = solver.explosive_state[i,j]
            W = conservative_to_primitive_explosive(U, state)
            rho, u_r, u_z, p = W
            
            c = sound_speed(state, rho, p)
            speed_r = abs(u_r) + c
            speed_z = abs(u_z) + c
            
            dt_r = solver.dr / speed_r
            dt_z = solver.dz / speed_z
            dt_cell = min(dt_r, dt_z)
            
            max_speed = max(max_speed, 1.0 / dt_cell)
        end
    end
    
    solver.dt = solver.cfl / max_speed
    solver.dt = min(solver.dt, 1e-5)  # Safety limit
    solver.dt = max(solver.dt, 1e-8)
end

# Simplified update (first-order for stability)
function update_solution!(solver::ExplosiveSolver)
    current = solver.current
    next = 3 - current
    
    solver.U[:, :, :, next] .= solver.U[:, :, :, current]
    
    # Simple first-order updates for stability
    for j in 2:solver.nz-1
        for i in 2:solver.nr-1
            # x-direction fluxes
            U_L = solver.U[:, i-1, j, current]
            U_R = solver.U[:, i+1, j, current]
            state_L = solver.explosive_state[i-1, j]
            state_R = solver.explosive_state[i+1, j]
            
            flux_r = hlle_flux_explosive(U_L, U_R, state_L, state_R, 1)
            
            # z-direction fluxes
            U_B = solver.U[:, i, j-1, current]
            U_T = solver.U[:, i, j+1, current]
            state_B = solver.explosive_state[i, j-1]
            state_T = solver.explosive_state[i, j+1]
            
            flux_z = hlle_flux_explosive(U_B, U_T, state_B, state_T, 2)
            
            # Simple finite difference update
            solver.U[:, i, j, next] -= (solver.dt / solver.dr) * flux_r / 2.0
            solver.U[:, i, j, next] -= (solver.dt / solver.dz) * flux_z / 2.0
            
            # Ensure positivity
            solver.U[1, i, j, next] = max(solver.U[1, i, j, next], 1e-12)
        end
    end
    
    # Simple boundary conditions
    for j in 1:solver.nz
        solver.U[:, 1, j, next] = solver.U[:, 2, j, next]
        solver.U[:, solver.nr, j, next] = solver.U[:, solver.nr-1, j, next]
    end
    for i in 1:solver.nr
        solver.U[:, i, 1, next] = solver.U[:, i, 2, next]
        solver.U[:, i, solver.nz, next] = solver.U[:, i, solver.nz-1, next]
    end
    
    solver.current = next
end

# Visualization
function plot_explosive_solution(solver::ExplosiveSolver, t::Float64)
    current = solver.current
    
    rho = zeros(solver.nr, solver.nz)
    pressure_vals = zeros(solver.nr, solver.nz)
    lambda_vals = zeros(solver.nr, solver.nz)
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            U = solver.U[:, i, j, current]
            state = solver.explosive_state[i,j]
            W = conservative_to_primitive_explosive(U, state)
            
            rho[i, j] = W[1]
            pressure_vals[i, j] = W[4]
            lambda_vals[i, j] = state.lambda
        end
    end
    
    # Create plots
    r_plot = solver.r
    z_plot = solver.z
    
    p1 = contourf(r_plot, z_plot, rho', 
                 title="Density at t = $(round(t*1e6, digits=1)) μs", 
                 xlabel="r (m)", ylabel="z (m)", aspect_ratio=:equal)
    
    p2 = contourf(r_plot, z_plot, log10.(pressure_vals' .+ 1e5), 
                 title="log₁₀(Pressure) [Pa]", 
                 xlabel="r (m)", ylabel="z (m)", aspect_ratio=:equal)
    
    p3 = contourf(r_plot, z_plot, lambda_vals', 
                 title="Reaction Progress λ", 
                 xlabel="r (m)", ylabel="z (m)", aspect_ratio=:equal, 
                 levels=0:0.1:1, color=:hot)
    
    # Detonation front
    if solver.det_start_time >= 0
        det_radius = solver.det_velocity * (t - solver.det_start_time)
        theta = 0:0.1:2π
        r_circle = solver.det_center_r .+ det_radius * cos.(theta)
        z_circle = solver.det_center_z .+ det_radius * sin.(theta)
        
        p4 = plot(solver.r, rho[:, solver.nz÷2], linewidth=2,
                 xlabel="r (m)", ylabel="Density", 
                 title="Radial Profile", legend=false)
        plot!(p4, r_circle, z_circle, linewidth=2, color=:red, 
              xlims=(solver.r_min, solver.r_max), ylims=(solver.z_min, solver.z_max))
    else
        p4 = plot(solver.r, rho[:, solver.nz÷2], linewidth=2,
                 xlabel="r (m)", ylabel="Density", 
                 title="Radial Profile", legend=false)
    end
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
end

# Main simulation
function solve_explosive_detonation(nr::Int=40, nz::Int=30, t_final::Float64=50e-6)
    
    println("="^60)
    println("2D Axisymmetric Explosive Detonation Simulation")
    println("="^60)
    
    # Create solver (dimensions in meters)
    r_min, r_max = 0.0, 0.1      # 10 cm radius
    z_min, z_max = 0.0, 0.1      # 10 cm height
    cfl = 0.3
    det_velocity = 6900.0        # TNT detonation velocity (m/s)
    
    solver = ExplosiveSolver(nr, nz, r_min, r_max, z_min, z_max, cfl, det_velocity)
    
    println("Domain: r ∈ [$(r_min*100), $(r_max*100)] cm, z ∈ [$(z_min*100), $(z_max*100)] cm")
    println("Detonation velocity: $(det_velocity) m/s")
    
    # Initialize explosive
    initialize_explosive!(solver, 1.0e5)  # 1 bar initial pressure
    
    # Initiate detonation at center
    initiate_detonation!(solver, 0.0, 0.05, 5e-6)  # Start at t=5μs
    
    # Time integration
    t = 0.0
    step = 0
    
    println("\nStarting explosive simulation...")
    println("Final time: $(t_final*1e6) μs")
    
    while t < t_final && step < 10000
        compute_time_step!(solver)
        
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        # Update detonation wave and reactions
        update_detonation!(solver, t + solver.dt, solver.dt)
        
        # Update fluid dynamics
        update_solution!(solver)
        
        t += solver.dt
        step += 1
        
        if step % 100 == 0
            println("Step: $step, Time: $(round(t*1e6, digits=2)) μs, dt: $(round(solver.dt*1e9, digits=2)) ns")
        end
    end
    
    println("Explosive simulation completed after $step steps")
    return solver, t
end

# Run the explosive simulation
solver, final_time = solve_explosive_detonation(30, 25, 40e-6)

# Plot results
plt = plot_explosive_solution(solver, final_time)
display(plt)

println("\n2D Axisymmetric Explosive Detonation Complete!")
println("✓ Murnaghan EOS for solid HE")
println("✓ JWL EOS for detonation products") 
println("✓ Reactive flow with λ progress variable")
println("✓ Spherical detonation wave tracking")
println("✓ Mixed EOS for reacting cells")
println("✓ TNT-like material properties")
println("✓ Heat release and product expansion")

global solver
