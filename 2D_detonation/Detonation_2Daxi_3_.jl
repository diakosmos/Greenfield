using Plots
using LinearAlgebra

# ========== ALL STRUCT DEFINITIONS FIRST ==========

# Material types for multi-material simulation
@enum MaterialType AIR_MATERIAL EXPLOSIVE_MATERIAL

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

# Air EOS (ideal gas)
struct AirEOS <: EquationOfState
    gamma::Float64
    R::Float64  # Specific gas constant (J/kg/K)
end

# Mixed explosive state for reacting cells
mutable struct ExplosiveState
    lambda::Float64           # Reaction progress: 0=solid HE, 1=products
    eos_solid::MurnaghanEOS   # Solid HE EOS
    eos_products::JWLEOS      # Product EOS
    det_time::Float64         # Time when detonation arrived (-1 if not yet)
    reaction_rate::Float64    # Reaction rate parameter (1/s)
end

# Mixed cell state: either air or explosive
mutable struct CellState
    material::MaterialType
    
    # For explosive cells
    explosive_state::Union{ExplosiveState, Nothing}
    
    # For air cells  
    air_eos::Union{AirEOS, Nothing}
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
    
    # Cell state for each cell (air or explosive)
    cell_states::Matrix{CellState}
    
    # Detonation parameters
    det_velocity::Float64   # Detonation velocity (m/s)
    det_center_r::Float64   # Detonation initiation point
    det_center_z::Float64
    det_start_time::Float64
    
    # Grid
    r::Vector{Float64}
    z::Vector{Float64}
end

# ========== ALL FUNCTIONS START HERE ==========

# CellState constructor
function CellState(material::MaterialType)
    if material == EXPLOSIVE_MATERIAL
        # Typical HMX parameters
        murnaghan = MurnaghanEOS(4.0e9, 8.0, 1630.0, 0.0)
        jwl = JWLEOS(1.0e11, 1.0e9, 4.15, 0.95, 0.3, 1630.0, 2.0e6)
        explosive_state = ExplosiveState(0.0, murnaghan, jwl, -1.0, 5.0e5)
        return CellState(material, explosive_state, nothing)
    else  # AIR_MATERIAL
        air_eos = AirEOS(1.4, 287.0)  # Standard air
        return CellState(material, nothing, air_eos)
    end
end

# Murnaghan EOS functions
function pressure(eos::MurnaghanEOS, rho::Float64, e::Float64)
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
    if rho <= 0
        return 1.0
    end
    
    compression = rho / eos.rho0
    if compression <= 0
        return 1.0
    end
    
    dpdρ = eos.K0 * compression^(eos.K0_prime - 1.0) / eos.rho0
    c_squared = max(dpdρ, 1.0)
    
    return sqrt(c_squared)
end

# JWL EOS functions
function pressure(eos::JWLEOS, rho::Float64, e::Float64)
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
    gamma_eff = 1.4  # Reasonable approximation for products
    c_squared = gamma_eff * p / rho
    return sqrt(max(c_squared, 1.0))
end

# Air EOS functions
function pressure(eos::AirEOS, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
    u = rho_u / rho
    v = rho_v / rho
    kinetic = 0.5 * rho * (u^2 + v^2)
    e_internal = rho_E / rho - 0.5 * (u^2 + v^2)
    
    # Ideal gas: p = (γ-1)ρe
    p = (eos.gamma - 1) * rho * e_internal
    return max(p, 1e-12)
end

function sound_speed(eos::AirEOS, rho::Float64, p::Float64)
    c_squared = eos.gamma * p / rho
    return sqrt(max(c_squared, 1.0))
end

function total_energy(eos::AirEOS, rho::Float64, u::Float64, v::Float64, p::Float64)
    kinetic = 0.5 * rho * (u^2 + v^2)
    internal = p / ((eos.gamma - 1) * rho)
    return rho * (internal + 0.5 * (u^2 + v^2))
end

# Mixed EOS for reacting explosive
function pressure(state::ExplosiveState, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
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
        e_products = e_internal + state.eos_products.Q
        return pressure(state.eos_products, rho, e_products)
        
    else
        # Mixed state: pressure equilibrium assumption
        p_solid = pressure(state.eos_solid, rho, e_internal)
        e_products = e_internal + state.lambda * state.eos_products.Q
        p_products = pressure(state.eos_products, rho, e_products)
        
        # Simple mixing rule
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
        c_solid = sound_speed(state.eos_solid, rho, p)
        c_products = sound_speed(state.eos_products, rho, p)
        c_mixed_sq = (1.0 - state.lambda) * c_solid^2 + state.lambda * c_products^2
        return sqrt(max(c_mixed_sq, 1.0))
    end
end

function total_energy(state::ExplosiveState, rho::Float64, u::Float64, v::Float64, p::Float64)
    kinetic = 0.5 * rho * (u^2 + v^2)
    
    if state.lambda <= 0.0
        internal = p / ((1.4 - 1) * rho)  # Approximate
        return internal + kinetic
        
    elseif state.lambda >= 1.0
        internal = p / ((1.4 - 1) * rho) - state.eos_products.Q  # Approximate
        return internal + kinetic + state.eos_products.Q
        
    else
        e_solid = p / ((1.4 - 1) * rho)
        e_products = p / ((1.4 - 1) * rho) - state.eos_products.Q
        internal = (1.0 - state.lambda) * e_solid + state.lambda * (e_products + state.eos_products.Q)
        return internal + kinetic
    end
end

# Unified pressure function for any cell state
function pressure(cell::CellState, rho::Float64, rho_E::Float64, rho_u::Float64, rho_v::Float64)
    if cell.material == AIR_MATERIAL
        return pressure(cell.air_eos, rho, rho_E, rho_u, rho_v)
    else  # EXPLOSIVE_MATERIAL
        return pressure(cell.explosive_state, rho, rho_E, rho_u, rho_v)
    end
end

function sound_speed(cell::CellState, rho::Float64, p::Float64)
    if cell.material == AIR_MATERIAL
        return sound_speed(cell.air_eos, rho, p)
    else  # EXPLOSIVE_MATERIAL
        return sound_speed(cell.explosive_state, rho, p)
    end
end

function total_energy(cell::CellState, rho::Float64, u::Float64, v::Float64, p::Float64)
    if cell.material == AIR_MATERIAL
        return total_energy(cell.air_eos, rho, u, v, p)
    else  # EXPLOSIVE_MATERIAL
        return total_energy(cell.explosive_state, rho, u, v, p)
    end
end

# MUSCL reconstruction with slope limiting
function muscl_reconstruct(U_L::Vector{Float64}, U_C::Vector{Float64}, U_R::Vector{Float64})
    # Van Leer slope limiter
    function van_leer_limit(a, b)
        if a * b <= 0
            return 0.0
        else
            return 2 * a * b / (a + b)
        end
    end
    
    # Calculate slopes
    slope_L = zeros(4)
    slope_R = zeros(4)
    
    for i in 1:4
        delta_L = U_C[i] - U_L[i]
        delta_R = U_R[i] - U_C[i]
        
        slope_L[i] = van_leer_limit(delta_L, delta_R)
        slope_R[i] = van_leer_limit(delta_R, delta_L)
    end
    
    # Reconstruct interface values
    U_L_recon = U_C + 0.25 * slope_L
    U_R_recon = U_C - 0.25 * slope_R
    
    return U_L_recon, U_R_recon
end

# ExplosiveSolver constructor
function ExplosiveSolver(nr::Int, nz::Int, r_min::Float64, r_max::Float64,
                        z_min::Float64, z_max::Float64, cfl::Float64,
                        det_velocity::Float64, charge_radius::Float64, 
                        charge_center_r::Float64, charge_center_z::Float64)
    
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    
    r = [r_min + (i-0.5)*dr for i in 1:nr]
    z = [z_min + (j-0.5)*dz for j in 1:nz]
    
    U = zeros(4, nr, nz, 2)
    
    # Create cell states: explosive inside sphere, air outside
    cell_states = Matrix{CellState}(undef, nr, nz)
    explosive_cells = 0
    air_cells = 0
    
    for j in 1:nz, i in 1:nr
        distance = sqrt((r[i] - charge_center_r)^2 + (z[j] - charge_center_z)^2)
        
        if distance <= charge_radius
            cell_states[i,j] = CellState(EXPLOSIVE_MATERIAL)
            explosive_cells += 1
        else
            cell_states[i,j] = CellState(AIR_MATERIAL)
            air_cells += 1
        end
    end
    
    println("Multi-material setup:")
    println("  Explosive cells: $explosive_cells")
    println("  Air cells: $air_cells")
    println("  Charge radius: $(charge_radius*100) cm")
    println("  Charge center: ($(charge_center_r*100), $(charge_center_z*100)) cm")
    
    return ExplosiveSolver(nr, nz, r_min, r_max, z_min, z_max, dr, dz, cfl, 0.0,
                          U, 1, cell_states, det_velocity, 0.0, 0.0, 0.0, r, z)
end

# Initialize spherical explosive charge surrounded by air
function initialize_explosive_and_air!(solver::ExplosiveSolver, p_initial::Float64=1.0e5)
    current = solver.current
    
    rho_air = 1.225      # kg/m³ at STP
    p_air = p_initial    # Pa
    u_initial = 0.0
    v_initial = 0.0
    
    rho_he = 1630.0      # kg/m³
    p_he = p_initial     # Pa
    
    println("Initializing multi-material system:")
    println("  Air: ρ=$(rho_air) kg/m³, p=$(p_air/1e5) bar")
    println("  HE:  ρ=$(rho_he) kg/m³, p=$(p_he/1e5) bar")
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            cell = solver.cell_states[i,j]
            
            if cell.material == AIR_MATERIAL
                internal = p_air / ((cell.air_eos.gamma - 1) * rho_air)
                rho_E = rho_air * (internal + 0.5 * (u_initial^2 + v_initial^2))
                
                solver.U[:, i, j, current] = [rho_air, 
                                             rho_air * u_initial,
                                             rho_air * v_initial, 
                                             rho_E]
                                             
            else  # EXPLOSIVE_MATERIAL
                e_internal = p_he / ((1.4-1) * rho_he)  # Approximate
                rho_E = rho_he * (e_internal + 0.5 * (u_initial^2 + v_initial^2))
                
                solver.U[:, i, j, current] = [rho_he, 
                                             rho_he * u_initial,
                                             rho_he * v_initial, 
                                             rho_E]
            end
        end
    end
    
    println("Multi-material initialization complete")
end

# Initiate detonation at center
function initiate_detonation!(solver::ExplosiveSolver, r_det::Float64, z_det::Float64, current_time::Float64)
    solver.det_center_r = r_det
    solver.det_center_z = z_det
    solver.det_start_time = current_time
    
    println("Detonation initiated at (r=$(r_det), z=$(z_det)) at t=$(current_time)")
    println("Detonation velocity: $(solver.det_velocity) m/s")
end

# Update detonation wave (only affects explosive cells)
function update_detonation!(solver::ExplosiveSolver, current_time::Float64, dt::Float64)
    if solver.det_start_time < 0
        return
    end
    
    det_radius = solver.det_velocity * (current_time - solver.det_start_time)
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            cell = solver.cell_states[i,j]
            
            if cell.material == EXPLOSIVE_MATERIAL
                r = solver.r[i]
                z = solver.z[j]
                
                distance = sqrt((r - solver.det_center_r)^2 + (z - solver.det_center_z)^2)
                
                state = cell.explosive_state
                
                if distance <= det_radius && state.det_time < 0
                    state.det_time = current_time
                end
                
                if state.det_time >= 0
                    reaction_time = current_time - state.det_time
                    state.lambda = 1.0 - exp(-state.reaction_rate * reaction_time)
                    state.lambda = clamp(state.lambda, 0.0, 1.0)
                end
            end
        end
    end
end

# Conservative to primitive conversion
function conservative_to_primitive_mixed(U::Vector{Float64}, cell::CellState)
    rho = max(U[1], 1e-12)
    u_r = U[2] / rho
    u_z = U[3] / rho
    p = max(pressure(cell, rho, U[4], U[2], U[3]), 1e-12)
    return [rho, u_r, u_z, p]
end

# HLLE flux for mixed materials
function hlle_flux_mixed(U_L::Vector{Float64}, U_R::Vector{Float64}, 
                        cell_L::CellState, cell_R::CellState, direction::Int)
    
    W_L = conservative_to_primitive_mixed(U_L, cell_L)
    W_R = conservative_to_primitive_mixed(U_R, cell_R)
    
    rho_L, u_r_L, u_z_L, p_L = W_L
    rho_R, u_r_R, u_z_R, p_R = W_R
    
    c_L = sound_speed(cell_L, rho_L, p_L)
    c_R = sound_speed(cell_R, rho_R, p_R)
    
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
            cell = solver.cell_states[i,j]
            W = conservative_to_primitive_mixed(U, cell)
            rho, u_r, u_z, p = W
            
            c = sound_speed(cell, rho, p)
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

# MUSCL-HLLE update with proper axisymmetric source terms
function update_solution!(solver::ExplosiveSolver)
    current = solver.current
    next = 3 - current
    
    # Initialize next timestep
    solver.U[:, :, :, next] .= solver.U[:, :, :, current]
    
    # MUSCL-HLLE reconstruction and flux computation
    for j in 2:solver.nz-1
        for i in 2:solver.nr-1
            # r-direction update with MUSCL reconstruction
            if i >= 3 && i <= solver.nr-2
                # Get neighboring states for MUSCL
                U_LL = solver.U[:, i-2, j, current]
                U_L = solver.U[:, i-1, j, current]
                U_C = solver.U[:, i, j, current]
                U_R = solver.U[:, i+1, j, current]
                U_RR = solver.U[:, i+2, j, current]
                
                # MUSCL reconstruction at interfaces
                U_L_left, U_L_right = muscl_reconstruct(U_LL, U_L, U_C)
                U_R_left, U_R_right = muscl_reconstruct(U_L, U_C, U_R)
                
                # Flux at i-1/2 interface
                cell_L = solver.cell_states[i-1, j]
                cell_C = solver.cell_states[i, j]
                flux_L = hlle_flux_mixed(U_L_right, U_R_left, cell_L, cell_C, 1)
                
                # Flux at i+1/2 interface
                U_RR_left, U_RR_right = muscl_reconstruct(U_C, U_R, U_RR)
                cell_R = solver.cell_states[i+1, j]
                flux_R = hlle_flux_mixed(U_R_left, U_RR_left, cell_C, cell_R, 1)
                
                # Update with flux difference
                solver.U[:, i, j, next] -= (solver.dt / solver.dr) * (flux_R - flux_L)
            else
                # Fallback to first-order at boundaries
                U_L = solver.U[:, max(i-1, 1), j, current]
                U_R = solver.U[:, min(i+1, solver.nr), j, current]
                cell_L = solver.cell_states[max(i-1, 1), j]
                cell_R = solver.cell_states[min(i+1, solver.nr), j]
                
                flux = hlle_flux_mixed(U_L, U_R, cell_L, cell_R, 1)
                solver.U[:, i, j, next] -= (solver.dt / solver.dr) * flux / 2.0
            end
            
            # z-direction update with MUSCL reconstruction
            if j >= 3 && j <= solver.nz-2
                U_BB = solver.U[:, i, j-2, current]
                U_B = solver.U[:, i, j-1, current]
                U_C = solver.U[:, i, j, current]
                U_T = solver.U[:, i, j+1, current]
                U_TT = solver.U[:, i, j+2, current]
                
                U_B_left, U_B_right = muscl_reconstruct(U_BB, U_B, U_C)
                U_T_left, U_T_right = muscl_reconstruct(U_B, U_C, U_T)
                
                cell_B = solver.cell_states[i, j-1]
                cell_C = solver.cell_states[i, j]
                flux_B = hlle_flux_mixed(U_B_right, U_T_left, cell_B, cell_C, 2)
                
                U_TT_left, U_TT_right = muscl_reconstruct(U_C, U_T, U_TT)
                cell_T = solver.cell_states[i, j+1]
                flux_T = hlle_flux_mixed(U_T_left, U_TT_left, cell_C, cell_T, 2)
                
                solver.U[:, i, j, next] -= (solver.dt / solver.dz) * (flux_T - flux_B)
            else
                U_B = solver.U[:, i, max(j-1, 1), current]
                U_T = solver.U[:, i, min(j+1, solver.nz), current]
                cell_B = solver.cell_states[i, max(j-1, 1)]
                cell_T = solver.cell_states[i, min(j+1, solver.nz)]
                
                flux = hlle_flux_mixed(U_B, U_T, cell_B, cell_T, 2)
                solver.U[:, i, j, next] -= (solver.dt / solver.dz) * flux / 2.0
            end
            
            # Axisymmetric source terms
            r = solver.r[i]
            if r > 1e-10  # Avoid division by zero
                U_current = solver.U[:, i, j, current]
                rho = U_current[1]
                rho_ur = U_current[2]
                rho_uz = U_current[3]
                rho_E = U_current[4]
                
                cell = solver.cell_states[i, j]
                p = pressure(cell, rho, rho_E, rho_ur, rho_uz)
                
                # Axisymmetric source: -1/r * [0, p, 0, 0]
                source = [0.0, p, 0.0, 0.0]
                solver.U[:, i, j, next] -= solver.dt * source / r
            end
            
            # Ensure positivity
            solver.U[1, i, j, next] = max(solver.U[1, i, j, next], 1e-12)
            solver.U[4, i, j, next] = max(solver.U[4, i, j, next], 1e-12)
        end
    end
    
    # Boundary conditions
    for j in 1:solver.nz
        # r boundaries
        solver.U[:, 1, j, next] = solver.U[:, 2, j, next]
        solver.U[:, solver.nr, j, next] = solver.U[:, solver.nr-1, j, next]
        
        # Set radial velocity to zero at centerline
        solver.U[2, 1, j, next] = 0.0
    end
    
    for i in 1:solver.nr
        # z boundaries (outflow)
        solver.U[:, i, 1, next] = solver.U[:, i, 2, next]
        solver.U[:, i, solver.nz, next] = solver.U[:, i, solver.nz-1, next]
    end
    
    solver.current = next
end

# Add source terms for chemical reaction
function add_reaction_source!(solver::ExplosiveSolver, dt::Float64)
    current = solver.current
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            cell = solver.cell_states[i,j]
            
            if cell.material == EXPLOSIVE_MATERIAL
                state = cell.explosive_state
                
                if state.det_time >= 0  # Cell is reacting
                    # Get current state
                    U = solver.U[:, i, j, current]
                    rho = U[1]
                    
                    # Rate of reaction progress
                    lambda_old = state.lambda
                    dlambda_dt = state.reaction_rate * (1.0 - state.lambda)
                    
                    # Energy release rate
                    dE_dt = rho * state.eos_products.Q * dlambda_dt
                    
                    # Add energy source
                    solver.U[4, i, j, current] += dt * dE_dt
                end
            end
        end
    end
end

# Visualization with robust error handling
function plot_explosive_solution(solver::ExplosiveSolver, t::Float64)
    current = solver.current
    
    rho = zeros(solver.nr, solver.nz)
    pressure_vals = zeros(solver.nr, solver.nz)
    lambda_vals = zeros(solver.nr, solver.nz)
    
    for j in 1:solver.nz
        for i in 1:solver.nr
            U = solver.U[:, i, j, current]
            cell = solver.cell_states[i,j]
            
            # Check for problematic values
            if any(.!isfinite.(U)) || U[1] <= 0
                println("WARNING: Non-finite or invalid values at cell ($i,$j): $U")
                # Use safe values based on material type
                if cell.material == AIR_MATERIAL
                    rho[i, j] = 1.225
                    pressure_vals[i, j] = 1e5
                else
                    rho[i, j] = 1630.0
                    pressure_vals[i, j] = 1e5
                end
                lambda_vals[i, j] = 0.0
                continue
            end
            
            try
                W = conservative_to_primitive_mixed(U, cell)
                rho[i, j] = W[1]
                pressure_vals[i, j] = W[4]
                
                # Lambda only for explosive cells
                if cell.material == EXPLOSIVE_MATERIAL
                    lambda_vals[i, j] = cell.explosive_state.lambda
                else
                    lambda_vals[i, j] = 0.0  # Air doesn't react
                end
                
            catch e
                println("Error converting to primitive at ($i,$j): $e")
                if cell.material == AIR_MATERIAL
                    rho[i, j] = 1.225
                    pressure_vals[i, j] = 1e5
                else
                    rho[i, j] = 1630.0
                    pressure_vals[i, j] = 1e5
                end
                lambda_vals[i, j] = 0.0
            end
        end
    end
    
    # Check data quality
    rho_finite = all(isfinite.(rho))
    p_finite = all(isfinite.(pressure_vals))
    lambda_finite = all(isfinite.(lambda_vals))
    
    println("Solution diagnostics:")
    println("  Density range: $(extrema(rho))")
    println("  Pressure range: $(extrema(pressure_vals))")
    println("  Lambda range: $(extrema(lambda_vals))")
    println("  All finite: ρ=$rho_finite, p=$p_finite, λ=$lambda_finite")
    
    if !rho_finite || !p_finite
        println("ERROR: Non-finite values in solution!")
        return plot(title="Error: Non-finite values in solution", 
                   xlabel="Simulation failed", ylabel="Check console output")
    end
    
    # Clean data for plotting
    rho = clamp.(rho, 100.0, 5000.0)
    pressure_vals = clamp.(pressure_vals, 1e4, 1e12)
    lambda_vals = clamp.(lambda_vals, 0.0, 1.0)
    
    try
        # Create plots
        r_plot = solver.r * 100  # Convert to cm
        z_plot = solver.z * 100  # Convert to cm
        
        p1 = contourf(r_plot, z_plot, rho', 
                     title="Density [kg/m³] at t = $(round(t*1e6, digits=1)) μs", 
                     xlabel="r (cm)", ylabel="z (cm)", aspect_ratio=:equal, c=:viridis)
        
        p2 = contourf(r_plot, z_plot, log10.(pressure_vals' .+ 1e4), 
                     title="log₁₀(Pressure) [Pa]", 
                     xlabel="r (cm)", ylabel="z (cm)", aspect_ratio=:equal, c=:plasma)
        
        p3 = contourf(r_plot, z_plot, lambda_vals',
                     title="Reaction Progress λ", 
                     xlabel="r (cm)", ylabel="z (cm)", aspect_ratio=:equal, c=:hot,
                     levels=0:0.1:1)
        
        # Velocity magnitude
        u_mag = zeros(solver.nr, solver.nz)
        for j in 1:solver.nz
            for i in 1:solver.nr
                U = solver.U[:, i, j, current]
                if U[1] > 0
                    u_r = U[2] / U[1]
                    u_z = U[3] / U[1]
                    u_mag[i, j] = sqrt(u_r^2 + u_z^2)
                end
            end
        end
        
        p4 = contourf(r_plot, z_plot, u_mag', 
                     title="Velocity Magnitude [m/s]", 
                     xlabel="r (cm)", ylabel="z (cm)", aspect_ratio=:equal, c=:turbo)
        
        return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
        
    catch e
        println("Error creating plots: $e")
        return plot(title="Error creating visualization", 
                   xlabel="Check console for details")
    end
end

# Run simulation
function run_simulation(; nr=50, nz=50, 
                       r_max=0.05, z_max=0.05,  # 5cm domain
                       charge_radius=0.01,      # 1cm charge
                       det_velocity=8000.0,     # 8 km/s
                       cfl=0.3,
                       t_final=2e-5,            # 20 μs
                       plot_interval=5e-6)      # Plot every 5 μs
    
    println("Setting up 2D axisymmetric explosive simulation...")
    println("Domain: $(r_max*100) cm × $(z_max*100) cm")
    println("Grid: $nr × $nz cells")
    println("Charge radius: $(charge_radius*100) cm")
    
    # Create solver
    solver = ExplosiveSolver(nr, nz, 0.0, r_max, -z_max/2, z_max/2, cfl,
                           det_velocity, charge_radius, 0.0, 0.0)
    
    # Initialize
    initialize_explosive_and_air!(solver)
    
    # Start simulation
    t = 0.0
    step = 0
    next_plot_time = 0.0
    plots_made = 0
    
    println("\nStarting simulation...")
    println("Final time: $(t_final*1e6) μs")
    
    # Initiate detonation at center immediately
    initiate_detonation!(solver, 0.0, 0.0, 0.0)
    
    while t < t_final
        # Update detonation wave
        update_detonation!(solver, t, solver.dt)
        
        # Compute time step
        compute_time_step!(solver)
        
        # Don't overshoot final time
        if t + solver.dt > t_final
            solver.dt = t_final - t
        end
        
        # Update solution
        update_solution!(solver)
        
        # Add reaction source terms
        add_reaction_source!(solver, solver.dt)
        
        t += solver.dt
        step += 1
        
        # Progress output
        if step % 100 == 0
            println("Step $step: t = $(round(t*1e6, digits=2)) μs, dt = $(round(solver.dt*1e9, digits=2)) ns")
        end
        
        # Plot at specified intervals
        if t >= next_plot_time && plots_made < 5
            println("Creating plot at t = $(round(t*1e6, digits=1)) μs")
            plt = plot_explosive_solution(solver, t)
            display(plt)
            next_plot_time += plot_interval
            plots_made += 1
        end
        
        # Safety check
        if step > 100000
            println("Maximum steps reached, stopping simulation")
            break
        end
    end
    
    println("\nSimulation complete!")
    println("Final time: $(round(t*1e6, digits=2)) μs")
    println("Total steps: $step")
    
    # Final plot
    plt = plot_explosive_solution(solver, t)
    display(plt)
    
    return solver
end

# Example usage
println("2D Axisymmetric Explosive Detonation Solver")
println("MUSCL-HLLE scheme with Murnaghan + JWL EOS")
println("========================================")

# Uncomment to run simulation
solver = run_simulation()