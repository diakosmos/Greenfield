using Plots

# Define flux functions for Euler equations
function compute_flux(u)
    ρ = u[1]
    m = u[2]
    E = u[3]

    v = m/ρ
    p = 0.4 * (E - 0.5*ρ*v^2) # γ = 1.4 for ideal gas 

    f1 = m 
    f2 = m*v + p
    f3 = (E + p)*v 

    return [f1, f2, f3]
end

# CESE space-time flux integration
function compute_flux_integral(ul, ur, Δt, Δx)
    # compute average state
    u_avg = 0.5*(ul + ur)

    # compute flux at average state 
    f_avg = compute_flux(u_avg) 

    # time step coefficient 
    τ = Δt/Δx
    
    return f_avg*τ 
end

# Main CESE solver 
function solve_sod(nx=200, cfl=0.5, t_end=0.2)
    # domain setup 
    x_min, x_max = -0.5, 0.5 
    Δx = (x_max - x_min)/nx 
    x = range(x_min + Δx/2, x_max - Δx/2, length=nx) 

    # initial conditions (Sod problem)
    ρ = zeros(nx)
    v = zeros(nx) 
    p = zeros(nx) 

    for i in 1:nx
        if x[i] < 0
            ρ[i] = 1.0
            p[i] = 1.0
        else
            ρ[i] = 0.125
            p[i] = 0.1
        end
    end

    # convert to conserved variables 
    E = @. p/0.4 + 0.5*ρ*v^2 
    u = hcat(ρ, ρ.*v, E) 

    # time stepping 
    t = 0.0
    Δt = cfl*Δx/10.0 # initial time step (will be adjusted) 

    #while t < t_end 
    for j in 1:500
        # Adjust time step if needed 
        if t + Δt > t_end 
            Δt = t_end - t 
        end

        # store old solution
        u_old = copy(u)

        # update solution using CESE method 
        for i in 2:nx-1
            # compute space-time flux integrals 
            flux_left = compute_flux_integral(u_old[i-1,:],u_old[i,:], Δt, Δx)
            flux_right = compute_flux_integral(u_old[i,:],u_old[i+1,:], Δt, Δx)

            # update solution 
            u[i,:] = u_old[i,:] - (flux_right - flux_left)
        end

        # update time 
        t += Δt 

        # apply boundary conditions (transmissive) 
        u[1,:] = u[2,:]
        u[nx,:] = u[nx-1,:]
    end

    return x, u 
end

# run simulation and plot results 
x, u = solve_sod()

# extract primitive variables 
ρ = u[:,1]
v = @. u[:,2]/u[:,1]
p = @. 0.4*(u[:,3] - 0.5*u[:,1]*v^2)

# create plots 
p1 = plot(x, ρ, label="Density",title="Sod problem at t=0.2", xlabel="x", ylabel="ρ")
p2 = plot(x, v, label="Velocity", xlabel="x", ylabel="v")
p3 = plot(x, p, label="Pressure", xlabel="x", ylabel="p")
fig = plot(p1,p2,p3,layout=(3,1)) #,size=(800,600))
display(fig)