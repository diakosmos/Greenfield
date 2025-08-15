using Plots
using Printf
global N = 1000
#=
This program is a direct literal translation from FORTRAN 77 to Julia of the code in Sin-Chung Chang's paper (Chung, 1995) in Journal of Computational Physics 119, 295-324.
=#

# "Allocating" arrays
q  = zeros(3,N)
qn = zeros(3,N)
qx = zeros(3,N)
qt = zeros(3,N)
s  = zeros(3,N)
vxl = zeros(3)
vxr = zeros(3)
xx = zeros(N)

it = 100
dt = 4.0e-3
dx = 1.0e-2 
ga = 1.4 
rhol = 1.0
ul = 0.0
pl = 1.0
rhor = 0.125
ur = 0.0
pr = 0.1
ia = 1

hdt = dt/2
tt = hdt*it 
qdt = dt/4
hdx = dx/2
qdx = dx/4
dtx = dt/dx 
a1 = ga - 1 
a2 = 3 - ga 
a3 = a2/2
a4 = 1.5*a1
q[1,1] = rhol 
q[2,1] * rhol*ul
q[3,1] = pl/a1 + 0.5*rhol*ul^2
itp = it + 1
for j in 1:itp 
    q[1,j+1] = rhor 
    q[2,j+1] = rhor*ur 
    q[3,j+1] = pr/a1 + 0.5*rhor*ur^2 
    for i in 1:3 
        qx[i,j] = 0.0
    end
end

fh8 = open("for008","a")
#write(fh8) tt, it, ia 
@printf fh8 " t = %.3g  it = %.3g  ia = %.3g\n" tt it ia 
@printf fh8 "dt = %.3g  dx = %.3g  gamma = %.3g\n" dt dx ga 
@printf fh8 "rhol = %.3g  ul = %.3g  pl = %.3g\n" rhol ul pl
@printf fh8 "rhor = %.3g  ur = %.3g  pr = %.3g\n" rhor ur pr

m = 2
for i in 1:it
    global m
    for j in 1:m
        w2 = q[2,j]/q[1,j]
        w3 = q[3,j]/q[1,j]
        f21 = -a3*w2^2
        f22 = a2*w2 
        f31 = a1*w2^3 - ga*w2*w3 
        f32 = ga*w3 - a4*w2^2 
        f33 = ga*w2 
        qt[1,j] = -qx[2,j]
        qt[2,j] = -(f21*qx[1,j] + f22*qx[2,j] + a1*qx[3,j])
        qt[3,j] = -(f31*qx[1,j] + f32*qx[2,j] + f33*qx[3,j])
        s[1,j] = qdx*qx[1,j] + dtx*(q[2,j] + qdt*qt[2,j])
        s[2,j] = qdx*qx[2,j] + dtx*(
            f21*(q[1,j] + qdt*qt[1,j]) + f22*(q[2,j] + qdt*qt[2,j]) + a1*(q[3,j] + qdt*qt[3,j])
        )
        s[3,j] = qdx*qx[3,j] + dtx*(
            f31*(q[1,j] + qdt*qt[1,j]) + f32*(q[2,j] + qdt*qt[2,j]) + f33*(q[3,j] + qdt*qt[3,j])
        )
    end
    mm = m - 1
    for j in 1:mm
        for k in 1:3
            qn[k,j+1] = 0.5 * (q[k,j] + q[k,j+1] + s[k,j] - s[k,j+1])
            vxl[k] = (qn[k,j+1] - q[k,j] - hdt*qt[k,j]) / hdx
            vxr[k] = (q[k,j+1] + hdt*qt[k,j+1] - qn[k,j+1]) / hdx
            qx[k,j+1] = (vxl[k]* abs(vxr[k])^ia + vxr[k]* abs(vxl[k])^ia) /
                (abs(vxl[k])^ia + abs(vxr[k])^ia + 1.0e-60)
        end
    end
    for j in 1:m 
        for k in 1:3
            q[k,j] = qn[k,j]
        end
    end
    m += 1
end
t2 = dx*itp 
xx[1] = -0.5*t2 
for j in 1:itp 
    xx[j+1] = xx[j] + dx 
end
for j in 1:m 
    x = q[2,j] / q[1,j] 
    z = a1*(q[3,j] - 0.5*x^2*q[1,j])
    @printf fh8 " x = %.3g  rho = %.3g  u = %.3g   p = %.3g\n" xx[j] q[1,j] x z
end 

close(fh8)