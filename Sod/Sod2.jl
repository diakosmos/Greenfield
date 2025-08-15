using Plots 

NXD = 1000

q = zeros(3,NXD)
qn = zeros(3,NXD)
qx = zeros(3,NXD)
qt = zeros(3,NXD)
s  = zeros(3,NXD)
vxl = zeros(3)
vxr = zeros(3)
xx = zeros(NXD)

nx = 11 # must be odd 
it = 0 # 100
dt = 0.4e-2
dx = 0.1e-1 
ga = 1.4 
rhol = 1.0
ul = 0.0
pl = 1.0
rhor = 0.125
ur = 0.0
pr = 0.1
ia = 1
#
nx1 = nx + 1 
nx2 = nx ÷ 2
hdt = dt/2.0
tt = hdt * it 
qdt = dt/4.0
hdx = dx/2.0
qdx = dx/4.0
dtx = dt/dx 
a1 = ga - 1.0
a2 = 3.0 - ga 
a3 = a2/2.0 
a4 = 1.5 * a1 
u2l = rhol * ul 
u3l = pl/a1 + 0.5*rhol*ul^2 
u2r = rhor*ur 
u3r = pr/a1 + 0.5*rhor*ur^2 
for j in 1:nx2 
    q[1,j] = rhol 
    q[2,j] = u2l 
    q[3,j] = u3l 
    q[1,nx2+j] = rhor
    q[2,nx2+j] = u2r 
    q[3,nx2+j] = u3r 
    for i in 1:3
        qx[i,j] = 0.0 
        qx[i,nx2+j] = 0.0
    end
end
# write tt,it ia, nx,, dt, dx, ga, rhol ul, pl, rhor, ur, pr 
for i in 1:it 
    local m = nx + i - (i÷2)*2 # 102, 101, 102, 101, 102, 101, ...
    for j in 1:m 
        w2 = q[2,j]/q[1,j]
        w3 = q[3,j]/q[1,j]
        f21 = -a3*w2^2 
        f22 = a2*w2 
        f31 = a1*w2^3 - ga*w2*w3 
        f32 = ga*w3 - a4*w2^2 
        f33 = ga*w2 
        q[1,j] = -qx[2,j]
        qt[2,j] = -(f21*qx[1,j] + f22*qx[2,j] + a1*qx[3,j])
        qt[3,j] = -(f31*qx[1,j] + f32*qx[3,j] + f33*qx[3,j])
        s[1,j] = qdx*qx[1,j] + dtx*(q[2,j] + qdt*qt[2,j])
        s[2,j] = qdx*qx[2,j] + dtx*(f21*(q[1,j] + qdt*qt[1,j]) +
                f22*(q[2,j] + qdt*qt[2,j]) + a1*(q[3,j] + qdt*qt[3,j]))
        s[3,j] = qdx*qx[3,j] + dtx*(f31*(q[1,j] + qdt*qt[1,j]) + 
                f32*(q[2,j] + qdt*qt[2,j]) + f33*(q[3,j] + qdt*qt[3,j]))
    end
    if i%2 == 0 # i == (i÷2)*2
        for k in 1:3
            qx[k,nx1] = qx[k,nx]
            qn[k,1] = q[k,1]
            qn[k,nx1] = q[k,nx]
        end
    end
    j1 = 1 - i + (i÷2)*2 
    local mm = m - 1
    for j in 1:mm 
        for k in 1:3
            qn[k,j+j1] = 0.5*(q[k,j] + q[k,j+1] + s[k,j] - s[k,j+1])
            vxl[k] = (qn[k,j+j1] - q[k,j] - hdt*qt[k,j])/hdx 
            vxr[k] = (q[k,j+1] + hdt*qt[k,j+1] - qn[k,j+j1])/hdx 
            qx[k,j+j1] = (vxl[k]*(abs(vxr[k]))^ia + vxr[k]*(abs(vxl[k]))^ia) / 
                    ((abs(vxl[k]))^ia + (abs(vxr[k]))^ia + 1.0e-60)
        end
    end
    m = nx1 - i + (i÷2)*2 
    for j in 1:m 
        for k in 1:3
            q[k,j] = qn[k,j]
        end
    end
end

m = nx1 - it + (it÷2)*2 
mm = m - 1
xx[1] = -0.5*dx*mm 
for j in 1:mm 
    xx[j+1] = xx[j] + dx 
end
for j in 1:m
    x = q[2,j]/q[1,j]
    y = a1*(q[3,j] - 0.5*x^2*q[1,j])
    z = x/sqrt(ga*y/q[1,j])
    print(xx[j],"  ",q[1,j],"  ",x,"  ",y,"  ",z,"\n")
end
