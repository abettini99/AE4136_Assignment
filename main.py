#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 14 12:25:58 2020
@author: mgerritsma

Modified: 2020-01-01
Modified: Andrea Bettini (4656970)
          Elrawy Soliman (4684443)
"""

## Import libaries
from scipy.sparse import diags
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import math as m
import numba as nb
from lib.incidence_mat import *
from lib.hodge_mat import *
from lib.grid_gen import *

def main() -> None:
    ## Set up problem variables
    L       = float(1.0)
    Re      = float(1000)   # Reynolds number
    N       = int(31)  	# mesh cells in x- and y-direction
    tol     = float(1e-6)
    diff    = np.inf

    ## Set up spaces
    u   = np.zeros([2*N*(N+1)],   dtype = np.float64)
    p   = np.zeros([N*N+4*N],     dtype = np.float64)
    tx  = np.zeros([N+1],         dtype = np.float64)   # grid points on primal grid
    x   = np.zeros([N+2],         dtype = np.float64)   # grid points on dual grid
    th  = np.zeros([N],           dtype = np.float64)   # mesh width primal grid
    h   = np.zeros([N+1],         dtype = np.float64)   # mesh width dual grid

    ## Generation of a non-uniform grid
    domain = Grid2D(N,N,'cosine')
    domain.scale_x_domain(L)
    domain.scale_y_domain(L)

    tx, _, th, _    = domain.get_primal_lengths()
    x, _, h, _      = domain.get_dual_lengths()

    h_min           = domain.hmin
    dt              = min(h_min,0.5*Re*h_min**2)            # dt for stable integration
    dt              = 5.*dt                                 # conservative timestep -> adjustable

    #  Boundary conditions for the lid driven acvity test case
    U_wall_top      = -1
    U_wall_bot      = 0
    U_wall_left     = 0
    U_wall_right    = 0
    V_wall_top      = 0
    V_wall_bot      = 0
    V_wall_left     = 0
    V_wall_right    = 0

    # Set up the sparse incidence matrix tE21. Use the orientations described
    # in the assignment.
    # Make sure to use sparse matrices to avoid memory problems
    tE21, ntE21 = construct_tE21_2D(domain)

    #  Insert the normal boundary conditions and split of the vector u_norm
    M = ntE21.shape[1]
    u_norm = np.empty((M))

    for face in domain.Pface_array:
        if face.type != 'internal':
            idx = face.Tidx

            idx0, idx1 = face.vertices_idx[0], face.vertices_idx[1]
            x0, x1 = domain.Pvert_array[idx0].coordinates[dir2D.x], domain.Pvert_array[idx1].coordinates[dir2D.x]
            y0, y1 = domain.Pvert_array[idx0].coordinates[dir2D.y], domain.Pvert_array[idx1].coordinates[dir2D.y]
            hface = m.sqrt( (x0-x1)**2 + (y0-y1)**2 )

            if face.type == 'north':
                u_norm[idx] = V_wall_top*hface
            elif face.type == 'east':
                u_norm[idx] = U_wall_left*hface
            elif face.type == 'south':
                u_norm[idx] = V_wall_bot*hface
            elif face.type == 'west':
                u_norm[idx] = U_wall_right*hface

    u_norm = ntE21@u_norm

    #  Set up the sparse, inner-oriented  incidence matrix E10
    E10 = construct_E10_2D(domain)

    #  Set up the (extended) sparse, inner-oriented incidence matrix E21
    E21, nE21 = construct_E21_2D(domain)

    #  Set up the outer-oriented incidence matrix tE10
    tE10 = E21.transpose()

    #  Split off the prescribed tangential velocity and store this in
    #  the vector u_pres
    M = nE21.shape[1]
    u_pres = np.empty((M))

    for face in domain.Dface_array:
        if face.type != 'internal':
            idx = face.Tidx

            idx0, idx1 = face.vertices_idx[0], face.vertices_idx[1]
            x0, x1 = domain.Dvert_array[idx0].coordinates[dir2D.x], domain.Dvert_array[idx1].coordinates[dir2D.x]
            y0, y1 = domain.Dvert_array[idx0].coordinates[dir2D.y], domain.Dvert_array[idx1].coordinates[dir2D.y]
            hface = m.sqrt( (x0-x1)**2 + (y0-y1)**2 )

            if face.type == 'north':
                u_pres[idx] = -U_wall_top*hface
            elif face.type == 'east':
                u_pres[idx] = V_wall_left*hface
            elif face.type == 'south':
                u_pres[idx] = U_wall_bot*hface
            elif face.type == 'west':
                u_pres[idx] = -V_wall_right*hface
    u_pres = nE21@u_pres

    #  Set up the Hodge matrices Ht11 and H1t1
    H1t1, Ht11 = construct_H11_2D(domain)

    #  Set up the Hodge matrix Ht02
    Ht02 = construct_Ht02_2D(domain)

    A = tE21@Ht11@E10
    A = A.tocsc()
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition

    u_pres_vort = Ht02@u_pres
    temp = H1t1@tE10@Ht02@u_pres
    u_pres = temp

    VLaplace = H1t1@tE10@Ht02@E21
    DIV = tE21@Ht11

    ux_xi = np.zeros(((N+1)*(N+1)), dtype = np.float64)
    uy_xi = np.zeros(((N+1)*(N+1)), dtype = np.float64)
    convective = np.zeros((2*N*(N+1)), dtype = np.float64)

    iter = 0
    while (diff>tol):
        xi = Ht02@E21@u + u_pres_vort

        # Using JIT from numba to speedup this component
        convective  = construct_convective(N, u, ux_xi, uy_xi, xi, h, convective, U_wall_bot, V_wall_left, U_wall_top, V_wall_right)

        # Set up the right hand side for the equation for the pressure
        rhs_Pois    = DIV@( u/dt - convective - VLaplace@u/Re - u_pres/Re) + u_norm/dt

        # Solve for the pressure
        p           = LU.solve(rhs_Pois)

        # Store the velocity from the previous time level in the vector uold
        uold        = u

        # Update the velocity field
        u           = u - dt*(convective + E10@p + (VLaplace@u)/Re + u_pres/Re)

        # Every other 1000 iterations check whether you approach steady state and
        # check whether you satsify conservation of mass. The largest rate at which
        # mass is created or destroyed is denoted my 'maxdiv'. This number should
        # be close to machine precision.

        if (iter%15000==0):
            maxdiv  = max(abs(DIV@u+u_norm))
            diff    = max(abs(u-uold))/dt

            if not iter == 0: print("\n")
            print("max(divU) \t| max(DeltaU)")
            print(f"{maxdiv:.7e} \t| {diff:.7e}")

        elif (iter%1000 == 0):
            maxdiv  = max(abs(DIV@u+u_norm))
            diff    = max(abs(u-uold))/dt
            print(f"{maxdiv:.7e} \t| {diff:.7e}")

        iter += 1

    #### ============= ####
    #### Plot contours ####
    #### ============= ####

    ## Pressure contours
    # Setup pressure arrays for plots
    x_psurf = np.zeros((N+2,N+2))
    y_psurf = np.zeros((N+2,N+2))
    p_psurf = np.zeros((N+2,N+2))
    p_mask  = np.ones((p.shape),dtype=bool) # We need to mask some gridpoints due to dual boundary vertices counting as internal points

    # We create the mask by going through all faces, which contain information on whether they are boundary or not
    for face in domain.Dface_array:
        if face.type != 'internal':

            idx0,idx1   = face.vertices_idx[0],face.vertices_idx[1]
            vert0,vert1 = domain.Dvert_array[idx0],domain.Dvert_array[idx1]
            Tidx0,Tidx1 = vert0.Tidx, vert1.Tidx

            if vert0.type == 'internal': p_mask[Tidx0] = False # Virtual corner cells exists
            if vert1.type == 'internal': p_mask[Tidx1] = False

    # We fill in the surfaces now
    x_psurf[:], y_psurf[:]  = x, x
    y_psurf                 = y_psurf.transpose()
    p_psurf[1:-1,1:-1]      = p[p_mask].reshape(N,N)
    p_psurf[0,:]            = p_psurf[1,:]
    p_psurf[-1,:]           = p_psurf[-2,:]
    p_psurf[:,0]            = p_psurf[0,:1]
    p_psurf[:,-1]           = p_psurf[:,-2]

    # Note the near singular matrix allows for a solution to be solved. The constant factor needs to be removed to match the solutions by Botella
    xmid_idx = int(N/2+1) if N%2 != 0 else int(N/2)
    ymid_idx = int(N/2+1) if N%2 != 0 else int(N/2)
    p_sub    = p_psurf[xmid_idx,ymid_idx]

    # Plot
    plevels = np.sort( np.asarray([0.3,0.17,0.12,0.11,0.09,0.07,0.05,0.02,0.0,-0.002]) )
    plt.contour(x_psurf,y_psurf,p_psurf - p_sub,levels=plevels)
    plt.show()


    ## Vorticity contours
    # Setup vorticity arrays for plots
    vort = xi
    x_vortsurf = np.zeros((N+1,N+1))
    y_vortsurf = np.zeros((N+1,N+1))
    x_vortsurf[:] = tx
    y_vortsurf[:] = tx
    y_vortsurf = y_vortsurf.transpose()
    vort_vortsurf = np.zeros((N+1,N+1))
    vort_vortsurf = vort.reshape(N+1,N+1)

    # Plot
    vortlevels = np.sort( np.asarray([5.0,4.0,3.0,2.0,1.0,0.5,0,-0.5,-1.0,-2.0,-3.0]) )
    plt.contour(x_vortsurf,y_vortsurf,vort_vortsurf,levels=vortlevels)
    plt.show()

    ## Stream function contours
    # Define least-square problem to solve for
    def min_prob(psi):
        return Ht11@u - tE10@psi # by definition: (u v) = grad_perp(psi). This is to be solved on the primal grid.

    # Setup  stream arrays for plots
    x0 = np.zeros((tE10.shape[1]))
    upperbound = np.ones((tE10.shape[1]))*0.5
    lowerbound = -np.ones((tE10.shape[1]))*0.5
    psi = opt.least_squares(min_prob,x0,bounds=(lowerbound,upperbound))
    psi = psi.x

    x_psisurf = np.zeros((N+1,N+1))
    y_psisurf = np.zeros((N+1,N+1))
    x_psisurf[:] = tx
    y_psisurf[:] = tx
    y_psisurf = y_psisurf.transpose()
    psi_psisurf = np.zeros((N+1,N+1))
    psi_psisurf = psi.reshape(N+1,N+1)
    psi_sub = psi_psisurf[0,0]

    # Plot
    psilevels = np.sort( np.asarray([0.1175,0.115,0.11,0.1,9e-2,7e-2,5e-2,3e-2,1e-2,1e-4,1e-5,1e-10,0,-1e-6,-1e-5,-5e-5,-1e-4,-2.5e-4,-5e-4,-1e-3,-1.5e-3]) )
    plt.contour(x_psisurf, y_psisurf, psi_psisurf-psi_sub, levels=psilevels)
    plt.show()

@nb.jit(nopython=True)
def construct_convective(N, u, ux_xi, uy_xi, xi, h, convective, U_wall_bot, V_wall_left, U_wall_top, V_wall_right):
    for i in range(N+1):
        for j in range(N+1):
            k = j + i*(N+1);
            if j==0:
                ux_xi[k] = U_wall_bot*xi[i+j*(N+1)]
                uy_xi[k] = V_wall_left*xi[j+i*(N+1)]
            elif j==N:
                ux_xi[k] = U_wall_top*xi[i+j*(N+1)]
                uy_xi[k] = V_wall_right*xi[j+i*(N+1)]
            else:
                ux_xi[k] = (u[i+j*(N+1)] + u[i+(j-1)*(N+1)]) * xi[i+j*(N+1)]/(2.*h[i])                       # Klopt
                uy_xi[k] = (u[N*(N+1)+j+i*N] + u[N*(N+1)+j-1+i*N]) * xi[j+i*(N+1)]/(2.*h[i])

    for  i in range(N):
        for j in range(N+1):
            convective[j+i*(N+1)] = -(uy_xi[j+i*(N+1)]+uy_xi[j+(i+1)*(N+1)])*h[j]/2.
            convective[N*(N+1)+i+j*N] = (ux_xi[j+i*(N+1)]+ux_xi[j+(i+1)*(N+1)])*h[j]/2.

    return convective

if __name__ == "__main__":
    main()
