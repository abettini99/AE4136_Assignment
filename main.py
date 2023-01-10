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
import os
from lib.incidence_mat import *
from lib.hodge_mat import *
from lib.grid_gen import *
from lib.utils import *

def main(N_input) -> None:
    ## Set up problem variables
    L       = float(1.0)
    Re      = float(1000)   # Reynolds number
    N       = int(N_input)  	# mesh cells in x- and y-direction
    tol     = float(1e-6)
    diff_Ut = np.inf

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
                u_norm[idx] = V_wall_bot*hface#-V_wall_bot*hface # No need for negatives as these are integrals defined in our predefined direction.
            elif face.type == 'west':
                u_norm[idx] = U_wall_right*hface#-U_wall_right*hface

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
                u_pres[idx] = U_wall_top*hface #-U_wall_top*hface # No need for negatives as these are integrals defined in our predefined direction.
            elif face.type == 'east':
                u_pres[idx] = V_wall_left*hface
            elif face.type == 'south':
                u_pres[idx] = U_wall_bot*hface
            elif face.type == 'west':
                u_pres[idx] = V_wall_right*hface#-V_wall_right*hface
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
    xi   = 0
    t    = 0
    while (diff_Ut>tol):
        # Store old xi
        xiold = xi

        # Calculate vorticity (/circulation)
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

        # Update time
        t           = t + dt

        # Every other 1000 iterations check whether you approach steady state and
        # check whether you satsify conservation of mass. The largest rate at which
        # mass is created or destroyed is denoted my 'maxdiv'. This number should
        # be close to machine precision.

        if (iter%15000==0):
            maxdiv   = max(abs(DIV@u+u_norm))
            diff_Ut  = max(abs(u-uold))/dt
            diff_xit = max(abs(xi-xiold))/dt

            if not iter == 0: print("\n")
            print(f"t \t\t| max(div_U) \t\t| max(Delta_U/Delta_t) \t| max(Delta_xi/Delta_t)  \t\t(N = {N_input}, dt = {dt:.2e}, h_min = {h_min:.2e})")
            print(f"{t:.3e} \t| {maxdiv:.7e} \t| {diff_Ut:.7e} \t| {diff_xit:.7e}")

        elif (iter%1000 == 0):
            maxdiv   = max(abs(DIV@u+u_norm))
            diff_Ut  = max(abs(u-uold))/dt
            diff_xit = max(abs(xi-xiold))/dt
            print(f"{t:.3e} \t| {maxdiv:.7e} \t| {diff_Ut:.7e} \t| {diff_xit:.7e}")

        iter += 1
    print("tolerance reached")
    print("")

    #### ==================== ####
    #### Create contours data ####
    #### ==================== ####

    ## Pressure contours
    # Setup pressure arrays for plots
    x_psurf = np.zeros((N+2,N+2))
    y_psurf = np.zeros((N+2,N+2))
    p_psurf = np.zeros((N+2,N+2))
    u_psurf = np.zeros(p.shape)
    v_psurf = np.zeros(p.shape)
    p_mask  = np.ones((p.shape),dtype=bool) # We need to mask some gridpoints due to dual boundary vertices counting as internal points

    # We create the mask by going through all faces, which contain information on whether they are boundary or not
    for face in domain.Dface_array:
        if face.type != 'internal': # if not internal

            idx0,idx1   = face.vertices_idx[0],face.vertices_idx[1]
            vert0,vert1 = domain.Dvert_array[idx0],domain.Dvert_array[idx1]
            Tidx0,Tidx1 = vert0.Tidx, vert1.Tidx # grab vertex indices

            if vert0.type == 'internal': p_mask[Tidx0] = False # Basically, all dual vertices are either internal or virtual. The virtual ones are the ones in the corner entries
            if vert1.type == 'internal': p_mask[Tidx1] = False # So basically we need to account for the fact that the boundary vertices are also considered internal for this assignment.

    # We need to subtract dynamic pressure from p to get static pressure.
    for cell in domain.Dcell_array:
        # we basically need information at the cell level to deduce information at the vertex level at this point.
        idx  = cell.Tidx
        idxN = cell.faces_idx[dir2D.N]; faceN = domain.Dface_array[idxN]
        idxE = cell.faces_idx[dir2D.E]; faceE = domain.Dface_array[idxE]
        idxS = cell.faces_idx[dir2D.S]; faceS = domain.Dface_array[idxS]
        idxW = cell.faces_idx[dir2D.W]; faceW = domain.Dface_array[idxW]

        idxN0,idxN1   = faceN.vertices_idx[0],faceN.vertices_idx[1]
        idxE0,idxE1   = faceE.vertices_idx[0],faceE.vertices_idx[1]
        idxS0,idxS1   = faceS.vertices_idx[0],faceS.vertices_idx[1]
        idxW0,idxW1   = faceW.vertices_idx[0],faceW.vertices_idx[1]

        vertN0,vertN1 = domain.Dvert_array[idxN0],domain.Dvert_array[idxN1]
        vertE0,vertE1 = domain.Dvert_array[idxE0],domain.Dvert_array[idxE1]
        vertS0,vertS1 = domain.Dvert_array[idxS0],domain.Dvert_array[idxS1]
        vertW0,vertW1 = domain.Dvert_array[idxW0],domain.Dvert_array[idxW1]

        TidxN0,TidxN1 = vertN0.Tidx, vertN1.Tidx
        TidxE0,TidxE1 = vertE0.Tidx, vertE1.Tidx
        TidxS0,TidxS1 = vertS0.Tidx, vertS1.Tidx
        TidxW0,TidxW1 = vertW0.Tidx, vertW1.Tidx

        if faceN.type == 'internal':
            if vertN0.type == 'internal': u_psurf[TidxN0] += u[faceN.Tidx]/cell.h[dir2D.x]
            if vertN1.type == 'internal': u_psurf[TidxN1] += u[faceN.Tidx]/cell.h[dir2D.x]
        elif faceN.type == 'north':
            if vertN0.type == 'internal': u_psurf[TidxN0] += U_wall_top
            if vertN1.type == 'internal': u_psurf[TidxN1] += U_wall_top

        if faceS.type == 'internal':
            if vertS0.type == 'internal': u_psurf[TidxS0] += u[faceS.Tidx]/cell.h[dir2D.x]
            if vertS1.type == 'internal': u_psurf[TidxS1] += u[faceS.Tidx]/cell.h[dir2D.x]
        elif faceS.type == 'south':
            if vertS0.type == 'internal': u_psurf[TidxS0] += U_wall_bot
            if vertS1.type == 'internal': u_psurf[TidxS1] += U_wall_bot

        if faceE.type == 'internal':
            if vertE0.type == 'internal': v_psurf[TidxE0] += u[faceE.Tidx]/cell.h[dir2D.y]
            if vertE1.type == 'internal': v_psurf[TidxE1] += u[faceE.Tidx]/cell.h[dir2D.y]
        elif faceE.type == 'east':
            if vertE0.type == 'internal': v_psurf[TidxE0] += V_wall_right
            if vertE1.type == 'internal': v_psurf[TidxE1] += V_wall_right

        if faceW.type == 'internal':
            if vertW0.type == 'internal': v_psurf[TidxW0] += u[faceW.Tidx]/cell.h[dir2D.y]
            if vertW1.type == 'internal': v_psurf[TidxW1] += u[faceW.Tidx]/cell.h[dir2D.y]
        elif faceW.type == 'west':
            if vertW0.type == 'internal': v_psurf[TidxW0] += V_wall_left
            if vertW1.type == 'internal': v_psurf[TidxW1] += V_wall_left

    # Each vertex surrounded by four cells! so each vertex counted 4 times as we loop over all cells! We take the average of the four cells
    u_psurf = u_psurf/4
    v_psurf = v_psurf/4

    # We fill in the surfaces now
    rho = 1
    V2_psurf = u_psurf**2 + v_psurf**2
    x_psurf[:], y_psurf[:]  = x, x
    y_psurf                 = y_psurf.transpose()
    p_psurf[1:-1,1:-1]      = p[p_mask].reshape(N,N) - 0.5*rho*V2_psurf[p_mask].reshape(N,N)
    p_psurf[0,:]            = p_psurf[1,:]
    p_psurf[-1,:]           = p_psurf[-2,:]
    p_psurf[:,0]            = p_psurf[:,1]
    p_psurf[:,-1]           = p_psurf[:,-2]

    # Note the near singular matrix allows for a solution to be solved. The constant factor needs to be removed to match the solutions by Botella
    xmid_idx = int(N/2+1) if N%2 != 0 else int(N/2)
    ymid_idx = int(N/2+1) if N%2 != 0 else int(N/2)
    p_sub    = p_psurf[xmid_idx,ymid_idx]

    # Plot
    # plevels = np.sort( np.asarray([0.3,0.17,0.12,0.11,0.09,0.07,0.05,0.02,0.0,-0.002]) )
    # plt.contour(x_psurf,y_psurf,p_psurf - p_sub,levels=plevels)
    # plt.show()


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
    # vortlevels = np.sort( np.asarray([5.0,4.0,3.0,2.0,1.0,0.5,0,-0.5,-1.0,-2.0,-3.0]) )
    # plt.contour(x_vortsurf,y_vortsurf,vort_vortsurf,levels=vortlevels)
    # plt.show()

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
    # psilevels = np.sort( np.asarray([0.1175,0.115,0.11,0.1,9e-2,7e-2,5e-2,3e-2,1e-2,1e-4,1e-5,1e-10,0,-1e-6,-1e-5,-5e-5,-1e-4,-2.5e-4,-5e-4,-1e-3,-1.5e-3]) )
    # plt.contour(x_psisurf, y_psisurf, psi_psisurf-psi_sub, levels=psilevels)
    # plt.show()

    ## u, v
    x_uvsurf = np.zeros((N+3,N+3))
    y_uvsurf = np.zeros((N+3,N+3))
    x_uvsurf[:,0], x_uvsurf[:,1:-1], x_uvsurf[:,-1] = x[0], (x[1:]+x[:-1])/2, x[-1]
    y_uvsurf[:,0], y_uvsurf[:,1:-1], y_uvsurf[:,-1] = x[0], (x[1:]+x[:-1])/2, x[-1]
    y_uvsurf = y_uvsurf.transpose()
    u_tmp = np.zeros(((N+1)*(N+1)))
    v_tmp = np.zeros(((N+1)*(N+1)))
    u_uvsurf = np.zeros((N+3,N+3))
    v_uvsurf = np.zeros((N+3,N+3))

    for cell in domain.Dcell_array:
        idx  = cell.Tidx
        idxN = cell.faces_idx[dir2D.N]; faceN = domain.Dface_array[idxN]
        idxE = cell.faces_idx[dir2D.E]; faceE = domain.Dface_array[idxE]
        idxS = cell.faces_idx[dir2D.S]; faceS = domain.Dface_array[idxS]
        idxW = cell.faces_idx[dir2D.W]; faceW = domain.Dface_array[idxW]

        if faceN.type == 'internal':    u_tmp[idx] += u[faceN.Tidx]/cell.h[dir2D.x]/2
        elif faceN.type == 'north':     u_tmp[idx] += U_wall_top/2

        if faceS.type == 'internal':    u_tmp[idx] += u[faceS.Tidx]/cell.h[dir2D.x]/2
        elif faceS.type == 'south':     u_tmp[idx] += U_wall_bot/2

        if faceE.type == 'internal':    v_tmp[idx] += u[faceE.Tidx]/cell.h[dir2D.y]/2
        elif faceE.type == 'east':      v_tmp[idx] += V_wall_right/2

        if faceW.type == 'internal':    v_tmp[idx] += u[faceW.Tidx]/cell.h[dir2D.y]/2
        elif faceW.type == 'west':      v_tmp[idx] += V_wall_left/2

    u_uvsurf[1:-1,1:-1] = u_tmp.reshape(N+1,N+1)
    v_uvsurf[1:-1,1:-1] = v_tmp.reshape(N+1,N+1)

    u_uvsurf[0,:], u_uvsurf[-1,:], u_uvsurf[:,0], u_uvsurf[:,-1] = U_wall_bot, U_wall_top, U_wall_left, U_wall_right
    v_uvsurf[0,:], v_uvsurf[-1,:], v_uvsurf[:,0], v_uvsurf[:,-1] = V_wall_bot, V_wall_top, V_wall_left, V_wall_right
    # for i in range(u_uvsurf.shape[0]):
    #     print(np.round(v_uvsurf[i],3))

    # plt.contour(x_uvsurf, y_uvsurf, u_uvsurf, levels=20)
    # plt.show()
    #
    # plt.contour(x_uvsurf, y_uvsurf, v_uvsurf, levels=20)
    # plt.show()

    #### =========== ####
    #### Export data ####
    #### =========== ####
    folder_name = f'N{N}'
    if not os.path.exists('exports/{}'.format(folder_name)):
        os.makedirs('exports/{}'.format(folder_name))
    np.savez('exports/{}/pressure.npz'.format(folder_name), x_pres=x_psurf, y_pres=y_psurf, pres=p_psurf-p_sub)
    np.savez('exports/{}/vorticity.npz'.format(folder_name), x_vort=x_vortsurf, y_vort=y_vortsurf, vort=vort_vortsurf)
    np.savez('exports/{}/psi.npz'.format(folder_name), x_psi=x_psisurf, y_psi=y_psisurf, psi=psi_psisurf-psi_sub)
    np.savez('exports/{}/uv.npz'.format(folder_name), x_uv=x_uvsurf, y_uv=y_uvsurf, u=u_uvsurf, v=v_uvsurf)


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

    # Check if main export folder exists
    if not os.path.exists('exports'):
        os.makedirs('exports')

    main(15)
    main(31)
    main(47)
    main(55)
    main(63)
