#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Import libaries
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plevels = np.asarray([0.3,0.17,0.12,0.11,0.09,0.07,0.05,0.02,0.0,-0.002]) [::-1]
ptext   = np.asarray(['a','b','c','d','e','f','g','h','i','j']) [::-1]
vortlevels = np.asarray([5.0,4.0,3.0,2.0,1.0,0.5,0,-0.5,-1.0,-2.0,-3.0]) [::-1]
vorttext   = np.asarray(['a','b','c','d','e','f','g','h','i','j','k']) [::-1]
psilevels = np.asarray([0.1175,0.115,0.11,0.1,9e-2,7e-2,5e-2,3e-2,1e-2,1e-4,1e-5,1e-10,0,-1e-6,-1e-5,-5e-5,-1e-4,-2.5e-4,-5e-4,-1e-3,-1.5e-3]) [::-1]
psitext   = np.asarray(['a','','b','','c','','d','','e','','f','','g','','h','','i','','j']) [::-1]

# https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_label_demo.html
fmt = {}


N15pressure = np.load('exports/N15/pressure.npz')
N15psi = np.load('exports/N15/psi.npz')
N15uv = np.load('exports/N15/uv.npz')
N15vorticity = np.load('exports/N15/vorticity.npz')

N31pressure = np.load('exports/N31/pressure.npz')
N31psi = np.load('exports/N31/psi.npz')
N31uv = np.load('exports/N31/uv.npz')
N31vorticity = np.load('exports/N31/vorticity.npz')

N47pressure = np.load('exports/N47/pressure.npz')
N47psi = np.load('exports/N47/psi.npz')
N47uv = np.load('exports/N47/uv.npz')
N47vorticity = np.load('exports/N47/vorticity.npz')

N55pressure = np.load('exports/N55/pressure.npz')
N55psi = np.load('exports/N55/psi.npz')
N55uv = np.load('exports/N55/uv.npz')
N55vorticity = np.load('exports/N55/vorticity.npz')

N63pressure = np.load('exports/N63/pressure.npz')
N63psi = np.load('exports/N63/psi.npz')
N63uv = np.load('exports/N63/uv.npz')
N63vorticity = np.load('exports/N63/vorticity.npz')

refx05 = np.genfromtxt("./ref_data/ref_x05.txt", skip_header = 1)


# Check if main figure folder exists
if not os.path.exists('figures'):
    os.makedirs('figures')

## Define text sizes for **SAVED** pictures (texpsize -- text export size)
texpsize    = [26,28,30]
levels      = 15



## Graphing Parameters
SMALL_SIZE  = texpsize[0]
MEDIUM_SIZE = texpsize[1]
BIGGER_SIZE = texpsize[2]

plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False


fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N15pressure['x_pres'],N15pressure['y_pres'],N15pressure['pres'],levels=plevels,colors='black')
for l,s in zip(C.levels, ptext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_pressure15.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N15vorticity['x_vort'],N15vorticity['y_vort'],N15vorticity['vort'],levels=vortlevels,colors='black')
for l,s in zip(C.levels, vorttext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_vorticity15.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N15psi['x_psi'],N15psi['y_psi'],N15psi['psi'], levels=psilevels,colors='black')
for l,s in zip(C.levels, psitext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels[::2],inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_psi15.png", bbox_inches='tight')
plt.close(fig)
fmt = {}


fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N31pressure['x_pres'],N31pressure['y_pres'],N31pressure['pres'],levels=plevels,colors='black')
for l,s in zip(C.levels, ptext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_pressure31.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N31vorticity['x_vort'],N31vorticity['y_vort'],N31vorticity['vort'],levels=vortlevels,colors='black')
for l,s in zip(C.levels, vorttext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_vorticity31.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N31psi['x_psi'],N31psi['y_psi'],N31psi['psi'], levels=psilevels,colors='black')
for l,s in zip(C.levels, psitext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels[::2],inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_psi31.png", bbox_inches='tight')
plt.close(fig)
fmt = {}




fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N47pressure['x_pres'],N47pressure['y_pres'],N47pressure['pres'],levels=plevels,colors='black')
for l,s in zip(C.levels, ptext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_pressure47.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N47vorticity['x_vort'],N47vorticity['y_vort'],N47vorticity['vort'],levels=vortlevels,colors='black')
for l,s in zip(C.levels, vorttext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_vorticity47.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N47psi['x_psi'],N47psi['y_psi'],N47psi['psi'], levels=psilevels,colors='black')
for l,s in zip(C.levels, psitext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels[::2],inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_psi47.png", bbox_inches='tight')
plt.close(fig)
fmt = {}



fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N55pressure['x_pres'],N55pressure['y_pres'],N55pressure['pres'],levels=plevels,colors='black')
for l,s in zip(C.levels, ptext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_pressure55.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N55vorticity['x_vort'],N55vorticity['y_vort'],N55vorticity['vort'],levels=vortlevels,colors='black')
for l,s in zip(C.levels, vorttext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_vorticity55.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N55psi['x_psi'],N55psi['y_psi'],N55psi['psi'], levels=psilevels,colors='black')
for l,s in zip(C.levels, psitext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels[::2],inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_psi55.png", bbox_inches='tight')
plt.close(fig)
fmt = {}




fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N63pressure['x_pres'],N63pressure['y_pres'],N63pressure['pres'],levels=plevels,colors='black')
for l,s in zip(C.levels, ptext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_pressure63.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N63vorticity['x_vort'],N63vorticity['y_vort'],N63vorticity['vort'],levels=vortlevels,colors='black')
for l,s in zip(C.levels, vorttext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels,inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_vorticity63.png", bbox_inches='tight')
plt.close(fig)
fmt = {}

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
C = ax[0,0].contour(N63psi['x_psi'],N63psi['y_psi'],N63psi['psi'], levels=psilevels,colors='black')
for l,s in zip(C.levels, psitext):
    fmt[l] = s
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
ax[0,0].clabel(C,C.levels[::2],inline=True,fmt=fmt,fontsize=SMALL_SIZE)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
plt.savefig("figures/cont_psi63.png", bbox_inches='tight')
plt.close(fig)
fmt = {}



fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15pressure['y_pres'][:,8],N15pressure['pres'][:,8], color='red', linewidth=2, label=r'$N=15$') # xconst
ax[0,0].plot(N31pressure['y_pres'][:,16],N31pressure['pres'][:,16], color='blue', linewidth=2, label=r'$N=31$') # xconst
ax[0,0].plot(N47pressure['y_pres'][:,24],N47pressure['pres'][:,24], color='green', linewidth=2, label=r'$N=47$') # xconst
ax[0,0].plot(N55pressure['y_pres'][:,28],N55pressure['pres'][:,28], color='cyan', linewidth=2, label=r'$N=55$') # xconst
ax[0,0].plot(N63pressure['y_pres'][:,32],N63pressure['pres'][:,32], color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$p\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/x05pressure.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15pressure['x_pres'][8,:],N15pressure['pres'][8,:], color='red', linewidth=2, label=r'$N=15$') # yconst
ax[0,0].plot(N31pressure['x_pres'][16,:],N31pressure['pres'][16,:], color='blue', linewidth=2, label=r'$N=31$') # yconst
ax[0,0].plot(N47pressure['x_pres'][24,:],N47pressure['pres'][24,:], color='green', linewidth=2, label=r'$N=47$') # yconst
ax[0,0].plot(N55pressure['x_pres'][28,:],N55pressure['pres'][28,:], color='cyan', linewidth=2, label=r'$N=55$') # yconst
ax[0,0].plot(N63pressure['x_pres'][32,:],N63pressure['pres'][32,:], color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$p\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/y05pressure.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)




N15_xvort = np.zeros(N15vorticity['vort'].shape[0])
for i in range(N15vorticity['vort'].shape[0]):
    N15_xvort[i] = np.interp([0.5], N15vorticity['x_vort'][i,:], N15vorticity['vort'][i,:])
N31_xvort = np.zeros(N31vorticity['vort'].shape[0])
for i in range(N31vorticity['vort'].shape[0]):
    N31_xvort[i] = np.interp([0.5], N31vorticity['x_vort'][i,:], N31vorticity['vort'][i,:])
N47_xvort = np.zeros(N47vorticity['vort'].shape[0])
for i in range(N47vorticity['vort'].shape[0]):
    N47_xvort[i] = np.interp([0.5], N47vorticity['x_vort'][i,:], N47vorticity['vort'][i,:])
N55_xvort = np.zeros(N55vorticity['vort'].shape[0])
for i in range(N55vorticity['vort'].shape[0]):
    N55_xvort[i] = np.interp([0.5], N55vorticity['x_vort'][i,:], N55vorticity['vort'][i,:])
N63_xvort = np.zeros(N63vorticity['vort'].shape[0])
for i in range(N63vorticity['vort'].shape[0]):
    N63_xvort[i] = np.interp([0.5], N63vorticity['x_vort'][i,:], N63vorticity['vort'][i,:])

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15vorticity['y_vort'][:,0],N15_xvort, color='red', linewidth=2, label=r'$N=15$') # xconst
ax[0,0].plot(N31vorticity['y_vort'][:,0],N31_xvort, color='blue', linewidth=2, label=r'$N=31$') # xconst
ax[0,0].plot(N47vorticity['y_vort'][:,0],N47_xvort, color='green', linewidth=2, label=r'$N=47$') # xconst
ax[0,0].plot(N55vorticity['y_vort'][:,0],N55_xvort, color='cyan', linewidth=2, label=r'$N=55$') # xconst
ax[0,0].plot(N63vorticity['y_vort'][:,0],N63_xvort, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$\omega\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/x05vorticity.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

N15_yvort = np.zeros(N15vorticity['vort'].shape[1])
for i in range(N15vorticity['vort'].shape[1]):
    N15_yvort[i] = np.interp([0.5], N15vorticity['y_vort'][:,i], N15vorticity['vort'][:,i])
N31_yvort = np.zeros(N31vorticity['vort'].shape[1])
for i in range(N31vorticity['vort'].shape[1]):
    N31_yvort[i] = np.interp([0.5], N31vorticity['y_vort'][:,i], N31vorticity['vort'][:,i])
N47_yvort = np.zeros(N47vorticity['vort'].shape[1])
for i in range(N47vorticity['vort'].shape[1]):
    N47_yvort[i] = np.interp([0.5], N47vorticity['y_vort'][:,i], N47vorticity['vort'][:,i])
N55_yvort = np.zeros(N55vorticity['vort'].shape[1])
for i in range(N55vorticity['vort'].shape[1]):
    N55_yvort[i] = np.interp([0.5], N55vorticity['y_vort'][:,i], N55vorticity['vort'][:,i])
N63_yvort = np.zeros(N63vorticity['vort'].shape[1])
for i in range(N63vorticity['vort'].shape[1]):
    N63_yvort[i] = np.interp([0.5], N63vorticity['y_vort'][:,i], N63vorticity['vort'][:,i])

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15vorticity['x_vort'][0,:],N15_yvort, color='red', linewidth=2, label=r'$N=15$') # yconst
ax[0,0].plot(N31vorticity['x_vort'][0,:],N31_yvort, color='blue', linewidth=2, label=r'$N=31$') # yconst
ax[0,0].plot(N47vorticity['x_vort'][0,:],N47_yvort, color='green', linewidth=2, label=r'$N=47$') # yconst
ax[0,0].plot(N55vorticity['x_vort'][0,:],N55_yvort, color='cyan', linewidth=2, label=r'$N=55$') # yconst
ax[0,0].plot(N63vorticity['x_vort'][0,:],N63_yvort, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$\omega\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/y05vorticity.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)


N15_xu = np.zeros(N15uv['u'].shape[0])
N15_xv = np.zeros(N15uv['v'].shape[0])
for i in range(N15uv['u'].shape[0]):
    N15_xu[i] = np.interp([0.5], N15uv['x_uv'][i,:], N15uv['u'][i,:])
for i in range(N15uv['v'].shape[0]):
    N15_xv[i] = np.interp([0.5], N15uv['x_uv'][i,:], N15uv['v'][i,:])
N31_xu = np.zeros(N31uv['u'].shape[0])
N31_xv = np.zeros(N31uv['v'].shape[0])
for i in range(N31uv['u'].shape[0]):
    N31_xu[i] = np.interp([0.5], N31uv['x_uv'][i,:], N31uv['u'][i,:])
for i in range(N31uv['v'].shape[0]):
    N31_xv[i] = np.interp([0.5], N31uv['x_uv'][i,:], N31uv['v'][i,:])
N47_xu = np.zeros(N47uv['u'].shape[0])
N47_xv = np.zeros(N47uv['v'].shape[0])
for i in range(N47uv['u'].shape[0]):
    N47_xu[i] = np.interp([0.5], N47uv['x_uv'][i,:], N47uv['u'][i,:])
for i in range(N47uv['v'].shape[0]):
    N47_xv[i] = np.interp([0.5], N47uv['x_uv'][i,:], N47uv['v'][i,:])
N55_xu = np.zeros(N55uv['u'].shape[0])
N55_xv = np.zeros(N55uv['v'].shape[0])
for i in range(N55uv['u'].shape[0]):
    N55_xu[i] = np.interp([0.5], N55uv['x_uv'][i,:], N55uv['u'][i,:])
for i in range(N55uv['v'].shape[0]):
    N55_xv[i] = np.interp([0.5], N55uv['x_uv'][i,:], N55uv['v'][i,:])
N63_xu = np.zeros(N63uv['u'].shape[0])
N63_xv = np.zeros(N63uv['v'].shape[0])
for i in range(N63uv['u'].shape[0]):
    N63_xu[i] = np.interp([0.5], N63uv['x_uv'][i,:], N63uv['u'][i,:])
for i in range(N63uv['v'].shape[0]):
    N63_xv[i] = np.interp([0.5], N63uv['x_uv'][i,:], N63uv['v'][i,:])


fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15uv['y_uv'][:,0],N15_xu, color='red', linewidth=2, label=r'$N=15$') # xconst
ax[0,0].plot(N31uv['y_uv'][:,0],N31_xu, color='blue', linewidth=2, label=r'$N=31$') # xconst
ax[0,0].plot(N47uv['y_uv'][:,0],N47_xu, color='green', linewidth=2, label=r'$N=47$') # xconst
ax[0,0].plot(N55uv['y_uv'][:,0],N55_xu, color='cyan', linewidth=2, label=r'$N=55$') # xconst
ax[0,0].plot(N63uv['y_uv'][:,0],N63_xu, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$u\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/x05u.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15uv['y_uv'][:,0],N15_xv, color='red', linewidth=2, label=r'$N=15$') # xconst
ax[0,0].plot(N31uv['y_uv'][:,0],N31_xv, color='blue', linewidth=2, label=r'$N=31$') # xconst
ax[0,0].plot(N47uv['y_uv'][:,0],N47_xv, color='green', linewidth=2, label=r'$N=47$') # xconst
ax[0,0].plot(N55uv['y_uv'][:,0],N55_xv, color='cyan', linewidth=2, label=r'$N=55$') # xconst
ax[0,0].plot(N63uv['y_uv'][:,0],N63_xv, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$v\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/x05v.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

N15_yu = np.zeros(N15uv['u'].shape[1])
N15_yv = np.zeros(N15uv['v'].shape[1])
for i in range(N15uv['u'].shape[1]):
    N15_yu[i] = np.interp([0.5], N15uv['y_uv'][:,i], N15uv['u'][:,i])
for i in range(N15uv['v'].shape[1]):
    N15_yv[i] = np.interp([0.5], N15uv['y_uv'][:,i], N15uv['v'][:,i])
N31_yu = np.zeros(N31uv['u'].shape[1])
N31_yv = np.zeros(N31uv['v'].shape[1])
for i in range(N31uv['u'].shape[1]):
    N31_yu[i] = np.interp([0.5], N31uv['y_uv'][:,i], N31uv['u'][:,i])
for i in range(N31uv['v'].shape[1]):
    N31_yv[i] = np.interp([0.5], N31uv['y_uv'][:,i], N31uv['v'][:,i])
N47_yu = np.zeros(N47uv['u'].shape[1])
N47_yv = np.zeros(N47uv['v'].shape[1])
for i in range(N47uv['u'].shape[1]):
    N47_yu[i] = np.interp([0.5], N47uv['y_uv'][:,i], N47uv['u'][:,i])
for i in range(N47uv['v'].shape[1]):
    N47_yv[i] = np.interp([0.5], N47uv['y_uv'][:,i], N47uv['v'][:,i])
N55_yu = np.zeros(N55uv['u'].shape[1])
N55_yv = np.zeros(N55uv['v'].shape[1])
for i in range(N55uv['u'].shape[1]):
    N55_yu[i] = np.interp([0.5], N55uv['y_uv'][:,i], N55uv['u'][:,i])
for i in range(N55uv['v'].shape[1]):
    N55_yv[i] = np.interp([0.5], N55uv['y_uv'][:,i], N55uv['v'][:,i])
N63_yu = np.zeros(N63uv['u'].shape[1])
N63_yv = np.zeros(N63uv['v'].shape[1])
for i in range(N63uv['u'].shape[1]):
    N63_yu[i] = np.interp([0.5], N63uv['y_uv'][:,i], N63uv['u'][:,i])
for i in range(N63uv['v'].shape[1]):
    N63_yv[i] = np.interp([0.5], N63uv['y_uv'][:,i], N63uv['v'][:,i])

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15uv['x_uv'][0,:],N15_yu, color='red', linewidth=2, label=r'$N=15$') # yconst
ax[0,0].plot(N31uv['x_uv'][0,:],N31_yu, color='blue', linewidth=2, label=r'$N=31$') # yconst
ax[0,0].plot(N47uv['x_uv'][0,:],N47_yu, color='green', linewidth=2, label=r'$N=47$') # yconst
ax[0,0].plot(N55uv['x_uv'][0,:],N55_yu, color='cyan', linewidth=2, label=r'$N=55$') # yconst
ax[0,0].plot(N63uv['x_uv'][0,:],N63_yu, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$u\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/y05u.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N15uv['x_uv'][0,:],N15_yv, color='red', linewidth=2, label=r'$N=15$') # yconst
ax[0,0].plot(N31uv['x_uv'][0,:],N31_yv, color='blue', linewidth=2, label=r'$N=31$') # yconst
ax[0,0].plot(N47uv['x_uv'][0,:],N47_yv, color='green', linewidth=2, label=r'$N=47$') # yconst
ax[0,0].plot(N55uv['x_uv'][0,:],N55_yv, color='cyan', linewidth=2, label=r'$N=55$') # yconst
ax[0,0].plot(N63uv['x_uv'][0,:],N63_yv, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$v\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
plt.savefig("figures/y05v.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)


#%%------------------------------------
#pressure plot with ref

refx05 = np.genfromtxt("./ref_data/ref_x05.txt", skip_header = 1)
yx05 =refx05[:,0]
ux05 =refx05[:,2]
px05 =refx05[:,3]   
omegax05 =refx05[:,4]

refy05 = np.genfromtxt("./ref_data/ref_y05.txt", skip_header = 1)
xy05 =refy05[:,0]
vy05 =refy05[:,2]
py05 =refy05[:,3]
omegay05 =refy05[:,4]

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63pressure['y_pres'][:,32],N63pressure['pres'][:,32], color='orange', linewidth=2, label=r'$N=63$')
ax[0,0].scatter(yx05,px05, s=200, facecolors='none', edgecolors='k', label=r'Ref') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$p\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/x05pressure_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63pressure['x_pres'][32,:],N63pressure['pres'][32,:], color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].scatter(xy05,py05, s=200, facecolors='none', edgecolors='k', label=r'Ref') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$p\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/y05pressure_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)


#vorticity plots with ref

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63vorticity['y_vort'][:,0],N63_xvort, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].scatter(yx05,omegax05, s=200, facecolors='none', edgecolors='k', label=r'Ref')
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$\omega\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/x05vorticity_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)


fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63vorticity['x_vort'][0,:],N63_yvort, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].scatter(xy05,omegay05, s=200, facecolors='none', edgecolors='k', label=r'Ref')
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$\omega\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/y05vorticity_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)


#Velocity at x =0.5
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63uv['y_uv'][:,0],N63_xu, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].scatter(yx05,ux05, s=200, facecolors='none', edgecolors='k', label=r'Ref')
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$u\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/x05u_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63uv['y_uv'][:,0],N63_xv, color='orange', linewidth=2, label=r'$N=63$') # xconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$y\,\,[-]$")
ax[0,0].set_ylabel(r"$v\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/x05v_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)



fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63uv['x_uv'][0,:],N63_yu, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$u\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/y05u_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,16))
ax[0,0].plot(N63uv['x_uv'][0,:],N63_yv, color='orange', linewidth=2, label=r'$N=63$') # yconst
ax[0,0].scatter(xy05,vy05, s=200, facecolors='none', edgecolors='k', label=r'Ref')
ax[0,0].set_xlim(0, 1)
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$v\,\,[-]$")
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
#ax[0,0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="lower left", mode="expand", ncol=4, framealpha=1.0).get_frame().set_edgecolor('k') # bbox_to_anchor = (x0,y0,width,height)
ax[0,0].legend()
plt.savefig("figures/y05v_withRef.png", bbox_inches='tight')
plt.rc('legend', fontsize=SMALL_SIZE)
plt.close(fig)

