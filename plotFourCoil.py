#!/usr/bin/env python3
from simsopt.geo.magneticfieldclasses import ToroidalField
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart
from poincareplot import compute_field_lines
import matplotlib.pyplot as plt
from math import pi
import numpy as np
############################################
spp           = 150    # number of phi steps for one stellarator revolution
nperiods      = 250     # how often to go around the stellarator
batch_size    = 5      # how many field lines are traced at the same time
max_thickness = 0.04   # how far out from the magnetic axis are we going
delta         = 0.001 # distance between starting points of fieldlines
magnetic_axis_radius = 0.983
qvfilename    = 'CaryHanson'
ResultsFolder = 'Results'
############################################
coils     = [CurveHelical(300, 2, 5, 2, 1., 0.3) for i in range(2)]
coils[0].set_dofs(np.concatenate(([0,0],[0,0])))
coils[1].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
currents  = [-3.07e5,3.07e5]
Bhelical  = BiotSavart(coils, currents)
Btoroidal = ToroidalField(1.0,1.0)
BField    = Bhelical+Btoroidal

######### COMPUTE POINCARE PLOT ############
rphiz, xyz, absB, phi_no_mod = compute_field_lines(BField, nperiods=nperiods, batch_size=batch_size, magnetic_axis_radius=magnetic_axis_radius, max_thickness=max_thickness, delta=delta, steps_per_period=spp)
nparticles = rphiz.shape[0]

######## PLOT ABSOLUTE VALUE OF B ##########
modBdata = np.hstack((phi_no_mod[:, None], absB.T))[0:(10*spp)]
plt.figure()
for i in range(min(modBdata.shape[1]-1, 10)):
    plt.plot(modBdata[:, 0], modBdata[:, i+1], zorder=100-i)
plt.savefig(ResultsFolder+'/'+qvfilename + "absB.png", dpi=300)
plt.close()

########### SAVE POINCARE PLOT ############
NFP=5
for k in range(4):
    plt.figure()
    for i in range(nparticles):
        plt.scatter(rphiz[i, range(k * spp//(NFP*4), nperiods*spp, spp), 0], rphiz[i, range(k * spp//(NFP*4), nperiods*spp, spp), 2], s=0.1)
    plt.savefig(ResultsFolder+'/'+qvfilename + f"poincare_{k}.png", dpi=300)
    plt.close()

############### 3D PLOT ##################
data0 = np.zeros((nperiods, nparticles*2))
data1 = np.zeros((nperiods, nparticles*2))
data2 = np.zeros((nperiods, nparticles*2))
data3 = np.zeros((nperiods, nparticles*2))
for i in range(nparticles):
    data0[:, 2*i+0] = rphiz[i, range(0, nperiods*spp, spp), 0]
    data0[:, 2*i+1] = rphiz[i, range(0, nperiods*spp, spp), 2]
    data1[:, 2*i+0] = rphiz[i, range(1*spp//(NFP*4), nperiods*spp, spp), 0]
    data1[:, 2*i+1] = rphiz[i, range(1*spp//(NFP*4), nperiods*spp, spp), 2]
    data2[:, 2*i+0] = rphiz[i, range(2*spp//(NFP*4), nperiods*spp, spp), 0]
    data2[:, 2*i+1] = rphiz[i, range(2*spp//(NFP*4), nperiods*spp, spp), 2]
    data3[:, 2*i+0] = rphiz[i, range(3*spp//(NFP*4), nperiods*spp, spp), 0]
    data3[:, 2*i+1] = rphiz[i, range(3*spp//(NFP*4), nperiods*spp, spp), 2]

import mayavi.mlab as mlab
mlab.options.offscreen = True # Show on screen or not
fig = mlab.figure(bgcolor=(1,1,1))
for count, coil in enumerate(BField.coils):
	if count==0:
		mlab.plot3d(coil.gamma()[:, 0], coil.gamma()[:, 1], coil.gamma()[:, 2], color=(1., 0., 0.), tube_radius=0.02)
	elif count==1:
		mlab.plot3d(coil.gamma()[:, 0], coil.gamma()[:, 1], coil.gamma()[:, 2], color=(0., 0.5, 1.), tube_radius=0.02)

colors = [
    (0.298, 0.447, 0.690), (0.866, 0.517, 0.321), (0.333, 0.658, 0.407), (0.768, 0.305, 0.321),
    (0.505, 0.447, 0.701), (0.576, 0.470, 0.376), (0.854, 0.545, 0.764), (0.549, 0.549, 0.549),
    (0.800, 0.725, 0.454), (0.392, 0.709, 0.803)
]
counter = 0
for i in range(0, nparticles):
    mlab.plot3d(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2], tube_radius=0.005, color=colors[counter%len(colors)])
    counter += 1
mlab.view(azimuth=0, elevation=0)
mlab.savefig(ResultsFolder+'/'+qvfilename + "poincare-3d_1.png", magnification=4)
mlab.view(azimuth=-10, elevation=75)
mlab.savefig(ResultsFolder+'/'+qvfilename + "poincare-3d_2.png", magnification=4)
mlab.close()