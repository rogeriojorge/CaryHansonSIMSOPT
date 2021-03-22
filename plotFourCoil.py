#!/usr/bin/env python3
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.core.optimizable import optimizable
from simsopt.geo.biotsavart import BiotSavart
from poincareplot import compute_field_lines
import matplotlib.pyplot as plt
import numpy as np
############################################
outputFile    = "CH_coil_data.dat"
currentHel    = 2.1e5
currentTor    = 1.91e5
ppp           = 12  # points for coils curve
spp           = 130 # number of phi steps for one stellarator revolution
nperiods      = 80 # how often to go around the stellarator
batch_size    = 8 # how many field lines are traced at the same time
max_thickness = 0.14 # how far out from the magnetic axis are we going
delta         = 0.003 # distance between starting points of fieldlines
nquadrature   = 101 # number of grid points for the axis
magnetic_axis_radius = 0.97
NFP           = 1
qvfilename    = 'CaryHanson'
############################################
coil_data = np.loadtxt(outputFile, delimiter=',')
Nt_coils=len(coil_data)-1
num_coils = int(len(coil_data[0])/6)
coils = [optimizable(CurveXYZFourier(Nt_coils*ppp, Nt_coils)) for i in range(num_coils)]
for ic in range(num_coils):
	dofs = coils[ic].dofs
	dofs[0][0] = coil_data[0, 6*ic + 1]
	dofs[1][0] = coil_data[0, 6*ic + 3]
	dofs[2][0] = coil_data[0, 6*ic + 5]
	for io in range(0, Nt_coils):
		dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
		dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
		dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
		dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
		dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
		dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
	coils[ic].set_dofs(np.concatenate(dofs))
currents = [-currentHel,currentHel,currentTor]
##########################################
biotsavart = BiotSavart(coils, currents)

axis = CurveRZFourier(nquadrature,0,1,True)
axis.set_dofs(np.concatenate(([1],[0])))
biotsavart.set_points(axis.gamma())
Bfield=biotsavart.B()
Bstrength=[np.sqrt(np.dot(Bfieldi,Bfieldi)) for Bfieldi in Bfield]
plt.plot(Bstrength)
plt.savefig(qvfilename + "BonAxis.png", dpi=300)

rphiz, xyz, absB, phi_no_mod = compute_field_lines(biotsavart, nperiods=nperiods, batch_size=batch_size, magnetic_axis_radius=magnetic_axis_radius, max_thickness=max_thickness, delta=delta, steps_per_period=spp)

nparticles = rphiz.shape[0]
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

modBdata = np.hstack((phi_no_mod[:, None], absB.T))[0:(10*spp)]
plt.figure()
for i in range(min(modBdata.shape[1]-1, 10)):
    plt.plot(modBdata[:, 0], modBdata[:, i+1], zorder=100-i)
plt.savefig(qvfilename + "absB.png", dpi=300)
plt.close()

import mayavi.mlab as mlab
#mlab.options.offscreen = True
fig = mlab.figure(bgcolor=(1,1,1))
for count, coil in enumerate(coils):
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
mlab.savefig(qvfilename + "poincare-3d_1.png", magnification=4)
mlab.view(azimuth=-10, elevation=75)
mlab.savefig(qvfilename + "poincare-3d_2.png", magnification=4)
mlab.close()

for k in range(4):
    plt.figure()
    for i in range(nparticles):
        plt.scatter(rphiz[i, range(k * spp//(NFP*4), nperiods*spp, spp), 0], rphiz[i, range(k * spp//(NFP*4), nperiods*spp, spp), 2], s=0.1)
    plt.savefig(qvfilename + f"poincare_{k}.png", dpi=300)
    plt.close()