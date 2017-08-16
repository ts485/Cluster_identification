import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.spatial import cKDTree
import pandas
import sys
import frogress

#ts:2017/04/26 simple code for Busch&White

def Rc_lambda(lamda,Rcfac=1.):
    return Rcfac*(lamda/100.)**0.2*(1.+zcosmo)


zcosmo = 0.25
dire = sys.argv[1]
projmax = float(sys.argv[2])
Rcfac = float(sys.argv[3])
iter=0


galaxy_frame = pandas.read_csv("./data/mydb_x.dat", delimiter=' ', header=None, usecols=(0, 1, 2, 3,4,5,6,7,8), names=("mstar","galID","haloID", "x", "y", "z","type","mvir","rvir"))
galaxy_frame.sort_values("mstar", ascending=False, inplace=1)

# Generate the tree
boxsize = 500.0
galaxy_frame.x = (galaxy_frame.x.values%boxsize)
galaxy_frame.y = (galaxy_frame.y.values%boxsize)
galaxy_frame.z = (galaxy_frame.z.values%boxsize)
tree_twod = cKDTree(zip(galaxy_frame.x.values, galaxy_frame.y.values), boxsize=boxsize)

nbar = galaxy_frame.x.size/boxsize**3
bg = nbar*2.*projmax

halo_frame_centrals = pandas.read_csv("./data/mydb_x.dat", delimiter=' ', header=None, usecols=(0, 1, 2, 3,4,5,6,7,8), names=("mstar","galID","haloID", "x", "y", "z","type","mvir","rvir"))
halo_frame_centrals.sort_values("mstar", ascending=False, inplace=1)


lamda = np.zeros(halo_frame_centrals.x.size)
funcval = np.zeros(halo_frame_centrals.x.size)
pfree = np.ones(galaxy_frame.x.size) 
rmemM16 = np.zeros(halo_frame_centrals.x.size)
flag = np.zeros(halo_frame_centrals.x.size)

for i in frogress.bar(range(0, halo_frame_centrals.x.values.size)):
#for i in frogress.bar(range(10000)):
    idx = tree_twod.query_ball_point([halo_frame_centrals.x.values[i], halo_frame_centrals.y.values[i]], 3.0)
    idx = np.array(idx)

    # Compute the dz
    dz = np.absolute(halo_frame_centrals.z.values[i] - galaxy_frame.z.values[idx])
    dz[dz>boxsize/2.] = boxsize-dz[dz>boxsize/2.]
    restidx = (dz<projmax)

    dx = np.absolute(halo_frame_centrals.x.values[i] - galaxy_frame.x.values[idx][restidx])
    dx[dx>boxsize/2.] = boxsize-dx[dx>boxsize/2.]

    dy = np.absolute(halo_frame_centrals.y.values[i] - galaxy_frame.y.values[idx][restidx])
    dy[dy>boxsize/2.] = boxsize-dy[dy>boxsize/2.]

    rsat2d = (dx**2 + dy**2)**0.5
    pfree_ind = pfree[idx][restidx]
    lamda_prev = 0.
    lamda_new = 1.
    while lamda_new > lamda_prev:
        lamda_prev = lamda_new        
        test = rsat2d < Rc_lambda(lamda_prev)
        Ng = np.shape(rsat2d[test])[0]
        lamda_new = Ng-bg*np.pi*Rc_lambda(lamda_prev)**2.
        if halo_frame_centrals.mstar.values[i] >= np.max(galaxy_frame.mstar.values[idx][restidx][test]):
            continue
        else:
            flag[i] = -1

    lamda[i] = lamda_prev 
    rmemM16[i] = np.sum(rsat2d[test])/np.sum(test)
    #ts: should I delete the member galaxies from the list of centrals?

    

halo_frame_centrals["lamda"] = lamda
halo_frame_centrals["rmemM16"] = rmemM16
halo_frame_centrals["flag"] = flag
halo_frame_centrals.sort_values("lamda", ascending=False, inplace=1)

halo_frame_centrals.to_csv("%s/Redmapper.Busch_proj60_x.%02d" % (dire,iter), sep=" ", index=False, index_label=False)

    
