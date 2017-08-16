import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.spatial import cKDTree
import pandas
import sys
import frogress

zcosmo = 0.25
dire = sys.argv[1]
projmax = float(sys.argv[2])
Rcfac = float(sys.argv[3])
print projmax

def get_lamda(R2d, bg, pfree_ind, verbose=0):

    from scipy.optimize import root
    from scipy.special import erf

    # Solve for the equation:
    # \Sum pmem_i = \lambda
    # pmem_i = lambda*theta_r*u(r)/(lambda*theta_r*u(r) + bg)
    # theta_r is a radial function which depends upon Rc the radius within which lambda is measured
    # And Rc(lambda) = (lambda/100.0)**0.2 in physical units
    # Busch and White assume u(r) = 1/(pi Raper**2)
    def funcroot(lamda, R2d, pfree_ind, ret_pmem=False, verbose=0):
        lamda = np.absolute(lamda)
        Raper_com = Rcfac*(lamda/100.0)**0.2*(1.+zcosmo) #physical unit
        #Raper_com = Rcfac*(lamda/100.0)**0.2 #comoving unit

        # The commented line below make it a tophat filter
        theta_r = (R2d<Raper_com).astype(int)
        #theta_r = 0.5*(1.+erf((Raper_com-R2d)/0.05))

        if ret_pmem:

            \return theta_r, pfree_ind*lamda/(lamda+bg*R2d**2*np.pi)

        res = 1. - np.sum(theta_r*pfree_ind/(lamda+bg*R2d**2*np.pi))

        if verbose:
            print lamda, res
            
        return res

    result = root(funcroot, 0.8*R2d.size, args=(R2d, pfree_ind, False, verbose), method='hybr', tol=1.E-2).x[0]
    result = np.absolute(result)
    theta_r, pmem = funcroot(result, R2d, pfree_ind, ret_pmem=True)
    funcval = funcroot(result, R2d, pfree_ind)
    if verbose:
        for lam in np.arange(0.05, 1.0, 0.01):
            print lam*R2d.size, funcroot(lam*R2d.size, R2d, sigma, pfree_ind)

    return result, theta_r, pmem, funcval

# Read galaxies: assume that the z value is in redshift space, and only red colored galaxies are selected, x, y, z in Mpc/h
#galaxy_frame = pandas.read_csv("test_Galaxy_file.dat", delim_whitespace=1, header=None, names=("mstar", "x", "y", "z"), skiprows=1, usecols=(0, 1, 2, 3))

galaxy_frame = pandas.read_csv("./data/mydb_z.dat", delimiter=' ', header=None, usecols=(0, 1, 2, 3,4,5,6,7,8), names=("mstar","galID","haloID", "x", "y", "z","type","mvir","rvir"))
print 'finish reading'

# Generate the tree
boxsize = 500.0
galaxy_frame.x = (galaxy_frame.x.values%boxsize)
galaxy_frame.y = (galaxy_frame.y.values%boxsize)
galaxy_frame.z = (galaxy_frame.z.values%boxsize)
tree_twod = cKDTree(zip(galaxy_frame.x.values, galaxy_frame.y.values), boxsize=boxsize)

nbar = galaxy_frame.x.size/boxsize**3
bg = nbar*2.*projmax

for itern in range(10):
    print itern
    if itern==0:
        halo_frame_centrals = pandas.read_csv("./data/mydb_z.dat", delimiter=' ', header=None, usecols=(0, 1, 2, 3,4,5,6,7,8), names=("mstar","galID","haloID", "x", "y", "z","type","mvir","rvir"))
        test = np.ones(halo_frame_centrals.x.size).astype('int')
        halo_frame_centrals.sort_values("mstar", ascending=False, inplace=1)
    else:
        halo_frame_centrals = pandas.read_csv("%s/Redmapper.lamdasorted_proj60_z.%02d" % (dire, itern-1),  delimiter=' ', header=None, usecols=(0, 1, 2, 3,4,5,6,7,8,9), names=("mstar","galID","haloID", "x", "y", "z","type","mvir","rvir","lamda"),low_memory=False,skiprows=1)
        test = (halo_frame_centrals.lamda.values > 1.) & (halo_frame_centrals.lamda.values < 400.)

    
    lamda = np.zeros(halo_frame_centrals.x.size)
    funcval = np.zeros(halo_frame_centrals.x.size)
    pfree = np.ones(galaxy_frame.x.size) 
    rmemM16 = np.zeros(halo_frame_centrals.x.size)    
    rmemM16_2 = np.zeros(halo_frame_centrals.x.size)
    flag = np.zeros(halo_frame_centrals.x.size)

    for i in frogress.bar(range(0, halo_frame_centrals.x.values[test].size)): #ts: added test-statement
    #for i in np.arange(20):
        #
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

        pfree_ind = pfree[idx[restidx]]
        res = get_lamda(rsat2d, bg, pfree_ind)
        # Set maximum lambda to be 400 or so (test what this does)
        if np.absolute(res[3]) <0.01 and res[0]<400.0:
            lamda[i], theta_r, thispmem, funcval[i] = res
            rmemM16[i] = np.sum(rsat2d*theta_r*thispmem)/np.sum(theta_r*thispmem) #ts: should I get pmem?
            rmemM16_2[i] = np.sum(rsat2d*theta_r)/np.sum(theta_r) #ts: closer to Busch's version

            #The formula is pfree_new = pfree_old (1.-pmem)
            pfree[idx[restidx]] = pfree[idx[restidx]]*(1.0-thispmem*theta_r)
            #print i, np.shape(galaxy_frame.mstar.values[idx]),np.shape(np.shape(galaxy_frame.mstar.values[idx][restidx])),np.shape(restidx),np.shape(theta_r),theta_r
            if (np.shape(theta_r)[0]==1):
                if halo_frame_centrals.mstar.values[i]<np.max(galaxy_frame.mstar.values[idx][restidx]):
                    flag[i] = -1.
            else:
                if halo_frame_centrals.mstar.values[i]<np.max(galaxy_frame.mstar.values[idx][restidx][theta_r]):
                    flag[i] = -1.
    
        else:
            continue

    # print "\n", halo_frame_centrals.x.values[itern], halo_frame_centrals.y.values[itern]

    halo_frame_centrals["lamda"] = lamda
    halo_frame_centrals["rmemM16"] = rmemM16    
    halo_frame_centrals["flag"] = flag
    halo_frame_centrals["rmemM16_busch"] = rmemM16_2
    halo_frame_centrals["funcval"] = funcval
    halo_frame_centrals.sort_values("lamda", ascending=False, inplace=1)

    # Now output the redmapper catalog
    #halo_frame_centrals.to_csv("%s/Redmapper.lamdasorted.%02d" % (dire, itern-1), sep=" ", index=False, index_label=False)
    halo_frame_centrals.to_csv("%s/Redmapper.lamdasorted_proj60_z.%02d" % (dire, itern), sep=" ", index=False, index_label=False)
