import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
h=128.0; w=257.0;
X = np.linspace(0, 257, int(w))
Y = np.linspace(0, 128, int(h))
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
#mu = np.array([20., 11.])
#Sigma = np.array([[ 3. , 0], [0.,  3.]]) # assume radius is around 333km
lon=np.load("lon.npy")
lat=np.load("lat.npy")
r=np.load("r.npy")

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    mat = np.exp(-fac / 2) / N
    print(type(mat)); 
    return np.asarray(mat) / np.amax(mat)
 

Z_all=[]
for j in range(len(lon)): #day
    for i in range(len(lon[j])): #objects in one shot
        lat_val=lat[j][i]*h
        lon_val=lon[j][i]*w
        r_val=float(r[j][i])
        variance=float((r_val+0.5)*5.0)
        print(i,j,variance)
        Sigma = np.array([[ variance , 0], [0., variance]]) 
        mu=np.array([lon_val,lat_val])
        if i==0:
            Z=np.asarray(multivariate_gaussian(pos, mu, Sigma))
        else:
            Z=Z+np.asarray(multivariate_gaussian(pos, mu, Sigma))
    Z_all.append(Z)
print(np.shape(Z_all))
np.save("label_image.npy",Z_all)

