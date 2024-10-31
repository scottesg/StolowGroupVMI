# DAVIS v7 - 20241023

import numpy as np
from math import sqrt
from scipy.special import legendre

# Functions for generating matrices for use with DAVIS Abel inversion

# Generates the matrix M(m,n) as given in Table 1 of the paper:
# The Journal of Chemical Physics 148, 194101 (2018); doi: 10.1063/1.5025057
# **** An error in the paper for M(2,0) has been fixed as noted below ****
#
# size: dimension of square matrix
# dtdeg: angular pixel size in degrees
#   Number of degrees per element in the polar image array
#   Smaller = better resolution
# dr: radial pixel size
#   Number of pixels per element in the polar image array
#   Smaller = better resolution
# save: save the matrix as "mxx_size_dtdeg_dr*10.npy" where xx are the m and n values
#
# matrices go up to size*dr pixels
def get_M(m, n, size, dtdeg, dr, save=False):
    m_matrix = np.zeros((size, size))
    for i in range(0, size):
        rp = (2 * i + 1 + dr / 2)
        rm = (2 * i + 1 - dr / 2)
        apw = np.deg2rad(dtdeg)

        if m == 0 and n == 0:
            for k in range(i, size):
                j = 2 * k + 1
                term = sqrt(j ** 2 - rm ** 2)
                if k != i:
                    term += - sqrt(j ** 2 - rp ** 2)
                m_matrix[i, k] = (2 * dr * apw / j) * term

        elif m == 2 and n == 2:
            for k in range(i, size):
                j = 2 * k + 1
                term = (2 * j ** 2 + rm ** 2) * sqrt(j ** 2 - rm ** 2)
                if k != i:
                    term += - (2 * j ** 2 + rp ** 2) * sqrt(j ** 2 - rp ** 2)
                m_matrix[i, k] = 2 * dr * apw / (3 * j ** 3) * term

        elif m == 2 and n == 0:
            for k in range(i, size):
                j = 2 * k + 1
                term = - (j ** 2 - rm ** 2) ** (3/2)
                if k != i:
                    term += (j ** 2 - rp ** 2) ** (3/2)
                # fixed error in DAVIS paper for above line:
                # Changed "+ rp ** 2" to "- rp ** 2"
                m_matrix[i, k] = dr * apw / (3 * j ** 3) * term

        elif m == 4 and n == 4:
            for k in range(i, size):
                j = 2 * k + 1
                term = sqrt(j ** 2 - rm ** 2) * (8 * j ** 4 + 4 * j ** 2 * rm ** 2 + 3 * rm ** 4)
                if k != i:
                    term += - sqrt(j ** 2 - rp ** 2) * (8 * j ** 4 + 4 * j ** 2 * rp ** 2 + 3 * rp ** 4)
                m_matrix[i, k] = 2 * dr * apw / (15 * j ** 5) * term

        elif m == 4 and n == 2:
            for k in range(i, size):
                j = 2 * k + 1
                term = - (j ** 2 - rm ** 2) ** (3/2) * (2 * j ** 2 + 3 * rm ** 2)
                if k != i:
                    term += (j ** 2 - rp ** 2) ** (3/2) * (2 * j ** 2 + 3 * rp ** 2)
                m_matrix[i, k] = dr * apw / (3 * j ** 5) * term

        elif m == 4 and n == 0:
            for k in range(i, size):
                j = 2 * k + 1
                term = - (j ** 2 - rm ** 2) ** (3/2) * (21 * rm ** 2 - j ** 2)
                if k != i:
                    term += (j ** 2 - rp ** 2) ** (3/2) * (21 * rp ** 2 - j ** 2)
                m_matrix[i, k] = dr * apw / (60 * j ** 5) * term
        else:
            print("Invalid M Matrix")
            break

    if save:
        np.save("m{}{}_{}_{}_{}.npy".format(m, n, size, int(dtdeg), int(dr*10)), m_matrix)
    return m_matrix

# Generates a matrix of Legendre polynomial values for even polynomials up to
# nmax. Angle runs from 0 to 360 by steps of dtdeg.
#
# nmax: upper limit for (even) Legendre polynomials to evaluate
# dtdeg: angular pixel size in degrees (must be the ssame dtdeg as the matrices)
#   Number of degrees per element in the polar image array
#   Smaller = better resolution
# save: saves matrix as "Legendre_dtdeg.npy"
def get_Legendre(nmax, dtdeg, save=False):
    LMatrix = np.zeros((int(nmax/2)+1, int(360/dtdeg)))
    arad = np.deg2rad(np.arange(0, 360, dtdeg))
    for i in range(0, nmax+1, 2):
        LMatrix[int(i/2),:] = [legendre(i)(np.cos(j)) for j in arad]
    if save:
        np.save("Legendre_{}.npy".format(int(dtdeg)), LMatrix)
    return LMatrix

# Saves Legendre, m00, m22, m20, m44, m42, and m40
# size, nmax, dtdeg and dr are as defined above
#
# matrices go up to size*dr pixels
def saveall(size, nmax, dtdeg, dr):
    
    print("Saving Legendre matrix...")
    get_Legendre(nmax, dtdeg, save=True)
    
    mn = [[0, 0], [2, 2], [2, 0],
         [4, 4], [4, 2], [4, 0]]
    
    for i, j in mn:
        print("Saving m{}{}...".format(i, j))
        get_M(i, j, size, dtdeg, dr, save=True)
    
    
    
    
    
    
    