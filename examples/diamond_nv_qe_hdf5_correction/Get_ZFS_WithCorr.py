
import numpy as np
from glob import iglob
from ase.io import read

def Get_ZFS_Tensor(filename):
    flines = open(filename).readlines()
    for i, line in enumerate(flines):
        if "Total D tensor (MHz):" in line:
            ind = i
    zfs_tensor = []
    for line in flines[ind+1:ind+4]:
        line = line.strip().replace("[", "").replace("]", "")
        line_arr = np.array([float(x) for x in line.split()])
        zfs_tensor.append(line_arr)
    zfs_tensor = np.asarray(zfs_tensor)
    return zfs_tensor


def Get_D_E(zfs):
    
    ev, evc = np.linalg.eig(zfs)

    # For triplet states, compute D and E parameters:
    # Denote three eigenvalues as Dx, Dy, Dz: |Dz| > |Dx| > |Dy|
    # D = 3/2 Dz, E = 1/2(Dx - Dy)
    args = np.abs(ev).argsort()
    dy, dx, dz = ev[args]
    Dvalue = 1.5 * dz
    Evalue = 0.5 * (dx - dy)
    
    return Dvalue, Evalue


def Get_ZFS_Corr(dir_prefix):

    # Uncorrected
    zfs_u = Get_ZFS_Tensor(f"{dir_prefix}/out/ZFS_OutFile.txt")
    D_u, E_u = Get_D_E(zfs_u)
    print("Uncorrected: ", D_u, E_u)

    # Corrected
    zfs1 = Get_ZFS_Tensor(f"C1/out/ZFS_OutFile.txt")
    zfs2 = Get_ZFS_Tensor(f"C2/out/ZFS_OutFile.txt")
    zfs3 = Get_ZFS_Tensor(f"C3/out/ZFS_OutFile.txt")
    zfs_avg = (zfs1 + zfs2 + zfs3) / 3

    zfs = zfs_u/2. - zfs_avg/2.
    D, E = Get_D_E(zfs)
    print("Corrected: ", D, E)

    return D_u, E_u, D, E



if __name__ == "__main__":

    Get_ZFS_Corr("NV")

