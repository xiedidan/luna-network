import numpy as np

def worldToVoxel(worldCoord, origin, spacing):
    voxelCoord = worldCoord - origin
    voxelCoord = np.absolute(voxelCoord)
    voxelCoord = np.rint(voxelCoord / spacing)
    voxelCoord = np.array(voxelCoord, dtype = int)
    return voxelCoord

def voxelToWorld(voxelCoord, origin, spacing):
    worldCoord = voxelCoord * spacing
    worldCoord += origin
    worldCoord = np.rint(worldCoord)
    worldCoord = np.array(worldCoord, dtype = int)
    return worldCoord

