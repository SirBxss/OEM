from plyfile import PlyData
plydata = PlyData.read('/Users/amin/PycharmProject/OEM/test/Image_20241026-114307.ply')
print(plydata['vertex'].data.dtype.names)