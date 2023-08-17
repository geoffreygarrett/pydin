# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def plot_frustum(ax, pos, target, up, hfov, aspect, near, far):
#     forward = (target - pos) / np.linalg.norm(target - pos)
#     right = np.cross(up, forward)
#     up = np.cross(forward, right)
#
#     fc = pos + forward * far
#     nc = pos + forward * near
#
#     hfar = 2 * np.tan(np.radians(hfov) / 2) * far
#     wfar = hfar * aspect
#     hnear = 2 * np.tan(np.radians(hfov) / 2) * near
#     wnear = hnear * aspect
#
#     ftl = fc + (up * hfar / 2) - (right * wfar / 2)
#     ftr = fc + (up * hfar / 2) + (right * wfar / 2)
#     fbl = fc - (up * hfar / 2) - (right * wfar / 2)
#     fbr = fc - (up * hfar / 2) + (right * wfar / 2)
#
#     ntl = nc + (up * hnear / 2) - (right * wnear / 2)
#     ntr = nc + (up * hnear / 2) + (right * wnear / 2)
#     nbl = nc - (up * hnear / 2) - (right * wnear / 2)
#     nbr = nc - (up * hnear / 2) + (right * wnear / 2)
#
#     lines = [ [ntl, ntr], [ntr, nbr], [nbr, nbl], [nbl, ntl],
#               [ftl, ftr], [ftr, fbr], [fbr, fbl], [fbl, ftl],
#               [ntl, ftl], [ntr, ftr], [nbr, fbr], [nbl, fbl]]
#
#     for line in lines:
#         ax.plot3D(*zip(*line), color='blue')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_frustum(ax, np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), 90, 1, 1, 10)
# plt.show()