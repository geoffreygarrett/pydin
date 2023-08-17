# from mayavi import mlab
# import numpy as np
#
# def plot_frustum(pos, target, up, hfov, aspect, near, far):
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
#     lines = [[ntl, ntr], [ntr, nbr], [nbr, nbl], [nbl, ntl],
#              [ftl, ftr], [ftr, fbr], [fbr, fbl], [fbl, ftl],
#              [ntl, ftl], [ntr, ftr], [nbr, fbr], [nbl, fbl]]
#
#     for line in lines:
#         mlab.plot3d(*zip(*line), color=(0, 0, 1), tube_radius=None)
