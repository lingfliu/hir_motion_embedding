import quaternion
import numpy as np
import scipy as sp
import transforms3d


e1 = np.array([45, 0, 0])*sp.pi/180
r1 = transforms3d.euler.euler2mat(e1[0], e1[1], e1[2], 'szxy')
q1 = quaternion.from_rotation_matrix(r1)

e2 = np.array([0, 45, 0])*sp.pi/180
r2 = transforms3d.euler.euler2mat(e2[0], e2[1], e2[2], 'szxy')
q2 = quaternion.from_rotation_matrix(r2)

# print(e1)
# print(r1)
# print(q1)
# print(np.norm(q1))

# print(e2)
# print(r2)
# print(q2)

# q3 = np.quaternion(1,0,1,0)
# q4 = np.quaternion(1,1,0,0)
#
# print(np.dot(q3,q4))
# print(q3*q4)
#
# print(q3.conjugate()*q4)
qr = q1.conjugate()*q2
angles = transforms3d.quaternions.quat2axangle(qr)
print(angles)

