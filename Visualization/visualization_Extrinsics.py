import torch
import numpy as np
import matplotlib.pyplot as plt

a = [1, 0, 0]
b = [0, 1, 0]
c = [0, 0, 1]

a = torch.tensor(a, dtype=torch.float32)
b = torch.tensor(b, dtype=torch.float32)
c = torch.tensor(c, dtype=torch.float32)



R = np.load("/.../R.npy", allow_pickle=True)
T = np.load("/.../T.npy", allow_pickle=True)

T = torch.tensor(T, dtype=torch.float32)
R = torch.tensor(R, dtype=torch.float32)



T_0 = T[0,:]
R_0 = R[0,:]

T_1 = T[1,:]
R_1 = R[1,:]

T_2 = T[2,:]
R_2 = R[2,:]

T_3 = T[3,:]
R_3 = R[3,:]

T_4 = T[4,:]
R_4 = R[4,:]

T_5 = T[5,:]
R_5 = R[5,:]


A_0 = R_0 @ a
B_0 = R_0 @ b
C_0 = R_0 @ c

A_1 = R_1 @ a
B_1 = R_1 @ b
C_1 = R_1 @ c

A_2 = R_2 @ a
B_2 = R_2 @ b
C_2 = R_2 @ c

A_3 = R_3 @ a
B_3 = R_3 @ b
C_3 = R_3 @ c

A_4 = R_4 @ a
B_4 = R_4 @ b
C_4 = R_4 @ c

A_5 = R_5 @ a
B_5 = R_5 @ b
C_5 = R_5 @ c

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(T_0[0], T_0[1], T_0[2], A_0[0], A_0[1], A_0[2], color='r')
ax.quiver(T_0[0], T_0[1], T_0[2], B_0[0], B_0[1], B_0[2], color='b')
ax.quiver(T_0[0], T_0[1], T_0[2], C_0[0], C_0[1], C_0[2], color='k')

ax.quiver(T_1[0], T_1[1], T_1[2], A_1[0], A_1[1], A_1[2], color='r')
ax.quiver(T_1[0], T_1[1], T_1[2], B_1[0], B_1[1], B_1[2], color='b')
ax.quiver(T_1[0], T_1[1], T_1[2], C_1[0], C_1[1], C_1[2], color='k')

ax.quiver(T_2[0], T_2[1], T_2[2], A_2[0], A_2[1], A_2[2], color='r')
ax.quiver(T_2[0], T_2[1], T_2[2], B_2[0], B_2[1], B_2[2], color='b')
ax.quiver(T_2[0], T_2[1], T_2[2], C_2[0], C_2[1], C_2[2], color='k')

ax.quiver(T_3[0], T_3[1], T_3[2], A_3[0], A_3[1], A_3[2], color='r')
ax.quiver(T_3[0], T_3[1], T_3[2], B_3[0], B_3[1], B_3[2], color='b')
ax.quiver(T_3[0], T_3[1], T_3[2], C_3[0], C_3[1], C_3[2], color='k')

ax.quiver(T_4[0], T_4[1], T_4[2], A_4[0], A_4[1], A_4[2], color='r')
ax.quiver(T_4[0], T_4[1], T_4[2], B_4[0], B_4[1], B_4[2], color='b')
ax.quiver(T_4[0], T_4[1], T_4[2], C_4[0], C_4[1], C_4[2], color='k')

ax.quiver(T_5[0], T_5[1], T_5[2], A_5[0], A_5[1], A_5[2], color='r')
ax.quiver(T_5[0], T_5[1], T_5[2], B_5[0], B_5[1], B_5[2], color='b')
ax.quiver(T_5[0], T_5[1], T_5[2], C_5[0], C_5[1], C_5[2], color='k')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()