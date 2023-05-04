import torch
import numpy as np


def Unproject():

    Un_project = np.array([[0.5, 0, 0, -50.0],
                             [0, 0.5, 0, -50.0],
                             [0, 0, 0.5, -5.0],
                             [0, 0, 0, 1]])

    return Un_project


def Intrinsics (i):

    intrins = np.load("/.../I.npy", allow_pickle=True)
    intrins = intrins[0,i,:]

    return intrins



def Extrinsics(i):

    last_line = [0, 0, 0, 1]
    last_line = np.reshape(last_line, [1, 4])

    R = np.load("/.../R.npy", allow_pickle=True)
    T = np.load("/.../T.npy", allow_pickle=True)
    R = R[0, :]
    T = T[0, :]

    if i == 0:
        R_0 = R[0, :]
        T_0 = T[0, :]
        T_0 = np.reshape(T_0, [3, 1])

        RT_0 = np.concatenate((R_0,T_0),axis=-1)
        RT_0 = np.concatenate((RT_0,last_line),axis=0)

        return RT_0

    if i == 1:
        R_1 = R[1, :]
        T_1 = T[1, :]
        T_1 = np.reshape(T_1, [3, 1])

        RT_1 = np.concatenate((R_1,T_1),axis=-1)
        RT_1 = np.concatenate((RT_1,last_line),axis=0)

        return RT_1

    if i == 2:
        R_2 = R[2, :]
        T_2 = T[2, :]
        T_2 = np.reshape(T_2, [3, 1])

        RT_2 = np.concatenate((R_2,T_2),axis=-1)
        RT_2 = np.concatenate((RT_2,last_line),axis=0)

        return RT_2


    if i == 3:
        R_3 = R[3, :]
        T_3 = T[3, :]
        T_3 = np.reshape(T_3, [3, 1])

        RT_3 = np.concatenate((R_3,T_3),axis=-1)
        RT_3 = np.concatenate((RT_3,last_line),axis=0)

        return RT_3


    if i == 4:
        R_4 = R[4, :]
        T_4 = T[4, :]
        T_4 = np.reshape(T_4, [3, 1])

        RT_4 = np.concatenate((R_4,T_4),axis=-1)
        RT_4 = np.concatenate((RT_4,last_line),axis=0)

        return RT_4


    if i == 5:
        R_5 = R[5, :]
        T_5 = T[5, :]
        T_5 = np.reshape(T_5, [3, 1])

        RT_5 = np.concatenate((R_5,T_5),axis=-1)
        RT_5 = np.concatenate((RT_5,last_line),axis=0)

        return RT_5




def IsVisible(x, y, z, p):

    Extrinsic = Extrinsics(i = p)           # shape( 4 * 4 )
    Intrinsic = Intrinsics (i = p)           # shape( 3 * 4 )


    un_project = Unproject()       # shape( 4 * 4 )

    pgrid = np.array([x, y, z, 1])

    pworld = np.matmul(un_project, pgrid)


    pcar = np.matmul(Extrinsic, pworld)

    pcam = pcar[0:3]
    depth = pcam[2]

    if depth <= 0:
        return 0
    pcam = pcam / depth
    ppix = np.matmul(Intrinsic, pcam)


    if (ppix[0] >= 0 and ppix[0] <= 225) and (ppix[1] >= 0 and ppix[1] < 400) :

        return 1
    else:
        return 0

def mask_one(p):

    occ = torch.zeros([200,200,16], dtype = torch.int8)

    for x in range(0,200,1):
        for y in range(0,200,1):
            for z in range(0,16,1):

                if IsVisible(x, y, z, p) == 1:

                    occ[x, y, z] = 1

    print(occ.shape)
    return occ

def mask_all():

    s = 6
    occ = torch.zeros([200,200,16], dtype = torch.int16)

    for p in range(6):

        for x in range(0, 200, 1):
            for y in range(0, 200, 1):
                for z in range(0, 16, 1):

                    if IsVisible(x, y, z, p) == 1:
                        occ[x, y, z] = 1

        print('   完成第{}个相机的处理  '.format(p + 1))


    return occ


# occ = []
# for i in range(6):
#
#     one = mask_one(i)
#     occ.append(one)
#     print('   完成第{}个相机的处理  '.format(i + 1))


all = mask_all()
# occ.append(all)
# occ = torch.stack(occ, dim=0)


np.save("/..../Fov_Mask.npy", all)


'''

dir = '/..../S_0.bin'
fov_mask = np.load("/..../Fov_Mask.npy", allow_pickle=True)
fov_mask = torch.tensor(fov_mask, dtype=torch.int16)

non_zero_a = torch.count_nonzero(fov_mask)
print('非零个数', non_zero_a)
print('非零个数占比', non_zero_a/(200*200*16) )

# p_dir = '/..../visualization_fov_mask'
# true_name = 'S_'  +  str(0) +'.bin'
# true_path = os.path.join(p_dir, true_name)

mask = np.array(fov_mask, dtype=numpy.int16)
mask.tofile(dir)

'''