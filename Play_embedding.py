import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import axes3d

train_dir = 'D:/Datasets/Turf Valley Resort/train/New folder/'
validation_dir = 'D:/Datasets/Turf Valley Resort/validation/New folder/'
test_dir = 'D:/Datasets/Turf Valley Resort/test/New folder/'
architecture_ID = "O-R-F-C"

set_id = "test"
movie_path = "D:/Datasets/Turf Valley Resort/movies/"

Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/Embedding data/Z_' + set_id + '_last-' + architecture_ID + '.npy'

Z = np.load(Z_save_path)

def count_files(path):
    Number_Of_Files = 0
    for file in os.walk(path):
        #     print(file)
        for files in file[2]:
            Number_Of_Files = Number_Of_Files + 1
    return Number_Of_Files

T_tr = count_files(train_dir)  # number of images in the dataset
T_val = count_files(validation_dir)
T_test = count_files(test_dir)


l=75
# determine real time start and end of the embedding in seconds

t_start = l - 1  # first frame inded in the valid range
t_end = T_test - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_tr = t_end - t_start + 1
samples = np.floor((t_end - t_start + 1))


def normalize0_1(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))


a = 7
fig2 = plt.figure(figsize=(1.7778*a, a))  #  e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
ax = fig2.gca(projection='3d')

dtheta = 0.1  # 0.02  #rotation rate deg
k=0
dk = 0  # 
theta = 60  # rotation angle
tail_points = 1
phi_0 = 10  # elevation angle
render_interval = 1

for t in range(tail_points, len(Z), render_interval):
    ax = fig2.gca(projection='3d')
    ax.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], 'k-', markerfacecolor='black', markersize=1, linewidth=1.4, label='Z')
    ax.plot3D(Z[t-tail_points:t, 0], Z[t-tail_points:t, 1], Z[t-tail_points:t, 2], '-o', markerfacecolor='red', mec='blue', markersize=15, label='Z(t)')

    phi = 20*np.sin(k) + phi_0
    # phi = 0
    ax.view_init(phi, theta)  #view_init(elev=None, azim=None)

    ax.dist = 8
    plt.draw()
    plt.pause(.01)
    fig2.savefig(movie_path + 'frame_' + f'{t:03}' + '.png', transparent=False, dpi='figure', bbox_inches=None)
    fig2.clear(keep_observers=True)

    theta += dtheta
    k += dk
    print(t)
