import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()

def plot_3d_motion2(save_path, kinematic_tree, joints1, joints2, title, figsize=(10, 10), fps=120, radius=4, foot1=None, foot2=None):
    matplotlib.use('Agg')
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = np.vstack([data1.min(axis=0).min(axis=0), data2.min(axis=0).min(axis=0)]).min(axis=0)
    MAXS = np.vstack([data1.max(axis=0).max(axis=0), data2.max(axis=0).max(axis=0)]).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    colors2 = ['green', 'brown', 'darkblue', 'green', 'brown']
    frame_number = data1.shape[0]
    #     print(data.shape)

    data1[:, :, 1] -= data1.min(axis=0).min(axis=0)[1]
    data2[:, :, 1] -= data2.min(axis=0).min(axis=0)[1]
    trajec1 = data1[:, 0, [0, 2]]
    trajec2 = data2[:, 0, [0, 2]]
    
    #data1[..., 0] -= data1[:, 0:1, 0]
    #data1[..., 2] -= data1[:, 0:1, 2]
    #data2[..., 0] -= data2[:, 0:1, 0]
    #data2[..., 2] -= data2[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        if index > 1:
            #ax.plot3D(trajec1[:index, 0]-trajec1[index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1]-trajec1[index, 1], linewidth=1.0,
            #          color='blue')
            #ax.plot3D(trajec2[:index, 0]-trajec2[index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1]-trajec2[index, 1], linewidth=1.0,
            #          color='darkred')
            if foot1 is not None and foot2 is not None:
                f1, f2 = foot1[index], foot2[index]
                if sum(f1)>2:
                    ax.plot3D(trajec1[:index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1], linewidth=1.0,
                        color='blue')
                else:
                    ax.plot3D(trajec1[:index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1], linewidth=1.0,
                        color='green')
                if sum(f2)>2:
                    ax.plot3D(trajec2[:index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1], linewidth=1.0,
                        color='darkred')
                else:
                    ax.plot3D(trajec2[:index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1], linewidth=1.0,
                        color='black')
            else:
                ax.plot3D(trajec1[:index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1], linewidth=1.0,
                        color='blue')
                ax.plot3D(trajec2[:index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1], linewidth=1.0,
                        color='darkred')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        
        for i, (chain, color, inv_color) in enumerate(zip(kinematic_tree, colors, colors2)):
#             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth, color=color)
            ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=inv_color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()