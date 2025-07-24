import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import chain
import ipdb


def plot_batch_with_ref(save_video=0):
    plt.cla() # Clear plot for each step

    npz = np.load("results/bT_state.npz")
    bT_state = npz["aircraft"]
    # print(bT_state)
    # TODO: time_list need to be saved with a key in simulate_batch
    time_list = np.load("results/time_list.npz")
    Ref_list = np.load("results/Ref_list.npz")
    Pos_des = Ref_list["Pos_des"]
    Pos_des = np.array(Ref_list["Pos_des"])
    Vel_des = np.array(Ref_list["Vel_des"])
    # print(Pos_des)
    Ref_arr = [
        Pos_des[:, 0],
        Pos_des[:, 1],
        Pos_des[:, 2],
        Vel_des[:, 0],
        Vel_des[:, 1],
        Vel_des[:, 2]
    ]
    # print(Ref_arr)
    bT_state = bT_state[::8]

    b, T, _ = bT_state.shape
    bT_pos = bT_state[:, :, 6:9]

    dt = 0.005
    T_t = np.arange(T) * dt * 10
    bT_t = np.broadcast_to(T_t, (b, T))

    arrs = [
        bT_pos[:, :, 0],
        bT_pos[:, :, 1],
        bT_pos[:, :, 2],
        bT_state[:, :, 0],
        bT_state[:, :, 1],
        bT_state[:, :, 2],
        bT_state[:, :, 3],
        bT_state[:, :, 4],
        bT_state[:, :, 5],
    ]
    labels = [r"$p_x$", r"$p_y$", r"$p_z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    # Dimensions of arrs and Ref_arr
    # print(len(arrs[0][0]))
    # print(len(Ref_arr[0]))
    # print(len(bT_t[0]))
    # print(bT_t[0][:10])
    # print(Ref_arr[0][:10])
    # print(arrs[1][0])

    # 1: Plot overhead 2d.
    fig, ax = plt.subplots(layout="constrained")
    line_col = LineCollection(bT_pos[:, :, :2][:, :, ::-1], color="C1", lw=0.01, alpha=0.9)
    ax.add_collection(line_col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel="East [ft]", ylabel="North [ft]")
    fig.savefig("results/batch_traj2d_square_with_ref.pdf")

    nrows = len(arrs)
    figsize = np.array([6, 1 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")


    def add_plot(frame, fig, axes, bT_t, Ref_arr, arrs):
        frame = frame*50+1
        print(frame)
        plt.cla() # Clear plot for each step
        # fig = fargs[0]
        # axes = fargs[1]
        # bT_t = fargs[2]
        # Ref_arr = fargs[3]
        # arrs = fargs[4]

        for ii, ax in enumerate(axes):

            # Add pos and vel reference as red dashed lines
            if ii in [0, 1, 2, 3, 4, 5]:
                ref_lines = np.stack([[bT_t[0][:frame]], [Ref_arr[ii][:frame]]], axis=-1)
                line_col = LineCollection(ref_lines, color="C3", lw=0.5, alpha=1, linestyle='--')
                ax.add_collection(line_col)
                ax.set_ylim([min(chain(Ref_arr[ii], arrs[ii][0])), max(chain(Ref_arr[ii], arrs[ii][0]))])
            else:
                ax.set_ylim([min(arrs[ii][0]), max(arrs[ii][0])])

            # (b, T, 2)
            lines = np.stack([[bT_t[0][:frame]], [arrs[ii][0][:frame]]], axis=-1)
            line_col = LineCollection(lines, color="C1", lw=0.5, alpha=1)
            ax.add_collection(line_col)
            ax.autoscale_view()
            ax.set_ylabel(labels[ii], rotation=0, ha="right")
            ax.set_xlim([min(bT_t[0]), max(bT_t[0])])

        axes[-1].set_xlabel("Time (s)")
        if frame == len(bT_t[0]):
            fig.savefig("results/batch_traj_with_ref.pdf")

    if save_video == 1:
        fargs = [fig, axes, bT_t, Ref_arr, arrs]
        ani = FuncAnimation(fig, add_plot, fargs = fargs, frames=201, interval=1, repeat=False)
        ani.save("results/batch_traj_with_ref.mp4", writer=FFMpegWriter(fps=30))
    elif save_video == 0:
         add_plot(10001, fig, axes, bT_t, Ref_arr, arrs)




if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        plot_batch_with_ref(save_video=1)
