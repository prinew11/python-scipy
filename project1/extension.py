from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import solver


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def solve_omega2_extension(gamma1, gamma2, gamma3, h):
    n = int(round(1.0 / h))
    half = int(round(0.5 / h))

    Nx = n + 1
    Ny = 2 * n + 1


    left_bc = np.full(Ny, 15.0)
    right_bc = np.full(Ny, 15.0)
    bottom_bc = np.full(Nx, 5.0)
    top_bc = np.full(Nx, 40.0)

    # Gamma1: left side, 0 < y < 1
    left_bc[1:n] = gamma1

    # Gamma3: right side, 0.5 < y < 1
    right_bc[half + 1:n] = gamma3

    # Gamma2: right side, 1 < y < 2
    right_bc[n + 1:2 * n] = gamma2

    return solver.solve_dirichlet_sparse(1.0, 2.0, h,left_bc, right_bc, bottom_bc, top_bc)


def solve_omega1_extension(q1, h):
    n = int(round(1.0 / h))
    N = n + 1

    left_bc = np.full(N, 40.0)
    bottom_bc = np.full(N, 15.0)
    top_bc = np.full(N, 15.0)

    return solver.solve_right_neumann_sparse(1.0, 1.0, h,left_bc, bottom_bc, top_bc, q1)


def solve_omega3_extension(q2, h):
    n = int(round(1.0 / h))
    N = n + 1

    right_bc = np.full(N, 40.0)
    bottom_bc = np.full(N, 15.0)
    top_bc = np.full(N, 15.0)

    return solver.solve_left_neumann_sparse(1.0, 1.0, h,right_bc, bottom_bc, top_bc, q2)


def solve_omega4(q3, h):
    #new Omega4 (0.5 x 0.5)
    half_n = int(round(0.5 / h))
    N4 = half_n + 1

    right_bc = np.full(N4, 15.0)
    bottom_bc = np.full(N4, 40.0)
    top_bc = np.full(N4, 15.0)

    return solver.solve_left_neumann_sparse(0.5, 0.5, h,right_bc, bottom_bc, top_bc, q3)


def compute_q1(U2, h):
    n = int(round(1.0 / h))
    return (U2[1:n, 1] - U2[1:n, 0]) / h


def compute_q2(U2, h):
    n = int(round(1.0 / h))
    return (U2[n + 1:2 * n, -2] - U2[n + 1:2 * n, -1]) / h


def compute_q3(U2, h):
    n = int(round(1.0 / h))
    half = int(round(0.5 / h))
    return (U2[half + 1:n, -2] - U2[half + 1:n, -1]) / h


def plot_temperature(U1, U2, U3, U4):
    x1 = np.linspace(0.0, 1.0, U1.shape[1])
    y1 = np.linspace(0.0, 1.0, U1.shape[0])
    X1, Y1 = np.meshgrid(x1, y1)

    x2 = np.linspace(1.0, 2.0, U2.shape[1])
    y2 = np.linspace(0.0, 2.0, U2.shape[0])
    X2, Y2 = np.meshgrid(x2, y2)

    x3 = np.linspace(2.0, 3.0, U3.shape[1])
    y3 = np.linspace(1.0, 2.0, U3.shape[0])
    X3, Y3 = np.meshgrid(x3, y3)

    x4 = np.linspace(2.0, 2.5, U4.shape[1])
    y4 = np.linspace(0.5, 1.0, U4.shape[0])
    X4, Y4 = np.meshgrid(x4, y4)

    levels = np.linspace(5.0, 40.0, 36)

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.contourf(X1, Y1, U1, levels=levels)
    axis.contourf(X2, Y2, U2, levels=levels)
    axis.contourf(X3, Y3, U3, levels=levels)
    c4 = axis.contourf(X4, Y4, U4, levels=levels)

    fig.colorbar(c4, ax=axis, label="Temperature (°C)")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Temperature distribution in the 2.5-room apartment")
    plt.show()



def main(h=1 / 20, omega=0.8, n_iter=10, make_plot=True):
    if size != 4:
        if rank == 0:
            print("Project 1a requires exactly 4 MPI processes.")
        return

    n = int(round(1.0 / h))
    N = n + 1
    half = int(round(0.5 / h))
    N4 = half + 1

    if rank == 0:
        U1 = np.full((N, N), 15.0)
        gamma1 = U1[1:-1, -1].copy()

    elif rank == 1:
        U2 = np.full((2 * n + 1, N), 15.0)

    elif rank == 2:
        U3 = np.full((N, N), 15.0)
        gamma2 = U3[1:-1, 0].copy()

    elif rank == 3:
        U4 = np.full((N4, N4), 15.0)
        gamma3 = U4[1:-1, 0].copy()

   
    for k in range(n_iter):
        # solve Omega2
        if rank == 0:
            comm.send(gamma1, dest=1, tag=10)
        elif rank == 2:
            comm.send(gamma2, dest=1, tag=20)
        elif rank == 3:
            comm.send(gamma3, dest=1, tag=30)

        # Omega2 solves Dirichlet problem 
        if rank == 1:
            gamma1_recv = comm.recv(source=0, tag=10)
            gamma2_recv = comm.recv(source=2, tag=20)
            gamma3_recv = comm.recv(source=3, tag=30)

            U2_new, A2, b2 = solve_omega2_extension(
                gamma1_recv, gamma2_recv, gamma3_recv, h
            )

            q1 = compute_q1(U2_new, h)
            q2 = compute_q2(U2_new, h)
            q3 = compute_q3(U2_new, h)

            comm.send(q1, dest=0, tag=40)
            comm.send(q2, dest=2, tag=50)
            comm.send(q3, dest=3, tag=60)

        # Outer rooms receive Neumann data and solve their problems
        if rank == 0:
            q1_recv = comm.recv(source=1, tag=40)
            U1_new, A1, b1 = solve_omega1_extension(q1_recv, h)
            U1 = omega * U1_new + (1.0 - omega) * U1

            U1[:, 0] = 40.0
            U1[0, :] = 15.0
            U1[-1, :] = 15.0
            gamma1 = U1[1:-1, -1].copy()

        elif rank == 2:
            q2_recv = comm.recv(source=1, tag=50)
            U3_new, A3, b3 = solve_omega3_extension(q2_recv, h)
            U3 = omega * U3_new + (1.0 - omega) * U3

            U3[:, -1] = 40.0
            U3[0, :] = 15.0
            U3[-1, :] = 15.0
            gamma2 = U3[1:-1, 0].copy()

        elif rank == 3:
            q3_recv = comm.recv(source=1, tag=60)
            U4_new, A4, b4 = solve_omega4(q3_recv, h)
            U4 = omega * U4_new + (1.0 - omega) * U4

            U4[:, -1] = 15.0
            U4[0, :] = 40.0
            U4[-1, :] = 15.0
            gamma3 = U4[1:-1, 0].copy()

        elif rank == 1:
            U2 = omega * U2_new + (1.0 - omega) * U2

            # boundary conditions stay constant
            U2[n:, 0] = 15.0
            U2[:half + 1, -1] = 15.0
            U2[n, -1] = 15.0
            U2[0, :] = 5.0
            U2[-1, :] = 40.0

        comm.Barrier()

    #final result 
    if rank == 1:
        comm.send(U2, dest=0, tag=100)
    elif rank == 2:
        comm.send(U3, dest=0, tag=200)
    elif rank == 3:
        comm.send(U4, dest=0, tag=300)
    elif rank == 0:
        U2_final = comm.recv(source=1, tag=100)
        U3_final = comm.recv(source=2, tag=200)
        U4_final = comm.recv(source=3, tag=300)

        print(f"Final result for h = {h}:")
        print("Omega1 min/max/mean:", U1.min(), U1.max(), U1.mean())
        print("Omega2 min/max/mean:", U2_final.min(), U2_final.max(), U2_final.mean())
        print("Omega3 min/max/mean:", U3_final.min(), U3_final.max(), U3_final.mean())
        print("Omega4 min/max/mean:", U4_final.min(), U4_final.max(), U4_final.mean())

        if make_plot:
            plot_temperature(U1, U2_final, U3_final, U4_final)


if __name__ == "__main__":
    main(h=1 / 20)
    # main(h=1/100)
