from mpi4py import MPI
import numpy as np
import solver
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def plot_temperature(U1,U2,U3):
    # Omega1 
    x1 = np.linspace(0, 1, U1.shape[1])
    y1 = np.linspace(0, 1, U1.shape[0])
    X1, Y1 = np.meshgrid(x1, y1)

    # Omega2 
    x2 = np.linspace(1, 2, U2.shape[1])
    y2 = np.linspace(0, 2, U2.shape[0])
    X2, Y2 = np.meshgrid(x2, y2)

    # Omega3 
    x3 = np.linspace(2, 3, U3.shape[1])
    y3 = np.linspace(1, 2, U3.shape[0])
    X3, Y3 = np.meshgrid(x3, y3)


    fig, axis = plt.subplots(figsize=(9, 6))

    levels = np.linspace(5, 40, 36)

    c1 = axis.contourf(X1, Y1, U1,levels=levels)
    c2 = axis.contourf(X2, Y2, U2,levels=levels)
    c3 = axis.contourf(X3, Y3, U3,levels=levels)

    fig.colorbar(c3, ax=axis, label="Temperature (°C)")

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Temperature distribution in the apartment")

    plt.show()



def main():

    if size != 3:
        if rank == 0:
            print("This program requires exactly 3 MPI processes.")
        return

    h = 1/20
    omega = 0.8
    n_iter = 10

    n = int(round(1 / h))   # 20 intervals
    N = n + 1               # 21 grid points

    if rank == 0:
        # Omega1
        U1 = np.full((N, N), 15.0)
        gamma1 = U1[1:-1, -1].copy()

    elif rank == 1:
        # Omega2
        U2 = np.full((2*n + 1, N), 15.0)

    elif rank == 2:
        # Omega3
        U3 = np.full((N, N), 15.0)
        gamma2 = U3[1:-1, 0].copy()


    for k in range(n_iter):

        # send last gamma to room2
        if rank == 0:
            comm.send(gamma1,dest=1,tag=10)

        elif rank == 2:
            comm.send(gamma2,dest=1,tag=20)


        # use last gamma to solve next omega2
        if rank == 1:
            gamma1_recv = comm.recv(source=0,tag=10)
            gamma2_recv = comm.recv(source=2,tag=20)

            U2_new, A2, b2 = solver.solve_omega2(gamma1_recv,gamma2_recv,h)

            # compute Neumann to send to room1 and room3
            q1 = solver.compute_q1(U2_new, h)
            q2 = solver.compute_q2(U2_new, h)

            comm.send(q1,dest=0,tag=30)
            comm.send(q2,dest=2,tag=40)


        # solve omega1 and omega3
        if rank == 0:

            q1_recv = comm.recv(source=1,tag=30)
            U1_new, A1, b1 = solver.solve_omega1(q1_recv,h)

            # relaxation
            U1 = (omega * U1_new + (1 - omega) * U1 )

            # boundary may stay constant
            U1[:, 0] = 40.0
            U1[0, :] = 15.0
            U1[-1, :] = 15.0

            # next gamma1
            gamma1 = U1[1:-1, -1].copy()


        elif rank == 2:

            q2_recv = comm.recv(source=1,tag=40)

            U3_new, A3, b3 = solver.solve_omega3(q2_recv,h)

            # relaxation
            U3 = (omega * U3_new+ (1 - omega) * U3)

            U3[:, -1] = 40.0
            U3[0, :] = 15.0
            U3[-1, :] = 15.0

            gamma2 = U3[1:-1, 0].copy()


        elif rank == 1:

            # relaxation for Omega2
            U2 = ( omega * U2_new+ (1 - omega) * U2)

            # side walls
            U2[n:, 0] = 15.0
            U2[:N, -1] = 15.0

            # physical top/bottom
            U2[0, :] = 5.0
            U2[-1, :] = 40.0

        comm.Barrier()


    #results
    if rank == 1:

        comm.send(U2,dest=0,tag=100)

    elif rank == 2:

        comm.send(U3, dest=0,tag=200)

    elif rank == 0:

        U2_final = comm.recv(source=1,tag=100)
        U3_final = comm.recv(source=2,tag=200)

        plot_temperature(U1,U2_final,U3_final)

        print("Final result:")
        print("Omega1:", U1.min(), U1.max())
        print("Omega2:", U2_final.min(), U2_final.max())
        print("Omega3:", U3_final.min(), U3_final.max())

        print("Average temperatures:")
        print("Omega1:", U1.mean())
        print("Omega2:", U2_final.mean())
        print("Omega3:", U3_final.mean())


if __name__ == "__main__":
    main()




