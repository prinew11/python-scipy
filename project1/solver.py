import numpy as np
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# h -> delta x 
def solve_dirichlet(Lx, Ly, h, left_bc, right_bc, bottom_bc, top_bc):

    nx = int(round(Lx / h))   
    ny = int(round(Ly / h))

    Nx = nx + 1  # number of grid points 
    Ny = ny + 1

    nx_inner = Nx - 2 # inner grid points index 
    ny_inner = Ny - 2

    n_unknowns = nx_inner * ny_inner 

    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j-1) * nx_inner + (i-1)

    # u_L + u_R + u_U + u_D - 4u_C = 0

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):

            p = idx(i, j)

            # center --> itself
            A[p, p] = -4.0

            # left
            if i - 1 == 0:
                b[p] -= left_bc[j] 
            else:
                q = idx(i - 1, j)
                A[p, q] = 1.0 

            # right
            if i + 1 == Nx - 1:
                b[p] -= right_bc[j] 
            else:
                q = idx(i + 1, j)
                A[p, q] = 1.0 

            # bottom neighbour
            if j - 1 == 0:
                b[p] -= bottom_bc[i] 
            else:
                q = idx(i, j - 1)
                A[p, q] = 1.0 

            # top
            if j + 1 == Ny - 1:
                b[p] -= top_bc[i] 
            else:
                q = idx(i, j + 1)
                A[p, q] = 1.0

    # Solve Au = b
    u_inner = solve(A, b)

    # Reconstruct full grid
    U = np.zeros((Ny, Nx))

    # Set boundary values directly
    U[:, 0] = left_bc
    U[:, -1] = right_bc

    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    # Fill interior
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            p = idx(i, j)
            U[j, i] = u_inner[p]

    return U, A, b

    



def solve_right_neumann(Lx,Ly,h,left_bc,bottom_bc,top_bc,q_right):


    #U_left + U_top+ U_down -3U_center = -h*U_right


    #grid points
    Nx = int(round(Lx / h)) + 1
    Ny = int(round(Ly / h)) + 1

    # unknown:
    # i = 1,...,Nx-1
    # j = 1,...,Ny-2

    nx_unknown = Nx - 1   # number of unknowns in x direction,right boundary is Neumann also unknown
    ny_unknown = Ny - 2   

    n_unknowns = nx_unknown * ny_unknown

    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j - 1) * nx_unknown + (i - 1)

    for j in range(1, Ny - 1):
        for i in range(1, Nx):

            p = idx(i, j)

            # right neumann 
            if i == Nx - 1:
                A[p, p] = -3.0

                # left
                A[p, idx(i - 1, j)] = 1.0

                # bottom
                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                # top
                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

                # Neumann contribution
                b[p] -= h * q_right[j - 1]


            # normal points
            else:

                A[p, p] = -4.0
                # left
                if i == 1:
                    b[p] -= left_bc[j]
                else:
                    A[p, idx(i - 1, j)] = 1.0

                # right
                A[p, idx(i + 1, j)] = 1.0

                # bottom
                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                # top
                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

    u_unknown = solve(A, b)

    U = np.zeros((Ny, Nx))

    U[:, 0] = left_bc
    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    for j in range(1, Ny - 1):
        for i in range(1, Nx):
            U[j, i] = u_unknown[idx(i, j)]

    return U, A, b



def solve_left_neumann(Lx,Ly,h,right_bc,bottom_bc,top_bc,q_left):


    #U_right + U_top+ U_down -3U_center = -h*U_left

    #grid points
    Nx = int(round(Lx / h)) + 1
    Ny = int(round(Ly / h)) + 1

    # unknown:
    # i = 1,...,Nx-1
    # j = 1,...,Ny-2

    nx_unknown = Nx - 1   # number of unknowns in x direction,left boundary is Neumann also unknown
    ny_unknown = Ny - 2   

    n_unknowns = nx_unknown * ny_unknown

    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j - 1) * nx_unknown + i

    for j in range(1, Ny - 1):
        for i in range(0, Nx-1):

            p = idx(i, j)

            # left neumann 
            if i == 0:
                A[p, p] = -3.0

                # right
                A[p, idx(i + 1, j)] = 1.0

                # bottom
                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                # top
                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

                # Neumann contribution
                b[p] -= h * q_left[j - 1]


            # normal points
            else:

                A[p, p] = -4.0
                # right
                if i == Nx - 2:
                    b[p] -= right_bc[j]
                else:
                    A[p, idx(i + 1, j)] = 1.0

                # left
                A[p, idx(i - 1, j)] = 1.0

                # bottom
                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                # top
                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

    u_unknown = solve(A, b)

    U = np.zeros((Ny, Nx))

    U[:, -1] = right_bc
    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    for j in range(1, Ny - 1):
        for i in range(0, Nx-1):
            U[j, i] = u_unknown[idx(i, j)]

    return U, A, b





def solve_omega2(gamma1, gamma2, h=1/20):

    Nx = int(round(1 / h)) + 1
    Ny = int(round(2 / h)) + 1

    left_bc = np.full(Ny, 15.0)
    right_bc = np.full(Ny, 15.0)

    bottom_bc = np.full(Nx, 5.0)
    top_bc = np.full(Nx, 40.0)


    left_bc[1:Nx-1] = gamma1
    right_bc[Nx: Ny - 1 ] = gamma2


    U2, A2, b2 = solve_dirichlet(Lx=1.0, Ly=2.0,h=h,left_bc=left_bc,right_bc=right_bc,bottom_bc=bottom_bc,top_bc=top_bc)
    # U2, A2, b2 = solve_dirichlet_sparse(Lx=1.0, Ly=2.0,h=h,left_bc=left_bc,right_bc=right_bc,bottom_bc=bottom_bc,top_bc=top_bc)

    return U2, A2, b2



def solve_omega1(q1, h):

    N = int(round(1 / h))

    Nx = N + 1
    Ny = N + 1

    left_bc = np.full(Ny, 40.0)
    bottom_bc = np.full(Nx, 15.0)
    top_bc = np.full(Nx, 15.0)

    return solve_right_neumann(1.0,1.0,h,left_bc,bottom_bc,top_bc,q1)
    # return solve_right_neumann_sparse(1.0,1.0,h,left_bc,bottom_bc,top_bc,q1)

def solve_omega3(q2, h):

    N = int(round(1 / h))

    Nx = N + 1
    Ny = N + 1

    right_bc = np.full(Ny, 40.0)
    bottom_bc = np.full(Nx, 15.0)
    top_bc = np.full(Nx, 15.0)


    return solve_left_neumann(1.0,1.0,h,right_bc, bottom_bc,top_bc,q2)
    # return solve_left_neumann_sparse(1.0,1.0,h,right_bc, bottom_bc,top_bc,q2)


def compute_q1(U2, h):
    N = int(round(1 / h))
    return (U2[1:N, 1] - U2[1:N, 0]) / h


def compute_q2(U2, h):
    N = int(round(1 / h))
    return (U2[N+1:2*N, -2]- U2[N+1:2*N, -1]) / h


def solve_dirichlet_sparse(Lx, Ly, h, left_bc, right_bc, bottom_bc, top_bc):
    nx = int(round(Lx / h))
    ny = int(round(Ly / h))

    Nx = nx + 1
    Ny = ny + 1

    nx_inner = Nx - 2
    ny_inner = Ny - 2
    n_unknowns = nx_inner * ny_inner

    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j - 1) * nx_inner + (i - 1)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            p = idx(i, j)
            A[p, p] = -4.0

            if i == 1:
                b[p] -= left_bc[j]
            else:
                A[p, idx(i - 1, j)] = 1.0

            if i == Nx - 2:
                b[p] -= right_bc[j]
            else:
                A[p, idx(i + 1, j)] = 1.0

            if j == 1:
                b[p] -= bottom_bc[i]
            else:
                A[p, idx(i, j - 1)] = 1.0

            if j == Ny - 2:
                b[p] -= top_bc[i]
            else:
                A[p, idx(i, j + 1)] = 1.0

    A = A.tocsr()  # convert to sparse
    u_inner = spsolve(A, b)

    U = np.zeros((Ny, Nx))
    U[:, 0] = left_bc
    U[:, -1] = right_bc
    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            U[j, i] = u_inner[idx(i, j)]

    return U, A, b


def solve_right_neumann_sparse(Lx, Ly, h, left_bc, bottom_bc, top_bc, q_right):
    Nx = int(round(Lx / h)) + 1
    Ny = int(round(Ly / h)) + 1

    nx_unknown = Nx - 1
    ny_unknown = Ny - 2
    n_unknowns = nx_unknown * ny_unknown

    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j - 1) * nx_unknown + (i - 1)

    for j in range(1, Ny - 1):
        for i in range(1, Nx):
            p = idx(i, j)

            if i == Nx - 1:
                A[p, p] = -3.0
                A[p, idx(i - 1, j)] = 1.0

                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

                b[p] -= h * q_right[j - 1]

            else:
                A[p, p] = -4.0

                if i == 1:
                    b[p] -= left_bc[j]
                else:
                    A[p, idx(i - 1, j)] = 1.0

                A[p, idx(i + 1, j)] = 1.0

                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

    A = A.tocsr()
    u_unknown = spsolve(A, b)

    U = np.zeros((Ny, Nx))
    U[:, 0] = left_bc
    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    for j in range(1, Ny - 1):
        for i in range(1, Nx):
            U[j, i] = u_unknown[idx(i, j)]

    return U, A, b


def solve_left_neumann_sparse(Lx, Ly, h, right_bc, bottom_bc, top_bc, q_left):
    Nx = int(round(Lx / h)) + 1
    Ny = int(round(Ly / h)) + 1

    nx_unknown = Nx - 1
    ny_unknown = Ny - 2
    n_unknowns = nx_unknown * ny_unknown

    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns)

    def idx(i, j):
        return (j - 1) * nx_unknown + i

    for j in range(1, Ny - 1):
        for i in range(0, Nx - 1):
            p = idx(i, j)

            if i == 0:
                A[p, p] = -3.0
                A[p, idx(i + 1, j)] = 1.0

                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

                b[p] -= h * q_left[j - 1]

            else:
                A[p, p] = -4.0

                if i == Nx - 2:
                    b[p] -= right_bc[j]
                else:
                    A[p, idx(i + 1, j)] = 1.0

                A[p, idx(i - 1, j)] = 1.0

                if j == 1:
                    b[p] -= bottom_bc[i]
                else:
                    A[p, idx(i, j - 1)] = 1.0

                if j == Ny - 2:
                    b[p] -= top_bc[i]
                else:
                    A[p, idx(i, j + 1)] = 1.0

    A = A.tocsr()
    u_unknown = spsolve(A, b)

    U = np.zeros((Ny, Nx))
    U[:, -1] = right_bc
    U[0, :] = bottom_bc
    U[-1, :] = top_bc

    for j in range(1, Ny - 1):
        for i in range(0, Nx - 1):
            U[j, i] = u_unknown[idx(i, j)]

    return U, A, b



if __name__ == "__main__":
    h = 1/100
    omega = 0.8
    n = int(round(1 / h))
    N  = n + 1

    gamma1 = np.full(n-1, 15.0)
    gamma2 = np.full(n-1, 15.0)
    U1 = np.full((N, N), 15.0)
    U2 = np.full((2 * n + 1, N), 15.0)
    U3 = np.full((N, N), 15.0)

    for i in range(10):
        U2_new, A2, b2 = solve_omega2(gamma1, gamma2, h)

        q1 = compute_q1(U2_new, h)
        q2 = compute_q2(U2_new, h)

        U1_new, A1, b1 = solve_omega1(q1, h)
        U3_new, A3, b3 = solve_omega3(q2, h)

        U1 = omega * U1_new + (1 - omega) * U1
        U2 = omega * U2_new + (1 - omega) * U2
        U3 = omega * U3_new + (1 - omega) * U3


        # boundary may stay constant
        U1[:, 0] = 40.0
        U1[0, :] = 15.0
        U1[-1, :] = 15.0

        U2[N-1:, 0] = 15.0
        U2[:N, -1] = 15.0
        U2[0, :] = 5.0
        U2[-1, :] = 40.0

        U3[:, -1] = 40.0
        U3[0, :] = 15.0
        U3[-1, :] = 15.0

        old_gamma1 = gamma1.copy()
        old_gamma2 = gamma2.copy()

        gamma1 = U1[1:N-1, -1].copy()
        gamma2 = U3[1:N-1, 0].copy()

        err1 = np.linalg.norm(gamma1 - old_gamma1)
        err2 = np.linalg.norm(gamma2 - old_gamma2)

        err = max(err1, err2)

        print(
            f"iteration {i+1}: "
            f"change={err:.6e}, "
            f"gamma1_mean={gamma1.mean():.4f}, "
            f"gamma2_mean={gamma2.mean():.4f}"
        )

    

    print(U1.min(), U1.max())
    print(U2.min(), U2.max())
    print(U3.min(), U3.max())




