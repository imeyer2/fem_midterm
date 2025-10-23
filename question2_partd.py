import numpy as np

def assemble_system_1d(
    x: np.ndarray,
    p: callable, 
    q: callable, 
    f: callable, 
    kappa_0: float,
    kappa_1: float,
    alpha_0: float,
    alpha_1: float,
    beta_0: float, 
    beta_1: float  
):
    """
    Assemble A = K + M + R and rhs = b + r for 1D P1 FEM with Robin BCs in the form:
        p(a)u'(a) = k0 (u(a) - alpha0) + alpha1
       -p(b)u'(b) = k1 (u(b) - beta0 ) + beta1
    Sign convention matches your derivation: r1 = -(alpha1 - k0*alpha0), rN = -(beta1 - k1*beta0).
    """
    N = len(x)
    A = np.zeros((N, N), dtype=float)
    rhs = np.zeros(N, dtype=float)

    # element loop
    for e in range(N-1):
        xe, xe1 = x[e], x[e+1]
        h = xe1 - xe
        xm = 0.5 * (xe + xe1)

        # coefficients at midpoint
        pm = p(xm)
        qm = q(xm)

        # local stiffness/mass
        Ke = (pm / h) * np.array([[ 1.0, -1.0],
                                  [-1.0,  1.0]])
        Me = (qm * h / 6.0) * np.array([[2.0, 1.0],
                                         [1.0, 2.0]])

        # consistent load (standard P1)
        fi, fj = f(xe), f(xe1)
        be = (h / 6.0) * np.array([2.0 * fi + fj, fi + 2.0 * fj])

        # assemble
        A[e:e+2, e:e+2] += Ke + Me
        rhs[e:e+2]      += be

    # Robin matrix R and vector r (only at boundary nodes)
    A[0, 0]       += kappa_0
    A[-1, -1]     += kappa_1
    rhs[0]        += -(alpha_1 - kappa_0 * alpha_0)
    rhs[-1]       += -(beta_1  - kappa_1 * beta_0)

    return A, rhs


def impose_dirichlet_strong(A: np.ndarray, rhs: np.ndarray, node: int, value: float):
    """
    Strongly impose u(node) = value by eliminating the DOF.
    Mutates A and rhs in place and returns them.
    """
    N = A.shape[0]
    # shift known value to RHS for all other rows
    for i in range(N):
        if i == node:
            continue
        rhs[i] -= A[i, node] * value
        A[i, node] = 0.0
        A[node, i] = 0.0
    A[node, node] = 1.0
    rhs[node] = value
    return A, rhs



# problem data
p = lambda x: 1.0
q = lambda x: 1.0
f = lambda x: x

# mesh
x = np.linspace(0.0, 1.0, 3)
# x = np.linspace(0.0, 1.0, 5000)

A, rhs = assemble_system_1d(
    x, p, q, f,
    kappa_0=0.0, kappa_1=0.0,
    alpha_0=0.0, alpha_1=0.0,   # unused for Dirichlet
    beta_0=0.0,  beta_1=0.0
)
A, rhs = impose_dirichlet_strong(A, rhs, node=0, value=1.0)
A, rhs = impose_dirichlet_strong(A, rhs, node=len(x)-1, value=2.0)
U_case1 = np.linalg.solve(A, rhs)  # ≈ [1, 18/13, 2]

A2, rhs2 = assemble_system_1d(
    x, p, q, f,
    kappa_0=0.0, kappa_1=0.0,
    alpha_0=0.0, alpha_1=1.0,   # left Neumann u'(0)=1
    beta_0=0.0,  beta_1=0.0     # right is Dirichlet, so no Robin data there
)
A2, rhs2 = impose_dirichlet_strong(A2, rhs2, node=len(x)-1, value=2.0)
U_case2 = np.linalg.solve(A2, rhs2)  # ≈ [0.6428, 1.2266, 2]


A3, rhs3 = assemble_system_1d(
    x, p, q, f,
    kappa_0=0.0, kappa_1=0.0,
    alpha_0=0.0, alpha_1=1.0, 
    beta_0=0.0,  beta_1=-2.0
)
U_case3 = np.linalg.solve(A3, rhs3)


if __name__ == "__main__":
    print("Solution Case 1 (Dirichlet-Dirichlet):", U_case1)
    print("Solution Case 2 (Neumann-Dirichlet):", U_case2)
    print("Solution Case 3 (Neumann-Neumann):", U_case3)

    # --- Visualization ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(x, U_case1, 'o-', label='Dirichlet-Dirichlet')
    plt.plot(x, U_case2, 's--', label='Neumann-Dirichlet')
    plt.plot(x, U_case3, 'd-.', label='Neumann-Neumann')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('FEM Solutions for Different Boundary Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('question2.png', dpi=150)
    plt.close()
