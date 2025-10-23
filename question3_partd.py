import numpy as np
from question2_partd import assemble_system_1d

def _element_L2_error(f, Ui, Uj, xi, xj):
    """
    Compute ||f - u_h||_{L2(I)} on one element I=[xi,xj] where u_h is linear
    with nodal values Ui at xi and Uj at xj.
    Uses 2-point Gauss quadrature on the physical element.
    """
    h = xj - xi
    # Gauss points on [-1,1] and weights
    gp = np.array([ -1.0/np.sqrt(3.0),  1.0/np.sqrt(3.0) ])
    w  = np.array([ 1.0,                1.0               ])

    # map to physical element: x = (xi+xj)/2 + (h/2)*t
    xm = 0.5*(xi + xj)
    hf = 0.5*h

    # linear shape functions on [-1,1]
    # phi0(t) = (1 - t)/2, phi1(t) = (1 + t)/2
    err2 = 0.0
    for t, wt in zip(gp, w):
        xq = xm + hf*t
        phi0 = 0.5*(1.0 - t)
        phi1 = 0.5*(1.0 + t)
        uhq = Ui*phi0 + Uj*phi1
        val = f(xq) - uhq
        err2 += wt * (val*val)
    # integral on physical element: sum w * (f-uh)^2 * (h/2)
    return np.sqrt(err2 * hf)


def adaptive_refine_1d(
    x: np.ndarray,
    p: callable, 
    q: callable, 
    f: callable, 
    kappa_0: float,
    kappa_1: float,
    alpha_0: float,
    alpha_1: float,
    beta_0: float, 
    beta_1: float,
    tol: float,
    max_iters: int = 20,
    min_h: float = 1e-10,
    verbose: bool = True,
):
    """
    Adaptive P1 FEM with fixed-rate strategy:
      - Indicator  eta_i = h_i * ||f - u_h||_{L2(I_i)}
      - Refine all elements with eta_i > 0.5 * max_j eta_j
      - Iterate until max(eta) <= tol or max_iters reached.

    Prints which elements are refined each round (indices and midpoints).

    Returns
    -------
    U : np.ndarray
        Nodal FEM solution on the final mesh.
    x : np.ndarray
        Final mesh nodes.
    eta : np.ndarray
        Element indicators on the final mesh.
    history : list[dict]
        Per-iteration diagnostics (max_eta, refined_indices, etc.).
    """
    history = []

    for it in range(1, max_iters+1):
        # (1) Assemble and solve on current mesh
        A, rhs = assemble_system_1d(
            x, p, q, f,
            kappa_0=kappa_0, kappa_1=kappa_1,
            alpha_0=alpha_0, alpha_1=alpha_1,
            beta_0=beta_0,   beta_1=beta_1
        )
        U = np.linalg.solve(A, rhs)

        # (2) Compute indicators eta_i = h_i * ||f - u_h||_{L2(I_i)}
        n_el = len(x) - 1
        eta = np.zeros(n_el)
        for i in range(n_el):
            xi, xj = x[i], x[i+1]
            h = xj - xi
            l2_err = _element_L2_error(f, U[i], U[i+1], xi, xj)
            eta[i] = h * l2_err

        max_eta = float(np.max(eta)) if n_el > 0 else 0.0
        history.append({
            "iter": it,
            "max_eta": max_eta,
        })

        if verbose:
            print(f"[Iter {it}] max eta = {max_eta:.6e}, elements = {n_el}")

        # (3) Check stopping criterion
        if max_eta <= tol or n_el == 0:
            if verbose:
                print(f"Reached tolerance: max eta = {max_eta:.6e} ≤ tol = {tol:.6e}")
            return U, x, eta, history

        # (4) Mark elements with eta_i > 0.5 * max_eta
        threshold = 0.5 * max_eta
        refine_mask = eta > threshold
        refine_indices = np.flatnonzero(refine_mask)

        if verbose:
            if refine_indices.size == 0:
                print("No elements exceeded 0.5 * max_eta; stopping to avoid stagnation.")
                return U, x, eta, history
            midpoints = 0.5*(x[refine_indices] + x[refine_indices+1])
            mids_str = ", ".join(f"{m:.6f}" for m in midpoints)
            idx_str = ", ".join(map(str, refine_indices.tolist()))
            print(f"  Refining {refine_indices.size} elements (indices: [{idx_str}])")
            print(f"  Midpoints inserted at: [{mids_str}]")

        # (5) Refine by inserting midpoints of marked elements
        new_nodes = []
        for i in refine_indices:
            xi, xj = x[i], x[i+1]
            if (xj - xi) <= min_h:
                continue  # skip if already too small
            new_nodes.append(0.5*(xi + xj))

        if not new_nodes:
            if verbose:
                print("All marked elements at min_h; stopping.")
            return U, x, eta, history

        x = np.sort(np.unique(np.concatenate([x, np.array(new_nodes)])))

    if verbose:
        print(f"Stopped after max_iters={max_iters} without reaching tol (max eta = {max_eta:.6e}).")
    return U, x, eta, history




# problem data (yours)
p = lambda x: 1.0
q = lambda x: 1.0
f = lambda x: np.exp(-100*(x-0.5)**2)

# initial mesh
x0 = np.linspace(0.0, 1.0, 3)

U, x_final, eta_final, history = adaptive_refine_1d(
    x0, p, q, f,
    kappa_0=100000.0, kappa_1=-100000.0,
    alpha_0=0.0, alpha_1=0.0,
    beta_0=0.0,  beta_1=0.0,
    tol=1e-3,        # your tolerance for max(eta)
    max_iters=20,
    verbose=True
)


print("Final mesh nodes:", x_final)
print("Final max eta:", np.max(eta_final))
print("Final U shape:", U.shape)

# --- Plot only FEM solution and save as question3.png ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(x_final, U, 'o-', label='FEM solution $u_h$')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Adaptive FEM Solution $u_h$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('question3.png', dpi=150)
plt.close()
