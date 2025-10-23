from fem.chapter1.main import L2Projector1D
import numpy as np
import matplotlib.pyplot as plt

# Our f from the exam

def f(x : np.ndarray) -> np.ndarray:
    return 1-x**2

# Define the uniform mesh from [0, 1] with N = 10, 25, 50
N_values = [10, 25, 50]
meshes = [np.linspace(0, 1, N+1) for N in N_values]

# Get the projections from each mesh
for mesh in meshes:
    projection = L2Projector1D(mesh, f)
    print(f"Mesh with {len(mesh)-1} elements: Projection coefficients = {projection}")
    # Save to a plt line graph
    plt.plot(mesh, projection, marker='o', label=f'N={len(mesh)-1}')
    plt.title("L2 Projection Coefficients")
    plt.xlabel("Mesh Points")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.savefig(f"question1_N{len(mesh)-1}.png", dpi=150)
    plt.close()