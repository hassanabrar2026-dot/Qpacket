import numpy as np
import matplotlib.pyplot as plt
import time

from main import (
    make_grid,
    gaussian_wavepacket,
    normalize,
    square_barrier,
    absorbing_boundary,
    build_hamiltonian,
    CrankNicolsonSolver,
    expectation_energy,
    momentum_distribution,
)


def run_real_barrier_case():
    # Grid and params
    x_min, x_max, N = -12.0, 12.0, 1200
    x, dx = make_grid(x_min, x_max, N)

    # Physical-like parameters (dimensionless)
    V0 = 6.0
    a = 1.0
    k0 = 6.0
    x0 = -8.0
    sigma = 0.6
    dt = 0.008
    steps = 2500

    # Potential: barrier + absorbing boundaries
    V_base = square_barrier(x, V0=V0, a=a)
    V_cap = absorbing_boundary(x, strength=2.0, width=3.0)
    V = V_base + V_cap

    # Initial wave packet
    psi = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    psi = normalize(psi, dx)

    # Hamiltonian and solver
    H = build_hamiltonian(x, dx, V)
    solver = CrankNicolsonSolver(H, dt)

    # Track energy for plotting
    energies = []
    times = []

    E0 = expectation_energy(psi, H, dx)

    t0 = time.time()
    for n in range(steps):
        psi = solver.step(psi)
        psi = normalize(psi, dx)

        if n % 10 == 0:
            energies.append(expectation_energy(psi, H, dx))
            times.append(n * dt)

    t1 = time.time()

    Ef = expectation_energy(psi, H, dx)
    energy_drift = abs(Ef - E0)

    # Reflection / transmission
    left_mask = x < 0.0
    right_mask = x > 0.0

    R = np.sum(np.abs(psi[left_mask]) ** 2) * dx
    T = np.sum(np.abs(psi[right_mask]) ** 2) * dx

    print("=== Realistic barrier case ===")
    print(f"Runtime: {t1 - t0:.3f} s")
    print(f"Initial energy: {E0:.4f}, Final energy: {Ef:.4f}")
    print(f"Energy drift: {energy_drift:.3e}")
    print(f"Reflection R: {R:.4f}")
    print(f"Transmission T: {T:.4f}")
    print(f"R + T: {R + T:.4f} (should be ~1)")

    # ============================
    # FIGURE 1: Position space
    # ============================
    plt.figure(figsize=(8, 4))
    plt.plot(x, np.abs(psi)**2, label="|ψ(x)|²")
    V_scaled = np.real(V_base) / np.max(np.real(V_base) + 1e-12) * np.max(np.abs(psi)**2)
    plt.plot(x, V_scaled, "k--", alpha=0.6, label="V(x) (scaled)")
    plt.xlabel("x")
    plt.ylabel("|ψ(x)|²")
    plt.title("Final Wave Packet (Position Space)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("barrier_position.png", dpi=300)
    plt.close()

    # ============================
    # FIGURE 2: Momentum space
    # ============================
    k, pk = momentum_distribution(psi, dx)
    plt.figure(figsize=(8, 4))
    plt.plot(k, pk)
    plt.xlabel("k")
    plt.ylabel("|ψ(k)|²")
    plt.title("Final Wave Packet (Momentum Space)")
    plt.tight_layout()
    plt.savefig("barrier_momentum.png", dpi=300)
    plt.close()

    # ============================
    # FIGURE 3: Energy vs time
    # ============================
    plt.figure(figsize=(8, 4))
    plt.plot(times, energies)
    plt.xlabel("t")
    plt.ylabel("⟨H⟩")
    plt.title("Energy Expectation vs Time")
    plt.tight_layout()
    plt.savefig("barrier_energy.png", dpi=300)
    plt.close()

    print("Saved figures:")
    print(" - barrier_position.png")
    print(" - barrier_momentum.png")
    print(" - barrier_energy.png")

    return {
        "runtime": t1 - t0,
        "E0": E0,
        "Ef": Ef,
        "energy_drift": energy_drift,
        "R": R,
        "T": T,
    }


if __name__ == "__main__":
    run_real_barrier_case()
