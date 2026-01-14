import time
import numpy as np
from main import (
    run_simulation,
    make_grid,
    gaussian_wavepacket,
    normalize,
    build_hamiltonian,
    expectation_energy,
    square_barrier,
    double_well,
    harmonic_oscillator,
    absorbing_boundary,
    CrankNicolsonSolver,
)


def run_case(name, potential_func, x0, k0, sigma, steps=800, dt=0.01):
    print(f"\n=== Running case: {name} ===")

    # Grid
    x_min, x_max, N = -12.0, 12.0, 1200
    x, dx = make_grid(x_min, x_max, N)

    # Potential
    V_base = potential_func(x)
    V_cap = absorbing_boundary(x, strength=2.0, width=3.0)
    V = V_base + V_cap

    # Initial state
    psi = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    psi = normalize(psi, dx)

    # Hamiltonian
    H = build_hamiltonian(x, dx, V)

    # Solver
    solver = CrankNicolsonSolver(H, dt)

    # Initial energy
    E0 = expectation_energy(psi, H, dx)

    # Time evolution
    t0 = time.time()
    for _ in range(steps):
        psi = solver.step(psi)
        psi = normalize(psi, dx)
    t1 = time.time()

    # Final energy
    Ef = expectation_energy(psi, H, dx)

    # Energy drift
    drift = abs(Ef - E0)

    # Tunneling probability (for barrier)
    left_prob = np.sum(np.abs(psi[x < 0]) ** 2) * dx
    right_prob = np.sum(np.abs(psi[x > 0]) ** 2) * dx

    print(f"Runtime: {t1 - t0:.3f} s")
    print(f"Energy drift: {drift:.3e}")
    print(f"Left prob: {left_prob:.3f}, Right prob: {right_prob:.3f}")

    return {
        "name": name,
        "runtime": t1 - t0,
        "energy_drift": drift,
        "left_prob": left_prob,
        "right_prob": right_prob,
    }


def main():
    results = []

    # 1. Harmonic oscillator
    results.append(
        run_case(
            "Harmonic Oscillator",
            lambda x: harmonic_oscillator(x, k=0.2),
            x0=-3.0,
            k0=3.0,
            sigma=1.0,
        )
    )

    # 2. Square barrier tunneling
    results.append(
        run_case(
            "Square Barrier",
            lambda x: square_barrier(x, V0=6.0, a=1.0),
            x0=-6.0,
            k0=4.0,
            sigma=0.8,
        )
    )

    # 3. Double well tunneling
    results.append(
        run_case(
            "Double Well",
            lambda x: double_well(x, a=2.0, k=0.02),
            x0=-2.0,
            k0=0.0,
            sigma=0.7,
        )
    )

    print("\n\n=== Summary Table ===")
    print(f"{'Case':20s} {'Runtime (s)':12s} {'Energy Drift':15s} {'Right Prob':12s}")
    for r in results:
        print(
            f"{r['name']:20s} "
            f"{r['runtime']:<12.3f} "
            f"{r['energy_drift']:<15.3e} "
            f"{r['right_prob']:<12.3f}"
        )


if __name__ == "__main__":
    main()
