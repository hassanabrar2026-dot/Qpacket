import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import splu


# ============================================================
#  GRID AND CONSTANTS
# ============================================================

def make_grid(x_min=-12.0, x_max=12.0, N=1200):
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    return x, dx


def get_constants():
    hbar = 1.0
    m = 1.0
    return hbar, m


# ============================================================
#  POTENTIALS
# ============================================================

def harmonic_oscillator(x, k=1.0):
    return 0.5 * k * x**2


def square_barrier(x, V0=6.0, a=1.0):
    V = np.zeros_like(x)
    V[np.abs(x) < a] = V0
    return V


def double_well(x, a=2.0, k=0.02):
    return k * (x**2 - a**2)**2


def custom_potential(x, expression):
    return eval(expression)


def time_dependent_barrier(x, t, V0=5.0, a=1.0, omega=2.0):
    V = np.zeros_like(x)
    mask = np.abs(x) < a
    V[mask] = V0 * (1 + 0.5 * np.sin(omega * t))
    return V


# ============================================================
#  ABSORBING BOUNDARY (CAP)
# ============================================================

def absorbing_boundary(x, strength=2.0, width=3.0):
    V_cap = np.zeros_like(x, dtype=complex)
    x_min, x_max = x[0], x[-1]

    left = x < (x_min + width)
    right = x > (x_max - width)

    V_cap[left] = -1j * strength * ((x[left] - (x_min + width)) / width)**2
    V_cap[right] = -1j * strength * ((x[right] - (x_max - width)) / width)**2

    return V_cap


# ============================================================
#  WAVE PACKET
# ============================================================

def gaussian_wavepacket(x, x0=-5.0, k0=5.0, sigma=1.0):
    pref = (1.0 / (sigma * np.sqrt(np.pi)))**0.5
    psi = pref * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    return psi


def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm


# ============================================================
#  HAMILTONIAN
# ============================================================

def build_hamiltonian(x, dx, V, hbar=1.0, m=1.0):
    N = len(x)
    pref = -hbar**2 / (2 * m * dx**2)

    diagonals = [
        np.ones(N - 1),
        -2 * np.ones(N),
        np.ones(N - 1),
    ]

    T = pref * diags(diagonals, [-1, 0, 1], dtype=complex)
    Vop = diags(V, 0, dtype=complex)

    return T + Vop


# ============================================================
#  CRANK–NICOLSON SOLVER
# ============================================================

class CrankNicolsonSolver:
    def __init__(self, H, dt, hbar=1.0):
        N = H.shape[0]
        I = diags([np.ones(N)], [0])

        self.A = I + 1j * dt * H / (2 * hbar)
        self.B = I - 1j * dt * H / (2 * hbar)

        self.A_lu = splu(self.A.tocsc())

    def step(self, psi):
        rhs = self.B @ psi
        return self.A_lu.solve(rhs)


# ============================================================
#  EXPECTATION VALUES
# ============================================================

def expectation_x(x, psi, dx):
    return np.real(np.sum(np.conjugate(psi) * x * psi) * dx)


def expectation_p(x, psi, dx, hbar=1.0):
    dpsi = np.zeros_like(psi, dtype=complex)
    dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2 * dx)
    dpsi[0] = (psi[1] - psi[0]) / dx
    dpsi[-1] = (psi[-1] - psi[-2]) / dx

    ppsi = -1j * hbar * dpsi
    return np.real(np.sum(np.conjugate(psi) * ppsi) * dx)


def expectation_energy(psi, H, dx):
    Hpsi = H @ psi
    return np.real(np.sum(np.conjugate(psi) * Hpsi) * dx)


# ============================================================
#  MOMENTUM DISTRIBUTION
# ============================================================

def momentum_distribution(psi, dx):
    N = len(psi)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    psi_k = np.fft.fft(psi)
    pk = np.abs(psi_k)**2
    pk /= pk.max()
    return k, pk


# ============================================================
#  MAIN SIMULATION FUNCTION
# ============================================================

def run_simulation(
    potential_type="double_well",
    use_time_dependent=False,
    x0=-5.0,
    k0=5.0,
    sigma=1.0,
    dt=0.01,
    steps=1500,
    plot_every=20,
    use_zoom_camera=False,
    save_frames=False,
):
    x_min, x_max, N = -12.0, 12.0, 1200
    x, dx = make_grid(x_min, x_max, N)
    hbar, m = get_constants()

    # Choose potential
    if potential_type == "harmonic":
        V_base = harmonic_oscillator(x, k=0.2)
    elif potential_type == "barrier":
        V_base = square_barrier(x, V0=6.0, a=1.0)
    elif potential_type == "custom":
        V_base = custom_potential(x, "0.1*x**4 - 0.5*x**2 + 2*np.exp(-x**2)")
    else:
        V_base = double_well(x, a=2.0, k=0.02)

    V_cap = absorbing_boundary(x, strength=2.0, width=3.0)
    V = V_base + V_cap

    psi = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    psi = normalize(psi, dx)

    H = build_hamiltonian(x, dx, V)
    solver = CrankNicolsonSolver(H, dt)

    fig = plt.figure(figsize=(10, 4))

    for n in range(steps):
        t = n * dt

        if use_time_dependent:
            V_td = time_dependent_barrier(x, t)
            H = build_hamiltonian(x, dx, V_base + V_cap + V_td)
            solver = CrankNicolsonSolver(H, dt)

        if n % plot_every == 0:
            fig.clf()

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(x, np.abs(psi)**2)
            ax1.set_title(f"Position space t={t:.2f}")

            k, pk = momentum_distribution(psi, dx)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(k, pk)
            ax2.set_title("Momentum space")

            plt.pause(0.01)

        psi = solver.step(psi)
        psi = normalize(psi, dx)

    plt.show()


# ============================================================
#  COMMAND-LINE INTERFACE (pip entry point)
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="QPacket 1D TDSE Simulator")

    parser.add_argument("--potential", type=str, default="double_well",
                        choices=["double_well", "harmonic", "barrier", "custom"])
    parser.add_argument("--time-dependent", action="store_true")
    parser.add_argument("--x0", type=float, default=-5.0)
    parser.add_argument("--k0", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--plot-every", type=int, default=20)

    args = parser.parse_args()

    run_simulation(
        potential_type=args.potential,
        use_time_dependent=args.time_dependent,
        x0=args.x0,
        k0=args.k0,
        sigma=args.sigma,
        dt=args.dt,
        steps=args.steps,
        plot_every=args.plot_every,
        use_zoom_camera=args.zoom,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()
