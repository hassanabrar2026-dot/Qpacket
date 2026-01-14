import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import splu


# ---------- Grid and constants ----------
def make_grid(x_min=-10.0, x_max=10.0, N=1000):
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    return x, dx


def get_constants():
    # Natural units: ℏ = m = 1
    hbar = 1.0
    m = 1.0
    return hbar, m


# ---------- Potentials ----------
def harmonic_oscillator(x, k=1.0):
    return 0.5 * k * x**2


def square_barrier(x, V0=5.0, a=1.0):
    V = np.zeros_like(x)
    V[np.abs(x) < a] = V0
    return V


def double_well(x, a=2.0, k=0.02):
    return k * (x**2 - a**2) ** 2


def custom_potential(x, expression):
    """
    Design your own potential using a Python expression in x and np.
    Example: "0.2*x**4 - 3*x**2 + 5*np.exp(-x**2)"
    """
    return eval(expression)


def time_dependent_barrier(x, t, V0=5.0, a=1.0, omega=2.0):
    """
    Example of a time-dependent potential:
    oscillating square barrier with frequency omega.
    """
    V = np.zeros_like(x)
    mask = np.abs(x) < a
    V[mask] = V0 * (1.0 + 0.5 * np.sin(omega * t))
    return V


# ---------- Absorbing boundary (complex absorbing potential) ----------
def absorbing_boundary(x, strength=1.0, width=3.0):
    """
    Smooth complex absorbing potential to suppress reflections at boundaries.
    V_cap is purely imaginary and negative: -i * something(x)
    """
    V_cap = np.zeros_like(x, dtype=complex)
    x_min, x_max = x[0], x[-1]

    left_mask = x < (x_min + width)
    right_mask = x > (x_max - width)

    # Quadratic profile inside absorbing region
    V_cap[left_mask] = -1j * strength * (
        (x[left_mask] - (x_min + width)) / width
    ) ** 2
    V_cap[right_mask] = -1j * strength * (
        (x[right_mask] - (x_max - width)) / width
    ) ** 2

    return V_cap


# ---------- Wave packet ----------
def gaussian_wavepacket(x, x0=-5.0, k0=5.0, sigma=1.0):
    prefactor = (1.0 / (sigma * np.sqrt(np.pi))) ** 0.5
    psi = prefactor * np.exp(-(x - x0) ** 2 / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    return psi


def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        return psi
    return psi / norm


# ---------- Hamiltonian & Crank–Nicolson solver ----------
def build_hamiltonian(x, dx, V, hbar=1.0, m=1.0):
    """
    Build 1D Hamiltonian H = T + V using finite-difference Laplacian.
    Ensures complex dtype so absorbing boundaries work.
    """
    N = len(x)
    kinetic_prefactor = -hbar**2 / (2.0 * m * dx**2)

    diagonals = [
        np.ones(N - 1),
        -2.0 * np.ones(N),
        np.ones(N - 1),
    ]

    # Force complex dtype to avoid float64/complex128 mismatch
    T = kinetic_prefactor * diags(diagonals, [-1, 0, 1], dtype=complex)
    V_op = diags(V, 0, dtype=complex)

    return T + V_op


class CrankNicolsonSolver:
    def __init__(self, H, dt, hbar=1.0):
        """
        Precompute LU factorization for efficient time stepping.
        """
        self.H = H
        self.dt = dt
        self.hbar = hbar
        N = H.shape[0]
        I = diags([np.ones(N)], [0])

        self.A = I + 1j * dt * H / (2.0 * hbar)
        self.B = I - 1j * dt * H / (2.0 * hbar)

        self.A_lu = splu(self.A.tocsc())

    def step(self, psi):
        rhs = self.B @ psi
        return self.A_lu.solve(rhs)


# ---------- Expectation values ----------
def expectation_x(x, psi, dx):
    return np.real(np.sum(np.conjugate(psi) * x * psi) * dx)


def expectation_p(x, psi, dx, hbar=1.0):
    """
    Approximate ⟨p⟩ using finite-difference for ∂/∂x.
    """
    dpsi_dx = np.zeros_like(psi, dtype=complex)
    dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dx)
    dpsi_dx[0] = (psi[1] - psi[0]) / dx
    dpsi_dx[-1] = (psi[-1] - psi[-2]) / dx

    p_op_psi = -1j * hbar * dpsi_dx
    return np.real(np.sum(np.conjugate(psi) * p_op_psi) * dx)


def expectation_energy(psi, H, dx):
    """
    ⟨H⟩ = ∑ ψ* (Hψ) dx
    """
    Hpsi = H @ psi
    return np.real(np.sum(np.conjugate(psi) * Hpsi) * dx)


# ---------- Momentum-space distribution ----------
def momentum_distribution(psi, dx):
    """
    Returns k grid and normalized |ψ(k)|^2 via FFT.
    """
    N = len(psi)
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    psi_k = np.fft.fft(psi)
    prob_k = np.abs(psi_k) ** 2
    if prob_k.max() > 0:
        prob_k /= prob_k.max()
    return k, prob_k


# ---------- Zoom camera ----------
def zoom_window(x, psi, window=4.0):
    """
    Returns (x_zoom, psi_zoom) within a window around ⟨x⟩.
    """
    x_center = expectation_x(x, psi, x[1] - x[0])
    mask = np.abs(x - x_center) < window
    return x[mask], psi[mask]


# ---------- Frame saving ----------
def save_frame(fig, n, folder="frames"):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"frame_{n:05d}.png"), dpi=120)


# ---------- Main simulation ----------
def run_simulation(
    potential_type="double_well",
    use_time_dependent=False,
    x_min=-12.0,
    x_max=12.0,
    N=1200,
    x0=-5.0,
    k0=5.0,
    sigma=1.0,
    dt=0.01,
    steps=1500,
    plot_every=20,
    use_zoom_camera=False,
    save_frames=False,
):
    # --- Grid and constants ---
    x, dx = make_grid(x_min, x_max, N)
    hbar, m = get_constants()

    # --- Choose base potential ---
    if potential_type == "harmonic":
        V_base = harmonic_oscillator(x, k=0.2)
    elif potential_type == "barrier":
        V_base = square_barrier(x, V0=6.0, a=1.0)
    elif potential_type == "custom":
        # You can modify this expression as you like
        V_base = custom_potential(x, "0.1*x**4 - 0.5*x**2 + 2*np.exp(-x**2)")
    else:
        # default: double well
        V_base = double_well(x, a=2.0, k=0.02)

    # --- Add absorbing boundary layer ---
    V_cap = absorbing_boundary(x, strength=2.0, width=3.0)
    V = V_base + V_cap

    # --- Initial wave packet ---
    psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    psi0 = normalize(psi0, dx)

    # --- Hamiltonian and solver (initial, for static V) ---
    H = build_hamiltonian(x, dx, V, hbar=hbar, m=m)
    solver = CrankNicolsonSolver(H, dt, hbar=hbar)

    # --- Time evolution ---
    psi = psi0.copy()

    times = []
    x_expect_vals = []
    p_expect_vals = []
    energy_expect_vals = []

    fig = plt.figure(figsize=(10, 4))

    for n in range(steps):
        t = n * dt

        if use_time_dependent:
            # Example: modulation on top of base + CAP
            V_td = time_dependent_barrier(x, t, V0=4.0, a=1.0, omega=2.0)
            H = build_hamiltonian(x, dx, V_base + V_cap + V_td, hbar=hbar, m=m)
            solver = CrankNicolsonSolver(H, dt, hbar=hbar)

        # --- Expectation values ---
        ex = expectation_x(x, psi, dx)
        ep = expectation_p(x, psi, dx, hbar=hbar)
        E = expectation_energy(psi, H, dx)

        times.append(t)
        x_expect_vals.append(ex)
        p_expect_vals.append(ep)
        energy_expect_vals.append(E)

        # --- Visualization ---
        if n % plot_every == 0 or n == 0:
            fig.clf()

            # Position-space subplot
            ax1 = fig.add_subplot(1, 2, 1)
            if use_zoom_camera:
                x_plot, psi_plot = zoom_window(x, psi, window=4.0)
                ax1.plot(x_plot, np.abs(psi_plot) ** 2, label="|ψ(x)|² (zoom)")
            else:
                ax1.plot(x, np.abs(psi) ** 2, label="|ψ(x)|²")

            # Rescaled real part of base potential for reference (ignore imaginary CAP)
            V_real = np.real(V_base)
            if np.max(V_real) > 0:
                V_scaled = V_real / np.max(V_real) * np.max(np.abs(psi) ** 2)
                ax1.plot(x, V_scaled, "k--", alpha=0.5, label="V(x) rescaled")

            ax1.set_xlabel("x")
            ax1.set_ylabel("|ψ(x)|²")
            ax1.set_title(f"Position space, t = {t:.2f}")
            ax1.legend(loc="upper right")

            # Momentum-space subplot
            ax2 = fig.add_subplot(1, 2, 2)
            k, pk = momentum_distribution(psi, dx)
            ax2.plot(k, pk)
            ax2.set_xlabel("k")
            ax2.set_ylabel("|ψ(k)|² (normalized)")
            ax2.set_title("Momentum space")

            plt.tight_layout()
            plt.pause(0.01)

            if save_frames:
                save_frame(fig, n)

        # --- Time step ---
        psi = solver.step(psi)
        psi = normalize(psi, dx)  # keep normalization under control

    plt.show()

    # --- Plot expectation values ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(times, x_expect_vals)
    plt.xlabel("t")
    plt.ylabel("⟨x⟩")

    plt.subplot(1, 3, 2)
    plt.plot(times, p_expect_vals)
    plt.xlabel("t")
    plt.ylabel("⟨p⟩")

    plt.subplot(1, 3, 3)
    plt.plot(times, energy_expect_vals)
    plt.xlabel("t")
    plt.ylabel("⟨H⟩")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="1D quantum wave packet simulator (Crank–Nicolson)"
    )

    parser.add_argument(
        "--potential",
        type=str,
        default="double_well",
        choices=["double_well", "harmonic", "barrier", "custom"],
        help="Type of potential to use",
    )
    parser.add_argument(
        "--time-dependent",
        action="store_true",
        help="Enable time-dependent barrier on top of base potential",
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=-5.0,
        help="Initial center position of the wave packet",
    )
    parser.add_argument(
        "--k0",
        type=float,
        default=5.0,
        help="Initial momentum of the wave packet",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Width of the initial wave packet",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of time steps to evolve",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step size",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Enable zoom camera that follows the wave packet",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames to ./frames for video creation",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=20,
        help="Plot every N steps",
    )

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
