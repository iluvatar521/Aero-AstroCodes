"""
nozzle_ui.py
============

Requires: numpy, scipy, matplotlib  (pip install numpy scipy matplotlib)

Run with:  python nozzle_ui.py
This interactive simulator makes the theory tangible - dragging sliders for geometry, stagnation conditions, and back pressure shows the flow regime shift in real time - unchoked, shock-in-duct, over-expanded, perfectly expanded -
while the schematic redraws shock locations and expansion fans instantly.
Watching Mach, pressure, and temperature profiles update alongside the governing equations connects abstract math to physical intuition far better than static diagrams.
The three critical back-pressure thresholds become tangible boundaries rather than numbers, building the intuition engineers need to design real propulsion and flow systems.

"""

import math
import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ======================================================================
#  GAS-DYNAMICS FUNCTIONS
# ======================================================================

class NozzleError(Exception):
    """Raised for any physically/numerically invalid nozzle configuration."""
    pass


def area_ratio(M, gamma):
    """Isentropic area-Mach relation: A/A* as a function of M."""
    M = max(M, 1e-8)
    return (1.0 / M) * ((2.0 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M ** 2)) \
        ** ((gamma + 1) / (2 * (gamma - 1)))


def mach_from_area_ratio(AR, gamma, branch):
    """Invert the area-Mach relation for the 'sub' or 'sup' root (A/A* >= 1)."""
    if AR < 1 - 1e-9:
        raise NozzleError(f"Area ratio A/A* must be >= 1 (got {AR:.6g}).")
    AR = max(AR, 1.0)

    def f(M):
        return area_ratio(M, gamma) - AR

    if branch == "sub":
        if AR <= 1 + 1e-12:
            return 1.0
        return brentq(f, 1e-6, 1 - 1e-12)
    else:
        if AR <= 1 + 1e-12:
            return 1.0
        return brentq(f, 1 + 1e-12, 80)


def p_p0(M, gamma):
    """Isentropic static-to-stagnation pressure ratio."""
    return (1 + (gamma - 1) / 2 * M ** 2) ** (-gamma / (gamma - 1))


def t_t0(M, gamma):
    """Isentropic static-to-stagnation temperature ratio."""
    return 1.0 / (1 + (gamma - 1) / 2 * M ** 2)


def mach_from_pressure_ratio(pr, gamma, branch):
    """Invert p/p0 = pr for M on the 'sub' or 'sup' branch."""
    def f(M):
        return p_p0(M, gamma) - pr

    if branch == "sub":
        return brentq(f, 1e-9, 1 - 1e-12)
    else:
        return brentq(f, 1 + 1e-12, 80)


def normal_shock(M1, gamma):
    """Normal-shock jump relations given the upstream Mach number M1 (>1)."""
    M2 = math.sqrt((1 + (gamma - 1) / 2 * M1 ** 2) / (gamma * M1 ** 2 - (gamma - 1) / 2))
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (M1 ** 2 - 1)
    rho2_rho1 = ((gamma + 1) * M1 ** 2) / ((gamma - 1) * M1 ** 2 + 2)
    T2_T1 = p2_p1 / rho2_rho1
    p02_p01 = (((gamma + 1) * M1 ** 2 / ((gamma - 1) * M1 ** 2 + 2)) ** (gamma / (gamma - 1))) * \
              ((gamma + 1) / (2 * gamma * M1 ** 2 - (gamma - 1))) ** (1 / (gamma - 1))
    return M2, p2_p1, T2_T1, p02_p01, rho2_rho1


def pressure_given_shock_at(Ash, At, Ae, P0, gamma):
    """
    Given a normal-shock location at area Ash (At <= Ash <= Ae) inside the
    diverging duct, compute the resulting exit static pressure Pe together
    with the Mach numbers just upstream/downstream of the shock (M1s, M2s),
    the exit Mach number Me, and the post-shock stagnation pressure P02.
    """
    M1s = mach_from_area_ratio(Ash / At, gamma, "sup")
    M2s, _, _, p02_p01, _ = normal_shock(M1s, gamma)
    P02 = P0 * p02_p01
    Astar2 = Ash / area_ratio(M2s, gamma)
    Me = mach_from_area_ratio(Ae / Astar2, gamma, "sub")
    Pe = P02 * p_p0(Me, gamma)
    return Pe, M1s, M2s, Me, P02


def choked_mass_flow(Astar, P0, T0, gamma, R):
    """Mass flow rate through an area Astar at which M=1."""
    return Astar * P0 * math.sqrt(gamma / (R * T0)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))


def pm_nu(M, gamma):
    """Prandtl-Meyer function (radians)."""
    M = max(M, 1.0)
    return math.sqrt((gamma + 1) / (gamma - 1)) * math.atan(math.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1))) \
        - math.atan(math.sqrt(M ** 2 - 1))


def pm_nu_inv(nu_target, gamma):
    """Invert the Prandtl-Meyer function for M (M>=1)."""
    numax = math.sqrt((gamma + 1) / (gamma - 1)) * math.pi / 2 - math.pi / 2
    nu_target = min(nu_target, numax - 1e-6)

    def f(M):
        return pm_nu(M, gamma) - nu_target

    return brentq(f, 1 + 1e-12, 80)


def theta_of_beta(M1, beta, gamma):
    """theta-beta-M relation: flow deflection angle for given M1 and shock angle beta."""
    tan_theta = 2 / math.tan(beta) * (M1 ** 2 * math.sin(beta) ** 2 - 1) / \
        (M1 ** 2 * (gamma + math.cos(2 * beta)) + 2)
    return math.atan(tan_theta)


def theta_max(M1, gamma):
    """Maximum flow-deflection angle sustaining an attached oblique shock at M1."""
    num = (gamma + 1) * M1 ** 2 / 4 - 1 + math.sqrt(
        (gamma + 1) * ((gamma + 1) * M1 ** 4 / 16 + (gamma - 1) * M1 ** 2 / 2 + 1))
    sin2b = num / (gamma * M1 ** 2)
    sin2b = min(max(sin2b, 0.0), 1.0)
    beta_at_theta_max = math.asin(math.sqrt(sin2b))
    theta_max_val = theta_of_beta(M1, beta_at_theta_max, gamma)
    return theta_max_val, beta_at_theta_max


def oblique_from_pressure(M1, p2_p1, gamma):
    """
    Given M1 and the required static pressure ratio p2/p1 across an oblique
    shock, find theta, downstream Mach M2, and stagnation-pressure ratio.
    """
    sin2b = ((p2_p1 - 1) * (gamma + 1) / (2 * gamma) + 1) / M1 ** 2
    sin2b = min(max(sin2b, 0.0), 1.0)
    beta = math.asin(math.sqrt(sin2b))
    Mn1 = M1 * math.sin(beta)
    if Mn1 < 1:
        raise NozzleError(
            f"Required pressure ratio cannot be produced by an oblique shock at "
            f"M1={M1:.3f} (would need Mn1<1)."
        )
    Mn2 = math.sqrt((1 + (gamma - 1) / 2 * Mn1 ** 2) / (gamma * Mn1 ** 2 - (gamma - 1) / 2))
    theta = theta_of_beta(M1, beta, gamma)
    M2 = Mn2 / math.sin(beta - theta)
    p02_p01 = (((gamma + 1) * Mn1 ** 2 / ((gamma - 1) * Mn1 ** 2 + 2)) ** (gamma / (gamma - 1))) * \
              ((gamma + 1) / (2 * gamma * Mn1 ** 2 - (gamma - 1))) ** (1 / (gamma - 1))
    return theta, M2, p02_p01


def oblique_from_theta(M1, theta, gamma):
    """
    Given M1 and a required flow deflection theta, find the weak oblique-shock
    solution. Raises if theta exceeds the max attached-shock angle (detached shock).
    """
    theta_max_val, beta_at_theta_max = theta_max(M1, gamma)
    if theta >= theta_max_val:
        raise NozzleError(
            f"Required deflection ({math.degrees(theta):.2f} deg) exceeds the max "
            f"attached-shock angle ({math.degrees(theta_max_val):.2f} deg) at "
            f"M1={M1:.3f}: shock would detach."
        )
    beta_mach = math.asin(1 / M1)

    def f(beta):
        return math.tan(theta) - 2 / math.tan(beta) * (M1 ** 2 * math.sin(beta) ** 2 - 1) / \
            (M1 ** 2 * (gamma + math.cos(2 * beta)) + 2)

    beta = brentq(f, beta_mach + 1e-8, beta_at_theta_max)
    Mn1 = M1 * math.sin(beta)
    Mn2 = math.sqrt((1 + (gamma - 1) / 2 * Mn1 ** 2) / (gamma * Mn1 ** 2 - (gamma - 1) / 2))
    M2 = Mn2 / math.sin(beta - theta)
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (Mn1 ** 2 - 1)
    p02_p01 = (((gamma + 1) * Mn1 ** 2 / ((gamma - 1) * Mn1 ** 2 + 2)) ** (gamma / (gamma - 1))) * \
              ((gamma + 1) / (2 * gamma * Mn1 ** 2 - (gamma - 1))) ** (1 / (gamma - 1))
    return M2, p2_p1, p02_p01


# ======================================================================
#  WALL GEOMETRY: CONE / CIRCULAR-ARC-FILLET / CONE
# ======================================================================

def solve_throat_fillet(A1, At, Ae, Lconv, Ldiv, Rc):
    """
    Build a wall-radius profile r(x) for x in [-Lconv, Ldiv] made of:
        - a straight cone from (-Lconv, r1) to a tangent point (x1, r(x1))
        - a circular arc of radius Rc, centered on the axis at (0, rt+Rc),
          running from x1 through the throat (x=0, r=rt) to x2
        - a straight cone from the tangent point (x2, r(x2)) to (Ldiv, re)

    The SAME arc carries the wall across the sonic point, so curvature
    (d^2r/dx^2) is continuous through the throat -- this is what removes
    the artificial Mach/pressure/temperature kink at M=1.

    The tangent stations x1 (<0) and x2 (>0) are found by solving the
    tangency condition "line slope == arc slope" directly with brentq,
    which is equivalent to (and simpler than) the general external-point-
    to-circle tangent-line construction, because we only ever need the
    tangent point on the lower arc r = yc - sqrt(Rc^2 - x^2).

    Returns
    -------
    r_of_x : callable, vectorized, valid on [-Lconv, Ldiv]
    x1, x2 : tangent stations (x1 < 0 < x2)
    rt     : throat radius (sanity return)
    """
    r1 = math.sqrt(A1 / math.pi)
    rt = math.sqrt(At / math.pi)
    re = math.sqrt(Ae / math.pi)

    if Rc <= 0:
        raise NozzleError("Throat fillet radius Rc must be positive.")
    if Rc >= Lconv or Rc >= Ldiv:
        raise NozzleError(
            f"Throat fillet radius Rc={Rc:.4g} m is too large for the plotted "
            f"duct lengths (Lconv={Lconv:.4g} m, Ldiv={Ldiv:.4g} m). Reduce "
            f"Rc/rt or increase Lconv/Ldiv."
        )

    yc = rt + Rc

    def upstream_eq(x1):
        s = math.sqrt(max(Rc ** 2 - x1 ** 2, 1e-14))
        r_x1 = yc - s
        m1 = x1 / s
        line_slope = (r_x1 - r1) / (x1 - (-Lconv))
        return line_slope - m1

    def downstream_eq(x2):
        s = math.sqrt(max(Rc ** 2 - x2 ** 2, 1e-14))
        r_x2 = yc - s
        m2 = x2 / s
        line_slope = (re - r_x2) / (Ldiv - x2)
        return line_slope - m2

    eps = 1e-9 * Rc
    try:
        x1 = brentq(upstream_eq, -Rc + eps, -eps)
        x2 = brentq(downstream_eq, eps, Rc - eps)
    except ValueError:
        raise NozzleError(
            "Cannot fit the requested throat fillet radius into the given "
            "geometry (Rc too large relative to A1/At/Ae, or Lconv/Ldiv too "
            "short). Try reducing Rc/rt or increasing Lconv/Ldiv."
        )

    if not (-Lconv < x1 < 0) or not (0 < x2 < Ldiv):
        raise NozzleError(
            "Throat fillet tangent points fall outside the plotted duct "
            "length. Reduce Rc/rt or increase Lconv/Ldiv."
        )

    r_x1 = yc - math.sqrt(Rc ** 2 - x1 ** 2)
    r_x2 = yc - math.sqrt(Rc ** 2 - x2 ** 2)
    m1 = x1 / math.sqrt(Rc ** 2 - x1 ** 2)
    m2 = x2 / math.sqrt(Rc ** 2 - x2 ** 2)

    def r_of_x(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        r = np.empty_like(x)
        left = x <= x1
        right = x >= x2
        mid = ~left & ~right

        r[left] = r1 + m1 * (x[left] - (-Lconv))
        r[right] = r_x2 + m2 * (x[right] - x2)
        xm = x[mid]
        r[mid] = yc - np.sqrt(np.clip(Rc ** 2 - xm ** 2, 0.0, None))
        return r

    return r_of_x, x1, x2, rt


# ======================================================================
#  CORE COMPUTATION
# ======================================================================

def compute_nozzle(P):
    """
    Given a dict of parameters P (see default_params()), return
    (log_str, res_dict, ok).
    """
    log_lines = []

    def L(fmt, *args):
        log_lines.append(fmt % args if args else fmt)

    res = {}
    try:
        gamma = P["gamma"]
        R = P["R"]
        supersonic_inlet = P["supersonicInlet"]
        At = P["At"]
        Ae = P["Ae"]
        P0 = P["P0"]
        T0 = P["T0"]
        Pb = P["Pb"]
        Lconv = P["Lconv"]
        Ldiv = P["Ldiv"]
        RcRatio = P["RcRatio"]

        if gamma <= 1:
            raise NozzleError("gamma must be greater than 1.")
        if R <= 0:
            raise NozzleError("Gas constant R must be positive.")
        if RcRatio <= 0:
            raise NozzleError("Throat fillet ratio Rc/rt must be positive.")

        if supersonic_inlet:
            M1_in = P["M1"]
            if M1_in <= 1:
                L("WARNING: M1 <= 1 is not supersonic; using 2.0 instead.")
                M1_in = 2.0
        else:
            A1 = P["A1"]

        if supersonic_inlet:
            if M1_in <= 1:
                raise NozzleError(f"Supersonic-inlet mode requires M1 > 1 (got {M1_in:.4g}).")
            A1 = At * area_ratio(M1_in, gamma)

        if At >= A1 or At >= Ae:
            raise NozzleError("Throat area At must be smaller than both A1 and Ae.")
        if P0 <= 0 or T0 <= 0 or Pb <= 0:
            raise NozzleError("P0, T0 and Pb must all be positive.")
        if Pb > P0:
            raise NozzleError("Back pressure Pb cannot exceed the plenum stagnation pressure P0 (no flow / backflow).")

        tol = 1e-3 * P0

        # INLET CONDITIONS
        if supersonic_inlet:
            M1_inlet = M1_in
        else:
            M1_inlet = mach_from_area_ratio(A1 / At, gamma, "sub")
        P1_inlet = P0 * p_p0(M1_inlet, gamma)
        T1_inlet = T0 * t_t0(M1_inlet, gamma)

        # THE 3 CRITICAL BACK PRESSURES
        epsE = Ae / At
        Msub_e = mach_from_area_ratio(epsE, gamma, "sub")
        Msup_e = mach_from_area_ratio(epsE, gamma, "sup")

        Pb1 = P0 * p_p0(Msub_e, gamma)
        Pe_design = P0 * p_p0(Msup_e, gamma)
        _, p2p1_exitshock, _, _, _ = normal_shock(Msup_e, gamma)
        Pb2 = Pe_design * p2p1_exitshock

        Ash = float("nan")

        # CLASSIFY THE REGIME AND SOLVE THE FLOW FIELD
        if Pb >= Pb1 - tol:
            regime = "Subsonic (unchoked)"
            if supersonic_inlet:
                L(
                    "WARNING: Back pressure is high enough for an all-subsonic/unchoked duct, but the "
                    "inlet was specified as supersonic (M1=%.3f, fixed externally). A fixed supersonic "
                    "inflow cannot simply \"unchoke\" the throat without a shock (diffuser unstart) somewhere "
                    "in the converging section -- that process is NOT modeled here. Results below assume the "
                    "throat still reaches M=1 (treat this as informational only, or lower Pb below Pb1)." % M1_inlet
                )
            if Pb >= P0 - tol:
                Me = 0.0
            else:
                Me = mach_from_pressure_ratio(Pb / P0, gamma, "sub")
            if Me < 1e-6:
                AstarEff, Mt, P02, T02, Pe, Te, mdot = 0.0, 0.0, P0, T0, Pb, T0, 0.0
            else:
                AstarEff = Ae / area_ratio(Me, gamma)
                Mt = mach_from_area_ratio(At / AstarEff, gamma, "sub")
                P02, T02, Pe, Te = P0, T0, Pb, T0 * t_t0(Me, gamma)
                mdot = choked_mass_flow(AstarEff, P0, T0, gamma, R)

        elif Pb > Pb2 + tol:
            regime = "Normal shock inside the diverging section"

            def f(Ashtest):
                return pressure_given_shock_at(Ashtest, At, Ae, P0, gamma)[0] - Pb

            Ash = brentq(f, At * (1 + 1e-6), Ae)
            Pe, M1s, M2s, Me, P02 = pressure_given_shock_at(Ash, At, Ae, P0, gamma)
            Mt = 1.0
            T02 = T0
            Te = T02 * t_t0(Me, gamma)
            AstarEff = At
            mdot = choked_mass_flow(At, P0, T0, gamma, R)

        elif Pb > Pe_design + tol:
            if abs(Pb - Pb2) <= tol:
                regime = "Normal shock exactly at the exit plane"
            else:
                regime = "Overexpanded (oblique shocks form OUTSIDE the nozzle)"
            Mt, Me, T02, P02 = 1.0, Msup_e, T0, P0
            Pe = Pe_design
            Te = T0 * t_t0(Me, gamma)
            AstarEff = At
            mdot = choked_mass_flow(At, P0, T0, gamma, R)

        else:
            if abs(Pb - Pe_design) <= tol:
                regime = "Correctly (perfectly) expanded"
            else:
                regime = "Underexpanded (expansion fans form OUTSIDE the nozzle)"
            Mt, Me, T02, P02 = 1.0, Msup_e, T0, P0
            Pe = Pe_design
            Te = T0 * t_t0(Me, gamma)
            AstarEff = At
            mdot = choked_mass_flow(At, P0, T0, gamma, R)

        # ---- wall geometry: cone / circular-arc fillet / cone ----
        rt = math.sqrt(At / math.pi)
        Rc = RcRatio * rt
        r_of_x, xt1, xt2, _ = solve_throat_fillet(A1, At, Ae, Lconv, Ldiv, Rc)

        # summary log
        L("=========================================================")
        L(" QUASI-1D CD NOZZLE ANALYSIS SUMMARY")
        L("=========================================================")
        L(" Geometry : A1=%.5g m^2   At=%.5g m^2   Ae=%.5g m^2   (Ae/At=%.4f)", A1, At, Ae, epsE)
        L(" Throat fillet: Rc=%.5g m (Rc/rt=%.3f)   tangent stations x1=%.5g m, x2=%.5g m",
          Rc, RcRatio, xt1, xt2)
        L(" Plenum   : P0=%.5g Pa    T0=%.5g K", P0, T0)
        L(" Back P.  : Pb=%.5g Pa", Pb)
        L("")
        L(" Critical back pressures for this geometry:")
        L("   P0        = %.6g Pa   (no flow)", P0)
        L("   Pb1       = %.6g Pa   (1st critical: sonic throat / all-subsonic)", Pb1)
        L("   Pb2       = %.6g Pa   (2nd critical: normal shock AT the exit)", Pb2)
        L("   Pe_design = %.6g Pa   (3rd critical: correctly expanded)", Pe_design)
        L("")
        L(" ---> Regime for the given Pb: %s", regime)
        L("")
        inlet_tag = "  (supersonic inlet)" if supersonic_inlet else ""
        L(" Inlet   (A1): M=%.4f   P=%.5g Pa   T=%.5g K%s", M1_inlet, P1_inlet, T1_inlet, inlet_tag)
        L(" Throat  (At): M=%.4f", Mt)
        if regime == "Normal shock inside the diverging section":
            L(" Shock location: Ash=%.5g m^2  (Ash/At=%.4f, between throat and exit)", Ash, Ash / At)
            L("   Just upstream of shock : M1=%.4f", M1s)
            L("   Just downstream of shock: M2=%.4f,  P02/P0=%.4f (stagnation pressure loss)", M2s, P02 / P0)
        L(" Exit    (Ae): M=%.4f   P=%.5g Pa   T=%.5g K   (P0,exit-plenum=%.5g Pa)", Me, Pe, Te, P02)
        L(" Mass flow rate: mdot = %.5g kg/s", mdot)
        L("=========================================================")

        # spatial distribution through the nozzle
        Nc, Nd = 200, 400
        xConv = np.linspace(-Lconv, 0, Nc)
        xDiv = np.linspace(0, Ldiv, Nd)
        x = np.concatenate([xConv, xDiv[1:]])

        rConv = r_of_x(xConv)
        rDiv = r_of_x(xDiv)
        Aconv = math.pi * rConv ** 2
        Adiv = math.pi * rDiv ** 2
        Ax = np.concatenate([Aconv, Adiv[1:]])

        Mx = np.zeros_like(x)
        Px = np.zeros_like(x)
        Tx = np.zeros_like(x)
        xshock = float("nan")

        if regime == "Subsonic (unchoked)":
            if AstarEff <= 0:
                Mx[:] = 0.0
                Px[:] = P0
                Tx[:] = T0
            else:
                for i in range(len(x)):
                    Mx[i] = mach_from_area_ratio(Ax[i] / AstarEff, gamma, "sub")
                    Px[i] = P0 * p_p0(Mx[i], gamma)
                    Tx[i] = T0 * t_t0(Mx[i], gamma)
        else:
            conv_branch = "sup" if supersonic_inlet else "sub"
            for i in range(Nc):
                Mx[i] = mach_from_area_ratio(Aconv[i] / At, gamma, conv_branch)
                Px[i] = P0 * p_p0(Mx[i], gamma)
                Tx[i] = T0 * t_t0(Mx[i], gamma)
            Mx[Nc - 1] = 1.0
            Px[Nc - 1] = P0 * p_p0(1.0, gamma)
            Tx[Nc - 1] = T0 * t_t0(1.0, gamma)

            if regime == "Normal shock inside the diverging section":
                Astar2 = Ash / area_ratio(M2s, gamma)

                rsh = math.sqrt(Ash / math.pi)

                def _r_resid(xx):
                    return float(r_of_x(xx)[0]) - rsh

                try:
                    xshock = brentq(_r_resid, 0.0, Ldiv)
                except ValueError:
                    # fallback (should not normally trigger): monotone area assumption
                    xshock = Ldiv * math.sqrt(max((Ash - At), 0.0) / (Ae - At))

                for j in range(1, Nd):
                    k = Nc + j - 1
                    if xDiv[j] < xshock:
                        Mx[k] = mach_from_area_ratio(Adiv[j] / At, gamma, "sup")
                        Px[k] = P0 * p_p0(Mx[k], gamma)
                        Tx[k] = T0 * t_t0(Mx[k], gamma)
                    else:
                        Mx[k] = mach_from_area_ratio(Adiv[j] / Astar2, gamma, "sub")
                        Px[k] = P02 * p_p0(Mx[k], gamma)
                        Tx[k] = T02 * t_t0(Mx[k], gamma)
            else:
                for j in range(1, Nd):
                    k = Nc + j - 1
                    Mx[k] = mach_from_area_ratio(Adiv[j] / At, gamma, "sup")
                    Px[k] = P0 * p_p0(Mx[k], gamma)
                    Tx[k] = T0 * t_t0(Mx[k], gamma)

        # ---- jet shock-cell / expansion-fan pattern (outside the nozzle) ----
        nCells = 15
        cellPR = np.zeros(nCells)
        cellWaveType = [None] * nCells
        cellMstart = np.zeros(nCells)
        cellMend = np.zeros(nCells)
        nValid = 0

        if regime in (
            "Overexpanded (oblique shocks form OUTSIDE the nozzle)",
            "Normal shock exactly at the exit plane",
            "Underexpanded (expansion fans form OUTSIDE the nozzle)",
        ):
            L("")
            L("Jet shock-cell / expansion-reflection sequence")
            L("(simplified single-streamtube model)")

            M_c, P_c, P0_c, thetaMag, prevType = Me, Pe, P02, 0.0, ""

            for cellIdx in range(1, nCells + 1):
                Mstart_local = M_c
                try:
                    if cellIdx % 2 == 1:
                        if P_c > Pb + tol:
                            Mnew = mach_from_pressure_ratio(Pb / P0_c, gamma, "sup")
                            thetaMag = pm_nu(Mnew, gamma) - pm_nu(M_c, gamma)
                            M_c, P_c, waveType = Mnew, Pb, "expansion"
                        elif P_c < Pb - tol:
                            thetaMag, Mnew, p02ratio = oblique_from_pressure(M_c, Pb / P_c, gamma)
                            if Mnew <= 1 + 1e-6:
                                L("  Cell %d: required compression approaches a normal shock; downstream flow becomes subsonic.", cellIdx)
                                L("  -> Stopping: the jet has effectively been recompressed to subsonic and the oblique-wave/expansion idealization no longer applies.")
                                break
                            P0_c, M_c, P_c, waveType = P0_c * p02ratio, Mnew, Pb, "shock"
                        else:
                            waveType, thetaMag = "none", 0.0
                    else:
                        if prevType == "expansion":
                            Mnew, p2p1, p02ratio = oblique_from_theta(M_c, thetaMag, gamma)
                            P0_c, P_c, M_c, waveType = P0_c * p02ratio, P_c * p2p1, Mnew, "shock"
                        else:
                            Mnew = pm_nu_inv(pm_nu(M_c, gamma) + thetaMag, gamma)
                            P_c, M_c, waveType = P0_c * p_p0(Mnew, gamma), Mnew, "expansion"
                except NozzleError as e2:
                    L("  Cell %d: %s", cellIdx, str(e2))
                    L("  -> Stopping: simple-wave idealization no longer valid beyond this point.")
                    break

                prevType = waveType
                nValid += 1
                idx0 = cellIdx - 1
                cellPR[idx0] = P_c / Pb
                cellWaveType[idx0] = waveType
                cellMstart[idx0] = Mstart_local
                cellMend[idx0] = M_c
                L("  cell %d (%-10s): M=%.4f   P/Pb=%.4f   P0_local/P0=%.4f",
                  cellIdx, waveType, M_c, P_c / Pb, P0_c / P0)
            L("---------------------------------------------------------------")

        rWall = np.sqrt(Ax / math.pi)
        rExit = rWall[-1]

        res.update(dict(
            x=x, Ax=Ax, Mx=Mx, Px=Px, Tx=Tx,
            P0=P0, T0=T0, Pb=Pb,
            regime=regime, Nc=Nc,
            rWall=rWall, rExit=rExit,
            xshock=xshock, Ash=Ash,
            xt1=xt1, xt2=xt2, Rc=Rc,
            nValid=nValid, cellPR=cellPR, cellWaveType=cellWaveType,
            cellMstart=cellMstart, cellMend=cellMend,
            supersonicInlet=supersonic_inlet, M1_inlet=M1_inlet,
        ))
        ok = True

    except NozzleError as e:
        L("")
        L("*** ERROR: %s", str(e))
        ok = False
    except Exception as e:  # numerical errors (e.g. brentq bracket failures)
        L("")
        L("*** ERROR: %s", str(e))
        ok = False

    return "\n".join(log_lines) + "\n", res, ok


# ======================================================================
#  DEFAULT PARAMETERS
# ======================================================================

def default_params():
    return dict(
        gamma=1.4, R=287.0, supersonicInlet=False,
        M1=2.0, A1=0.050, At=0.020, Ae=0.040,
        RcRatio=1.0,
        P0=1.0e6, T0=800.0, Pb=3.0e5,
        Lconv=0.10, Ldiv=0.20,
    )


# ======================================================================
#  PLOTTING
# ======================================================================

def draw_schematic(ax, res):
    x, rWall, rExit, Nc = res["x"], res["rWall"], res["rExit"], res["Nc"]
    ax.clear()
    ax.grid(True)

    ax.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([rWall, -rWall[::-1]]),
             color=(0.85, 0.85, 0.85), edgecolor=(0.3, 0.3, 0.3), linewidth=1.0,
             label="Nozzle wall")
    ax.plot([x[0], x[-1]], [0, 0], "k-.", linewidth=0.75)
    ax.plot([0, 0], [-rWall[Nc - 1], rWall[Nc - 1]], "k:", linewidth=1.1, label="Throat")

    xt1, xt2 = res.get("xt1"), res.get("xt2")
    if xt1 is not None and xt2 is not None and not (math.isnan(xt1) or math.isnan(xt2)):
        ax.axvline(xt1, color=(0.0, 0.5, 0.0), linestyle=":", linewidth=0.9, label="Fillet tangency")
        ax.axvline(xt2, color=(0.0, 0.5, 0.0), linestyle=":", linewidth=0.9)

    if res["supersonicInlet"]:
        inlet_label = f"M_1={res['M1_inlet']:.2f} (supersonic inlet)"
    else:
        inlet_label = f"M_1={res['M1_inlet']:.2f} (subsonic inlet)"
    ax.text(x[0], rWall[0] * 1.15, inlet_label, fontsize=8, ha="left")

    if res["regime"] == "Normal shock inside the diverging section" and not math.isnan(res["xshock"]):
        rShock = math.sqrt(res["Ash"] / math.pi)
        ax.plot([res["xshock"], res["xshock"]], [-rShock, rShock], "r-",
                 linewidth=2, label="Normal shock (in-duct)")

    if res["nValid"] > 0:
        xCur = x[-1]
        firstShockDone, firstExpDone = False, False
        for i in range(res["nValid"]):
            M0 = max(res["cellMstart"][i], 1.0001)
            M1v = max(res["cellMend"][i], 1.0001)
            mu0 = math.asin(1 / M0)
            mu1 = math.asin(1 / M1v)

            if (i + 1) % 2 == 1:
                y0, y1 = rExit, 0.0
            else:
                y0, y1 = 0.0, rExit

            if res["cellWaveType"][i] == "expansion":
                nRays = 5
                muRays = np.linspace(mu0, mu1, nRays)
                for k in range(nRays):
                    dxk = rExit / math.tan(muRays[k])
                    if k == 0 and not firstExpDone:
                        ax.plot([xCur, xCur + dxk], [y0, y1], "b--", linewidth=0.9, label="Expansion fan")
                        ax.plot([xCur, xCur + dxk], [-y0, -y1], "b--", linewidth=0.9)
                        firstExpDone = True
                    else:
                        ax.plot([xCur, xCur + dxk], [y0, y1], "b--", linewidth=0.6)
                        ax.plot([xCur, xCur + dxk], [-y0, -y1], "b--", linewidth=0.6)
                dxAdvance = rExit / math.tan(mu1)
            else:
                dxAdvance = rExit / math.tan(mu0)
                lw = max(2.2 - 0.2 * (i + 1), 0.5)
                if not firstShockDone:
                    ax.plot([xCur, xCur + dxAdvance], [y0, y1], "r-", linewidth=lw, label="Oblique shock")
                    ax.plot([xCur, xCur + dxAdvance], [-y0, -y1], "r-", linewidth=lw)
                    firstShockDone = True
                else:
                    ax.plot([xCur, xCur + dxAdvance], [y0, y1], "r-", linewidth=lw)
                    ax.plot([xCur, xCur + dxAdvance], [-y0, -y1], "r-", linewidth=lw)
            xCur += dxAdvance
        ax.set_xlim(x[0], xCur * 1.05)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.set_title("Nozzle schematic")
    ax.legend(loc="best", fontsize=7)


def update_plots(fig, res):
    fig.clear()
    fig.suptitle(f"Regime: {res['regime']}", fontweight="bold")
    axs = fig.subplots(3, 2)

    ax = axs[0, 0]
    ax.plot(res["x"], res["Ax"], "k-", linewidth=1.5)
    ax.grid(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("Area [m^2]")
    ax.set_title("Area distribution A(x)")

    ax = axs[0, 1]
    ax.plot(res["x"], res["Mx"], "b-", linewidth=1.5)
    ax.grid(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("Mach number [-]")
    ax.set_title("Mach number M(x)")

    ax = axs[1, 0]
    ax.plot(res["x"], res["Px"] / res["P0"], "r-", linewidth=1.5)
    ax.axhline(res["Pb"] / res["P0"], color="k", linestyle="--", label="P_b/P_0")
    ax.grid(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("P / P_0 [-]")
    ax.set_title("Static pressure ratio")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    ax.plot(res["x"], res["Tx"] / res["T0"], "m-", linewidth=1.5)
    ax.grid(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("T / T_0 [-]")
    ax.set_title("Static temperature ratio")

    ax = axs[2, 0]
    if res["nValid"] > 0:
        idx = np.arange(1, res["nValid"] + 1)
        ax.stem(idx, res["cellPR"][: res["nValid"]])
        ax.axhline(1, color="k", linestyle="--", label="P = P_b")
        ax.grid(True)
        ax.set_xlabel("Cell index"); ax.set_ylabel("P_local/P_b")
        ax.set_title("Jet shock-cell / expansion trace")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No shock-diamond / expansion-fan\npattern for this regime",
                ha="center", va="center", fontstyle="italic")

    draw_schematic(axs[2, 1], res)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


# ======================================================================
#  TKINTER UI
# ======================================================================

class ParamRow:
    """A label + numeric entry + slider, two-way linked (mirrors addParamRow)."""

    def __init__(self, parent, row, label_text, minv, maxv, initv, fmt, post_fcn=None):
        self.minv, self.maxv, self.fmt, self.post_fcn = minv, maxv, fmt, post_fcn
        self._suppress = False

        lbl = ttk.Label(parent, text=label_text, anchor="e", justify="right", wraplength=190)
        lbl.grid(row=row, column=0, sticky="e", padx=(0, 6), pady=3)

        self.value = initv
        self.entry_var = tk.StringVar(value=(fmt % initv))
        self.entry = ttk.Entry(parent, textvariable=self.entry_var, width=10)
        self.entry.grid(row=row, column=1, sticky="w", padx=(0, 6), pady=3)
        self.entry.bind("<Return>", self._on_entry_change)
        self.entry.bind("<FocusOut>", self._on_entry_change)

        self.scale_var = tk.DoubleVar(value=min(max(initv, minv), maxv))
        self.scale = ttk.Scale(parent, from_=minv, to=maxv, orient="horizontal",
                                variable=self.scale_var, command=self._on_scale_change)
        self.scale.grid(row=row, column=2, sticky="ew", pady=3)

    def _on_scale_change(self, value):
        if self._suppress:
            return
        v = float(value)
        self.value = v
        self._suppress = True
        self.entry_var.set(self.fmt % v)
        self._suppress = False
        if self.post_fcn:
            self.post_fcn(v)

    def _on_entry_change(self, event=None):
        if self._suppress:
            return
        try:
            v = float(self.entry_var.get())
        except ValueError:
            self.entry_var.set(self.fmt % self.value)
            return
        self.value = v  # the real (unclamped) value used downstream
        clamped = min(max(v, self.minv), self.maxv)
        self._suppress = True
        self.scale_var.set(clamped)
        self._suppress = False
        if self.post_fcn:
            self.post_fcn(v)

    def set(self, v):
        self.value = v
        self.entry_var.set(self.fmt % v)
        self._suppress = True
        self.scale_var.set(min(max(v, self.minv), self.maxv))
        self._suppress = False

    def set_limits(self, minv, maxv):
        self.minv, self.maxv = minv, maxv
        self.scale.configure(from_=minv, to=maxv)

    def set_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.scale.configure(state=state)

    def get(self):
        return self.value


class NozzleUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quasi-1D CD Nozzle - Inputs")
        self.rows = {}

        self.plot_win = None
        self.fig = None
        self.canvas = None

        self.log_win = None
        self.log_text = None

        self._build_input_ui()
        self.on_run()  # show a first result immediately

    # ---------------- UI construction ----------------

    def _build_input_ui(self):
        P = default_params()

        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(2, weight=1)

        ttl = ttk.Label(frame, text="Quasi-1D CD Nozzle Flow - Inputs",
                         font=("TkDefaultFont", 13, "bold"), anchor="center")
        ttl.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.supersonic_var = tk.BooleanVar(value=P["supersonicInlet"])
        chk = ttk.Checkbutton(frame, variable=self.supersonic_var,
                               text="Supersonic inlet (specify M1 at A1 instead of A1 directly)",
                               command=self._on_inlet_mode_changed)
        chk.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        r = 2
        self.rows["gamma"] = ParamRow(frame, r, "Gamma, gamma [-]", 1.05, 1.8, P["gamma"], "%.3f"); r += 1
        self.rows["R"] = ParamRow(frame, r, "Gas constant R [J/(kg*K)]", 100, 600, P["R"], "%.1f"); r += 1
        self.rows["M1"] = ParamRow(frame, r, "Inlet Mach M1 (supersonic)", 1.01, 6, P["M1"], "%.3f"); r += 1
        self.rows["A1"] = ParamRow(frame, r, "Inlet area A1 [m^2]", 0.001, 0.3, P["A1"], "%.4f"); r += 1
        self.rows["At"] = ParamRow(frame, r, "Throat area At [m^2]", 5e-4, 0.15, P["At"], "%.4f"); r += 1
        self.rows["Ae"] = ParamRow(frame, r, "Exit area Ae [m^2]", 0.001, 0.4, P["Ae"], "%.4f"); r += 1
        self.rows["RcRatio"] = ParamRow(frame, r, "Throat fillet radius, Rc/r_t [-]", 0.05, 3.0,
                                         P["RcRatio"], "%.3f"); r += 1
        self.rows["P0"] = ParamRow(frame, r, "Stagnation pressure P0 [Pa]", 1e4, 5e6, P["P0"], "%.0f",
                                    post_fcn=self._on_p0_changed); r += 1
        self.rows["T0"] = ParamRow(frame, r, "Stagnation temperature T0 [K]", 100, 3000, P["T0"], "%.1f"); r += 1
        self.rows["Pb"] = ParamRow(frame, r, "Back pressure Pb [Pa]", 0, P["P0"], P["Pb"], "%.0f"); r += 1
        self.rows["Lconv"] = ParamRow(frame, r, "Convergent length, plot [m]", 0.01, 1, P["Lconv"], "%.3f"); r += 1
        self.rows["Ldiv"] = ParamRow(frame, r, "Divergent length, plot [m]", 0.01, 1, P["Ldiv"], "%.3f"); r += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=r + 1, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        btn_frame.columnconfigure(0, weight=2)
        btn_frame.columnconfigure(1, weight=1)

        run_btn = ttk.Button(btn_frame, text="Run Analysis", command=self.on_run)
        run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        reset_btn = ttk.Button(btn_frame, text="Reset Defaults", command=self.on_reset)
        reset_btn.grid(row=0, column=1, sticky="ew")

        self._set_inlet_enable(P["supersonicInlet"])

    def _on_inlet_mode_changed(self):
        self._set_inlet_enable(self.supersonic_var.get())

    def _set_inlet_enable(self, is_sup):
        self.rows["M1"].set_enabled(is_sup)
        self.rows["A1"].set_enabled(not is_sup)

    def _on_p0_changed(self, new_p0):
        pb_row = self.rows["Pb"]
        if new_p0 > pb_row.minv:
            pb_row.set_limits(pb_row.minv, new_p0)
            if pb_row.value > new_p0:
                pb_row.set(new_p0)

    # ---------------- parameter gathering ----------------

    def _gather_params(self):
        return dict(
            gamma=self.rows["gamma"].get(),
            R=self.rows["R"].get(),
            supersonicInlet=self.supersonic_var.get(),
            M1=self.rows["M1"].get(),
            A1=self.rows["A1"].get(),
            At=self.rows["At"].get(),
            Ae=self.rows["Ae"].get(),
            RcRatio=self.rows["RcRatio"].get(),
            P0=self.rows["P0"].get(),
            T0=self.rows["T0"].get(),
            Pb=self.rows["Pb"].get(),
            Lconv=self.rows["Lconv"].get(),
            Ldiv=self.rows["Ldiv"].get(),
        )

    # ---------------- actions ----------------

    def on_run(self):
        try:
            P = self._gather_params()
            log_str, res, ok = compute_nozzle(P)
            self._show_log(log_str)
            if ok:
                self._show_plots(res)
        except Exception as e:
            tk.messagebox.showerror("Unexpected error", str(e))

    def on_reset(self):
        D = default_params()
        self.supersonic_var.set(D["supersonicInlet"])
        self.rows["Pb"].set_limits(0, D["P0"])
        for name in ("gamma", "R", "M1", "A1", "At", "Ae", "RcRatio", "P0", "T0", "Pb", "Lconv", "Ldiv"):
            self.rows[name].set(D[name])
        self._set_inlet_enable(D["supersonicInlet"])
        self.on_run()

    # ---------------- results windows ----------------

    def _show_plots(self, res):
        if self.plot_win is None or not self.plot_win.winfo_exists():
            self.plot_win = tk.Toplevel(self.root)
            self.plot_win.title("Quasi-1D CD Nozzle - Results")
            self.plot_win.geometry("900x760")
            self.fig = Figure(figsize=(9, 7.6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_win)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        update_plots(self.fig, res)
        self.canvas.draw()

    def _show_log(self, log_str):
        if self.log_win is None or not self.log_win.winfo_exists():
            self.log_win = tk.Toplevel(self.root)
            self.log_win.title("Quasi-1D CD Nozzle - Log / Output")
            self.log_win.geometry("620x760")
            self.log_text = scrolledtext.ScrolledText(self.log_win, font=("Consolas", 10), state="normal")
            self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", log_str)
        self.log_text.configure(state="disabled")


def main():
    root = tk.Tk()
    NozzleUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
