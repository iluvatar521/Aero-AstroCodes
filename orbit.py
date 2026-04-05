import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ============================================================
# CONFIG
# ============================================================
G = 6.67430e-11

EARTH_RADIUS = 6.3781e6
MOON_RADIUS = 1.7371e6

m1_0 = 5.9722e24   # Earth
m2_0 = 7.3477e22   # Moon
m3_0 = 25848.0     # Spacecraft

SIM_DAYS = 90
N_SAMPLES = 2000
FRAME_STEP = 6          # higher = faster playback
ANIM_INTERVAL_MS = 40   # lower = faster playback

# ============================================================
# INPUT DATA
# ============================================================
mass_change = {
    "m1": [[0, 0, 0]],
    "m2": [[0, 0, 0]],
    "m3": [[0, 0, 0], [0, 0, 0]]
}

forces = {
    "m1": [[0, 0, 0, 0, 0]],
    "m2": [[0, 0, 0, 0, 0]],
    "m3": [
        [-900,  500, -150, 3600 * 18,       3600 * 10],
        [-900, -200,  100, 3600 * 24 * 4.2, 3600 * 12],
        [ 100, -200,  100, 3600 * 24 * 22,  3600 * 12],
        [   0,  -20,  100, 3600 * 24 * 34,  3600 * 12],
    ]
}

# Relative to Earth
r2_0 = np.array([2.923e8, -2.624e8, -4.512e7], dtype=float)   # Moon initial position
r3_0 = np.array([7.021e7, -6.154e7, -1.240e7], dtype=float)   # Spacecraft initial position
v2_0 = np.array([705.4, 765.2, 122.1], dtype=float)           # Moon initial velocity
v3_0 = np.array([1150.2, -920.5, -180.3], dtype=float)        # Spacecraft initial velocity

y0 = np.concatenate([r2_0, r3_0, v2_0, v3_0])

# ============================================================
# HELPERS
# ============================================================
def get_mass(t, m0, mc_matrix):
    total_dm = 0.0
    for row in mc_matrix:
        rate, t_start, t_dur = row
        if t >= t_start:
            active_time = min(t - t_start, t_dur)
            if active_time > 0:
                total_dm += rate * active_time
    return max(1.0, m0 + total_dm)

def get_force(t, f_matrix):
    total_f = np.array([0.0, 0.0, 0.0], dtype=float)
    for row in f_matrix:
        fx, fy, fz, t_start, t_dur = row
        if t_start <= t <= (t_start + t_dur):
            total_f += np.array([fx, fy, fz], dtype=float)
    return total_f

# ============================================================
# DYNAMICS
# ============================================================
def relative_3body_dynamics(t, y, G, m1_0, m2_0, m3_0, mass_change, forces):
    r2 = y[0:3]    # Moon wrt Earth
    r3 = y[3:6]    # Spacecraft wrt Earth
    v2 = y[6:9]
    v3 = y[9:12]

    m1 = get_mass(t, m1_0, mass_change["m1"])
    m2 = get_mass(t, m2_0, mass_change["m2"])
    m3 = get_mass(t, m3_0, mass_change["m3"])

    a1_ext = get_force(t, forces["m1"]) / m1
    a2_ext = get_force(t, forces["m2"]) / m2
    a3_ext = get_force(t, forces["m3"]) / m3

    r23 = r3 - r2

    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)
    r23_mag = np.linalg.norm(r23)

    eps = 1e-9
    r2_mag = max(r2_mag, eps)
    r3_mag = max(r3_mag, eps)
    r23_mag = max(r23_mag, eps)

    # Earth's absolute acceleration due to Moon and spacecraft
    a1_total = (
        G * m2 * r2 / r2_mag**3
        + G * m3 * r3 / r3_mag**3
        + a1_ext
    )

    # Relative accelerations wrt Earth
    a2_rel = (
        -G * m1 * r2 / r2_mag**3
        + G * m3 * r23 / r23_mag**3
        + a2_ext
        - a1_total
    )

    a3_rel = (
        -G * m1 * r3 / r3_mag**3
        - G * m2 * r23 / r23_mag**3
        + a3_ext
        - a1_total
    )

    return np.concatenate([v2, v3, a2_rel, a3_rel])

# ============================================================
# SIMULATION
# ============================================================
def run_simulation():
    t_end = 3600 * 24 * SIM_DAYS
    t_eval = np.linspace(0, t_end, N_SAMPLES)

    print("Calculating trajectory...")

    sol = solve_ivp(
        relative_3body_dynamics,
        [0, t_end],
        y0,
        t_eval=t_eval,
        args=(G, m1_0, m2_0, m3_0, mass_change, forces),
        rtol=1e-9,
        atol=1e-11,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    Y = sol.y.T
    t = sol.t
    return t, Y

# ============================================================
# ANIMATION
# ============================================================
def create_animation(t, Y):
    moon_xyz = Y[:, 0:3]
    ship_xyz = Y[:, 3:6]

    # Dynamic bounds with margin
    all_xyz = np.vstack([moon_xyz, ship_xyz, np.zeros((1, 3))])
    xyz_max = np.max(np.abs(all_xyz), axis=0)
    bounds = 1.15 * np.maximum(xyz_max, np.array([1e8, 1e8, 1e8]))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(-bounds[0], bounds[0])
    ax.set_ylim(-bounds[1], bounds[1])
    ax.set_zlim(-bounds[2], bounds[2])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Earth–Moon–Spacecraft Trajectory")

    # Better aspect handling
    try:
        ax.set_box_aspect((1, 1, 0.7))
    except Exception:
        pass

    # Static Earth
    earth = ax.scatter(
        [0], [0], [0],
        s=180, color="royalblue", label="Earth", depthshade=True
    )

    # Moon trajectory line and current point
    moon_line, = ax.plot([], [], [], linestyle=":", linewidth=1.5, color="dimgray", label="Moon path")
    moon_point = ax.scatter([], [], [], s=55, color="gray", depthshade=True, label="Moon")

    # Spacecraft trajectory line and current point
    ship_line, = ax.plot([], [], [], linestyle="-", linewidth=1.8, color="crimson", label="Spacecraft path")
    ship_point = ax.scatter([], [], [], s=70, color="red", marker="x", label="Spacecraft")

    thrust_quiver = [None]
    thrust_text = [None]

    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, fontsize=12, fontweight="bold")
    info_text = ax.text2D(0.03, 0.91, "", transform=ax.transAxes, fontsize=10)

    ax.legend(loc="upper right")

    def set_scatter_position(scatter_obj, xyz):
        scatter_obj._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])

    def init():
        moon_line.set_data([], [])
        moon_line.set_3d_properties([])

        ship_line.set_data([], [])
        ship_line.set_3d_properties([])

        # tail_line.set_data([], [])
        # tail_line.set_3d_properties([])

        time_text.set_text("")
        info_text.set_text("")
        return moon_line, ship_line, time_text, info_text

    def update(frame_number):
        k = frame_number

        # Full path up to current frame
        moon_line.set_data(moon_xyz[:k+1, 0], moon_xyz[:k+1, 1])
        moon_line.set_3d_properties(moon_xyz[:k+1, 2])

        ship_line.set_data(ship_xyz[:k+1, 0], ship_xyz[:k+1, 1])
        ship_line.set_3d_properties(ship_xyz[:k+1, 2])

        # Current positions
        m_pos = moon_xyz[k]
        s_pos = ship_xyz[k]
        set_scatter_position(moon_point, m_pos)
        set_scatter_position(ship_point, s_pos)

        # Remove old thrust vector/text
        if thrust_quiver[0] is not None:
            thrust_quiver[0].remove()
            thrust_quiver[0] = None
        if thrust_text[0] is not None:
            thrust_text[0].remove()
            thrust_text[0] = None

        # Add new thrust vector if active
        active_f = get_force(t[k], forces["m3"])
        active_f_norm = np.linalg.norm(active_f)

        if active_f_norm > 0:
            # Scale for visibility
            arrow_len = 5e7
            f_dir = active_f / active_f_norm * arrow_len

            thrust_quiver[0] = ax.quiver(
                s_pos[0], s_pos[1], s_pos[2],
                f_dir[0], f_dir[1], f_dir[2],
                color="darkred",
                arrow_length_ratio=0.2,
                linewidth=2
            )

            thrust_text[0] = ax.text(
                s_pos[0], s_pos[1], s_pos[2] - 2.2e7,
                "THRUST",
                color="darkred",
                fontsize=9,
                fontweight="bold"
            )

        # Labels
        days = t[k] / 86400.0
        dist_ship_earth = np.linalg.norm(s_pos)
        dist_moon_earth = np.linalg.norm(m_pos)
        dist_ship_moon = np.linalg.norm(s_pos - m_pos)

        time_text.set_text(f"T + {days:6.2f} days")
        info_text.set_text(
            f"Ship-Earth: {dist_ship_earth/1e6:,.1f} Mm    "
            f"Moon-Earth: {dist_moon_earth/1e6:,.1f} Mm    "
            f"Ship-Moon: {dist_ship_moon/1e6:,.1f} Mm"
        )

        artists = [moon_line, ship_line, time_text, info_text]
        if thrust_quiver[0] is not None:
            artists.append(thrust_quiver[0])
        if thrust_text[0] is not None:
            artists.append(thrust_text[0])

        return artists

    frames = list(range(0, len(t), FRAME_STEP))

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=ANIM_INTERVAL_MS,
        blit=False,
        repeat=True
    )

    return fig, ani

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t, Y = run_simulation()
    fig, ani = create_animation(t, Y)
    
    # fig, ani = animate_trajectory(t, Y, forces)

    print("Saving video...")

    ani.save(
     "trajectory.mp4",
     writer="ffmpeg",
     fps=25,
     dpi=150,
     bitrate=2000
    )

    print("Saved as trajectory.mp4")



    plt.show()

    ani.save("trajectory.gif", writer="pillow", fps=20)