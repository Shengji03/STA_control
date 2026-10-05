"""MuJoCo overlays and optional plots for recorded TCP trajectories."""

from pathlib import Path
import mujoco
import numpy as np
from src.config.paths import OUTPUTS_DIR


def draw_trajectory(user_scn, points, rgba, sphere_size=0.008):
    """Draw trajectory points and line segments in MuJoCo viewer."""
    max_geom = user_scn.maxgeom
    for i, pt in enumerate(points):
        if user_scn.ngeom >= max_geom - 2:
            break
        g_idx = user_scn.ngeom
        mujoco.mjv_initGeom(
            user_scn.geoms[g_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.zeros(3),
            pt.astype(np.float64),
            np.zeros(9),
            np.array(rgba, dtype=np.float32),
        )
        user_scn.geoms[g_idx].size[:] = [sphere_size, 0, 0]
        user_scn.ngeom += 1

        if i > 0 and user_scn.ngeom < max_geom:
            prev = points[i - 1]
            mid = (pt + prev) * 0.5
            diff = pt - prev
            length = float(np.linalg.norm(diff))
            if length < 1e-6:
                continue
            g_idx2 = user_scn.ngeom
            mujoco.mjv_initGeom(
                user_scn.geoms[g_idx2],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([0.003, length / 2, 0]),
                mid.astype(np.float64),
                np.zeros(9),
                np.array(rgba, dtype=np.float32),
            )
            d = diff / length
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(d, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            right = np.cross(up, d)
            right /= np.linalg.norm(right)
            up2 = np.cross(d, right)
            rot = np.array([right, up2, d]).T
            user_scn.geoms[g_idx2].mat[:] = rot
            user_scn.ngeom += 1


def plot_tcp_trajectories(left_points, right_points, output_path=None):
    """Plot TCP trajectories in 3D after simulation."""
    if not left_points and not right_points:
        return
    try:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt
    except ImportError:
        print("[TaskRunner] matplotlib not available, skipping 3D plot")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if left_points:
        pts = np.array(left_points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                'r-', linewidth=1.5, alpha=0.8, label='L臂 (主臂)')
        ax.scatter(*pts[0], color='red', s=80, marker='o',
                   edgecolors='black', zorder=5)
        ax.scatter(*pts[-1], color='red', s=80, marker='s',
                   edgecolors='black', zorder=5)

    if right_points:
        pts = np.array(right_points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                'b-', linewidth=1.5, alpha=0.8, label='R臂 (从臂)')
        ax.scatter(*pts[0], color='blue', s=80, marker='o',
                   edgecolors='black', zorder=5)
        ax.scatter(*pts[-1], color='blue', s=80, marker='s',
                   edgecolors='black', zorder=5)

    pipe_x = np.linspace(-1, 4, 50)
    ax.plot(pipe_x, [0.4] * 50, [0.5] * 50,
            'gray', linewidth=3, alpha=0.3, label='管道A')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('双臂末端TCP轨迹')
    ax.legend(loc='upper left')
    plt.tight_layout()
    output_path = Path(output_path) if output_path is not None else OUTPUTS_DIR / 'tcp_trajectory_3d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f'[TaskRunner] 轨迹图已保存: {output_path}')
    plt.show()
