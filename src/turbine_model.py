from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go


def _solid_scale(color: str) -> list[list[object]]:
    return [[0.0, color], [1.0, color]]


def _cylinder_surface(
    x_start: float,
    x_end: float,
    radius_start: float,
    radius_end: float,
    *,
    y_center: float = 0.0,
    z_center: float = 0.0,
    start_angle: float = 0.0,
    end_angle: float = 2 * math.pi,
    axial_points: int = 12,
    radial_points: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axial = np.linspace(x_start, x_end, axial_points)
    angles = np.linspace(start_angle, end_angle, radial_points)
    x, angle = np.meshgrid(axial, angles)
    radii = np.linspace(radius_start, radius_end, axial_points)[None, :]
    y = y_center + radii * np.cos(angle)
    z = z_center + radii * np.sin(angle)
    return x, y, z


def _surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: str,
    opacity: float,
    name: str,
) -> go.Surface:
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=_solid_scale(color),
        cmin=0,
        cmax=1,
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        lighting={"ambient": 0.45, "diffuse": 0.8, "roughness": 0.55, "specular": 0.35},
        lightposition={"x": -2, "y": -5, "z": 8},
    )


def _ring_lines(x_position: float, radius: float, color: str, width: int = 2) -> go.Scatter3d:
    theta = np.linspace(0, 2 * math.pi, 80)
    return go.Scatter3d(
        x=np.full_like(theta, x_position),
        y=radius * np.cos(theta),
        z=radius * np.sin(theta),
        mode="lines",
        line={"color": color, "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def _blade_lines(
    stages: list[tuple[float, float, float, int]],
    rotation: float,
    *,
    skew: float,
    width: float,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for stage_number, (x_position, hub, tip, blade_count) in enumerate(stages):
        stage_offset = stage_number * 0.17
        for blade_number in range(blade_count):
            angle = rotation + stage_offset + blade_number * 2 * math.pi / blade_count
            root_angle = angle - 0.035
            tip_angle = angle + skew
            trailing_angle = angle + 0.06
            coordinates = [
                (x_position - width * 0.45, hub, root_angle),
                (x_position + width * 0.30, tip, tip_angle),
                (x_position + width * 0.48, hub, trailing_angle),
                (x_position - width * 0.45, hub, root_angle),
            ]
            for x, radius, theta in coordinates:
                x_values.append(x)
                y_values.append(radius * math.cos(theta))
                z_values.append(radius * math.sin(theta))
            x_values.append(None)
            y_values.append(None)
            z_values.append(None)
    return x_values, y_values, z_values


def _blade_trace(
    stages: list[tuple[float, float, float, int]],
    rotation: float,
    color: str,
    name: str,
    *,
    skew: float,
    width: float,
    line_width: int,
) -> go.Scatter3d:
    x, y, z = _blade_lines(stages, rotation, skew=skew, width=width)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": line_width},
        name=name,
        hovertemplate=f"{name}<extra></extra>",
        showlegend=False,
    )


def build_gas_turbine_figure(rpm: float = 3000.0) -> go.Figure:
    """Create a rotating conceptual cutaway with 17 + 20 + 4 exact stages/cans."""
    figure = go.Figure()

    compressor_stages = [
        (-3.75 + stage * 0.235, 0.34 + stage * 0.012, 1.06 - stage * 0.019, 18)
        for stage in range(17)
    ]
    turbine_stages = [
        (2.35 + stage * 0.46, 0.49 - stage * 0.028, 0.80 + stage * 0.095, 20)
        for stage in range(4)
    ]
    compressor_stators = [
        (x + 0.115, hub + 0.035, tip - 0.025, 13)
        for x, hub, tip, _ in compressor_stages[:-1]
    ]
    turbine_stators = [
        (x + 0.225, hub + 0.02, tip - 0.03, 15)
        for x, hub, tip, _ in turbine_stages[:-1]
    ]

    # Foundation and bearing pedestals.
    figure.add_trace(
        go.Mesh3d(
            x=[-5.0, 5.15, 5.15, -5.0, -5.0, 5.15, 5.15, -5.0],
            y=[-0.72, -0.72, 0.72, 0.72, -0.72, -0.72, 0.72, 0.72],
            z=[-1.47, -1.47, -1.47, -1.47, -1.34, -1.34, -1.34, -1.34],
            i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 1, 5, 4],
            j=[1, 2, 5, 6, 4, 5, 6, 7, 3, 2, 6, 7],
            k=[2, 3, 6, 7, 1, 2, 3, 0, 4, 5, 7, 0],
            color="#29343d",
            opacity=0.92,
            hoverinfo="skip",
            name="Foundation",
            showlegend=False,
        )
    )
    for bearing_x in (-3.75, -0.05, 2.30, 4.05):
        x, y, z = _cylinder_surface(
            bearing_x - 0.18,
            bearing_x + 0.18,
            0.43,
            0.43,
            z_center=-0.94,
            axial_points=3,
            radial_points=18,
        )
        figure.add_trace(_surface(x, y, z, "#52606a", 0.92, "Bearing pedestal"))

    # Shaft and rotor drums.
    x, y, z = _cylinder_surface(-4.4, 4.55, 0.19, 0.19, axial_points=30)
    figure.add_trace(_surface(x, y, z, "#b9c3c7", 0.96, "Common shaft"))
    x, y, z = _cylinder_surface(-3.90, 0.30, 0.33, 0.56, axial_points=32)
    figure.add_trace(_surface(x, y, z, "#5f8492", 0.92, "Compressor rotor drum"))
    x, y, z = _cylinder_surface(2.18, 4.10, 0.48, 0.35, axial_points=18)
    figure.add_trace(_surface(x, y, z, "#9d6d56", 0.92, "Turbine rotor drum"))

    # Fixed stator vanes stay stationary while rotor traces animate.
    figure.add_trace(
        _blade_trace(
            compressor_stators,
            0.0,
            "rgba(91,125,139,0.70)",
            "Compressor stators",
            skew=-0.12,
            width=0.09,
            line_width=2,
        )
    )
    figure.add_trace(
        _blade_trace(
            turbine_stators,
            0.0,
            "rgba(144,103,81,0.72)",
            "Turbine stators",
            skew=0.10,
            width=0.12,
            line_width=3,
        )
    )

    compressor_trace_index = len(figure.data)
    figure.add_trace(
        _blade_trace(
            compressor_stages,
            0.0,
            "#62d5e0",
            "17 compressor rotor stages",
            skew=0.15,
            width=0.12,
            line_width=3,
        )
    )
    turbine_trace_index = len(figure.data)
    figure.add_trace(
        _blade_trace(
            turbine_stages,
            0.0,
            "#f39a52",
            "4 turbine rotor stages",
            skew=-0.18,
            width=0.17,
            line_width=4,
        )
    )

    # Twenty can-annular combustors distributed uniformly around the shaft.
    for chamber in range(20):
        angle = chamber * 2 * math.pi / 20
        y_center = 1.01 * math.cos(angle)
        z_center = 1.01 * math.sin(angle)
        x, y, z = _cylinder_surface(
            0.60,
            1.88,
            0.115,
            0.095,
            y_center=y_center,
            z_center=z_center,
            axial_points=5,
            radial_points=12,
        )
        color = "#c29569" if chamber % 2 == 0 else "#9c765a"
        figure.add_trace(_surface(x, y, z, color, 0.85, f"Combustor {chamber + 1}"))
        flame_x, flame_y, flame_z = _cylinder_surface(
            1.52,
            1.85,
            0.09,
            0.035,
            y_center=y_center,
            z_center=z_center,
            axial_points=4,
            radial_points=10,
        )
        figure.add_trace(_surface(flame_x, flame_y, flame_z, "#ff8a37", 0.65, "Flame zone"))

    # Half-open casings keep the blade rows visible as an engineering cutaway.
    x, y, z = _cylinder_surface(-4.58, -3.91, 1.27, 1.08, start_angle=0.08, end_angle=math.pi - 0.08)
    figure.add_trace(_surface(x, y, z, "#405966", 0.42, "Inlet casing"))
    x, y, z = _cylinder_surface(-3.91, 0.35, 1.17, 0.84, start_angle=0.08, end_angle=math.pi - 0.08, axial_points=30)
    figure.add_trace(_surface(x, y, z, "#405966", 0.24, "Compressor casing"))
    x, y, z = _cylinder_surface(0.28, 2.18, 1.24, 1.15, start_angle=0.08, end_angle=math.pi - 0.08)
    figure.add_trace(_surface(x, y, z, "#69574c", 0.22, "Combustion casing"))
    x, y, z = _cylinder_surface(2.15, 4.12, 1.02, 1.28, start_angle=0.08, end_angle=math.pi - 0.08)
    figure.add_trace(_surface(x, y, z, "#684c43", 0.26, "Turbine casing"))
    x, y, z = _cylinder_surface(4.05, 5.02, 0.90, 1.22, start_angle=0.08, end_angle=math.pi - 0.08)
    figure.add_trace(_surface(x, y, z, "#73574f", 0.40, "Exhaust diffuser"))

    for x_position, radius, color in [
        (-4.58, 1.28, "#718a94"),
        (-3.91, 1.16, "#547786"),
        (0.30, 1.24, "#876d59"),
        (2.18, 1.14, "#9c7259"),
        (4.10, 1.25, "#956759"),
        (5.02, 1.23, "#8a6c62"),
    ]:
        figure.add_trace(_ring_lines(x_position, radius, color))

    figure.add_trace(
        go.Scatter3d(
            x=[-1.75, 1.23, 3.05],
            y=[-1.42, -1.42, -1.42],
            z=[0, 0, 0],
            mode="text",
            text=["17-STAGE COMPRESSOR", "20 COMBUSTORS", "4-STAGE TURBINE"],
            textfont={"color": ["#62d5e0", "#f1bd73", "#f39a52"], "size": 11},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    frame_count = 24
    figure.frames = [
        go.Frame(
            name=f"rotor-{frame}",
            data=[
                _blade_trace(
                    compressor_stages,
                    frame * 2 * math.pi / frame_count,
                    "#62d5e0",
                    "17 compressor rotor stages",
                    skew=0.15,
                    width=0.12,
                    line_width=3,
                ),
                _blade_trace(
                    turbine_stages,
                    frame * 2 * math.pi / frame_count,
                    "#f39a52",
                    "4 turbine rotor stages",
                    skew=-0.18,
                    width=0.17,
                    line_width=4,
                ),
            ],
            traces=[compressor_trace_index, turbine_trace_index],
        )
        for frame in range(frame_count)
    ]

    frame_duration = int(max(35, min(120, 180_000 / max(1.0, float(rpm)))))
    figure.update_layout(
        height=590,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="#0c1118",
        plot_bgcolor="#0c1118",
        font={"family": "Segoe UI, Arial", "color": "#dfe7e8"},
        title={
            "text": f"ROTATING CUTAWAY · SOURCE SPEED {rpm:,.0f} RPM · VISUAL SPEED SCALED",
            "font": {"size": 11, "color": "#83939b"},
            "x": 0.015,
            "y": 0.97,
        },
        scene={
            "xaxis": {"visible": False, "range": [-5.3, 5.35]},
            "yaxis": {"visible": False, "range": [-1.75, 1.75]},
            "zaxis": {"visible": False, "range": [-1.62, 1.62]},
            "aspectmode": "manual",
            "aspectratio": {"x": 3.2, "y": 1.0, "z": 1.0},
            "camera": {"eye": {"x": 1.35, "y": 1.65, "z": 0.72}, "center": {"x": 0, "y": 0, "z": -0.08}},
            "bgcolor": "#0c1118",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.02,
                "yanchor": "bottom",
                "showactive": False,
                "bgcolor": "#111b21",
                "bordercolor": "#34505a",
                "font": {"color": "#c9d5d7", "size": 10},
                "buttons": [
                    {
                        "label": "▶ ROTATE SHAFT",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "■ STOP",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
    )
    return figure
