from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE_HZ = 2000
EXPECTED_SAMPLES = 8000
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
COLOR_PATTERN = re.compile(r"fill: (#[0-9a-fA-F]{6})")

JET_CHANNELS = {
    "red": ((0.0, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.0, 0.5)),
    "green": ((0.0, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.0, 0.0)),
    "blue": ((0.0, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.0, 0.0)),
}


@dataclass(frozen=True)
class AffineCalibration:
    scale: float
    intercept: float

    @classmethod
    def from_points(
        cls,
        svg_start: float,
        data_start: float,
        svg_end: float,
        data_end: float,
    ) -> AffineCalibration:
        scale = (data_end - data_start) / (svg_end - svg_start)
        return cls(scale=scale, intercept=data_start - scale * svg_start)

    def convert(self, svg_value: float) -> float:
        return self.scale * svg_value + self.intercept

    def as_dict(self) -> dict[str, float | str]:
        return {
            "equation": "data_value = scale * svg_coordinate + intercept",
            "scale": self.scale,
            "intercept": self.intercept,
        }


@dataclass(frozen=True)
class BarGeometry:
    group_id: str
    left_x: float
    right_x: float
    baseline_y: float
    top_y: float

    @property
    def center_x(self) -> float:
        return (self.left_x + self.right_x) / 2.0


def format_number(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_svg(path: Path) -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser).getroot()


def require_group(root: ET.Element, group_id: str) -> ET.Element:
    group = root.find(f".//svg:g[@id='{group_id}']", SVG_NS)
    if group is None:
        raise ValueError(f"Missing SVG group {group_id!r}")
    return group


def path_vertices(path_node: ET.Element) -> list[tuple[float, float]]:
    values = [float(value) for value in NUMBER_PATTERN.findall(path_node.get("d", ""))]
    if len(values) % 2:
        raise ValueError("SVG path has an odd number of coordinates")
    return list(zip(values[0::2], values[1::2]))


def direct_path_vertices(group: ET.Element) -> list[tuple[float, float]]:
    path_node = group.find("./svg:path", SVG_NS)
    if path_node is None:
        raise ValueError(f"SVG group {group.get('id')!r} has no direct path")
    return path_vertices(path_node)


def numeric_comment(element: ET.Element) -> float | None:
    for node in element.iter():
        if node.tag is not ET.Comment or not node.text:
            continue
        text = node.text.strip().replace("\N{MINUS SIGN}", "-")
        try:
            return float(text)
        except ValueError:
            continue
    return None


def y_axis_calibration(axes: ET.Element) -> AffineCalibration:
    axis = next(
        (
            child
            for child in axes.findall("./svg:g", SVG_NS)
            if child.get("id", "").startswith("matplotlib.axis_")
            and any(grandchild.get("id", "").startswith("ytick_") for grandchild in child)
        ),
        None,
    )
    if axis is None:
        raise ValueError(f"Axes {axes.get('id')!r} has no y-axis group")

    anchors: list[tuple[float, float]] = []
    for tick in axis:
        if not tick.get("id", "").startswith("ytick_"):
            continue
        label = numeric_comment(tick)
        marker = tick.find(".//svg:use", SVG_NS)
        if label is not None and marker is not None and marker.get("y") is not None:
            anchors.append((float(marker.get("y", "nan")), label))

    if len(anchors) < 2:
        raise ValueError(f"Could not calibrate y-axis {axes.get('id')!r}")
    calibration = AffineCalibration.from_points(*anchors[0], *anchors[-1])
    residual = max(abs(calibration.convert(svg_y) - value) for svg_y, value in anchors)
    if residual > 1e-6:
        raise ValueError(f"Nonlinear y-axis calibration residual: {residual}")
    return calibration


def sample_x_calibration(svg_centers: Sequence[float]) -> AffineCalibration:
    if len(svg_centers) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples, found {len(svg_centers)}")
    if any(right <= left for left, right in zip(svg_centers, svg_centers[1:])):
        raise ValueError("Rendered sample x coordinates are not strictly increasing")
    return AffineCalibration.from_points(
        svg_centers[0],
        0.0,
        svg_centers[-1],
        (EXPECTED_SAMPLES - 1) / SAMPLE_RATE_HZ,
    )


def extract_red_bars(axes: ET.Element, y_calibration: AffineCalibration) -> list[BarGeometry]:
    baseline_y = -y_calibration.intercept / y_calibration.scale
    bars: list[BarGeometry] = []
    for group in axes.findall("./svg:g", SVG_NS):
        path_node = group.find("./svg:path", SVG_NS)
        if (
            not group.get("id", "").startswith("patch_")
            or path_node is None
            or "fill: #ff0000" not in path_node.get("style", "")
        ):
            continue
        vertices = path_vertices(path_node)
        x_values = [x for x, _ in vertices]
        y_values = [y for _, y in vertices]
        top_y = max(y_values, key=lambda value: abs(value - baseline_y))
        bars.append(
            BarGeometry(
                group_id=group.get("id", ""),
                left_x=min(x_values),
                right_x=max(x_values),
                baseline_y=baseline_y,
                top_y=top_y,
            )
        )

    if len(bars) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} red bars, found {len(bars)}")
    return bars


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def interpolate_channel(position: float, anchors: Sequence[tuple[float, float]]) -> float:
    for (left_x, left_value), (right_x, right_value) in zip(anchors, anchors[1:]):
        if position <= right_x:
            fraction = (position - left_x) / (right_x - left_x)
            return left_value + fraction * (right_value - left_value)
    return anchors[-1][1]


def jet_lut() -> list[tuple[int, int, int]]:
    colors = []
    for index in range(256):
        position = index / 255.0
        colors.append(
            tuple(
                round(255 * interpolate_channel(position, JET_CHANNELS[channel]))
                for channel in ("red", "green", "blue")
            )
        )
    return colors


def decode_jet(color_hex: str, lookup: Sequence[tuple[int, int, int]]) -> tuple[float, float]:
    rgb = tuple(int(color_hex[index : index + 2], 16) for index in (1, 3, 5))
    distances = [math.dist(rgb, candidate) for candidate in lookup]
    best_index = min(range(len(distances)), key=distances.__getitem__)
    return best_index / (len(lookup) - 1), distances[best_index]


def write_signal_paths(
    output_path: Path,
    root: ET.Element,
    series: Sequence[tuple[str, str]],
    x_calibration: AffineCalibration,
    y_calibration: AffineCalibration,
) -> tuple[int, float]:
    rows: list[tuple[object, ...]] = []
    predicted_peak = 0.0
    for series_name, group_id in series:
        vertices = direct_path_vertices(require_group(root, group_id))
        for vertex_index, (svg_x, svg_y) in enumerate(vertices):
            amplitude = y_calibration.convert(svg_y)
            if series_name == "predicted_ecg":
                predicted_peak = max(predicted_peak, abs(amplitude))
            rows.append(
                (
                    series_name,
                    group_id,
                    vertex_index,
                    format_number(x_calibration.convert(svg_x)),
                    format_number(amplitude),
                    format_number(svg_x),
                    format_number(svg_y),
                )
            )
    count = write_csv(
        output_path,
        (
            "series",
            "svg_group_id",
            "vertex_index",
            "time_sec_rendered",
            "amplitude_rendered",
            "svg_x",
            "svg_y",
        ),
        rows,
    )
    if predicted_peak == 0.0:
        raise ValueError("Could not estimate the rendered predicted-ECG peak")
    return count, predicted_peak


def write_activation_bars(
    output_path: Path,
    bars: Sequence[BarGeometry],
    x_calibration: AffineCalibration,
    y_calibration: AffineCalibration,
    predicted_peak: float,
) -> int:
    rows = []
    for sample_index, bar in enumerate(bars):
        scaled_amplitude = y_calibration.convert(bar.top_y)
        rows.append(
            (
                sample_index,
                format_number(sample_index / SAMPLE_RATE_HZ),
                format_number(x_calibration.convert(bar.left_x)),
                format_number(x_calibration.convert(bar.right_x)),
                format_number(scaled_amplitude),
                format_number(scaled_amplitude / predicted_peak),
                bar.group_id,
                format_number(bar.left_x),
                format_number(bar.right_x),
                format_number(bar.baseline_y),
                format_number(bar.top_y),
            )
        )
    return write_csv(
        output_path,
        (
            "sample_index",
            "time_sec",
            "bar_left_time_sec_rendered",
            "bar_right_time_sec_rendered",
            "gradcam_scaled_amplitude_rendered",
            "gradcam_normalized_approx",
            "svg_group_id",
            "svg_left_x",
            "svg_right_x",
            "svg_baseline_y",
            "svg_top_y",
        ),
        rows,
    )


def svg_date(root: ET.Element) -> str | None:
    date_node = root.find(".//{http://purl.org/dc/elements/1.1/}date")
    return date_node.text if date_node is not None else None


def write_metadata(
    output_dir: Path,
    source_path: Path,
    source_root: ET.Element,
    outputs: dict[str, int],
    calibrations: dict[str, AffineCalibration],
    selectors: dict[str, object],
    caveats: Sequence[str],
) -> None:
    metadata = {
        "schema_version": "1.0",
        "provenance": "render-derived from retained Matplotlib SVG; not raw model output",
        "source_file": source_path.name,
        "source_sha256": sha256(source_path),
        "source_svg_date": svg_date(source_root),
        "sample_rate_hz_from_source_code": SAMPLE_RATE_HZ,
        "expected_sample_count": EXPECTED_SAMPLES,
        "axis_calibrations": {
            name: calibration.as_dict() for name, calibration in calibrations.items()
        },
        "svg_selectors": selectors,
        "outputs": outputs,
        "caveats": list(caveats),
    }
    with (output_dir / "rendered_data_metadata.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def extract_s3() -> None:
    output_dir = ROOT / "supplement_figure_S3" / "retained_source_export"
    source_path = output_dir / "grad_cam_predicted_ecg.svg"
    root = parse_svg(source_path)
    upper_axes = require_group(root, "axes_1")
    lower_axes = require_group(root, "axes_2")
    upper_y = y_axis_calibration(upper_axes)
    lower_y = y_axis_calibration(lower_axes)

    collection = require_group(root, "PathCollection_1")
    uses = collection.findall(".//svg:use", SVG_NS)
    if len(uses) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} S3 scatter samples, found {len(uses)}")
    upper_x_values = [float(use.get("x", "nan")) for use in uses]
    upper_x = sample_x_calibration(upper_x_values)
    lookup = jet_lut()
    sample_rows = []
    max_color_distance = 0.0
    for sample_index, use in enumerate(uses):
        svg_x = float(use.get("x", "nan"))
        svg_y = float(use.get("y", "nan"))
        match = COLOR_PATTERN.search(use.get("style", ""))
        if match is None:
            raise ValueError(f"Scatter sample {sample_index} has no fill color")
        color_hex = match.group(1).lower()
        red, green, blue = (int(color_hex[index : index + 2], 16) for index in (1, 3, 5))
        jet_position, color_distance = decode_jet(color_hex, lookup)
        max_color_distance = max(max_color_distance, color_distance)
        sample_rows.append(
            (
                sample_index,
                format_number(sample_index / SAMPLE_RATE_HZ),
                format_number(upper_y.convert(svg_y)),
                color_hex,
                red,
                green,
                blue,
                format_number(jet_position),
                format_number(color_distance),
                format_number(svg_x),
                format_number(svg_y),
                use.get(XLINK_HREF, ""),
            )
        )
    if max_color_distance > 2.0:
        raise ValueError(f"S3 colors do not match the standard jet LUT: distance {max_color_distance}")

    outputs = {
        "rendered_upper_samples.csv": write_csv(
            output_dir / "rendered_upper_samples.csv",
            (
                "sample_index",
                "time_sec",
                "predicted_ecg_rendered",
                "marker_color_hex",
                "marker_red_8bit",
                "marker_green_8bit",
                "marker_blue_8bit",
                "jet_colormap_position_approx",
                "jet_rgb_distance_8bit",
                "svg_x",
                "svg_y",
                "svg_marker_path_id",
            ),
            sample_rows,
        )
    }

    region_rows = []
    region_group = require_group(root, "PolyCollection_1")
    for region_index, path_node in enumerate(region_group.findall("./svg:path", SVG_NS)):
        vertices = path_vertices(path_node)
        x_values = [x for x, _ in vertices]
        region_rows.append(
            (
                region_index,
                format_number(upper_x.convert(min(x_values))),
                format_number(upper_x.convert(max(x_values))),
                format_number(min(x_values)),
                format_number(max(x_values)),
                len(vertices),
                "grad_cam_upsampled > 0.3",
            )
        )
    outputs["rendered_upper_high_activation_regions.csv"] = write_csv(
        output_dir / "rendered_upper_high_activation_regions.csv",
        (
            "region_index",
            "start_time_sec_rendered",
            "end_time_sec_rendered",
            "start_svg_x",
            "end_svg_x",
            "polygon_vertex_count",
            "source_threshold_expression",
        ),
        region_rows,
    )

    lower_bars = extract_red_bars(lower_axes, lower_y)
    lower_x = sample_x_calibration([bar.center_x for bar in lower_bars])
    signal_count, predicted_peak = write_signal_paths(
        output_dir / "rendered_lower_signal_paths.csv",
        root,
        (("real_ecg", "line2d_69"), ("predicted_ecg", "line2d_70")),
        lower_x,
        lower_y,
    )
    outputs["rendered_lower_signal_paths.csv"] = signal_count
    outputs["rendered_lower_activation_bars.csv"] = write_activation_bars(
        output_dir / "rendered_lower_activation_bars.csv",
        lower_bars,
        lower_x,
        lower_y,
        predicted_peak,
    )
    write_metadata(
        output_dir,
        source_path,
        root,
        outputs,
        {
            "upper_x_time_sec": upper_x,
            "upper_y_predicted_ecg": upper_y,
            "lower_x_time_sec": lower_x,
            "lower_y_amplitude": lower_y,
        },
        {
            "upper_samples": "axes_1/PathCollection_1/use",
            "upper_high_activation_regions": "axes_1/PolyCollection_1/path",
            "lower_real_ecg": "axes_2/line2d_69/path",
            "lower_predicted_ecg": "axes_2/line2d_70/path",
            "lower_activation_bars": "axes_2 direct patch groups with fill #ff0000",
        },
        (
            "The ECG and region coordinates were recovered from rendered SVG geometry, so they retain only SVG export precision.",
            "jet_colormap_position_approx is the nearest position in the standard 256-level Matplotlib jet lookup table; it is not the original Grad-CAM tensor value.",
            "The lower signal paths were simplified by Matplotlib during SVG export and contain fewer than 8000 vertices.",
            "gradcam_normalized_approx divides rendered bar height by the peak absolute amplitude retained in the simplified predicted path.",
        ),
    )
    print(f"S3: wrote {sum(outputs.values())} rows across {len(outputs)} CSV files")


def extract_s4() -> None:
    output_dir = ROOT / "supplement_figure_S4" / "retained_source_export"
    source_path = output_dir / "grad_cam_activation_map.svg"
    root = parse_svg(source_path)
    axes = require_group(root, "axes_1")
    y_calibration = y_axis_calibration(axes)
    bars = extract_red_bars(axes, y_calibration)
    x_calibration = sample_x_calibration([bar.center_x for bar in bars])

    outputs: dict[str, int] = {}
    signal_count, predicted_peak = write_signal_paths(
        output_dir / "rendered_signal_paths.csv",
        root,
        (("real_ecg", "line2d_31"), ("predicted_ecg", "line2d_32")),
        x_calibration,
        y_calibration,
    )
    outputs["rendered_signal_paths.csv"] = signal_count
    outputs["rendered_activation_bars.csv"] = write_activation_bars(
        output_dir / "rendered_activation_bars.csv",
        bars,
        x_calibration,
        y_calibration,
        predicted_peak,
    )
    write_metadata(
        output_dir,
        source_path,
        root,
        outputs,
        {"x_time_sec": x_calibration, "y_amplitude": y_calibration},
        {
            "real_ecg": "axes_1/line2d_31/path",
            "predicted_ecg": "axes_1/line2d_32/path",
            "activation_bars": "axes_1 direct patch groups with fill #ff0000",
        },
        (
            "The values were recovered from rendered SVG geometry and are not raw Grad-CAM or ECG arrays; recovered raw arrays are packaged separately under original_data/.",
            "The real and predicted signal paths were simplified by Matplotlib during SVG export and contain fewer than 8000 vertices.",
            "gradcam_scaled_amplitude_rendered is the directly plotted bar height in amplitude-axis units.",
            "gradcam_normalized_approx divides rendered bar height by the peak absolute amplitude retained in the simplified predicted path.",
        ),
    )
    print(f"S4: wrote {sum(outputs.values())} rows across {len(outputs)} CSV files")


def main() -> None:
    extract_s3()
    extract_s4()


if __name__ == "__main__":
    main()