# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Geometry of the graphs: geodesic edge lengths, and the quality of the mesh triangles.

Triangle quality follows Shewchuk, "What is a good linear finite element?" (2002), and
https://scicomp.stackexchange.com/a/27095. The metrics are computed on the flat triangle through the three vertices in
Earth-centred coordinates; for ocean meshes (edges up to ~200 km) it differs from the spherical triangle by far less
than the metrics' variation. `gamma` is gmsh's quality measure (as in seamsh's tests), so the two report the same
numbers.

Quartiles hide a few poor triangles (0.15 % of them leave every quartile at 0.99), so the statistics also count the
faces beyond thresholds, say where the poor ones are (at the boundary, one row in, or deeper: poor triangles deep in
the ocean point to the size field, at the coast to the coastline), and check the mesh against quality targets.
"""

import logging
from typing import Any

import numpy as np
import pyproj

from mobydick.graph_statistics.config import QualityTargetsConfig
from mobydick.graph_statistics.graphs import Graph
from mobydick.graph_statistics.quantities import described, fraction, histogram, summarize
from mobydick.mesh_graph import MeshGraph, face_rows_from_boundary, small_components

logger = logging.getLogger(__name__)


def geodesic_lengths(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
  """Geodesic distances on the WGS84 ellipsoid (metres) between points given in degrees."""
  _, _, distances = pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
  return np.asarray(distances)


def edge_lengths_km(graph: Graph) -> np.ndarray:
  """Geodesic length of every edge of the graph (km)."""
  senders, receivers = graph.sender_nodes, graph.receiver_nodes
  return (
    geodesic_lengths(
      senders.lat[graph.senders],
      senders.lon[graph.senders],
      receivers.lat[graph.receivers],
      receivers.lon[graph.receivers],
    )
    / 1e3
  )


# Triangle quality metrics and their descriptions (also the JSON descriptions).
TRIANGLE_QUALITY = {
  "area_km2": "Area of the flat triangle through the vertices (km²)",
  "aspect_ratio": "Longest edge / smallest height (the height on the longest edge), times √3/2: 1 for an equilateral "
  "triangle, larger for worse ones",
  "gamma": "gmsh's gamma, 2 × inradius / circumradius: 1 for an equilateral triangle, 0 for a degenerate one",
  "volume_length": "4/√3 × area / mean squared edge length: 1 for an equilateral triangle, 0 for a degenerate one, "
  "negative when the normal points to the Earth's centre (inverted)",
  "angle": "Interior angles of all faces (degrees)",
  "min_angle": "Smallest angle of each face (degrees)",
  "max_angle": "Largest angle of each face (degrees)",
  "edge_ratio": "Longest / shortest edge of each face",
}


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
  """The metrics of `TRIANGLE_QUALITY` for each face (`angle`: three values per face).

  Args:
    vertices: Earth-centred coordinates [num_vertices, 3] (metres).
    faces: Vertex indices [num_faces, 3].
  """
  a, b, c = (vertices[faces[:, i]] for i in range(3))
  # Edge lengths [num_faces, 3], each opposite a vertex: |bc|, |ca|, |ab|.
  lengths = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
  normal = np.cross(b - a, c - a)
  area = 0.5 * np.linalg.norm(normal, axis=1)
  # +1 when the normal points away from the Earth's centre (the centroid is along a + b + c), -1 for inverted faces.
  orientation = np.sign(np.einsum("ij,ij->i", normal, a + b + c))
  longest, shortest = lengths.max(1), lengths.min(1)
  inradius = 2 * area / lengths.sum(1)
  circumradius = lengths.prod(1) / (4 * area)
  # Interior angles from the law of cosines, each at the vertex opposite the corresponding edge.
  la, lb, lc = lengths.T
  cosines = np.stack(
    [
      (lb**2 + lc**2 - la**2) / (2 * lb * lc),
      (lc**2 + la**2 - lb**2) / (2 * lc * la),
      (la**2 + lb**2 - lc**2) / (2 * la * lb),
    ],
    1,
  )
  angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
  return {
    "area_km2": area / 1e6,
    # The smallest height is the one on the longest edge: 2 * area / longest.
    "aspect_ratio": np.sqrt(3) / 2 * longest / (2 * area / longest),
    "gamma": 2 * inradius / circumradius,
    "volume_length": orientation * 4 / np.sqrt(3) * area / np.mean(lengths**2, axis=1),
    "angle": angles.reshape(-1),
    "min_angle": angles.min(1),
    "max_angle": angles.max(1),
    "edge_ratio": longest / shortest,
  }


# Shares of faces beyond thresholds: name -> (metric, comparison, threshold).
QUALITY_THRESHOLDS = {
  "gamma_below_0.5": ("gamma", "<", 0.5),
  "gamma_below_0.8": ("gamma", "<", 0.8),
  "min_angle_below_15": ("min_angle", "<", 15.0),
  "min_angle_below_30": ("min_angle", "<", 30.0),
  "max_angle_above_90": ("max_angle", ">", 90.0),
}

# A face is poor below this gamma (the threshold of the share target).
POOR_GAMMA = 0.8

# Where the poor faces are: row from the boundary (`mesh_graph.face_rows_from_boundary`) -> name.
POOR_LOCATIONS = {0: "touching_boundary", 1: "one_row_in", 2: "deeper"}


def threshold_shares(quality: dict[str, np.ndarray]) -> dict[str, Any]:
  """Number and percentage of the faces beyond each of `QUALITY_THRESHOLDS`."""
  result = {}
  for name, (metric, comparison, threshold) in QUALITY_THRESHOLDS.items():
    values = quality[metric]
    selected = values < threshold if comparison == "<" else values > threshold
    result[name] = described(f"Faces with {metric} {comparison} {threshold:g}", fraction(selected))
  return result


def poor_locations(gamma: np.ndarray, faces: np.ndarray, num_vertices: int) -> dict[str, Any]:
  """Where the faces with gamma < `POOR_GAMMA` are: number and percentage of them touching the boundary (a vertex on an
  edge of one face only), one row of faces in, or deeper."""
  poor = gamma < POOR_GAMMA
  rows = face_rows_from_boundary(faces, num_vertices)[poor]
  return {
    name: described(f"Faces with gamma < {POOR_GAMMA:g} {name.replace('_', ' ')}", fraction(rows == row))
    for row, name in POOR_LOCATIONS.items()
  }


def quality_targets(quality: dict[str, np.ndarray], targets: QualityTargetsConfig) -> dict[str, Any]:
  """Each target with the mesh's value and whether it is met, and whether all are (`passed`). Without faces, no target
  is met."""
  num_faces = len(quality["gamma"])
  worst_gamma = float(quality["gamma"].min()) if num_faces else float("nan")
  smallest_angle = float(quality["min_angle"].min()) if num_faces else float("nan")
  poor_percent = float(100 * (quality["gamma"] < POOR_GAMMA).mean()) if num_faces else float("nan")
  checks = {
    "min_gamma": ("Worst gamma", worst_gamma, ">=", targets.min_gamma, worst_gamma >= targets.min_gamma),
    "max_percent_poor": (
      f"Percentage of faces with gamma < {POOR_GAMMA:g}",
      poor_percent,
      "<",
      targets.max_percent_poor,
      poor_percent < targets.max_percent_poor,
    ),
    "min_angle": (
      "Smallest angle (degrees)",
      smallest_angle,
      ">=",
      targets.min_angle,
      smallest_angle >= targets.min_angle,
    ),
  }
  result: dict[str, Any] = {
    name: described(description, {"value": value, "comparison": comparison, "target": target, "passed": bool(passed)})
    for name, (description, value, comparison, target, passed) in checks.items()
  }
  result["passed"] = described("Every target is met", all(entry["passed"] for entry in result.values()))
  return result


def triangle_statistics(
  mesh: MeshGraph,
  bins: int,
  targets: QualityTargetsConfig | None = None,
  small_component_max_faces: int = 3,
  name: str = "mesh",
) -> dict[str, Any]:
  """Summaries of the triangle quality of a mesh, with histograms of the bounded metrics, the shares of faces beyond
  thresholds, the location of the poor faces, the quality targets, and the components of at most
  `small_component_max_faces` faces (each logged as a warning: an enclosed sea meshed as one triangle, say)."""
  vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
  quality = triangle_quality(vertices, faces)
  histogram_bins = {
    "gamma": np.linspace(0.0, 1.0, bins + 1),
    # Negative for inverted faces: over [-1, 1], so that the counts add up to the faces.
    "volume_length": np.linspace(-1.0, 1.0, 2 * bins + 1),
    "angle": np.linspace(0.0, 180.0, bins + 1),
  }
  result = {
    "faces": described("Triangles of the mesh", len(faces)),
    "inverted": described("Faces whose normal points to the Earth's centre", int((quality["volume_length"] < 0).sum())),
  }
  for metric, values in quality.items():
    entry: dict[str, Any] = summarize(values)
    if metric in histogram_bins:
      entry["histogram"] = histogram(values, histogram_bins[metric])
    result[metric] = described(TRIANGLE_QUALITY[metric], entry)
  result["thresholds"] = described("Faces beyond quality thresholds", threshold_shares(quality))
  result["poor_faces"] = described(
    f"Where the faces with gamma < {POOR_GAMMA:g} are (percentages of them)",
    poor_locations(quality["gamma"], faces, len(vertices)),
  )
  result["targets"] = described(
    "Quality targets (pass/fail)", quality_targets(quality, targets or QualityTargetsConfig())
  )
  small = small_components(vertices, faces, small_component_max_faces)
  for component in small:
    logger.warning(
      f"{name}: a connected component of {component['faces']} face(s) and {component['vertices']} vertices at "
      f"{component['lat']:.2f}° N, {component['lon']:.2f}° E (an enclosed sea smaller than the elements?)"
    )
  result["small_components"] = described(
    f"Connected components of at most {small_component_max_faces} faces, largest first (faces, vertices, latitude "
    "and longitude of their centre in degrees)",
    small,
  )
  return result
