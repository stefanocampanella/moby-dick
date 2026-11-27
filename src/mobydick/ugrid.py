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
"""Mesh files: the meshes of a model in one NetCDF-4 file, following the UGRID conventions, read and written with
xarray (h5netcdf), without gmsh.

UGRID (https://ugrid-conventions.github.io/ugrid-conventions/) describes unstructured meshes with variable
attributes, so the files open in tools that read it (xugrid, uxarray, QGIS, ParaView). One file holds one mesh or a
hierarchy, finest first, each mesh a topology of its own named `mesh<level>`, with dimensions and variables prefixed
by that name:

- `mesh<l>`: the topology variable (`cf_role = "mesh_topology"`), with the mesh's `description` and `license` (lists
  of strings, as `MeshData` has them);
- `mesh<l>_node_lon`, `mesh<l>_node_lat`: geodetic longitude and latitude of the nodes on WGS84 (degrees), the
  UGRID node coordinates for other tools;
- `mesh<l>_node_x`, `_y`, `_z`: the node coordinates the model uses, Earth-centred metres in the system of
  `mesh<l>_crs` (its `crs_wkt`), read back as written;
- `mesh<l>_face_nodes` [faces, 3] and `mesh<l>_edge_nodes` [edges, 2] (start index 0): the triangles and their
  edges, each once (UGRID's face and edge connectivity, which other tools read);
- `mesh<l>_processor_edges` [processor edges, 2]: the processor's directed edges, sender first (both directions of
  every face edge, or the multi-mesh's edges of all levels), outside the UGRID topology;
- `mesh<l>_boundary`: 1 at the nodes of the boundary (the coastline), when the mesh has one;
- `mesh<l>_mesh_size` and `mesh<l>_size_<field>`: the target mesh size (the minimum of the fields) and each size
  field at the nodes (metres).

Between two consecutive levels a (finer) and b (coarser), both ways: `mesh<a>_node_in_mesh<b>_face` (the face of
mesh b closest to each node of mesh a: containing it, or nearest for a node outside mesh b),
`mesh<a>_node_in_mesh<b>_weights` (the barycentric coordinates of the closest point in that face, of its three nodes
in the order of `mesh<b>_face_nodes`) and `mesh<a>_node_in_mesh<b>_distance` (metres, 0 inside). These are the
mesh-to-mesh decoder connectivity (3 edges per receiver) and the weights of an interpolating decoder.
"""

import itertools
import logging
import pathlib
from collections.abc import Mapping, Sequence

import click
import numpy as np
import pyproj
import xarray as xr

from mobydick import cli_utils
from mobydick.basis_utils import barycentric_coordinates
from mobydick.gis_utils import cartesian_crs, cartesian_to_lat_lon
from mobydick.mesh_graph import MeshData, MeshGraph

logger = logging.getLogger(__name__)

# Lossless compression of every array (HDF5 gzip with byte shuffling): 8 to 14 times smaller than text .msh files.
_ENCODING = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def level_name(level: int) -> str:
  """The name of the topology (and prefix of the variables) of a level of the hierarchy."""
  return f"mesh{level}"


def _link_name(a: int, b: int) -> str:
  return f"{level_name(a)}_node_in_{level_name(b)}"


def mesh_dataset(mesh: MeshData, level: int = 0, size_fields: Mapping[str, np.ndarray] | None = None) -> xr.Dataset:
  """The variables of one mesh as level `level` of a mesh file (see the module docstring).

  Args:
    mesh: The mesh, with Earth-centred vertices.
    level: Its level in the hierarchy (0: finest).
    size_fields: Size fields at the nodes by name (metres), besides `mesh.mesh_size`.
  """
  name = level_name(level)
  graph = mesh.mesh_graph
  if pyproj.CRS(graph.spatial_reference_system) != pyproj.CRS(cartesian_crs.to_wkt()):
    raise ValueError(
      f"Mesh of level {level}: mesh files hold Earth-centred WGS84 coordinates (metres), not "
      f"{pyproj.CRS(graph.spatial_reference_system).name}"
    )
  vertices = np.asarray(graph.vertices, dtype=np.float64)
  faces = np.asarray(graph.faces, dtype=np.int32)
  senders, receivers = graph.edges
  processor_edges = np.stack([np.asarray(senders), np.asarray(receivers)], axis=1).astype(np.int32)
  # Every edge of the faces once, for UGRID (undirected).
  face_edges = np.sort(np.stack([faces.reshape(-1), np.roll(faces, -1, axis=1).reshape(-1)], axis=1), axis=1)
  edges = np.unique(face_edges, axis=0).astype(np.int32) if len(face_edges) else np.empty((0, 2), np.int32)
  nodes, face_dim, edge_dim = f"{name}_nNodes", f"{name}_nFaces", f"{name}_nEdges"
  # Lists of strings; an empty one is left out (netCDF would store it as an empty number array).
  texts = {key: list(value) for key, value in (("description", mesh.description), ("license", mesh.license)) if value}
  lat, lon = cartesian_to_lat_lon(vertices) if len(vertices) else (np.empty(0), np.empty(0))
  crs = f"{name}_crs"
  variables: dict[str, xr.Variable] = {
    name: xr.Variable(
      (),
      np.int32(0),
      {
        "cf_role": "mesh_topology",
        "long_name": f"Topology of the triangular mesh of level {level} (0: finest)",
        "topology_dimension": np.int32(2),
        "node_coordinates": f"{name}_node_lon {name}_node_lat",
        "face_node_connectivity": f"{name}_face_nodes",
        "edge_node_connectivity": f"{name}_edge_nodes",
        "face_dimension": face_dim,
        "edge_dimension": edge_dim,
        **texts,
      },
    ),
    crs: xr.Variable((), np.int32(0), {"crs_wkt": graph.spatial_reference_system}),
    f"{name}_node_lon": xr.Variable(
      nodes, lon, {"standard_name": "longitude", "units": "degrees_east", "long_name": "Longitude of the nodes (WGS84)"}
    ),
    f"{name}_node_lat": xr.Variable(
      nodes, lat, {"standard_name": "latitude", "units": "degrees_north", "long_name": "Latitude of the nodes (WGS84)"}
    ),
    f"{name}_face_nodes": xr.Variable(
      (face_dim, f"{name}_nMax_face_nodes"),
      faces,
      {"cf_role": "face_node_connectivity", "start_index": np.int32(0), "long_name": "Nodes of each triangle"},
    ),
    f"{name}_edge_nodes": xr.Variable(
      (edge_dim, "Two"),
      edges,
      {"cf_role": "edge_node_connectivity", "start_index": np.int32(0), "long_name": "Edges of the triangles"},
    ),
    f"{name}_processor_edges": xr.Variable(
      (f"{name}_nProcessorEdges", "Two"),
      processor_edges,
      {"long_name": "Directed edges of the processor, sender then receiver (start index 0)", "mesh": name},
    ),
  }
  for axis, column in zip("xyz", vertices.T, strict=True):
    variables[f"{name}_node_{axis}"] = xr.Variable(
      nodes,
      column,
      {"units": "m", "grid_mapping": crs, "long_name": f"Earth-centred {axis} of the nodes", "mesh": name},
    )
  node_data = {"location": "node", "mesh": name, "units": "m"}
  if graph.boundary is not None:
    variables[f"{name}_boundary"] = xr.Variable(
      nodes,
      np.asarray(graph.boundary, dtype=np.int8),
      {
        "location": "node",
        "mesh": name,
        "flag_values": np.array([0, 1], dtype=np.int8),
        "long_name": "Node of the boundary",
      },
    )
  if mesh.mesh_size is not None:
    variables[f"{name}_mesh_size"] = xr.Variable(
      nodes, np.asarray(mesh.mesh_size, np.float64), {**node_data, "long_name": "Target mesh size"}
    )
  for field, values in (size_fields or {}).items():
    variables[f"{name}_size_{field}"] = xr.Variable(
      nodes, np.asarray(values, np.float64), {**node_data, "long_name": f"Mesh size field {field}"}
    )
  return xr.Dataset(variables)


def links_dataset(fine: MeshGraph, coarse: MeshGraph, fine_level: int) -> xr.Dataset:
  """The links between the nodes of `fine` (level `fine_level`) and the faces of `coarse` (the next level), both
  ways: closest face, barycentric weights and distance (see the module docstring)."""
  variables = {}
  for a, b, nodes_of, faces_of in (
    (fine_level, fine_level + 1, fine, coarse),
    (fine_level + 1, fine_level, coarse, fine),
  ):
    face, weights, distance = barycentric_coordinates(faces_of, np.asarray(nodes_of.vertices, np.float64))
    name, nodes = _link_name(a, b), f"{level_name(a)}_nNodes"
    attributes = {"location": "node", "mesh": level_name(a)}
    variables[f"{name}_face"] = xr.Variable(
      nodes,
      face.astype(np.int32),
      {**attributes, "long_name": f"Face of {level_name(b)} closest to each node (start index 0)"},
    )
    variables[f"{name}_weights"] = xr.Variable(
      (nodes, "Three"),
      weights,
      {**attributes, "long_name": f"Barycentric coordinates of the closest point, of the nodes of {name}_face"},
    )
    variables[f"{name}_distance"] = xr.Variable(
      nodes, distance, {**attributes, "units": "m", "long_name": f"Distance to {level_name(b)} (0: inside)"}
    )
  return xr.Dataset(variables)


def write_mesh_file(
  path: pathlib.Path | str,
  meshes: Sequence[MeshData],
  size_fields: Sequence[Mapping[str, np.ndarray] | None] | None = None,
) -> None:
  """Write meshes, finest first, to a NetCDF-4 file with the links between consecutive levels.

  Args:
    path: Output file (NetCDF-4, e.g. `.nc`).
    meshes: The levels, finest first.
    size_fields: For each level, its size fields by name (see `mesh_dataset`).
  """
  if not meshes:
    raise ValueError("No mesh to write")
  size_fields = size_fields or [None] * len(meshes)
  parts = [
    mesh_dataset(mesh, level, fields) for level, (mesh, fields) in enumerate(zip(meshes, size_fields, strict=True))
  ]
  for level in range(len(meshes) - 1):
    parts.append(links_dataset(meshes[level].mesh_graph, meshes[level + 1].mesh_graph, level))
  # Every variable has its own name (prefixed by its level): no merging needed, and attributes are kept.
  dataset = xr.Dataset({name: variable for part in parts for name, variable in part.variables.items()})
  dataset.attrs = {
    "Conventions": "CF-1.11 UGRID-1.0",
    "title": "Meshes of a Moby-Dick model, finest first",
    "levels": np.int32(len(meshes)),
  }
  encoding = {name: _ENCODING for name, variable in dataset.variables.items() if variable.ndim}
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  logger.info(f"Writing {len(meshes)} mesh(es) to {path}")
  dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def num_levels(path: pathlib.Path | str) -> int:
  """The number of meshes in a mesh file."""
  with xr.open_dataset(path, engine="h5netcdf") as dataset:
    return int(dataset.attrs["levels"])


def read_mesh_file(path: pathlib.Path | str, level: int = 0) -> MeshData:
  """One mesh of a mesh file (level 0: the finest), as written by `write_mesh_file`.

  The size fields other than the mesh size are read with `read_size_fields`.
  """
  name = level_name(level)
  logger.info(f"Reading {name} of {path}")
  with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as dataset:
    if name not in dataset:
      raise ValueError(f"{path} has no mesh of level {level} ({int(dataset.attrs.get('levels', 0))} levels)")
    topology = dataset[name].attrs
    vertices = np.stack([dataset[f"{name}_node_{axis}"].values for axis in "xyz"], axis=1)
    edges = dataset[f"{name}_processor_edges"].values.astype(np.int64)
    boundary = dataset[f"{name}_boundary"].values.astype(bool) if f"{name}_boundary" in dataset else None
    mesh_size = dataset[f"{name}_mesh_size"].values if f"{name}_mesh_size" in dataset else None
    graph = MeshGraph(
      vertices=vertices,
      faces=dataset[f"{name}_face_nodes"].values.astype(np.int64),
      edges=(edges[:, 0], edges[:, 1]),
      boundary=boundary,
      spatial_reference_system=str(dataset[f"{name}_crs"].attrs["crs_wkt"]),
    )

  def strings(value: object) -> list[str]:
    # netCDF reads a list of one string back as a string.
    if isinstance(value, str):
      return [value]
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []

  logger.info(f"{name}: {len(vertices):,} nodes, {len(graph.faces):,} faces, {len(edges):,} edges")
  return MeshData(
    mesh_graph=graph,
    mesh_size=mesh_size,
    description=strings(topology.get("description")),
    license=strings(topology.get("license")),
  )


def read_size_fields(path: pathlib.Path | str, level: int = 0) -> dict[str, np.ndarray]:
  """The size fields of a mesh of a mesh file at its nodes, by field name (metres)."""
  prefix = f"{level_name(level)}_size_"
  with xr.open_dataset(path, engine="h5netcdf") as dataset:
    return {
      str(name)[len(prefix) :]: dataset[name].values for name in dataset.data_vars if str(name).startswith(prefix)
    }


def read_links(path: pathlib.Path | str, level: int, other: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """For each node of mesh `level`, the closest face of mesh `other` (a neighbouring level), the barycentric weights
  of that face's nodes, and the distance to it (metres)."""
  if abs(level - other) != 1:
    raise ValueError(f"Links are stored between consecutive levels only, not {level} and {other}")
  name = _link_name(level, other)
  with xr.open_dataset(path, engine="h5netcdf") as dataset:
    if f"{name}_face" not in dataset:
      raise ValueError(f"{path} has no links between levels {level} and {other}")
    return dataset[f"{name}_face"].values, dataset[f"{name}_weights"].values, dataset[f"{name}_distance"].values


def median_edge_length(mesh: MeshGraph) -> float:
  """Median length of the triangle edges (straight line, metres)."""
  faces = np.asarray(mesh.faces)
  vertices = np.asarray(mesh.vertices, np.float64)
  lengths = np.linalg.norm(vertices[faces] - vertices[np.roll(faces, -1, axis=1)], axis=-1)
  return float(np.median(lengths)) if lengths.size else float("nan")


@click.command("combine-meshes")
@click.argument("output_path", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True))
@click.argument("mesh_paths", nargs=-1, required=True, type=click.Path(path_type=pathlib.Path, exists=True))
@click.option("--overwrite/--no-overwrite", default=False, help="Whether to overwrite an existing output")
@cli_utils.LOG_LEVEL_OPTION
def combine_meshes(output_path: pathlib.Path, mesh_paths: tuple[pathlib.Path, ...], overwrite: bool, log_level: str):
  """Write the meshes of MESH_PATHS (mesh files of one mesh each, finest first) to OUTPUT_PATH as one hierarchy, with
  the links between consecutive levels (closest face and barycentric weights, both ways).

  Each level is meshed on its own (its coastline coarsened for its size, its own seeds), so the levels are not nested:
  the links are computed here.
  """
  cli_utils.configure_logging(log_level)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  meshes, size_fields = [], []
  for path in mesh_paths:
    if num_levels(path) != 1:
      raise click.ClickException(f"{path} holds {num_levels(path)} meshes: combine files of one mesh each")
    meshes.append(read_mesh_file(path))
    size_fields.append(read_size_fields(path))
  lengths = [median_edge_length(mesh.mesh_graph) for mesh in meshes]
  if any(coarser < finer for finer, coarser in itertools.pairwise(lengths)):
    km = ", ".join(f"{length / 1e3:.1f}" for length in lengths)
    raise click.ClickException(f"The meshes must be given finest first; their median edges are {km} km")
  write_mesh_file(output_path, meshes, size_fields)
