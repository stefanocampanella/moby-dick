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
# Seamsh usage example are taken from: https://jlambrechts.git-page.immc.ucl.ac.be/seamsh/examples/6-stereographics.html
"""moby-dick make-ocean-mesh: the ocean between coastlines meshed with seamsh and gmsh (the `mesh` extra)."""

import logging
import pathlib
import sys

import click
import gmsh
import numpy as np
import seamsh

from mobydick import cli_utils
from mobydick.gis_utils import CRSRegistry
from mobydick.mesh_config import MeshGenerationConfig
from mobydick.mesh_graph import (
  MeshData,
  MeshGraph,
  boundary_vertices,
  faces_to_edges,
  format_position,
  small_components,
)
from mobydick.meshing.geospatial_mesh_utils import CompositeMeshSizeField, coarsen_boundaries, load_domain
from mobydick.ugrid import write_mesh_file

logger = logging.getLogger(__name__)

# Pieces of a mesh listed in the log after meshing, largest first.
MAX_PIECES_LOGGED = 20


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
)
@click.option(
  "--msh",
  "msh_path",
  type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
  default=None,
  help="Also write the mesh (without size fields) to this file in gmsh's format, to look at it in gmsh",
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@cli_utils.LOG_LEVEL_OPTION
def make_ocean_mesh(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  msh_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Mesh the ocean bounded by the coastline of CONFIG_PATH and write it, with its mesh size fields, to OUTPUT_PATH, a
  mesh file (UGRID, NetCDF-4, e.g. .nc)."""
  cli_utils.configure_logging(log_level)
  # Invalid settings and an existing output fail before meshing.
  configs = cli_utils.read_configs(config_path, schema=MeshGenerationConfig)
  for path in (output_path, msh_path):
    if path is not None and path.exists() and not overwrite:
      raise click.ClickException(f"{path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")
  write_ocean_mesh(configs, output_path, msh_path)


def gmsh_triangles() -> tuple[np.ndarray, np.ndarray]:
  """The nodes and triangles of the current gmsh model: coordinates [num_nodes, 3] of the nodes of the triangles only,
  in the order of their tags, and faces [num_faces, 3] as indices into them. Nodes in no triangle (gmsh puts one on
  every geometry point, e.g. the control points of B-spline coastlines) are left out."""
  _, node_tags = gmsh.model.mesh.getElementsByType(2)
  node_tags = node_tags.astype(np.int64)
  tags, coordinates, _ = gmsh.model.mesh.getNodes()
  tags = tags.astype(np.int64)
  used = np.unique(node_tags)
  # Index of each tag among all nodes, then among the used ones.
  position = np.full(int(max(tags.max(initial=0), used.max(initial=0))) + 1, -1)
  position[tags] = np.arange(len(tags))
  vertices = coordinates.reshape(-1, 3)[position[used]]
  index = np.full(len(position), -1)
  index[used] = np.arange(len(used))
  return vertices, index[node_tags].reshape(-1, 3)


def warn_small_components(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> list[dict[str, float | int]]:
  """Log a warning for each connected component of the triangles with at most `max_faces` faces
  (`mesh_graph.small_components`), and return them. seamsh keeps an enclosed sea smaller than the mesh size as one
  triangle; such a piece has no interior node, and later models (e.g. hierarchies) may not support it."""
  pieces = small_components(vertices, faces, len(faces))
  # Every piece, largest first: a seed given on land shows as a piece where no sea is.
  listed = ", ".join(f"{p['faces']} at {format_position(p['lat'], p['lon'])}" for p in pieces[:MAX_PIECES_LOGGED])
  more = f" and {len(pieces) - MAX_PIECES_LOGGED} more" if len(pieces) > MAX_PIECES_LOGGED else ""
  logger.info(f"The mesh has {len(pieces)} piece(s), by triangles: {listed}{more}")
  small = [p for p in pieces if p["faces"] <= max_faces]
  for component in small:
    logger.warning(
      f"A piece of the mesh has only {component['faces']} triangle(s) and {component['vertices']} nodes, at "
      f"{format_position(component['lat'], component['lon'])}: a basin smaller than the mesh size (lower the size "
      "there to resolve it)"
    )
  return small


def write_ocean_mesh(
  configs: MeshGenerationConfig, output_path: pathlib.Path, msh_path: pathlib.Path | None = None
) -> None:
  """Mesh the domain with seamsh and gmsh, and write the mesh with its size fields (and their minimum, the mesh size)
  at the nodes to a mesh file (`mobydick.ugrid`), and the mesh alone to `msh_path` (gmsh format) if given."""
  # Progress bars only on a terminal, not in batch logs.
  seamsh.show_progress(sys.stderr.isatty())
  # gmsh needs to be initialized before most seamsh functions can be used.
  if not gmsh.is_initialized():
    logger.info("Initialize gmsh.")
    gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", configs.gmsh.verbosity)
  gmsh.option.setNumber("Mesh.Algorithm", configs.gmsh.algorithm)
  gmsh.option.setNumber("Mesh.MshFileVersion", configs.gmsh.msh_file_version)
  gmsh.model.add(configs.model_name)
  domain_path = pathlib.Path(configs.domain.filepath)
  logger.info("Loading and coarsening boundary.")
  domain = load_domain(
    domain_path, physical_name_field=configs.domain.physical_name_field, curve_type=configs.domain.curve_type
  )
  boundary = coarsen_boundaries(
    domain,
    configs.coarsen_boundaries.fields,
    x0=configs.coarsen_boundaries.seeds(),
    x0_projection_name=configs.coarsen_boundaries.x0_projection,
  )
  mesh_size = CompositeMeshSizeField(configs.fields)
  cartesian = CRSRegistry["cartesian"]
  seamsh.gmsh.mesh(
    boundary, mesh_size, output_srs=cartesian, smoothness=configs.mesh.smoothness, optimize=configs.mesh.optimize
  )
  vertices, faces = gmsh_triangles()
  warn_small_components(vertices, faces, configs.mesh.small_component_max_faces)
  mesh = MeshData(
    mesh_graph=MeshGraph(
      vertices=vertices,
      faces=faces,
      edges=faces_to_edges(faces),
      boundary=boundary_vertices(faces, len(vertices)),
      spatial_reference_system=cartesian.ExportToWkt(),
    ),
    mesh_size=mesh_size.mesh_size_3d(vertices, projection=cartesian),
    description=[configs.description] if configs.description else [],
    license=[configs.license] if configs.license else [],
  )
  fields: dict[str, np.ndarray] = {
    str(name): field.mesh_size_3d(vertices, projection=cartesian) for name, field in mesh_size.fields.items()
  }
  write_mesh_file(output_path, [mesh], [fields])
  if msh_path is not None:
    logger.info(f"Writing the mesh in gmsh's format to {msh_path}")
    if configs.license:
      gmsh.model.setAttribute("license", [configs.license])
    if configs.description:
      gmsh.model.setAttribute("description", [configs.description])
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
