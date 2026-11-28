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

import itertools
import logging
import pathlib
import sys

import click
import gmsh
import numpy as np
import seamsh

from mobydick import cli_utils
from mobydick.gis_utils import CRSRegistry
from mobydick.mesh_commands import median_edge_length
from mobydick.mesh_config import LevelConfig, MeshGenerationConfig
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
  """Mesh the ocean bounded by the coastline of CONFIG_PATH, one mesh or a hierarchy (`levels`, finest first), and
  write it, with its mesh size fields and the links between levels, to OUTPUT_PATH, a mesh file (UGRID, NetCDF-4,
  e.g. .nc). With --msh and a hierarchy, each level goes to a file of its own (<stem>_level<l>.msh)."""
  cli_utils.configure_logging(log_level)
  # Invalid settings and an existing output fail before meshing.
  configs = cli_utils.read_configs(config_path, schema=MeshGenerationConfig)
  outputs = [output_path, *(msh_paths(msh_path, len(configs.mesh_levels())) if msh_path else [])]
  for path in outputs:
    if path.exists() and not overwrite:
      raise click.ClickException(f"{path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")
  try:
    write_ocean_mesh(configs, output_path, msh_path)
  except LevelOrderError as error:
    raise click.ClickException(str(error)) from error


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


class LevelOrderError(ValueError):
  """The levels of a hierarchy are not finest first."""


def msh_paths(msh_path: pathlib.Path, levels: int) -> list[pathlib.Path]:
  """The gmsh files of --msh: the path itself for one mesh, one file per level (`<stem>_level<l><suffix>`) for a
  hierarchy (a .msh file holds one mesh)."""
  if levels == 1:
    return [msh_path]
  return [msh_path.with_name(f"{msh_path.stem}_level{level}{msh_path.suffix}") for level in range(levels)]


def mesh_level(
  configs: MeshGenerationConfig, level: LevelConfig, domain, number: int
) -> tuple[MeshData, dict[str, np.ndarray], str]:
  """Mesh one level: coarsen the coastline of `domain` with the level's settings and seeds, mesh it with gmsh in a
  model of its own, and return the mesh (nodes of the triangles only, boundary from the faces) with its size fields at
  the nodes, and the name of the gmsh model (left in gmsh, for `write_msh`)."""
  label = level.name or f"level {number}"
  logger.info(f"{label}: coarsening the coastline and meshing")
  model = f"{configs.model_name}_{number}"
  gmsh.model.add(model)
  boundary = coarsen_boundaries(
    domain,
    level.coarsen_boundaries.fields,
    x0=level.coarsen_boundaries.seeds(),
    x0_projection_name=level.coarsen_boundaries.x0_projection,
  )
  mesh_size = CompositeMeshSizeField(level.fields)
  cartesian = CRSRegistry["cartesian"]
  seamsh.gmsh.mesh(
    boundary, mesh_size, output_srs=cartesian, smoothness=configs.mesh.smoothness, optimize=configs.mesh.optimize
  )
  vertices, faces = gmsh_triangles()
  logger.info(f"{label}: {len(vertices):,} nodes, {len(faces):,} triangles")
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
  return mesh, fields, model


def write_msh(configs: MeshGenerationConfig, model: str, path: pathlib.Path) -> None:
  """Write the gmsh model `model` (one level) to `path` in gmsh's format, with the license and description."""
  gmsh.model.setCurrent(model)
  logger.info(f"Writing {model} in gmsh's format to {path}")
  if configs.license:
    gmsh.model.setAttribute("license", [configs.license])
  if configs.description:
    gmsh.model.setAttribute("description", [configs.description])
  path.parent.mkdir(parents=True, exist_ok=True)
  gmsh.write(str(path))


def write_ocean_mesh(
  configs: MeshGenerationConfig, output_path: pathlib.Path, msh_path: pathlib.Path | None = None
) -> None:
  """Mesh the domain with seamsh and gmsh, one mesh or each level of a hierarchy (finest first), and write the meshes
  with their size fields (and their minimum, the mesh size) at the nodes, and the links between levels, to a mesh
  file (`mobydick.ugrid`); with `msh_path`, also each mesh alone in gmsh's format (`msh_paths`)."""
  # Progress bars only on a terminal, not in batch logs.
  seamsh.show_progress(sys.stderr.isatty())
  # gmsh needs to be initialized before most seamsh functions can be used.
  if not gmsh.is_initialized():
    logger.info("Initialize gmsh.")
    gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", configs.gmsh.verbosity)
  gmsh.option.setNumber("Mesh.Algorithm", configs.gmsh.algorithm)
  gmsh.option.setNumber("Mesh.MshFileVersion", configs.gmsh.msh_file_version)
  logger.info("Loading the coastline.")
  # Loaded once: each level coarsens it into a new domain.
  domain = load_domain(
    pathlib.Path(configs.domain.filepath),
    physical_name_field=configs.domain.physical_name_field,
    curve_type=configs.domain.curve_type,
  )
  levels = configs.mesh_levels()
  paths: list[pathlib.Path | None] = list(msh_paths(msh_path, len(levels))) if msh_path else [None] * len(levels)
  meshes, size_fields, models = [], [], []
  try:
    for number, level in enumerate(levels):
      mesh, fields, model = mesh_level(configs, level, domain, number)
      meshes.append(mesh)
      size_fields.append(fields)
      models.append(model)
    # Checked before anything is written: no output is left behind by a refused hierarchy.
    lengths = [median_edge_length(mesh.mesh_graph) for mesh in meshes]
    if any(coarser < finer for finer, coarser in itertools.pairwise(lengths)):
      km = ", ".join(f"{length / 1e3:.1f}" for length in lengths)
      raise LevelOrderError(f"The levels must be given finest first; their median edges are {km} km")
    write_mesh_file(output_path, meshes, size_fields)
    for model, path in zip(models, paths, strict=True):
      if path is not None:
        write_msh(configs, model, path)
  finally:
    for model in models:
      gmsh.model.setCurrent(model)
      gmsh.model.remove()
