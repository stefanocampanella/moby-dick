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
"""The mesh commands that need no mesh generator (no gmsh or seamsh): the icosahedral multi-mesh and the hierarchy
files. `moby-dick make-ocean-mesh` is in `mobydick.meshing` (the `mesh` extra)."""

import itertools
import logging
import pathlib

import click
import numpy as np

from mobydick import cli_utils, icosahedral_mesh
from mobydick.mesh_config import IcosahedralMeshConfig
from mobydick.mesh_graph import MeshGraph
from mobydick.ugrid import num_levels, read_mesh_file, read_size_fields, write_mesh_file

logger = logging.getLogger(__name__)


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
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@cli_utils.LOG_LEVEL_OPTION
def make_icosahedral_mesh(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Write the icosahedral multi-mesh of CONFIG_PATH (experimental), on the WGS84 ellipsoid, to OUTPUT_PATH, a mesh
  file (UGRID, NetCDF-4, e.g. .nc).

  The file holds the finest triangles, the edges of all levels and the mesh size (the longest edge of the finest mesh),
  and can be used as the mesh of a training configuration.
  """
  cli_utils.configure_logging(log_level)
  configs = cli_utils.read_configs(config_path, schema=IcosahedralMeshConfig)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")
  mesh_data = icosahedral_mesh.get_multi_mesh_data(
    configs.splits,
    description=[configs.description] if configs.description else None,
    license=[configs.license] if configs.license else None,
  )
  logger.info(f"Writing {len(mesh_data.mesh_graph.vertices)} vertices and {len(mesh_data.mesh_graph.edges[0])} edges")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  write_mesh_file(output_path, [mesh_data])


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
