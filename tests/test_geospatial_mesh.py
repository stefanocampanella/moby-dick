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
"""Tests of `moby-dick make-icosahedral-mesh`: the multi-mesh written to a mesh file and read back."""

import pathlib

import numpy as np
import pyproj
import pytest
from click.testing import CliRunner

from mobydick import icosahedral_mesh
from mobydick.mesh_commands import make_icosahedral_mesh
from mobydick.ugrid import read_mesh_file

SPLITS = 2


def _same_crs(wkt_a: str, wkt_b: str) -> bool:
  return pyproj.CRS(wkt_a) == pyproj.CRS(wkt_b)


def _write_config(tmp_path: pathlib.Path, extra: str = "") -> pathlib.Path:
  path = tmp_path / "icosahedral.toml"
  path.write_text(f'description = "A test mesh"\nlicense = "A test license"\nsplits = {SPLITS}\n' + extra)
  return path


def test_make_icosahedral_mesh(tmp_path):
  """The CLI writes the multi-mesh of the config so that reading it gives it back: vertices, finest faces, the edges
  of all levels in order (also those that are not edges of the finest faces), GraphCast's mesh size, no boundary, the
  spatial reference system, description and license."""
  output = tmp_path / "out" / "icosahedral.nc"
  result = CliRunner().invoke(
    make_icosahedral_mesh, [str(_write_config(tmp_path)), str(output), "--log-level", "warning"]
  )
  assert result.exit_code == 0, result.output

  mesh_data = read_mesh_file(output)

  expected = icosahedral_mesh.get_multi_mesh_data(SPLITS)
  graph, expected_graph = mesh_data.mesh_graph, expected.mesh_graph
  np.testing.assert_array_equal(graph.vertices, expected_graph.vertices)
  np.testing.assert_array_equal(graph.faces, expected_graph.faces)
  np.testing.assert_array_equal(graph.edges[0], expected_graph.edges[0])
  np.testing.assert_array_equal(graph.edges[1], expected_graph.edges[1])
  # The coarser levels add edges that the finest faces do not have.
  assert len(graph.edges[0]) > 3 * len(graph.faces)
  assert graph.boundary is None
  assert _same_crs(graph.spatial_reference_system, expected_graph.spatial_reference_system)
  assert mesh_data.mesh_size is not None and expected.mesh_size is not None
  np.testing.assert_allclose(mesh_data.mesh_size, expected.mesh_size)
  assert mesh_data.description == ["A test mesh"] and mesh_data.license == ["A test license"]


def test_make_icosahedral_mesh_refuses_existing_output(tmp_path):
  """An existing output is replaced only with --overwrite."""
  output = tmp_path / "icosahedral.nc"
  output.write_text("existing")
  arguments = [str(_write_config(tmp_path)), str(output), "--log-level", "warning"]
  result = CliRunner().invoke(make_icosahedral_mesh, arguments)
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert output.read_text() == "existing"
  assert CliRunner().invoke(make_icosahedral_mesh, [*arguments, "--overwrite"]).exit_code == 0


@pytest.mark.parametrize(
  "settings", ["splits = 2\nunknown = 1\n", "splits = -1\n"], ids=["unknown_key", "negative_splits"]
)
def test_invalid_icosahedral_config_is_a_click_error(tmp_path, settings):
  """Unknown keys and a negative number of refinements are reported as a CLI error before anything is written."""
  config = tmp_path / "icosahedral.toml"
  config.write_text(settings)
  output = tmp_path / "icosahedral.nc"
  result = CliRunner().invoke(make_icosahedral_mesh, [str(config), str(output)])
  assert result.exit_code != 0 and "Invalid configuration" in result.output
  assert not output.exists()
