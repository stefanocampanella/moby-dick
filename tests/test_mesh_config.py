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
"""Tests of the settings of the ocean mesh generator (mesh_config.py) and of the shipped mesh configs (no mesh
generator needed: mesh generation is tested in test_ocean_mesh.py)."""

import pathlib

import pytest
from pydantic import ValidationError

from mobydick import cli_utils
from mobydick.mesh_config import (
  IcosahedralMeshConfig,
  MeshGenerationConfig,
  ShoreProximityFieldConfig,
  field_arguments,
)

MESH_CONFIGS = pathlib.Path(__file__).parents[1] / "configs" / "mesh"
COAST_LATITUDE = -70.0
UNIFORM_COARSENING = {"name": "uniform", "value": 200_000.0}
MINIMAL = {
  "domain": {"filepath": "coast.shp"},
  "coarsen_boundaries": {"fields": [UNIFORM_COARSENING]},
  "fields": [{"name": "uniform", "value": 1_000_000.0}],
}


REFINEMENT = {"name": "point_refinement", "points": [[41.12, 29.07]], "size": 250.0, "growth": 0.02}


def _settings(**changes) -> dict:
  return {**MINIMAL, **changes}


def test_defaults():
  """Defaults: the gmsh options and smoothness used before they were configurable, one seed at the North Pole."""
  configs = MeshGenerationConfig.model_validate(MINIMAL)
  assert (configs.gmsh.verbosity, configs.gmsh.algorithm, configs.gmsh.msh_file_version) == (2, 6, 4.1)
  assert configs.mesh.smoothness == 0.3 and configs.mesh.optimize and configs.mesh.small_component_max_faces == 3
  assert configs.model_name == "OceanMesh"
  assert configs.domain.physical_name_field == "featurecla" and configs.domain.curve_type == "bspline"
  assert configs.coarsen_boundaries.x0 == [(0.0, 0.0)] and configs.coarsen_boundaries.x0_projection == "stereographic"


@pytest.mark.parametrize(
  ("x0", "expected"),
  [([1.0, 2.0], [(1.0, 2.0)]), ([[1.0, 2.0]], [(1.0, 2.0)]), ([[90, 0], [43.5, 34.0]], [(90.0, 0.0), (43.5, 34.0)])],
  ids=["one_point", "list_of_one", "two_points"],
)
def test_seeds_are_one_point_or_a_list(x0, expected):
  """`coarsen_boundaries.x0` takes one point, as before seamsh accepted several, or a list of points: either way a
  list of points."""
  coarsening = {"fields": [UNIFORM_COARSENING], "x0": x0, "x0_projection": "equirectangular"}
  configs = MeshGenerationConfig.model_validate(_settings(coarsen_boundaries=coarsening))
  assert configs.coarsen_boundaries.x0 == expected


@pytest.mark.parametrize(
  ("x0", "projection"),
  [
    ([], "stereographic"),
    ([[1.0]], "stereographic"),
    ([[1.0, 2.0, 3.0]], "stereographic"),
    ([True, False], "stereographic"),
    ([[95.0, 20.0]], "equirectangular"),
    ([[0.0, 0.0]], "cartesian"),
  ],
  ids=["empty", "one_coordinate", "three_coordinates", "booleans", "latitude_out_of_range", "cartesian"],
)
def test_invalid_seeds_are_rejected(x0, projection):
  """At least one seed is needed, each a pair of numbers (not booleans); equirectangular seeds are a (latitude,
  longitude) in range; only stereographic and equirectangular coordinates are seeds (cartesian points have three)."""
  coarsening = {"fields": [UNIFORM_COARSENING], "x0": x0, "x0_projection": projection}
  with pytest.raises(ValidationError):
    MeshGenerationConfig.model_validate(_settings(coarsen_boundaries=coarsening))


def test_equirectangular_seeds_are_latitude_then_longitude():
  """Seeds in degrees are (latitude, longitude), as the points of point_refinement fields, and reach seamsh in the axis
  order of the equirectangular CRS (longitude, latitude); stereographic seeds are passed as given."""
  coarsening = {"fields": [UNIFORM_COARSENING], "x0": [[90.0, 0.0], [43.5, 34.0]], "x0_projection": "equirectangular"}
  configs = MeshGenerationConfig.model_validate(_settings(coarsen_boundaries=coarsening))
  assert configs.coarsen_boundaries.seeds() == [(0.0, 90.0), (34.0, 43.5)]
  stereographic = MeshGenerationConfig.model_validate(MINIMAL).coarsen_boundaries
  assert stereographic.seeds() == [(0.0, 0.0)]


@pytest.mark.parametrize(
  "path", sorted(set(MESH_CONFIGS.glob("*.toml")) - set(MESH_CONFIGS.glob("icosahedral_*.toml"))), ids=lambda p: p.name
)
def test_shipped_configs_are_valid(path):
  """Every shipped ocean mesh config validates, and a uniform config's name and description give its value in km."""
  configs = cli_utils.read_configs(path, schema=MeshGenerationConfig)
  if path.name.startswith("uniform_") and path.stem.endswith("km"):
    (field,) = [field for field in configs.fields if field.name == "uniform"]
    km = int(path.stem.removeprefix("uniform_").removesuffix("km"))
    assert field.value == km * 1000
    assert f"value of {km} km" in " ".join(configs.description.split())
  # Every ocean mesh keeps the Black Sea: coarsening and mesh size refined at the Turkish straits and Kerch.
  for fields in (configs.coarsen_boundaries.fields, configs.fields):
    (refinement,) = [field for field in fields if field.name == "point_refinement"]
    assert len(refinement.points) == 3
  (coarsening,) = [field for field in configs.coarsen_boundaries.fields if field.name == "uniform"]
  assert coarsening.value == 10_000.0
  # Paths are container paths, used as they are.
  assert configs.domain.filepath.startswith("/data/")
  assert "Black Sea" in configs.description


@pytest.mark.parametrize("path", sorted(MESH_CONFIGS.glob("icosahedral_*.toml")), ids=lambda path: path.name)
def test_shipped_icosahedral_configs_are_valid(path):
  """Every shipped icosahedral mesh config validates, and its name and description give its number of refinements."""
  configs = cli_utils.read_configs(path, schema=IcosahedralMeshConfig)
  assert configs.splits == int(path.stem.removeprefix("icosahedral_"))
  assert f"refined {configs.splits} times" in " ".join(configs.description.split())
  assert configs.license


@pytest.mark.parametrize(
  ("settings", "message"),
  [
    (_settings(unknown=1), "Extra inputs"),
    (_settings(domain={"filepath": "coast.shp", "sampling": 5000.0}), "Extra inputs"),
    (_settings(domain={"filepath": "coast.gpkg"}), "shapefile"),
    (_settings(domain={"filepath": "coast.shp", "curve_type": "spiral"}), "curve_type"),
    (_settings(fields=[]), "at least 1"),
    (_settings(fields=[{"name": "constant", "value": 1.0}]), "name"),
    (_settings(fields=[{"name": "uniform", "value": -1.0}]), "greater than 0"),
    (_settings(fields=[{"name": "uniform", "value": 1.0}, {"name": "uniform", "value": 2.0}]), "repeated"),
    (_settings(coarsen_boundaries={"fields": []}), "at least 1"),
    (_settings(coarsen_boundaries={"fields": [{"name": "uniform", "value": 0.0}]}), "greater than 0"),
    (
      _settings(coarsen_boundaries={"fields": [UNIFORM_COARSENING], "x0_projection": "mercator"}),
      "x0_projection",
    ),
    (_settings(gmsh={"algorithm": 0}), "greater than or equal to 1"),
    (
      _settings(
        coarsen_boundaries={"fields": [UNIFORM_COARSENING, {**REFINEMENT, "points": [[95.0, 0.0]]}]},
      ),
      "latitude",
    ),
    (_settings(coarsen_boundaries={"fields": [{**REFINEMENT, "points": []}]}), "at least 1"),
    (_settings(coarsen_boundaries={"fields": [REFINEMENT, REFINEMENT]}), "repeated"),
  ],
)
def test_invalid_settings(settings, message):
  """Unknown keys, a non-shapefile domain, missing or repeated fields and out-of-range values are rejected."""
  with pytest.raises(ValidationError, match=message):
    MeshGenerationConfig.model_validate(settings)


SHORE = {"name": "shore_proximity", "filepath": "coast.shp", "sampling": 5000.0, "field_min": 0.0}
BATHYMETRY = {"name": "bathymetry", "filepath": "static", "var_name": "deptho"}


@pytest.mark.parametrize(
  ("field", "message"),
  [
    ({**SHORE, "field_max": 500_000.0, "size_min": 2.0, "size_max": 1.0}, "must not exceed size_max"),
    ({**SHORE, "field_max": 0.0, "size_min": 1.0, "size_max": 2.0}, "greater than 0"),
    (
      {**SHORE, "field_min": 600_000.0, "field_max": 500_000.0, "size_min": 1.0, "size_max": 2.0},
      "less than field_max",
    ),
    ({**BATHYMETRY, "q_low": 0.9, "q_high": 0.1, "size_min": 1.0, "size_max": 2.0}, "less than q_high"),
    ({**BATHYMETRY, "q_high": 1.5, "size_min": 1.0, "size_max": 2.0}, "less than or equal to 1"),
    ({**BATHYMETRY, "size_min": 1.0}, "size_max"),
  ],
)
def test_invalid_field_settings(field, message):
  """Bounded fields need size_min <= size_max; distances and quantiles must be ordered and in range."""
  with pytest.raises(ValidationError, match=message):
    MeshGenerationConfig.model_validate(_settings(fields=[field]))


def test_field_arguments():
  """A field's keyword arguments exclude its name, and its file path is used as it is (a container path)."""
  field = ShoreProximityFieldConfig.model_validate(
    {**SHORE, "filepath": "/data/geometry/coast.shp", "field_max": 500_000.0, "size_min": 1.0, "size_max": 2.0}
  )
  arguments = field_arguments(field)
  assert "name" not in arguments
  assert arguments["filepath"] == "/data/geometry/coast.shp"
