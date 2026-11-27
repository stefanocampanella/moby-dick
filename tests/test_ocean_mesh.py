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
"""Tests of moby-dick make-ocean-mesh (mobydick.meshing, the `mesh` extra) on synthetic coastlines: a ring at 70°S,
meshed with a coarse uniform field, and a continent with an inland sea behind a narrow channel. Skipped without the
extra."""

import itertools
import pathlib

import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner
from osgeo import ogr, osr

pytest.importorskip("seamsh", reason="mesh generation needs the mesh extra")
pytest.importorskip("gmsh", reason="mesh generation needs the mesh extra")

from mobydick.gis_utils import (
  cartesian_crs,
  cartesian_to_lat_lon,
  get_transform,
  lat_lon_to_cartesian,
  stereographic_crs,
)
from mobydick.mesh_graph import face_components, faces_to_edges
from mobydick.meshing.geospatial_mesh import make_ocean_mesh
from mobydick.meshing.geospatial_mesh_utils import PointRefinementField
from mobydick.ugrid import num_levels, read_links, read_mesh_file, read_size_fields

COAST_LATITUDE = -70.0


@pytest.fixture
def coastline(tmp_path) -> pathlib.Path:
  """A shapefile with one closed coastline, a ring at 70°S: the ocean is north of it, including the North Pole (the
  origin of the stereographic plane, where boundary coarsening starts)."""
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  source = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(tmp_path / "coast.shp"))
  layer = source.CreateLayer("coast", srs, ogr.wkbLineString)
  layer.CreateField(ogr.FieldDefn("featurecla", ogr.OFTString))
  line = ogr.Geometry(ogr.wkbLineString)
  for lon in np.append(np.arange(-180.0, 180.0, 5.0), -180.0):
    line.AddPoint_2D(float(lon), COAST_LATITUDE)
  feature = ogr.Feature(layer.GetLayerDefn())
  feature.SetField("featurecla", "Coastline")
  feature.SetGeometry(line)
  layer.CreateFeature(feature)
  source = None  # Closes and writes the shapefile.
  return tmp_path


def _write_config(path: pathlib.Path, extra: str = "") -> pathlib.Path:
  path.write_text(
    'description = "A test mesh"\nlicense = "A test license"\n'
    "[gmsh]\nverbosity = 0\n"
    f'[domain]\nfilepath = "{path.parent / "coast.shp"}"\ncurve_type = "polyline"\n'
    '[[coarsen_boundaries.fields]]\nname = "uniform"\nvalue = 200_000.0\n'
    '[[fields]]\nname = "uniform"\nvalue = 1_000_000.0\n' + extra
  )
  return path


def test_make_ocean_mesh(coastline):
  """The CLI meshes the ocean north of the coastline and stores the description and the mesh size at every node."""
  config = _write_config(coastline / "mesh.toml")
  output = coastline / "out" / "mesh.nc"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  mesh_data = read_mesh_file(output)
  vertices = mesh_data.mesh_graph.vertices
  latitudes = np.degrees(np.arcsin(vertices[:, 2] / np.linalg.norm(vertices, axis=1)))
  assert len(vertices) > 100
  # Boundary nodes lie on the (piecewise linear) coastline, a little north of 70°S between its points.
  assert latitudes.min() > COAST_LATITUDE - 0.5
  assert latitudes.max() > 80.0
  assert mesh_data.mesh_size is not None
  np.testing.assert_allclose(mesh_data.mesh_size, 1_000_000.0)
  assert " ".join(mesh_data.description) == "A test mesh" and " ".join(mesh_data.license) == "A test license"
  # The boundary is the coastline, every node is in a triangle, and the edges are those of the triangles.
  graph = mesh_data.mesh_graph
  assert graph.boundary is not None and graph.boundary.any() and not graph.boundary.all()
  assert latitudes[graph.boundary].max() < COAST_LATITUDE + 0.5
  assert np.isin(np.arange(len(vertices)), graph.faces).all()
  senders, receivers = faces_to_edges(graph.faces)
  np.testing.assert_array_equal(graph.edges[0], senders)
  np.testing.assert_array_equal(graph.edges[1], receivers)


def test_point_refinement_field():
  """The size is `size` at a point and grows by `growth` per metre of distance from the nearest point, whether the
  points are given in Earth-centred or stereographic coordinates (seamsh uses the latter, moby-dick the former)."""
  field = PointRefinementField(points=[(0.0, 0.0), (0.0, 90.0)], size=250.0, growth=0.02)
  # At the first point, 0.9° east of it, and 0.9° west of the second point (the nearest one).
  at = lat_lon_to_cartesian(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.9, 89.1]))
  sizes = field.mesh_size_3d(at, cartesian_crs)
  chord = np.linalg.norm(at[1] - at[0])  # ~100 km
  np.testing.assert_allclose(sizes, [250.0, 250.0 + 0.02 * chord, 250.0 + 0.02 * chord], rtol=1e-6)
  stereographic = get_transform(cartesian_crs, stereographic_crs)(at)
  np.testing.assert_allclose(field.mesh_size_3d(stereographic, stereographic_crs), sizes, rtol=1e-6)


@pytest.fixture
def channel_coastline(tmp_path) -> pathlib.Path:
  """The 70°S ring (the ocean's outer boundary in the North Pole stereographic plane) and a continent at the equator
  (lon 0-40, lat ±20) holding an inland sea (lon 15-25, lat ±5) joined to the ocean by a channel 0.2° (22 km) wide
  along the equator from lon 25 to 40, all as Natural Earth-like coastline lines."""
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  source = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(tmp_path / "coast.shp"))
  layer = source.CreateLayer("coast", srs, ogr.wkbLineString)
  layer.CreateField(ogr.FieldDefn("featurecla", ogr.OFTString))
  ring = [(float(lon), COAST_LATITUDE) for lon in np.append(np.arange(-180.0, 180.0, 5.0), -180.0)]
  continent = [(40, 0.1), (40, 20), (0, 20), (0, -20), (40, -20), (40, -0.1), (25, -0.1), (25, -5), (15, -5), (15, 5)]
  continent += [(25, 5), (25, 0.1), (40, 0.1)]

  def densified(points, step=0.5):
    # Points every `step` degrees at most, as a real coastline has.
    out = []
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
      n = max(1, int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) / step)))
      out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    return out + [points[-1]]

  for points in (ring, densified(continent)):
    line = ogr.Geometry(ogr.wkbLineString)
    for lon, lat in points:
      line.AddPoint_2D(float(lon), float(lat))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("featurecla", "Coastline")
    feature.SetGeometry(line)
    layer.CreateFeature(feature)
  source = None  # Closes and writes the shapefile.
  return tmp_path


@pytest.mark.parametrize("refined", [False, True], ids=["channel_closed", "channel_refined"])
def test_a_narrow_channel_needs_refinement_to_keep_the_sea_behind_it(channel_coastline, refined):
  """A coarsening size larger than a channel's width closes it, and the sea behind it is not meshed; refining the
  coarsening size and the mesh size at the channel (point_refinement, as for the Bosporus) keeps the sea in the one
  connected mesh."""
  channel = '{name = "point_refinement", points = [[0.0, 32.5]], size = 5_000.0, growth = 0.01}'
  coarsening = '{name = "uniform", value = 200_000.0}' + (f", {channel}" if refined else "")
  config = channel_coastline / "mesh.toml"
  config.write_text(
    "[gmsh]\nverbosity = 0\n"
    f'[domain]\nfilepath = "{channel_coastline / "coast.shp"}"\ncurve_type = "polyline"\n'
    f"[coarsen_boundaries]\nfields = [{coarsening}]\n"
    '[[fields]]\nname = "uniform"\nvalue = 300_000.0\n'
    + (
      '[[fields]]\nname = "point_refinement"\npoints = [[0.0, 32.5]]\nsize = 10_000.0\ngrowth = 0.3\n'
      if refined
      else ""
    )
  )
  output = channel_coastline / "mesh.nc"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  graph = read_mesh_file(output).mesh_graph
  lat, lon = cartesian_to_lat_lon(graph.vertices)
  in_sea = (np.abs(lat) < 4.5) & (lon > 15.5) & (lon < 24.5)
  assert in_sea.any() == refined
  if refined:
    # One mesh: the sea is connected to the ocean through the channel.
    assert nx.number_connected_components(nx.Graph(zip(*graph.edges, strict=True))) == 1


def test_make_ocean_mesh_refuses_existing_output(coastline):
  """An existing output is replaced only with --overwrite."""
  config = _write_config(coastline / "mesh.toml")
  output = coastline / "mesh.nc"
  output.write_text("existing")
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output)])
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert output.read_text() == "existing"


def test_invalid_config_is_a_click_error(coastline):
  """Invalid settings are reported as a CLI error before anything is meshed or written."""
  config = _write_config(coastline / "mesh.toml", extra="unknown = 1\n")
  output = coastline / "mesh.nc"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output)])
  assert result.exit_code != 0 and "Invalid configuration" in result.output
  assert not output.exists()


SHORE_FIELD = (
  '[[fields]]\nname = "shore_proximity"\nfilepath = "coast.shp"\ncurve_type = "polyline"\nsampling = 50_000.0\n'
  "field_min = 0.0\nfield_max = 3_000_000.0\nsize_min = 500_000.0\nsize_max = 2_000_000.0\n"
)


def _mesh(coastline: pathlib.Path, name: str, extra: str = "") -> pathlib.Path:
  """Mesh the synthetic coastline with the test settings (plus extra TOML), into coastline / name.nc."""
  # Paths are used as they are: the shapefile by its absolute path.
  extra = extra.replace('"coast.shp"', f'"{coastline / "coast.shp"}"')
  config = _write_config(coastline / f"{name}.toml", extra=extra)
  output = coastline / f"{name}.nc"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output)])
  assert result.exit_code == 0, result.output
  return output


def test_the_size_fields_are_written_at_every_node(coastline):
  """With two fields, the mesh file holds each at every node of the triangles, and the mesh size is their minimum."""
  output = _mesh(coastline, "two_fields", extra=SHORE_FIELD)
  mesh_data = read_mesh_file(output)
  sizes = read_size_fields(output)
  assert set(sizes) == {"uniform", "shore_proximity"} and mesh_data.mesh_size is not None
  for values in (*sizes.values(), mesh_data.mesh_size):
    assert values.shape == (len(mesh_data.mesh_graph.vertices),)
  np.testing.assert_allclose(sizes["uniform"], 1_000_000.0)
  np.testing.assert_allclose(mesh_data.mesh_size, np.minimum(sizes["uniform"], sizes["shore_proximity"]))


def test_the_msh_option_writes_the_mesh_for_gmsh(coastline):
  """--msh also writes the mesh in gmsh's format, for looking at it in gmsh, with its license and description."""
  config = _write_config(coastline / "mesh.toml")
  output, msh = coastline / "mesh.nc", coastline / "mesh.msh"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--msh", str(msh), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  text = msh.read_text()
  assert "$Nodes" in text and "$Elements" in text and "A test license" in text
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(coastline / "other.nc"), "--msh", str(msh)])
  assert result.exit_code != 0 and "--overwrite" in result.output


def _mesh_channel(coastline: pathlib.Path, seeds: str, coarsening: float, size: float) -> tuple[pathlib.Path, str]:
  """Mesh the channel coastline with seeds in latitude and longitude, a uniform coarsening and a uniform mesh size;
  the output file and the command's output (with the log: the command configures its own handler, which caplog does
  not see)."""
  config = coastline / "seeds.toml"
  config.write_text(
    "[gmsh]\nverbosity = 0\n"
    f'[domain]\nfilepath = "{coastline / "coast.shp"}"\ncurve_type = "polyline"\n'
    f'[coarsen_boundaries]\nx0 = {seeds}\nx0_projection = "equirectangular"\n'
    f'fields = [{{name = "uniform", value = {coarsening}}}]\n'
    f'[[fields]]\nname = "uniform"\nvalue = {size}\n'
  )
  output = coastline / "seeds.nc"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--overwrite", "--log-level", "info"])
  assert result.exit_code == 0, result.output
  return output, result.output


def test_a_seed_in_an_enclosed_sea_meshes_it_as_a_piece_of_its_own(channel_coastline):
  """With a second seed in the inland sea, whose channel closes at the coarsening size, the sea is meshed as a second
  connected component instead of being lost, and the log lists both pieces; seeds are given as (latitude,
  longitude)."""
  output, log = _mesh_channel(channel_coastline, "[[90.0, 0.0], [0.0, 20.0]]", 200_000.0, 300_000.0)
  graph = read_mesh_file(output).mesh_graph
  count, labels = face_components(graph.faces, len(graph.vertices))
  lat, lon = cartesian_to_lat_lon(graph.vertices)
  in_sea = (np.abs(lat) < 4.5) & (lon > 15.5) & (lon < 24.5)
  assert count == 2 and in_sea.any()
  assert len(np.unique(labels[in_sea])) == 1 and labels[in_sea][0] != labels[np.argmax(lat)]
  assert "The mesh has 2 piece(s)" in log
  assert "A piece of the mesh has only" not in log and "not in the coarsened domain" not in log


def test_a_sea_smaller_than_the_mesh_size_is_one_triangle_with_a_warning(channel_coastline):
  """A seeded sea smaller than the mesh size is kept by seamsh as a single triangle (no interior node), which models
  may not support later: make-ocean-mesh logs a warning that gives its position."""
  output, log = _mesh_channel(channel_coastline, "[[90.0, 0.0], [0.0, 20.0]]", 200_000.0, 3_000_000.0)
  graph = read_mesh_file(output).mesh_graph
  _, labels = face_components(graph.faces, len(graph.vertices))
  faces_per_component = np.bincount(labels[graph.faces[:, 0]])
  assert faces_per_component[faces_per_component > 0].min() == 1
  assert "A piece of the mesh has only 1 triangle(s) and 3 nodes" in log


def test_a_seed_whose_sea_the_coarsening_loses_gives_a_warning(channel_coastline):
  """A coarsening size about the width of the seeded sea thins it to nothing: seamsh logs that the seed is not in the
  coarsened domain, and the ocean alone is meshed."""
  output, log = _mesh_channel(channel_coastline, "[[90.0, 0.0], [0.0, 20.0]]", 1_500_000.0, 3_000_000.0)
  assert "of x0 is not in the coarsened domain" in log
  graph = read_mesh_file(output).mesh_graph
  assert face_components(graph.faces, len(graph.vertices))[0] == 1


@pytest.mark.parametrize("optimize", [True, False])
def test_the_optimize_setting_reaches_seamsh(coastline, monkeypatch, optimize):
  """`mesh.optimize` is passed to seamsh.gmsh.mesh (gmsh's optimiser after meshing, on by default)."""
  import seamsh

  calls = []
  original = seamsh.gmsh.mesh

  def spy(*args, **kwargs):
    calls.append(kwargs)
    return original(*args, **kwargs)

  monkeypatch.setattr(seamsh.gmsh, "mesh", spy)
  _mesh(coastline, f"optimize_{optimize}", extra=f"[mesh]\noptimize = {str(optimize).lower()}\n")
  assert [call["optimize"] for call in calls] == [optimize]


def _hierarchy_config(coastline: pathlib.Path, levels: list[tuple[str, float, float, str]]) -> pathlib.Path:
  """A hierarchy config on the channel coastline: per level, its name, coarsening and mesh size, and seeds (TOML)."""
  text = f'[gmsh]\nverbosity = 0\n[domain]\nfilepath = "{coastline / "coast.shp"}"\ncurve_type = "polyline"\n'
  channel = '{name = "point_refinement", points = [[0.0, 32.5]], size = 5_000.0, growth = 0.01}'
  for name, coarsening, size, seeds in levels:
    refined = name == "fine"
    text += f'[[levels]]\nname = "{name}"\n'
    text += f'[levels.coarsen_boundaries]\nx0 = {seeds}\nx0_projection = "equirectangular"\n'
    text += f'fields = [{{name = "uniform", value = {coarsening}}}' + (f", {channel}]\n" if refined else "]\n")
    text += f'[[levels.fields]]\nname = "uniform"\nvalue = {size}\n'
    if refined:
      text += '[[levels.fields]]\nname = "point_refinement"\npoints = [[0.0, 32.5]]\nsize = 10_000.0\ngrowth = 0.3\n'
  path = coastline / "hierarchy.toml"
  path.write_text(text)
  return path


def test_one_config_makes_a_hierarchy(channel_coastline):
  """A config with levels makes every mesh from one coastline and writes them, with the links, to one file: the fine
  level keeps the channel open (one connected ocean), the coarse one coarsens the coastline at its own size, so the
  sea is a piece of its own, kept by its seed; with --msh each level gets a gmsh file."""
  config = _hierarchy_config(
    channel_coastline,
    [("fine", 200_000.0, 300_000.0, "[[90.0, 0.0]]"), ("coarse", 400_000.0, 800_000.0, "[[90.0, 0.0], [0.0, 20.0]]")],
  )
  output, msh = channel_coastline / "hierarchy.nc", channel_coastline / "hierarchy.msh"
  arguments = [str(config), str(output), "--msh", str(msh), "--log-level", "warning"]
  result = CliRunner().invoke(make_ocean_mesh, arguments)
  assert result.exit_code == 0, result.output
  assert num_levels(output) == 2
  fine, coarse = (read_mesh_file(output, level).mesh_graph for level in (0, 1))
  assert face_components(fine.faces, len(fine.vertices))[0] == 1
  assert face_components(coarse.faces, len(coarse.vertices))[0] == 2
  assert len(fine.vertices) > len(coarse.vertices)
  face, weights, _, _ = read_links(output, 0, 1)
  assert face.shape == (len(fine.vertices),) and np.allclose(weights.sum(axis=1), 1.0)
  assert sorted(path.name for path in channel_coastline.glob("hierarchy_level*.msh")) == [
    "hierarchy_level0.msh",
    "hierarchy_level1.msh",
  ]
  # The outputs exist: a second run needs --overwrite, also for the per-level gmsh files.
  result = CliRunner().invoke(make_ocean_mesh, arguments)
  assert result.exit_code != 0 and "--overwrite" in result.output


def test_the_levels_of_a_hierarchy_must_be_finest_first(channel_coastline):
  """Levels given coarsest first are refused after meshing, with the median edges, and nothing is written (neither the
  mesh file nor the gmsh files of --msh, so that a corrected run needs no --overwrite)."""
  config = _hierarchy_config(
    channel_coastline,
    [("coarse", 400_000.0, 800_000.0, "[[90.0, 0.0]]"), ("fine", 200_000.0, 300_000.0, "[[90.0, 0.0]]")],
  )
  output = channel_coastline / "reversed.nc"
  arguments = [str(config), str(output), "--msh", str(channel_coastline / "reversed.msh"), "--log-level", "warning"]
  result = CliRunner().invoke(make_ocean_mesh, arguments)
  assert result.exit_code != 0 and "finest first" in result.output
  assert not output.exists() and not list(channel_coastline.glob("reversed*.msh"))
