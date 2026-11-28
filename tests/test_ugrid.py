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
"""Tests of the mesh files (`mobydick.ugrid`): one mesh or a hierarchy in a UGRID NetCDF-4 file, the links between
levels, and `moby-dick combine-meshes`."""

import dataclasses

import h5py
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from mobydick import icosahedral_mesh, ugrid
from mobydick.gis_utils import cartesian_to_lat_lon, cartesian_unit_sphere_crs
from mobydick.mesh_commands import combine_meshes
from mobydick.mesh_graph import MeshData, boundary_vertices, faces_to_edges


def _ocean_like(splits: int = 2) -> MeshData:
  """An icosahedral mesh with its northern cap removed: an open mesh with a boundary, like an ocean mesh."""
  # Entries as TOML multi-line strings give them, with line breaks inside.
  mesh = icosahedral_mesh.get_multi_mesh_data(
    splits, description=["  A mesh\n  on two lines\n  ", "second"], license=["A license\n"]
  )
  graph = mesh.mesh_graph
  lat, _ = cartesian_to_lat_lon(graph.vertices)
  keep = (lat[graph.faces] < 60.0).all(axis=1)
  used = np.unique(graph.faces[keep])
  index = np.full(len(graph.vertices), -1)
  index[used] = np.arange(len(used))
  faces = index[graph.faces[keep]]
  graph = dataclasses.replace(
    graph,
    vertices=graph.vertices[used],
    faces=faces,
    edges=faces_to_edges(faces),
    boundary=boundary_vertices(faces, len(used)),
  )
  assert mesh.mesh_size is not None
  return dataclasses.replace(mesh, mesh_graph=graph, mesh_size=mesh.mesh_size[used])


def test_a_mesh_reads_back_as_written(tmp_path):
  """Every array of a mesh comes back unchanged (vertices bit for bit, faces, directed edges, boundary, mesh size,
  size fields, the constant one too), with its spatial reference system, description and license (one text each:
  the strings of a list come back joined by line breaks, line breaks inside them kept)."""
  mesh = _ocean_like()
  fields = {"uniform": np.full(len(mesh.mesh_graph.vertices), 1e6), "shore_proximity": 2 * np.asarray(mesh.mesh_size)}
  path = tmp_path / "mesh.nc"
  ugrid.write_mesh_file(path, [mesh], [fields])
  read = ugrid.read_mesh_file(path)
  graph, expected = read.mesh_graph, mesh.mesh_graph
  np.testing.assert_array_equal(graph.vertices, expected.vertices)
  np.testing.assert_array_equal(graph.faces, expected.faces)
  for got, want in zip(graph.edges, expected.edges, strict=True):
    np.testing.assert_array_equal(got, want)
  assert graph.boundary is not None and expected.boundary is not None
  np.testing.assert_array_equal(graph.boundary, expected.boundary)
  assert graph.spatial_reference_system == expected.spatial_reference_system
  assert read.mesh_size is not None and mesh.mesh_size is not None
  np.testing.assert_array_equal(read.mesh_size, mesh.mesh_size)
  assert read.description == ["\n".join(mesh.description)] and read.license == mesh.license
  sizes = ugrid.read_size_fields(path)
  assert set(sizes) == set(fields)
  for name, values in fields.items():
    np.testing.assert_array_equal(sizes[name], values)
  assert ugrid.num_levels(path) == 1


def test_the_file_follows_ugrid(tmp_path):
  """The topology variable names the UGRID roles of the other variables, connectivity starts at 0, the UGRID edges are
  the faces' edges once each (the processor's directed edges are a variable of their own), and the node coordinates
  are the geodetic longitude and latitude of the nodes, so that UGRID tools can open the file."""
  mesh = _ocean_like()
  path = tmp_path / "mesh.nc"
  ugrid.write_mesh_file(path, [mesh])
  with xr.open_dataset(path) as dataset:
    assert "UGRID-1.0" in dataset.attrs["Conventions"]
    topology = dataset["mesh0"].attrs
    assert topology["cf_role"] == "mesh_topology" and topology["topology_dimension"] == 2
    lon_name, lat_name = topology["node_coordinates"].split()
    for role in ("face_node_connectivity", "edge_node_connectivity"):
      variable = dataset[topology[role]]
      assert variable.attrs["cf_role"] == role and variable.attrs["start_index"] == 0
    assert dataset[topology["face_node_connectivity"]].dims[0] == topology["face_dimension"]
    edges = dataset[topology["edge_node_connectivity"]].values
    assert len(edges) == len(mesh.mesh_graph.edges[0]) // 2 and len(np.unique(np.sort(edges, axis=1), axis=0)) == len(
      edges
    )
    assert "mesh0_processor_edges" not in topology.values()
    lat, lon = cartesian_to_lat_lon(mesh.mesh_graph.vertices)
    np.testing.assert_allclose(dataset[lat_name].values, lat)
    np.testing.assert_allclose(dataset[lon_name].values, lon)
    assert dataset[lat_name].attrs["standard_name"] == "latitude"
    assert dataset["mesh0_boundary"].attrs["location"] == "node"


def test_the_file_opens_in_qgis_and_paraview(tmp_path):
  """What QGIS 3.44 (MDAL) and ParaView 6.0 need, found by opening mesh files in them on 28 Nov 2026 (the apps are not
  in CI): text attributes are NC_CHAR, as both read them with `nc_get_att_text` and found no mesh with NC_STRING
  (h5netcdf's default for `str`); node fields are int32 or floating point, as MDAL reads no smaller integers; a node
  field with one value everywhere is a scalar, as QGIS crashed on one (it comes back at every node); no variable is
  named `<mesh>_node_z`, MDAL's fallback for the elevation; the [nodes, 3] link weights are no node field, which the
  viewers would read as 3 values at as many times as nodes."""
  fine, coarse = _ocean_like(2), _ocean_like(1)
  constant = np.full(len(fine.mesh_graph.vertices), 1e6)
  path = tmp_path / "hierarchy.nc"
  ugrid.write_mesh_file(path, [fine, coarse], [{"uniform": constant}, None])
  hdf5_internal = {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST", "_Netcdf4Dimid", "_Netcdf4Coordinates"}

  def check_texts(name: str, item: h5py.HLObject) -> None:
    for key in item.attrs:
      kind = item.attrs.get_id(key).get_type()
      if isinstance(kind, h5py.h5t.TypeStringID) and key not in hdf5_internal:
        assert not kind.is_variable_str(), f"{name}: {key} is NC_STRING"

  with h5py.File(path) as file:
    check_texts("/", file)
    file.visititems(check_texts)
  with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as dataset:
    fields = [name for name, variable in dataset.variables.items() if variable.attrs.get("location") == "node"]
    assert {"mesh0_boundary", "mesh0_node_in_mesh1_distance", "mesh0_node_in_mesh1_inside"} <= set(fields)
    for name in fields:
      values = dataset[name].values
      assert values.dtype in (np.int32, np.float32, np.float64), name
      assert values.ndim == 1 and not (values == values[0]).all(), name
    # The uniform size field, and the coarse nodes' distances (0: they are fine nodes) and `inside`, are scalars.
    for name in ("mesh0_size_uniform", "mesh1_node_in_mesh0_distance", "mesh1_node_in_mesh0_inside"):
      assert dataset[name].ndim == 0, name
    assert not [name for name in dataset.variables if str(name).endswith("_node_z")]
    assert "location" not in dataset["mesh0_node_in_mesh1_weights"].attrs
  np.testing.assert_array_equal(ugrid.read_size_fields(path)["uniform"], constant)
  _, _, distance, inside = ugrid.read_links(path, 1, 0)
  assert distance.shape == inside.shape == (len(coarse.mesh_graph.vertices),)
  assert (distance == 0).all() and inside.all()
  read = ugrid.read_mesh_file(path, 1)
  assert read.mesh_size is not None and coarse.mesh_size is not None
  np.testing.assert_array_equal(read.mesh_size, coarse.mesh_size)


def test_a_hierarchy_holds_every_level_and_the_links_between_them(tmp_path):
  """A hierarchy file holds each level as a mesh of its own, and both ways between consecutive levels the closest
  face and the barycentric weights: they sum to 1, reproduce a node's position inside the other mesh (exactly for a
  linear function, up to the mesh's curvature for positions on the sphere), and point at faces of the right mesh."""
  fine, coarse = icosahedral_mesh.get_multi_mesh_data(3), icosahedral_mesh.get_multi_mesh_data(1)
  path = tmp_path / "hierarchy.nc"
  ugrid.write_mesh_file(path, [fine, coarse])
  assert ugrid.num_levels(path) == 2
  for level, mesh in enumerate((fine, coarse)):
    np.testing.assert_array_equal(ugrid.read_mesh_file(path, level).mesh_graph.faces, mesh.mesh_graph.faces)
  for level, other, nodes, faces in ((0, 1, fine, coarse), (1, 0, coarse, fine)):
    face, weights, distance, inside = ugrid.read_links(path, level, other)
    # Both meshes cover the sphere: every node is within the other one, also the fine nodes off its flat faces.
    assert inside.all()
    if level == 0:
      assert (distance > 1e3).any()
    vertices = nodes.mesh_graph.vertices
    assert face.shape == (len(vertices),) and weights.shape == (len(vertices), 3)
    assert face.max() < len(faces.mesh_graph.faces) and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # The weights interpolate the other mesh's vertices to the closest point on it: within `distance` of the node.
    corners = faces.mesh_graph.vertices[faces.mesh_graph.faces[face]]
    closest = np.einsum("ij,ijk->ik", weights, corners)
    np.testing.assert_allclose(np.linalg.norm(closest - vertices, axis=1), distance, rtol=1e-6, atol=1e-3)
  # Coarse vertices are vertices of the fine mesh: they lie on it, at one of its faces' corners.
  _, weights, distance, _ = ugrid.read_links(path, 1, 0)
  np.testing.assert_allclose(distance, 0.0, atol=1e-3)
  np.testing.assert_allclose(weights.max(axis=1), 1.0, atol=1e-6)


def test_nodes_outside_the_other_mesh_are_flagged(tmp_path):
  """Nodes of a closed mesh north of an open mesh's edge (60° N) are outside the open mesh: flagged so, with the
  distance to its nearest face, while the nodes well inside it are within it."""
  closed, open_mesh = icosahedral_mesh.get_multi_mesh_data(3), _ocean_like(2)
  path = tmp_path / "pair.nc"
  ugrid.write_mesh_file(path, [closed, open_mesh])
  _, _, distance, inside = ugrid.read_links(path, 0, 1)
  lat, _ = cartesian_to_lat_lon(closed.mesh_graph.vertices)
  assert not inside[lat > 75.0].any() and distance[lat > 75.0].min() > 500e3
  assert inside[lat < 45.0].all()


def test_only_earth_centred_metres_are_written(tmp_path):
  """Mesh files hold WGS84 Earth-centred coordinates in metres (node lon/lat and the links are computed from them): a
  mesh on the unit sphere is refused."""
  graph = icosahedral_mesh.get_multi_mesh_graph(1, crs=cartesian_unit_sphere_crs)
  mesh = MeshData(mesh_graph=graph, mesh_size=None, description=[], license=[])
  with pytest.raises(ValueError, match="Earth-centred WGS84"):
    ugrid.write_mesh_file(tmp_path / "unit.nc", [mesh])


def test_missing_levels_and_links_are_errors(tmp_path):
  """Asking for a level the file does not have, or for links between levels that are not consecutive, is an error
  that says why."""
  path = tmp_path / "mesh.nc"
  ugrid.write_mesh_file(path, [icosahedral_mesh.get_multi_mesh_data(1)])
  with pytest.raises(ValueError, match="no mesh of level 1"):
    ugrid.read_mesh_file(path, 1)
  with pytest.raises(ValueError, match="no links"):
    ugrid.read_links(path, 0, 1)
  with pytest.raises(ValueError, match="consecutive"):
    ugrid.read_links(path, 0, 2)
  with pytest.raises(ValueError, match="No mesh"):
    ugrid.write_mesh_file(tmp_path / "empty.nc", [])
  # A hierarchy written before the links held `_inside` is to be regenerated.
  hierarchy, old = tmp_path / "hierarchy.nc", tmp_path / "old.nc"
  ugrid.write_mesh_file(hierarchy, [icosahedral_mesh.get_multi_mesh_data(2), icosahedral_mesh.get_multi_mesh_data(1)])
  with xr.open_dataset(hierarchy, engine="h5netcdf", decode_cf=False) as dataset:
    dataset.drop_vars(["mesh0_node_in_mesh1_inside"]).to_netcdf(old, engine="h5netcdf")
  with pytest.raises(ValueError, match="regenerate"):
    ugrid.read_links(old, 0, 1)


def test_combine_meshes(tmp_path):
  """moby-dick combine-meshes writes files of one mesh each, finest first, as one hierarchy with their size fields and
  the links; coarsest first, a file that is already a hierarchy, or an existing output are refused."""
  fine, coarse = icosahedral_mesh.get_multi_mesh_data(2), icosahedral_mesh.get_multi_mesh_data(1)
  fine_path, coarse_path = tmp_path / "fine.nc", tmp_path / "coarse.nc"
  ugrid.write_mesh_file(fine_path, [fine], [{"uniform": np.ones(len(fine.mesh_graph.vertices))}])
  ugrid.write_mesh_file(coarse_path, [coarse])
  output = tmp_path / "hierarchy.nc"
  run = CliRunner().invoke
  result = run(combine_meshes, [str(output), str(fine_path), str(coarse_path), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  assert ugrid.num_levels(output) == 2 and list(ugrid.read_size_fields(output, 0)) == ["uniform"]
  assert ugrid.read_links(output, 0, 1)[0].shape == (len(fine.mesh_graph.vertices),)
  result = run(combine_meshes, [str(tmp_path / "reversed.nc"), str(coarse_path), str(fine_path)])
  assert result.exit_code != 0 and "finest first" in result.output
  result = run(combine_meshes, [str(tmp_path / "nested.nc"), str(output), str(coarse_path)])
  assert result.exit_code != 0 and "holds 2 meshes" in result.output
  result = run(combine_meshes, [str(output), str(fine_path), str(coarse_path)])
  assert result.exit_code != 0 and "--overwrite" in result.output
