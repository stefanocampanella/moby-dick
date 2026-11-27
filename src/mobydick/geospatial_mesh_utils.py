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

# TODO:
#   1. Some StereoMeshSizeField classes require several samples to average otherwise noisy criterion fields. These are
#      The one based on the hessian norm (SST might be a good target) and the one leveraging the
#      Courant–Friedrichs–Lewy condition. These require both to be implemented and some other machinery (a command in
#      geospatial_mesh.py, and possibly a slurm script in leonardo/scripts) to compute the relevant statistics before a
#      field object can be instantiated.
#   3. Seamsh can ingest raster fields, use this feature to implement HessianField, BathymetryField and CourantField.
import abc
import atexit
import logging
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import gmsh
import numpy as np
import seamsh
import xarray as xr
from osgeo import osr
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from mobydick.gis_utils import (
  CoordinateReferenceSystem,
  CRSName,
  CRSRegistry,
  cartesian_crs,
  get_transform,
  lat_lon_to_cartesian,
  stereographic_crs,
  xarray_to_gdal_raster,
)
from mobydick.mesh_config import FieldConfig, field_arguments

logger = logging.getLogger(__name__)


@atexit.register
def _maybe_gmsh_finalize():
  if gmsh.is_initialized():
    logger.info("Finalize gmsh.")
    gmsh.finalize()


class StereoMeshSizeField(abc.ABC):
  """
  Base class that computes a mesh size field in stereographic projection coordinates.

  This object depends on the reference system in two ways:
    1. how it interprets the coordinates of the points on which is evaluated, and
    2. whether the mesh size (which is an edge length) is defined on the stereographic plane or in 3D.

  As the point coordinates are represented by numpy arrays, not holding information of the reference system, all the
  methods take a projection argument specifying it.

  About the second point, as the meshing procedure is performed by gmsh on the stereographic plane, the `__call__`
  method returns the value of the mesh size in that space.

  First, the value in 3D space of the criterion field is computed; then the mesh size field is rescaled to its value on
  the stereographic projection.

  See https://doi.org/10.1007/s10236-008-0148-3.
  """

  @abc.abstractmethod
  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the mesh size field in 3D space."""

  def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the mesh size field in stereographic projection coordinates,
    possibly as a function of the coordinates in parametric space."""
    mesh_size = self.mesh_size_3d(x, projection)
    projection_crs = CoordinateReferenceSystem.from_osr(projection)
    if not stereographic_crs.IsSame(projection_crs):
      transform = get_transform(projection_crs, stereographic_crs)
      x = transform(x)
    assert isinstance(x, np.ndarray) and x.shape[1] == 2
    earth_radius_squared = stereographic_crs.GetSemiMajor() * stereographic_crs.GetSemiMinor()
    stereo_factor = (4 * earth_radius_squared) / (4 * earth_radius_squared + x[:, 0] ** 2 + x[:, 1] ** 2)
    return mesh_size / stereo_factor


class UniformField(StereoMeshSizeField):
  """Stereographic mesh size field with a constant value."""

  def __init__(self, value: float):
    self.value = value

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    return np.full(x.shape[0], self.value)


class BoundedStereoMeshSizeField(StereoMeshSizeField):
  """
  Base class that computes a mesh size field for using a criterion field in stereographic projection coordinates.
  Computes the criterion field in 3D space, then clips it, and finally rescale it to interpolate the values
  between `size_min` and `size_max`.

  """

  def __init__(self, size_min, size_max):
    self.size_min = size_min
    self.size_max = size_max

  @abc.abstractmethod
  def criterion(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the criterion field in 3D space."""

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the mesh size field in 3D space."""
    alpha = self.criterion(x, projection)
    delta = self.size_min + (self.size_max - self.size_min) * alpha
    return delta


class BoundaryProximityField(BoundedStereoMeshSizeField):
  """Stereographic mesh size field based on the distance from the coast."""

  def __init__(
    self,
    filepath: str | pathlib.Path,
    physical_name_field: str,
    curve_type: str,
    sampling: float,
    field_min: float,
    field_max: float,
    size_min,
    size_max,
  ):
    super().__init__(size_min, size_max)
    if isinstance(filepath, str):
      filepath = pathlib.Path(filepath)
    domain = load_domain(filepath, physical_name_field=physical_name_field, curve_type=curve_type)

    self.field_min = field_min
    self.field_max = field_max
    self.distance_from_boundary = seamsh.field.Distance(domain, sampling, projection=cartesian_crs)

  def criterion(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    value = np.clip(self.distance_from_boundary(x, projection), self.field_min, self.field_max)
    alpha = (value - self.field_min) / (self.field_max - self.field_min)
    return alpha


class RasterField(BoundedStereoMeshSizeField):
  """Stereographic mesh size field whose criterion field is interpolated from values defined on a regular grid in
  plate carree projection. The grid is assumed to be an xarray DataArray with dimensions (latitude_dim, longitude_dim).
  The field extrema are computed from the quantiles q_low and q_high.
  """

  def __init__(
    self,
    grid: xr.DataArray,
    size_min,
    size_max,
    q_low: float = 0.0,
    q_high: float = 1.0,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
  ):
    super().__init__(size_min, size_max)
    if q_low <= 0.0:
      self.field_min = np.nanmin(grid)
    else:
      self.field_min = np.nanquantile(grid, q_low)
    if q_high >= 1.0:
      self.field_max = np.nanmax(grid)
    else:
      self.field_max = np.nanquantile(grid, q_high)
    grid = grid.fillna(self.field_max)
    gdal_raster = xarray_to_gdal_raster(da=grid, latitude_dim=latitude_dim, longitude_dim=longitude_dim)
    self.field = seamsh.field.Raster(gdal_raster)

  def criterion(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    value = np.clip(self.field(x, projection), self.field_min, self.field_max)
    alpha = (value - self.field_min) / (self.field_max - self.field_min)
    return alpha


class BathymetryField(RasterField):
  """Stereographic mesh size field based on the bathymetry."""

  def __init__(self, filepath: str | pathlib.Path, var_name: str, size_min, size_max, **kwargs):
    """It assumes that variable contains positive depth values."""
    grid_ds = xr.open_dataset(filepath, engine="zarr")
    grid_da = grid_ds[var_name].load()
    grid_da = grid_da / grid_da.max()
    assert np.all(np.logical_or(grid_da.isnull(), grid_da >= 0.0)), (
      f"Variable {var_name} contain negative depth values."
    )
    bathy_sqrt: xr.DataArray = xr.apply_ufunc(np.sqrt, grid_da)
    super().__init__(bathy_sqrt, size_min, size_max, **kwargs)


class BathymetryHessianField(RasterField):
  def __init__(
    self,
    filepath: str | pathlib.Path,
    var_name: str,
    size_min,
    size_max,
    eps: float = 1.0e-5,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
    spline_kwargs: dict[str, Any] | None = None,
    filter_kwargs: dict[str, Any] | None = None,
    **kwargs,
  ):
    grid_ds = xr.open_dataset(filepath, engine="zarr")
    grid_da = grid_ds[var_name].load()
    grid_da = grid_da / grid_da.max()
    hnorm = self.norm_of_hessian(
      grid_da,
      longitude_dim=longitude_dim,
      latitude_dim=latitude_dim,
      filter_kwargs=filter_kwargs,
      spline_kwargs=spline_kwargs,
    )
    assert np.all(hnorm >= 0.0), "Computed Hessian contains negative values."
    hnorm_invsqrt = 1 / np.clip(np.sqrt(hnorm), a_min=eps, a_max=None)
    hnorm_invsqrt = xr.where(grid_da.isnull(), np.nan, hnorm_invsqrt)
    super().__init__(
      hnorm_invsqrt,
      size_min,
      size_max,
      longitude_dim=longitude_dim,
      latitude_dim=latitude_dim,
      **kwargs,
    )

  @staticmethod
  def norm_of_hessian(
    da: xr.DataArray,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
    filter_kwargs: dict[str, Any] | None = None,
    spline_kwargs: dict[str, Any] | None = None,
  ) -> xr.DataArray:
    """Computes the norm of the Hessian matrix of a 2D array. The Hessian matrix is approximated by a spline,
    and assumes latitude and longitude are in degrees."""

    spline_kwargs = spline_kwargs or {}
    filter_kwargs = filter_kwargs or {}
    # Use canonical coordinates order and fill missing values.
    da = da.transpose(latitude_dim, longitude_dim)
    da = da.fillna(0.0)
    data = da.to_numpy()
    sigma = filter_kwargs.pop("sigma", 1.0)
    data = gaussian_filter(data, sigma, **filter_kwargs)
    latitudes = da[latitude_dim].to_numpy()
    longitudes = da[longitude_dim].to_numpy()
    hessian_matrix = np.empty(data.shape + (2, 2), dtype=data.dtype)

    def direction(coord: str) -> tuple[int, int]:
      if coord == longitude_dim:
        return 0, 1
      elif coord == latitude_dim:
        return 1, 0
      else:
        raise ValueError(f"Unknown coordinate {coord}.")

    def fix_for_latitudes(z_di: np.ndarray, eps=1e-10) -> np.ndarray:
      latitudes_grid = np.tile(latitudes.reshape(-1, 1), (1, len(longitudes)))
      corrective_factor = 1 / np.clip(np.cos(latitudes_grid * np.pi / 180.0), a_min=eps, a_max=None)
      return np.where(
        np.logical_or(np.isclose(latitudes_grid, 90.0), np.isclose(latitudes_grid, -90.0)),
        np.zeros_like(z_di),
        z_di * corrective_factor,
      )

    def grad(z: np.ndarray, coord: str) -> np.ndarray:
      z_spline = RectBivariateSpline(latitudes, longitudes, z, **spline_kwargs)
      z_di = z_spline.partial_derivative(*direction(coord))(latitudes, longitudes)
      # FIXME: document why this is needed, and in which approximation solves the problem.
      if coord == longitude_dim:
        z_di = fix_for_latitudes(z_di)
      return z_di

    for i, coord_i in enumerate([longitude_dim, latitude_dim]):
      for j, coord_j in enumerate([longitude_dim, latitude_dim]):
        hessian_matrix[..., i, j] = grad(grad(data, coord_i), coord_j)

    # FIXME: document why this is needed, and in which approximation solves the problem.
    hessian_matrix = 0.5 * (hessian_matrix + np.swapaxes(hessian_matrix, -1, -2))
    hessian_matrix_singular_values = np.linalg.svd(hessian_matrix, compute_uv=False, hermitian=True)
    hessian_matrix_norm = np.max(hessian_matrix_singular_values, axis=-1)
    hessian_matrix_norm = xr.DataArray(
      data=hessian_matrix_norm, coords=da.coords, dims=da.dims, name=f"{da.name}_hessian_norm"
    )

    return hessian_matrix_norm


class PointRefinementField(StereoMeshSizeField):
  """Size `size` at the points, growing by `growth` per unit of distance from the nearest point (chord distance in
  Earth-centred coordinates). Used, combined with other fields by the minimum, to resolve straits narrower than the
  other fields' size: the coastline coarsening closes a strait narrower than its size, and gmsh needs elements about
  as wide as the strait to mesh it."""

  def __init__(self, points: list[tuple[float, float]], size: float, growth: float):
    self.size = size
    self.growth = growth
    lat, lon = np.asarray(points, dtype=np.float64).T
    self.points = lat_lon_to_cartesian(lat, lon)  # [num_points, 3], metres

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    projection_crs = CoordinateReferenceSystem.from_osr(projection)
    if not cartesian_crs.IsSame(projection_crs):
      x = get_transform(projection_crs, cartesian_crs)(np.asarray(x))
    # CHORD DISTANCE (straight line through the Earth), not geodesic. A chord of geodesic length d on a sphere of
    # radius R is shorter by about d³ / (24 R²): 0.003 % at 100 km, 0.08 % at 500 km, beyond which every refinement in
    # the configs has reached its base size ((10 km - 250 m) / 0.02 = 490 km; (120 km - 3 km) / 0.3 = 390 km). The
    # ellipsoid changes this by less (flattening 1/298). The geodesic (pyproj.Geod.inv) would cost a call per point
    # and refinement point for no visible difference. See also mesh_graph.longest_incident_edges.
    distance = np.min(np.linalg.norm(np.asarray(x)[:, None, :3] - self.points[None], axis=-1), axis=1)
    return self.size + self.growth * distance


class MinimumField(StereoMeshSizeField):
  """The minimum of several fields."""

  def __init__(self, components: Iterable[StereoMeshSizeField]):
    self.components = list(components)

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    return np.minimum.reduce([field.mesh_size_3d(x, projection) for field in self.components])


FieldName = Literal["uniform", "shore_proximity", "bathymetry", "bathymetry_hessian", "point_refinement"]
FieldsRegistry = {
  "uniform": UniformField,
  "shore_proximity": BoundaryProximityField,
  "bathymetry": BathymetryField,
  "bathymetry_hessian": BathymetryHessianField,
  "point_refinement": PointRefinementField,
}


class CompositeMeshSizeField(MinimumField):
  """The minimum of the fields of a config, by name (`fields`: each is also written as a view of the mesh)."""

  def __init__(self, fields_config: Iterable[FieldConfig]):
    """Build each field from its validated settings (mesh_config)."""
    fields = {}
    for config in fields_config:
      arguments = field_arguments(config)
      logger.info("Creating field %s with config %s.", config.name, arguments)
      fields[config.name] = FieldsRegistry[config.name](**arguments)
    self.fields = fields
    super().__init__(fields.values())


def load_domain(path: pathlib.Path, physical_name_field: str = "featurecla", curve_type: str = "bspline"):
  # Path must be a shapefile.
  if not path.name.endswith(".shp"):
    raise ValueError(f"Path must be a shapefile, got {path}")
  domain = seamsh.geometry.Domain(projection=stereographic_crs)
  domain.add_boundary_curves_shp(str(path), physical_name_field, getattr(seamsh.geometry.CurveType, curve_type.upper()))
  return domain


def coarsen_boundaries(
  domain: seamsh.geometry.Domain,
  fields: Iterable[FieldConfig],
  x0: Sequence[tuple[float, float]] | tuple[float, float] = ((0.0, 0.0),),
  x0_projection_name: CRSName = "stereographic",
):
  """Creates a new Domain with the same projection and coarsened boundaries: resampled at the minimum of `fields`,
  keeping the water reachable from any point of x0 (one point or several, see mesh_config.CoarsenBoundariesConfig).
  seamsh logs a warning for each point whose basin the coarsening loses. A single uniform field gives exactly the
  coarsening of the configs with one size (the minimum of one field is that field)."""
  x0_projection = CRSRegistry[x0_projection_name]
  size = CompositeMeshSizeField(fields)
  return seamsh.geometry.coarsen_boundaries(domain, x0=x0, x0_projection=x0_projection, mesh_size=size)
