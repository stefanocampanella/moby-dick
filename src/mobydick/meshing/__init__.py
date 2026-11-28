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
"""Mesh generation with seamsh and gmsh: `moby-dick make-ocean-mesh`, its size fields and the coastline coarsening.

This subpackage needs the `mesh` extra (`pip install "moby-dick[mesh]"`): gmsh (GPL-2.0-or-later) and seamsh (GPL-3.0).
No module outside it imports it, and the CLI imports it only when `make-ocean-mesh` runs, so that training and the
other commands never load GPL code (a test and the CI job `core` check it). Its output, the mesh files, is read with
`mobydick.ugrid`.
"""
