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
"""Tests of the separation of the `mesh` extra (gmsh and seamsh, GPL): no command but make-ocean-mesh loads them,
and without them the CLI says what to install."""

import os
import subprocess
import sys

from click.testing import CliRunner

# The subprocesses import the sources pytest imports (`pythonpath` in pyproject.toml), not an installed copy.
ENVIRONMENT = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

# The modules of every command and of training, as the CLI and the training processes import them.
CORE_MODULES = (
  "mobydick",
  "mobydick.training",
  "mobydick.training_utils",
  "mobydick.graph_statistics.cli",
  "mobydick.basis",
  "mobydick.mesh_commands",
  "mobydick.ugrid",
  "mobydick.downloader",
  "mobydick.leviathan.model",
)


def test_core_modules_do_not_import_the_mesh_extra():
  """Importing the CLI, training and every command but make-ocean-mesh leaves gmsh and seamsh unloaded (in a fresh
  interpreter: this test process may have loaded them already), so that a training process never holds GPL code."""
  code = "import sys\n" + "".join(f"import {module}\n" for module in CORE_MODULES)
  code += "loaded = sorted({'gmsh', 'seamsh'} & set(sys.modules))\nassert not loaded, loaded\n"
  result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=ENVIRONMENT)
  assert result.returncode == 0, result.stderr


def test_without_the_extra_make_ocean_mesh_says_what_to_install():
  """Without gmsh and seamsh, --help still lists every command (make-ocean-mesh as not installed) and make-ocean-mesh
  fails with the extra to install; the other mesh commands work."""
  code = """
import sys
sys.modules["gmsh"] = None  # Importing it raises ModuleNotFoundError, as when it is not installed.
sys.modules["seamsh"] = None
from click.testing import CliRunner
import mobydick
runner = CliRunner()
listed = runner.invoke(mobydick.cli, ["--help"])
assert listed.exit_code == 0 and "needs the mesh extra" in listed.output, listed.output
failed = runner.invoke(mobydick.cli, ["make-ocean-mesh", "config.toml", "mesh.nc"])
assert failed.exit_code == 1 and 'pip install "moby-dick[mesh]"' in failed.output, failed.output
assert runner.invoke(mobydick.cli, ["make-icosahedral-mesh", "--help"]).exit_code == 0
"""
  result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=ENVIRONMENT)
  assert result.returncode == 0, result.stderr


def test_a_broken_gmsh_does_not_break_help(tmp_path):
  """A gmsh whose native libraries are missing raises OSError on import (the PyPI wheel without libGLU): --help still
  lists every command, and make-ocean-mesh says why it cannot run."""
  (tmp_path / "gmsh.py").write_text('raise OSError("libGLU.so.1: cannot open shared object file")\n')
  code = """
from click.testing import CliRunner
import mobydick
runner = CliRunner()
listed = runner.invoke(mobydick.cli, ["--help"])
assert listed.exit_code == 0 and "make-ocean-mesh" in listed.output, listed.output
failed = runner.invoke(mobydick.cli, ["make-ocean-mesh", "config.toml", "mesh.nc"])
assert failed.exit_code == 1 and "libGLU" in failed.output, failed.output
"""
  environment = {**ENVIRONMENT, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
  result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=environment)
  assert result.returncode == 0, result.stderr


def test_with_the_extra_make_ocean_mesh_is_the_command():
  """With the extra installed, make-ocean-mesh is the real command (its help names the mesh file it writes)."""
  import pytest

  pytest.importorskip("seamsh", reason="needs the mesh extra")
  import mobydick

  result = CliRunner().invoke(mobydick.cli, ["make-ocean-mesh", "--help"])
  assert result.exit_code == 0 and "--msh" in result.output
