# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
#
# What every test file shares, loaded by its setup (`load common`, then `common_setup`). The libraries are git
# submodules in helpers/ (bats-core's own layout): bats-support (output), bats-assert (assert_success,
# assert_output, assert_line, ...) and bats-file (assert_file_exists, ...). Run the tests with
#
#     git submodule update --init
#     tests/bats/helpers/bats-core/bin/bats tests/bats
#
# The scripts never reach Apptainer, Slurm or the module system: stand-ins first on PATH record their calls.

# The checkout under test.
REPO="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"

common_setup() {
    load "${BATS_TEST_DIRNAME}/helpers/bats-support/load"
    load "${BATS_TEST_DIRNAME}/helpers/bats-assert/load"
    load "${BATS_TEST_DIRNAME}/helpers/bats-file/load"
    # Each test has its own project directory, and its stand-ins first on PATH.
    PROJECT="${BATS_TEST_TMPDIR}/project"
    STUBS="${BATS_TEST_TMPDIR}/stubs"
    mkdir -p "${PROJECT}" "${STUBS}"
    PATH="${STUBS}:${PATH}"
    export CALLS="${BATS_TEST_TMPDIR}/calls"
    # Module systems such as Lmod export `module` as a shell function, which would win over the stand-in on PATH.
    unset -f module ml 2> /dev/null || true
    # Nothing of the caller's site environment reaches the scripts.
    unset APPTAINER_IMAGE APPTAINERENV_PREPEND_PATH DATA_ROOT DATA_DIR DATASET MASK TENDENCY_STD GEOMETRY_DIR \
        PROJECT_DIR CONFIG_DIR OUTPUT_DIR TRAIN_DIR TB_DIR LOG_DIR LOG_LEVEL SLURM_SUBMIT_DIR SLURM_JOB_ID \
        SLURM_CPUS_PER_TASK
    export HOME="${BATS_TEST_TMPDIR}/home"
    mkdir -p "${HOME}"
}

# A stand-in for a command, first on PATH: it records its name and arguments, one call per line, in $CALLS, then runs
# BODY (shell code; "$@" are its arguments). An argument that is empty or holds whitespace is recorded in single
# quotes ('a b'), so that a split or lost argument shows; the others as they are (not with printf %q, whose escapes
# differ between bash versions, e.g. for commas). Usage: stub NAME [BODY].
stub() {
    stub_at "${STUBS}/$1" "${2:-}"
}

# The same at a given path (e.g. a script of the project replaced by a stand-in).
# shellcheck disable=SC2094 # basename reads the path, not the file
stub_at() {
    mkdir -p "$(dirname "$1")"
    {
        echo '#!/usr/bin/env bash'
        echo "{ printf %s $(basename "$1"); for a in \"\$@\"; do"
        echo "    if [[ -z \$a || \$a == *[[:space:]]* ]]; then printf \" '%s'\" \"\$a\"; else printf ' %s' \"\$a\"; fi"
        echo "done; echo; } >> \"\${CALLS}\""
        printf '%s\n' "${2:-}"
    } > "$1"
    chmod +x "$1"
}

# The calls the stand-ins recorded, one per line (none: nothing).
calls() {
    cat "${CALLS}" 2>/dev/null || true
}

# Copies of the given files of the checkout in the project, at the same relative paths, for scripts that find the
# project from their own location or from the submit directory. Usage: copy_into_project PATH...
copy_into_project() {
    local path
    for path in "$@"; do
        mkdir -p "${PROJECT}/$(dirname "${path}")"
        cp -p "${REPO}/${path}" "${PROJECT}/${path}"
    done
}

# Skip a test that needs GNU getopt (scripts/train.sh parses its options with it; macOS has BSD getopt).
require_gnu_getopt() {
    local status=0
    getopt --test > /dev/null 2>&1 || status=$?
    ((status == 4)) || skip 'GNU getopt is not installed'
}
