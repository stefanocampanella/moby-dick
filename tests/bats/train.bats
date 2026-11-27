#!/usr/bin/env bats
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
# scripts/train.sh: the Apptainer binds it builds for the training data (directories and SquashFS images), the
# options it passes to `moby-dick train`, and the checks that fail before the container starts. The options are
# parsed with GNU getopt: those tests are skipped where it is missing (macOS).

setup() {
    load common
    common_setup
    stub apptainer
    # A project with an image, a training config and the four data inputs (the mask as a SquashFS image).
    mkdir -p "${PROJECT}/configs/training" "${PROJECT}/data/dataset" "${PROJECT}/data/tendency_std" \
        "${PROJECT}/data/geometry"
    touch "${PROJECT}/moby-dick.sif" "${PROJECT}/configs/training/demo.toml" "${PROJECT}/data/mask.sqsh"
    export DATASET=data/dataset MASK=data/mask.sqsh TENDENCY_STD="${PROJECT}/data/tendency_std" \
        GEOMETRY_DIR=data/geometry
}

train() {
    run "${REPO}/scripts/train.sh" --project-dir "${PROJECT}" "$@"
}

@test "--help works without getopt or Apptainer" {
    # Only bash (for the shebang) and cat (for the usage) on PATH.
    local bin="${BATS_TEST_TMPDIR}/bin"
    mkdir -p "${bin}"
    ln -s "$(command -v bash)" "${bin}/bash"
    ln -s "$(command -v cat)" "${bin}/cat"
    rm "${STUBS}/apptainer"
    PATH="${bin}" run "${REPO}/scripts/train.sh" --help
    assert_success
    assert_output --partial 'Usage: scripts/train.sh [options] CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME'
}

@test "directories are bound read-only under /data, SquashFS images with image-src" {
    require_gnu_getopt
    train demo model ckpt
    assert_success
    run calls
    assert_output --partial "--bind ${PROJECT}/data/dataset:/data/dataset:ro"
    assert_output --partial "--bind ${PROJECT}/data/mask.sqsh:/data/mask:image-src=/,ro"
    assert_output --partial "--bind ${PROJECT}/data/tendency_std:/data/tendency_std:ro"
    assert_output --partial "--bind ${PROJECT}/data/geometry:/data/geometry:ro"
}

@test "moby-dick train gets the config, output and checkpoint paths and --no-overwrite by default" {
    require_gnu_getopt
    train demo model ckpt
    assert_success
    run calls
    assert_output --regexp "^apptainer run --nv --pwd ${PROJECT} .* ${PROJECT}/moby-dick.sif moby-dick train \
--log-level=info --tensorboard-logdir=${PROJECT}/data/tensorboard --no-overwrite -- \
${PROJECT}/configs/training/demo.toml ${PROJECT}/data/models/model ${PROJECT}/data/training/ckpt$"
}

@test "--overwrite, --continue and --log-level reach moby-dick train" {
    require_gnu_getopt
    train --continue demo --overwrite model ckpt --log-level=debug
    assert_success
    run calls
    assert_output --partial 'moby-dick train --log-level=debug'
    assert_output --partial ' --overwrite --continue -- '
}

@test "the output, checkpoint and TensorBoard directories are created and bound" {
    require_gnu_getopt
    train --output-dir out --train-dir "${BATS_TEST_TMPDIR}/ckpts" --tb-dir tb demo runs/model ckpt
    assert_success
    assert_dir_exists "${PROJECT}/out/runs"
    assert_dir_exists "${BATS_TEST_TMPDIR}/ckpts"
    assert_dir_exists "${PROJECT}/tb"
    run calls
    assert_output --partial "--bind ${PROJECT}/out:${PROJECT}/out"
    assert_output --partial "--bind ${BATS_TEST_TMPDIR}/ckpts:${BATS_TEST_TMPDIR}/ckpts"
    assert_output --partial "${PROJECT}/out/runs/model ${BATS_TEST_TMPDIR}/ckpts/ckpt"
}

@test "extra binds: relative sources resolve in the project, destinations must be absolute" {
    require_gnu_getopt
    mkdir -p "${PROJECT}/extra"
    train --bind extra:/data/extra:ro --bind extra demo model ckpt
    assert_success
    run calls
    assert_output --partial "--bind ${PROJECT}/extra:/data/extra:ro"
    assert_output --partial "--bind ${PROJECT}/extra:${PROJECT}/extra"
    train --bind extra:relative demo model ckpt
    assert_failure 1
    assert_output --partial 'bind destination must be absolute'
}

@test "a data input that is missing or contains an Apptainer separator fails before Apptainer starts" {
    require_gnu_getopt
    unset GEOMETRY_DIR
    train demo model ckpt
    assert_failure 1
    assert_output --partial 'GEOMETRY_DIR must name a readable directory'
    export GEOMETRY_DIR=data/geometry
    mkdir -p "${PROJECT}/data/a:b"
    DATASET='data/a:b' train demo model ckpt
    assert_failure 1
    assert_output --partial 'bind path contains an Apptainer separator'
    assert_equal "$(calls)" ''
}

@test "a missing training config fails" {
    require_gnu_getopt
    train missing model ckpt
    assert_failure 1
    assert_output --partial "training config is not a readable file: ${PROJECT}/configs/training/missing.toml"
}

@test "wrong positional arguments, log levels and option values fail" {
    require_gnu_getopt
    train demo model
    assert_failure 1
    assert_output --partial 'expected exactly CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME'
    train --log-level=loud demo model ckpt
    assert_failure 1
    assert_output --partial 'invalid log level: loud'
    train --tb-dir --continue demo model ckpt
    assert_failure 1
    assert_output --partial 'missing value for --tb-dir'
    train --unknown demo model ckpt
    assert_failure 1
    assert_output --partial 'unknown option or missing value'
}

@test "without Apptainer on PATH the script fails with a message" {
    require_gnu_getopt
    rm "${STUBS}/apptainer"
    PATH="${STUBS}:$(dirname "$(command -v getopt)"):/usr/bin:/bin" train demo model ckpt
    assert_failure 1
    assert_output --partial 'apptainer not found'
}

@test "the exit status of the container reaches the caller (75: preempted, resume with --continue)" {
    require_gnu_getopt
    stub apptainer 'exit 75'
    train demo model ckpt
    assert_failure 75
}
