#!/usr/bin/env bats
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
# scripts/download.sh: the moby-dick command it picks, the arguments it forwards to Apptainer unchanged, and its
# checks before starting the container.

setup() {
    load common
    common_setup
    IMAGE="${BATS_TEST_TMPDIR}/moby-dick.sif"
    touch "${IMAGE}"
    export APPTAINER_IMAGE="${IMAGE}"
    stub apptainer
}

@test "--help prints the usage and succeeds without an image" {
    unset APPTAINER_IMAGE
    run "${REPO}/scripts/download.sh" --help
    assert_success
    assert_output --partial 'Usage: scripts/download.sh DATASET CLIMATOLOGY ANOMALY_STD OUTPUT'
    assert_equal "$(calls)" ''
}

@test "too few arguments print the usage and fail" {
    run "${REPO}/scripts/download.sh" dataset climatology anomaly_std
    assert_failure 1
    assert_output --partial 'Usage:'
    assert_equal "$(calls)" ''
}

@test "the standardized anomalies: moby-dick download with every argument forwarded" {
    run "${REPO}/scripts/download.sh" s3://bucket/dataset /local/climatology anomaly_std out \
        --config configs/download/zarr_v2.toml --overwrite
    assert_success
    assert_equal "$(calls)" "apptainer exec ${IMAGE} moby-dick download s3://bucket/dataset /local/climatology \
anomaly_std out --config configs/download/zarr_v2.toml --overwrite"
}

@test "--mask runs download-mask with two positional arguments" {
    run "${REPO}/scripts/download.sh" --mask dataset mask --config configs/download/mask.toml
    assert_success
    assert_equal "$(calls)" "apptainer exec ${IMAGE} moby-dick download-mask dataset mask --config \
configs/download/mask.toml"
}

@test "--tendency-std runs compute-tendency-std" {
    run "${REPO}/scripts/download.sh" --tendency-std standardized tendency_std
    assert_success
    assert_equal "$(calls)" "apptainer exec ${IMAGE} moby-dick compute-tendency-std standardized tendency_std"
}

@test "--mask with one argument fails" {
    run "${REPO}/scripts/download.sh" --mask dataset
    assert_failure 1
    assert_equal "$(calls)" ''
}

@test "an unreadable image fails before Apptainer starts" {
    export APPTAINER_IMAGE="${BATS_TEST_TMPDIR}/missing.sif"
    run "${REPO}/scripts/download.sh" dataset climatology anomaly_std out
    assert_failure 1
    assert_output --partial "image is not readable: ${BATS_TEST_TMPDIR}/missing.sif"
    assert_equal "$(calls)" ''
}

@test "the image defaults to moby-dick.sif in the project" {
    unset APPTAINER_IMAGE
    copy_into_project scripts/download.sh
    touch "${PROJECT}/moby-dick.sif"
    run "${PROJECT}/scripts/download.sh" dataset climatology anomaly_std out
    assert_success
    assert_equal "$(calls)" "apptainer exec ${PROJECT}/moby-dick.sif moby-dick download dataset climatology \
anomaly_std out"
}

@test "the container finds AWS credential helpers in ~/.local/bin unless the environment says otherwise" {
    stub apptainer 'echo "PREPEND_PATH=${APPTAINERENV_PREPEND_PATH}" >> "${CALLS}"'
    run "${REPO}/scripts/download.sh" dataset climatology anomaly_std out
    assert_success
    run calls
    assert_line --index 1 "PREPEND_PATH=${HOME}/.local/bin"
    : > "${CALLS}"
    APPTAINERENV_PREPEND_PATH=/opt/helpers run "${REPO}/scripts/download.sh" dataset climatology anomaly_std out
    assert_success
    run calls
    assert_line --index 1 'PREPEND_PATH=/opt/helpers'
}

@test "the exit status of the container reaches the caller" {
    stub apptainer 'exit 3'
    run "${REPO}/scripts/download.sh" dataset climatology anomaly_std out
    assert_failure 3
}
