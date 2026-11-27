#!/usr/bin/env bats
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
# The launchers in clusters/jupiter/slurm: how train.slurm and download.slurm source the site environment and
# hand their arguments to the portable scripts, and the configs and jobs mesh_study.sh makes from the real demo
# config. Slurm, the module system and the scripts they start are stand-ins. The launchers run with bash, as sbatch
# runs them (download.slurm is not executable).

setup() {
    load common
    common_setup
    copy_into_project clusters/jupiter/slurm/train.slurm clusters/jupiter/slurm/download.slurm \
        clusters/jupiter/slurm/mesh_study.sh configs/training/demo_5000.toml
    # A site environment like clusters/jupiter/.env, with this test's paths.
    ENV_FILE="${PROJECT}/clusters/jupiter/.env"
    cat > "${ENV_FILE}" <<EOF
MODULES=(CUDA/13 Other/1)
APPTAINERENV_PYTHONPATH=/site/src
DATA_ROOT=${BATS_TEST_TMPDIR}/data
GEOMETRY_DIR=${BATS_TEST_TMPDIR}/geometry
DATASET=/site/dataset
EOF
    mkdir -p "${BATS_TEST_TMPDIR}/geometry"
    stub module
    stub srun 'echo "DATASET=${DATASET} SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK}" >> "${CALLS}"'
    stub sbatch
    stub_at "${PROJECT}/scripts/train.sh"
    stub_at "${PROJECT}/scripts/download.sh"
}

# --- train.slurm --------------------------------------------------------------------------------------------------

@test "train.slurm --help needs no site environment" {
    rm "${ENV_FILE}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --help
    assert_success
    assert_output --partial 'Usage: sbatch clusters/jupiter/slurm/train.slurm'
    assert_equal "$(calls)" ''
}

@test "train.slurm starts one step of train.sh with the runner's arguments unchanged and exports the site settings" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --continue --bind 'a b:/data/x' --log-level debug demo model ckpt
    assert_success
    run calls
    assert_line --index 2 --regexp "^srun --export=ALL --kill-on-bad-exit=1 --output=${PROJECT}/logs/training/%j_%N_%t.out \
--error=${PROJECT}/logs/training/%j_%N_%t.err ${PROJECT}/scripts/train.sh --project-dir ${PROJECT} --continue --bind \
'a b:/data/x' --log-level debug demo model ckpt$"
    assert_line --index 3 'DATASET=/site/dataset SRUN_CPUS_PER_TASK=72'
}

@test "train.slurm loads the site's modules and unloads them in reverse order" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" demo model ckpt
    assert_success
    run calls
    assert_line --index 0 'module load CUDA/13'
    assert_line --index 1 'module load Other/1'
    assert_line --index 4 'module unload Other/1'
    assert_line --index 5 'module unload CUDA/13'
}

@test "train.slurm takes --env-file and --log-dir for itself, also as --option=value" {
    local other="${BATS_TEST_TMPDIR}/other.env"
    echo 'DATASET=/other/dataset' > "${other}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --project-dir "${PROJECT}" --env-file="${other}" \
        --log-dir logs/here demo model ckpt
    assert_success
    assert_dir_exists "${PROJECT}/logs/here"
    run calls
    assert_line --index 0 --partial "--output=${PROJECT}/logs/here/%j_%N_%t.out"
    assert_line --index 0 --partial "train.sh --project-dir ${PROJECT} demo model ckpt"
    assert_line --index 1 --partial 'DATASET=/other/dataset'
}

@test "train.slurm forwards a runner option whose value looks like a launcher option" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --bind --env-file demo model ckpt
    assert_success
    run calls
    assert_line --index 2 --partial 'train.sh --project-dir '"${PROJECT}"' --bind --env-file demo model ckpt'
}

@test "train.slurm merges the task logs of the job into one file per stream" {
    export SLURM_SUBMIT_DIR="${PROJECT}" SLURM_JOB_ID=123
    local logs="${PROJECT}/logs/training"
    stub srun "echo first > ${logs}/123_node1_0.out; echo second > ${logs}/123_node1_1.out; echo oops > ${logs}/123_node1_0.err"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" demo model ckpt
    assert_success
    assert_file_not_exists "${logs}/123_node1_0.out"
    assert_file_not_exists "${logs}/123_node1_1.out"
    run cat "${logs}/123.out"
    assert_output "$(printf '===== 123_node1_0.out =====\nfirst\n===== 123_node1_1.out =====\nsecond')"
    run cat "${logs}/123.err"
    assert_output "$(printf '===== 123_node1_0.err =====\noops')"
}

@test "train.slurm exits with the status of the step (75: preempted) after cleaning up" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    stub srun 'exit 75'
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" demo model ckpt
    assert_failure 75
    run calls
    assert_line --index 3 'module unload Other/1'
}

@test "train.slurm fails without a readable site environment or a runner" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --env-file "${BATS_TEST_TMPDIR}/missing.env" demo model ckpt
    assert_failure 1
    assert_output --partial "Bash environment file not readable: ${BATS_TEST_TMPDIR}/missing.env"
    chmod -x "${PROJECT}/scripts/train.sh"
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" demo model ckpt
    assert_failure 1
    assert_output --partial 'runner is not executable'
    run bash "${PROJECT}/clusters/jupiter/slurm/train.slurm" --env-file
    assert_failure 1
    assert_output --partial 'missing value for --env-file'
}

# --- download.slurm -----------------------------------------------------------------------------------------------

@test "download.slurm sources the site environment and runs download.sh with its arguments" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    stub_at "${PROJECT}/scripts/download.sh" 'echo "PYTHONPATH=${APPTAINERENV_PYTHONPATH}" >> "${CALLS}"'
    run bash "${PROJECT}/clusters/jupiter/slurm/download.slurm" --mask dataset mask --config configs/download/mask.toml
    assert_success
    run calls
    assert_line --index 0 'download.sh --mask dataset mask --config configs/download/mask.toml'
    assert_line --index 1 'PYTHONPATH=/site/src'
}

@test "download.slurm takes another environment file with --env-file" {
    export SLURM_SUBMIT_DIR="${PROJECT}"
    local other="${BATS_TEST_TMPDIR}/other.env"
    echo 'APPTAINERENV_PYTHONPATH=/worktree/src' > "${other}"
    stub_at "${PROJECT}/scripts/download.sh" 'echo "PYTHONPATH=${APPTAINERENV_PYTHONPATH}" >> "${CALLS}"'
    run bash "${PROJECT}/clusters/jupiter/slurm/download.slurm" --env-file "${other}" a b c d
    assert_success
    run calls
    assert_line --index 0 'download.sh a b c d'
    assert_line --index 1 'PYTHONPATH=/worktree/src'
}

@test "download.slurm fails outside the project directory" {
    export SLURM_SUBMIT_DIR="${BATS_TEST_TMPDIR}"
    run bash "${PROJECT}/clusters/jupiter/slurm/download.slurm" a b c d
    assert_failure 1
    assert_output --partial 'submit from the project directory'
}

# --- mesh_study.sh ------------------------------------------------------------------------------------------------

@test "mesh_study.sh writes one config per case and seed from demo_5000.toml and submits each" {
    touch "${BATS_TEST_TMPDIR}/geometry/uniform_120km.nc" "${BATS_TEST_TMPDIR}/geometry/icosahedral_6.nc"
    run "${PROJECT}/clusters/jupiter/slurm/mesh_study.sh" --account acc study 1,2 120km:8 icosahedral_6:16:4096:06:00:00
    assert_success
    local configs="${PROJECT}/runs/study/configs"
    local settings='^(seed|gnn_msg_steps|path = "/data/geometry/|edge_chunk_size)'
    run grep -E "${settings}" "${configs}/demo_5000_120km_msg8_seed2.toml"
    assert_output "$(printf '%s\n' 'seed = 2' 'gnn_msg_steps = 8' 'path = "/data/geometry/uniform_120km.nc"' \
        'edge_chunk_size = 0')"
    run grep -E "${settings}" "${configs}/demo_5000_icosahedral_6_msg16_seed1.toml"
    assert_output "$(printf '%s\n' 'seed = 1' 'gnn_msg_steps = 16' 'path = "/data/geometry/icosahedral_6.nc"' \
        'edge_chunk_size = 4096')"
    run calls
    assert_equal "${#lines[@]}" 4
    assert_line --index 0 "sbatch --account=acc --nodes=8 --time=03:00:00 --job-name=study-120km-msg8-s1 \
${PROJECT}/clusters/jupiter/slurm/train.slurm --env-file ${configs}/study.env --config-dir ${configs} --tb-dir \
${BATS_TEST_TMPDIR}/data/tensorboard/study/demo_5000_120km_msg8_seed1 demo_5000_120km_msg8_seed1 \
study_demo_5000_120km_msg8_seed1 study_demo_5000_120km_msg8_seed1"
    assert_line --index 3 --partial '--time=06:00:00 --job-name=study-icosahedral_6-msg16-s2 '
}

@test "mesh_study.sh points the jobs at the code of its own checkout" {
    touch "${BATS_TEST_TMPDIR}/geometry/uniform_50km.nc"
    run "${PROJECT}/clusters/jupiter/slurm/mesh_study.sh" study 42 50km:16
    assert_success
    run grep '^APPTAINERENV_PYTHONPATH=' "${PROJECT}/runs/study/configs/study.env"
    assert_output "APPTAINERENV_PYTHONPATH=${PROJECT}/src"
    run calls
    assert_line --index 0 --partial '--account=training2647 '
}

@test "mesh_study.sh fails on a missing mesh or a case without message passing steps, before submitting" {
    run "${PROJECT}/clusters/jupiter/slurm/mesh_study.sh" study 1 75km:8
    assert_failure 1
    assert_output --partial "no mesh uniform_75km.nc or 75km.nc in ${BATS_TEST_TMPDIR}/geometry"
    run "${PROJECT}/clusters/jupiter/slurm/mesh_study.sh" study 1 75km
    assert_failure 1
    assert_output --partial 'no message passing steps in 75km'
    run "${PROJECT}/clusters/jupiter/slurm/mesh_study.sh" study 1
    assert_failure 1
    assert_output --partial 'no cases given'
    assert_equal "$(calls)" ''
}
