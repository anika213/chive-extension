#!/bin/bash
# GPU preflight check - source this at the top of Slurm GPU job scripts.
# Fails immediately if Slurm assigned this job a GPU that already has too little
# free memory, before expensive model loading can OOM later.
#
# Safety model: nvidia-smi is node-wide on this cluster, even inside Slurm jobs.
# Only inspect GPUs named by SLURM_JOB_GPUS. Do not kill or reset processes here.
#
# Usage in Slurm scripts:
#   source /workspace-vast/adamk/chive/scripts/gpu_preflight.sh
#
# Set GPU_MIN_FREE_MIB to override the free-memory threshold (default: 130000 MiB).

GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-130000}"

_gpu_preflight_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

_gpu_preflight() {
    local min_free="$1"

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "GPU preflight FAILED: SLURM_JOB_ID is not set; this check must run inside a Slurm job."
        exit 1
    fi

    if [[ -z "${SLURM_JOB_GPUS:-}" ]]; then
        echo "GPU preflight FAILED: SLURM_JOB_GPUS is not set for job ${SLURM_JOB_ID}."
        exit 1
    fi

    declare -A uuid_by_index
    declare -A free_by_uuid
    local idx uuid mem_free
    while IFS=',' read -r idx uuid mem_free; do
        idx=$(_gpu_preflight_trim "$idx")
        uuid=$(_gpu_preflight_trim "$uuid")
        mem_free=$(_gpu_preflight_trim "$mem_free")
        [[ -z "$idx" || -z "$uuid" || -z "$mem_free" ]] && continue
        uuid_by_index["$idx"]="$uuid"
        free_by_uuid["$uuid"]="$mem_free"
    done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits)

    local assigned_raw="${SLURM_JOB_GPUS}"
    local assigned_token
    local assigned_uuids=()
    IFS=',' read -ra assigned_tokens <<< "$assigned_raw"

    for assigned_token in "${assigned_tokens[@]}"; do
        assigned_token=$(_gpu_preflight_trim "$assigned_token")
        [[ -z "$assigned_token" ]] && continue

        if [[ "$assigned_token" =~ ^[0-9]+$ ]]; then
            if [[ -z "${uuid_by_index[$assigned_token]:-}" ]]; then
                echo "GPU preflight FAILED: assigned GPU index ${assigned_token} was not found by nvidia-smi."
                exit 1
            fi
            assigned_uuids+=("${uuid_by_index[$assigned_token]}")
        elif [[ "$assigned_token" =~ ^GPU- ]]; then
            if [[ -z "${free_by_uuid[$assigned_token]:-}" ]]; then
                echo "GPU preflight FAILED: assigned GPU UUID ${assigned_token} was not found by nvidia-smi."
                exit 1
            fi
            assigned_uuids+=("$assigned_token")
        else
            echo "GPU preflight FAILED: unsupported SLURM_JOB_GPUS token '${assigned_token}'."
            exit 1
        fi
    done

    if (( ${#assigned_uuids[@]} == 0 )); then
        echo "GPU preflight FAILED: SLURM_JOB_GPUS='${SLURM_JOB_GPUS}' did not contain any GPU IDs."
        exit 1
    fi

    local dirty=false
    local target_uuid target_free
    for target_uuid in "${assigned_uuids[@]}"; do
        target_free="${free_by_uuid[$target_uuid]}"
        if (( target_free < min_free )); then
            dirty=true
            echo "GPU preflight dirty: assigned GPU ${target_uuid} has ${target_free} MiB free (<${min_free} MiB)."
        fi
    done

    if ! $dirty; then
        echo "GPU preflight OK: assigned GPU(s) have >=${min_free} MiB free on $(hostname)"
        return 0
    fi

    echo ""
    echo "GPU preflight FAILED on $(hostname)"
    echo "Job: ${SLURM_JOB_ID} | SLURM_JOB_GPUS=${SLURM_JOB_GPUS} | $(date)"
    echo "Assigned GPU compute apps:"

    local app_uuid pid mem_used pname app_matched
    for target_uuid in "${assigned_uuids[@]}"; do
        app_matched=false
        while IFS=',' read -r app_uuid pid mem_used pname; do
            app_uuid=$(_gpu_preflight_trim "$app_uuid")
            pid=$(_gpu_preflight_trim "$pid")
            mem_used=$(_gpu_preflight_trim "$mem_used")
            pname=$(_gpu_preflight_trim "$pname")
            [[ "$app_uuid" != "$target_uuid" ]] && continue

            app_matched=true
            if ps -p "$pid" > /dev/null 2>&1; then
                local pid_user
                pid_user=$(ps -p "$pid" -o user= | tr -d ' ')
                echo "  ${target_uuid}: live pid=${pid} user=${pid_user} mem=${mem_used} name=${pname}"
            else
                echo "  ${target_uuid}: zombie pid=${pid} mem=${mem_used} name=${pname}"
            fi
        done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader 2>/dev/null || true)

        if ! $app_matched; then
            echo "  ${target_uuid}: no compute apps reported, but memory is below threshold"
        fi
    done

    echo "Exiting immediately so Slurm can retry on a different GPU/node."
    exit 1
}

_gpu_preflight "$GPU_MIN_FREE_MIB"
