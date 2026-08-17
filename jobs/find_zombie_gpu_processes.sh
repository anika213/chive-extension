#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Read-only scanner for GPU processes that nvidia-smi reports but ps cannot find.
#
# Usage:
#   ./jobs/find_zombie_gpu_processes.sh              # scan all available nodes
#   ./jobs/find_zombie_gpu_processes.sh node-11      # scan specific node(s)
#   ./jobs/find_zombie_gpu_processes.sh --exclude    # print an sbatch exclude flag
#
# Set SSH_TIMEOUT to override the per-node SSH timeout in seconds (default: 5).

set -euo pipefail

SSH_TIMEOUT="${SSH_TIMEOUT:-5}"
MODE="report"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,10p' "$0"
    exit 0
fi

if [[ "${1:-}" == "--exclude" ]]; then
    MODE="exclude"
    shift
fi

if (( $# > 0 )); then
    NODES=("$@")
else
    mapfile -t NODES < <(sinfo -N -o "%N %T" --noheader 2>/dev/null \
        | awk '$2 !~ /down|drain/ {print $1}' \
        | sort -t'-' -k2 -n \
        | uniq)
fi

TMPDIR_SCAN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SCAN"' EXIT

scan_node() {
    local node="$1"
    local outfile="$TMPDIR_SCAN/$node"
    local errfile="$TMPDIR_SCAN/$node.err"

    if ! ssh -o ConnectTimeout="$SSH_TIMEOUT" -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$node" 'bash -s' > "$outfile" 2> "$errfile" <<'REMOTE'
set -euo pipefail

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

declare -A GPU_INDEX_BY_UUID
while IFS=',' read -r idx uuid; do
    idx=$(trim "$idx")
    uuid=$(trim "$uuid")
    [[ -z "$idx" || -z "$uuid" ]] && continue
    GPU_INDEX_BY_UUID["$uuid"]="$idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

found=false
while IFS=',' read -r uuid pid mem_used name; do
    uuid=$(trim "$uuid")
    pid=$(trim "$pid")
    mem_used=$(trim "$mem_used")
    name=$(trim "$name")
    [[ -z "$uuid" || -z "$pid" ]] && continue

    if ! ps -p "$pid" > /dev/null 2>&1; then
        found=true
        gpu_index="${GPU_INDEX_BY_UUID[$uuid]:-unknown}"
        kill_check=$(kill -0 "$pid" 2>&1 || true)
        if [[ -z "$kill_check" ]]; then
            echo "hidden:gpu:${gpu_index} uuid:${uuid} pid:${pid} mem:${mem_used} name:${name} reason:ps_failed_but_pid_exists"
        elif [[ "$kill_check" == *"Operation not permitted"* ]]; then
            echo "hidden:gpu:${gpu_index} uuid:${uuid} pid:${pid} mem:${mem_used} name:${name} reason:pid_exists_permission_denied"
        else
            echo "zombie:gpu:${gpu_index} uuid:${uuid} pid:${pid} mem:${mem_used} name:${name} reason:pid_not_found"
        fi
    fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader 2>/dev/null || true)

if ! $found; then
    echo "clean"
fi
REMOTE
    then
        echo "UNREACHABLE" > "$outfile"
    fi
}

for node in "${NODES[@]}"; do
    scan_node "$node" &
done
wait

zombie_nodes=()
hidden_nodes=()
unreachable_nodes=()
warning_nodes=()

for node in "${NODES[@]}"; do
    content=$(cat "$TMPDIR_SCAN/$node")
    errfile="$TMPDIR_SCAN/$node.err"
    if [[ -s "$errfile" ]]; then
        warning_nodes+=("$node")
    fi

    if [[ "$content" == "UNREACHABLE" ]]; then
        unreachable_nodes+=("$node")
        continue
    fi

    if grep -q '^zombie:' "$TMPDIR_SCAN/$node"; then
        zombie_nodes+=("$node")
        if [[ "$MODE" == "report" ]]; then
            echo "$node:"
            grep '^zombie:' "$TMPDIR_SCAN/$node" | sed 's/^/  /'
        fi
    fi

    if grep -q '^hidden:' "$TMPDIR_SCAN/$node"; then
        hidden_nodes+=("$node")
        if [[ "$MODE" == "report" ]]; then
            echo "$node (hidden/live PID, not excluded):"
            grep '^hidden:' "$TMPDIR_SCAN/$node" | sed 's/^/  /'
        fi
    fi
done

if (( ${#zombie_nodes[@]} == 0 )); then
    if [[ "$MODE" == "report" ]]; then
        if (( ${#unreachable_nodes[@]} > 0 )); then
            echo "No zombie GPU processes found on reachable nodes."
        else
            echo "No zombie GPU processes found."
        fi
    fi
else
    if [[ "$MODE" == "report" ]]; then
        echo ""
        echo "Zombie GPU nodes: ${zombie_nodes[*]}"
    fi
fi

if (( ${#unreachable_nodes[@]} > 0 )); then
    if [[ "$MODE" == "report" ]]; then
        echo "Unreachable nodes (not treated as zombies): ${unreachable_nodes[*]}"
    fi
fi

if (( ${#hidden_nodes[@]} > 0 )); then
    if [[ "$MODE" == "report" ]]; then
        echo "Hidden/live GPU PID nodes (not excluded): ${hidden_nodes[*]}"
    fi
fi

if (( ${#warning_nodes[@]} > 0 )); then
    if [[ "$MODE" == "report" ]]; then
        echo "SSH warnings on nodes (not treated as zombies): ${warning_nodes[*]}"
    fi
fi

if (( ${#unreachable_nodes[@]} > 0 )); then
    if [[ "$MODE" == "report" ]]; then
        echo "WARNING: direct SSH failed for ${#unreachable_nodes[@]} node(s), so live zombie scanning is incomplete."
        echo "         These nodes are not included in --exclude unless a live scan succeeds and finds zombies."
    else
        echo "WARNING: direct SSH failed for ${#unreachable_nodes[@]} node(s), so live zombie scanning is incomplete: ${unreachable_nodes[*]}" >&2
    fi
fi

if [[ "$MODE" == "exclude" ]]; then
    exclude_nodes=("${zombie_nodes[@]}")

    declare -A seen_exclude
    unique_exclude=()
    for node in "${exclude_nodes[@]}"; do
        if [[ -z "${seen_exclude[$node]:-}" ]]; then
            unique_exclude+=("$node")
            seen_exclude["$node"]=1
        fi
    done

    if (( ${#unique_exclude[@]} > 0 )); then
        exclude_str=$(IFS=,; echo "${unique_exclude[*]}")
        echo "--exclude=${exclude_str}"
    fi
fi
