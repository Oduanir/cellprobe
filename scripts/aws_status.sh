#!/usr/bin/env bash
#
# Real-time status + cost estimate for all AWS resources tagged Project=cellprobe
# in eu-west-3 (Paris).
#
# AWS billing has a 6-24h delay on the console; this script computes the
# cellprobe subset live by multiplying current uptime by the listed on-demand
# rates. It is an *estimate* (does not include data egress, snapshots, etc.)
# but covers > 95% of what this project will spend.
#
# Important: EC2 cost shown is for the **current running session only**
# (since the latest start). AWS bills aggregate actual running time across
# all sessions — for historical cumulative cost, use Cost Explorer in the
# console (24h delay accepted). EBS cost is since-creation (always-on).
#
# Usage:
#   ./scripts/aws_status.sh
#
# Requires `aws` CLI and `jq`. Uses --profile admin --region eu-west-3 by
# default; override with AWS_PROFILE / AWS_REGION env vars.
set -euo pipefail

PROFILE="${AWS_PROFILE:-admin}"
REGION="${AWS_REGION:-eu-west-3}"
PROJECT_TAG="${PROJECT_TAG:-cellprobe}"

# eu-west-3 on-demand rates (USD/h). Update if AWS rates change.
# Using a function instead of an associative array for bash 3.2 compatibility (macOS default).
rate_for() {
  case "$1" in
    g6.xlarge)    echo 0.805 ;;
    g6.2xlarge)   echo 1.210 ;;
    g5.xlarge)    echo 1.262 ;;
    g4dn.xlarge)  echo 0.526 ;;
    t3.medium)    echo 0.0464 ;;
    *)            echo 0 ;;
  esac
}

EBS_GP3_PER_GB_MONTH=0.0928   # USD per GB-month, eu-west-3

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
dim()  { printf '\033[2m%s\033[0m\n' "$1"; }

bold "=== EC2 instances (Project=$PROJECT_TAG) ==="
INSTANCES_JSON=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=$PROJECT_TAG" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,PublicIpAddress,Placement.AvailabilityZone,Tags[?Key==`Name`]|[0].Value]' \
  --output json)

TOTAL_COMPUTE_COST=0
if [[ "$(echo "$INSTANCES_JSON" | jq 'length')" == "0" ]]; then
  dim "no EC2 instances tagged Project=$PROJECT_TAG"
else
  printf '%-22s %-13s %-9s %-12s %-17s %-12s %s\n' \
    INSTANCE_ID TYPE STATE UPTIME RATE_USD_PER_H COST_SO_FAR NAME
  while IFS=$'\t' read -r id type state launched ip az name; do
    rate=$(rate_for "$type")
    if [[ "$state" == "running" ]]; then
      start=$(date -u -j -f '%Y-%m-%dT%H:%M:%S' "${launched%.*}" '+%s' 2>/dev/null || date -d "$launched" '+%s')
      now=$(date -u '+%s')
      hours=$(echo "scale=3; ($now - $start) / 3600" | bc)
      cost=$(echo "scale=2; $hours * $rate" | bc)
      TOTAL_COMPUTE_COST=$(echo "scale=2; $TOTAL_COMPUTE_COST + $cost" | bc)
      uptime="${hours}h"
    else
      uptime="-"
      cost="0.00"
    fi
    printf '%-22s %-13s %-9s %-12s %-17s %-12s %s\n' \
      "$id" "$type" "$state" "$uptime" "\$$rate" "\$$(printf '%.3f' "$cost")" "${name:-(no name)}"
  done < <(echo "$INSTANCES_JSON" | jq -r '.[] | @tsv')
fi
echo

bold "=== EBS volumes (Project=$PROJECT_TAG or attached to a tagged instance) ==="
# 1. Directly-tagged volumes
VOLUMES_DIRECT=$(aws ec2 describe-volumes \
  --filters "Name=tag:Project,Values=$PROJECT_TAG" \
  --query 'Volumes[].VolumeId' --output text)
# 2. Volumes attached to tagged instances (most of our volumes won't have the tag — they were auto-created by run-instances)
INSTANCE_IDS=$(echo "$INSTANCES_JSON" | jq -r '.[][0]')
VOLUMES_ATTACHED=""
if [[ -n "$INSTANCE_IDS" ]]; then
  VOLUMES_ATTACHED=$(aws ec2 describe-volumes \
    --filters "Name=attachment.instance-id,Values=${INSTANCE_IDS// /,}" \
    --query 'Volumes[].VolumeId' --output text)
fi
VOLUMES=$(echo "$VOLUMES_DIRECT $VOLUMES_ATTACHED" | tr ' ' '\n' | sort -u | grep -v '^$' || true)

TOTAL_EBS_COST=0
if [[ -z "$VOLUMES" ]]; then
  dim "no EBS volumes found"
else
  printf '%-25s %-8s %-6s %-12s %s\n' VOLUME_ID SIZE_GB TYPE COST_SO_FAR ATTACHED_TO
  for vol in $VOLUMES; do
    read -r size type created attached <<< "$(aws ec2 describe-volumes --volume-ids "$vol" \
      --query 'Volumes[0].[Size,VolumeType,CreateTime,Attachments[0].InstanceId]' --output text)"
    start=$(date -u -j -f '%Y-%m-%dT%H:%M:%S' "${created%.*}" '+%s' 2>/dev/null || date -d "$created" '+%s')
    now=$(date -u '+%s')
    hours=$(echo "scale=3; ($now - $start) / 3600" | bc)
    # cost = size_gb * rate_per_gb_month * (hours / 24 / 30.44)
    cost=$(echo "scale=2; $size * $EBS_GP3_PER_GB_MONTH * $hours / (24 * 30.44)" | bc)
    TOTAL_EBS_COST=$(echo "scale=2; $TOTAL_EBS_COST + $cost" | bc)
    printf '%-25s %-8s %-6s %-12s %s\n' "$vol" "${size}GB" "$type" "\$$(printf '%.3f' "$cost")" "${attached:-none}"
  done
fi
echo

bold "=== Estimated cellprobe cost so far ==="
TOTAL=$(echo "scale=2; $TOTAL_COMPUTE_COST + $TOTAL_EBS_COST" | bc)
printf '  Compute (current session):    $%.3f\n' "$TOTAL_COMPUTE_COST"
printf '  EBS storage (since creation): $%.3f\n' "$TOTAL_EBS_COST"
printf '  \033[1mTotal estimate:               $%.3f\033[0m\n' "$TOTAL"
dim "Excludes: data egress, EBS snapshots, NAT/ELB if any."
dim "AWS console figures lag by 6-24h; this is the live computed view."
