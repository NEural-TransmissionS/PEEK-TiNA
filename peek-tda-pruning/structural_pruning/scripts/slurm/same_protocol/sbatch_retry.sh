# sbatch with retries, sourced by submit_yolov5.sh and submit_recovery.sh. The Slurm controller refuses jobs
# while it is busy ("Resource temporarily unavailable"; sbatch's own retries ran out during the 2026-11-07
# re-run burst), and one refused job used to stop the whole submission. Here: up to 20 tries, 30 s apart.
sbatch() {
  local i
  for i in $(seq 1 20); do
    command sbatch "$@" && return 0
    echo "sbatch refused (try $i), retrying in 30 s" >&2; sleep 30
  done
  return 1
}
