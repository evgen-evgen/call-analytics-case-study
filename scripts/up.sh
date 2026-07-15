#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

./scripts/check_network.sh
docker compose up -d --build "$@"
./scripts/check_network.sh
docker compose ps
