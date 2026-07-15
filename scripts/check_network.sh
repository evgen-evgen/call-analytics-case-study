#!/usr/bin/env bash
set -euo pipefail

docker_probe_ip="172.19.0.9"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or is not in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is unavailable for the current user." >&2
  exit 1
fi

if command -v tailscale >/dev/null 2>&1; then
  prefs="$(tailscale debug prefs 2>/dev/null || true)"
  if grep -q '"RouteAll": true' <<<"${prefs}" \
    && grep -q '"ExitNodeAllowLANAccess": false' <<<"${prefs}"; then
    cat >&2 <<'EOF'
ERROR: Tailscale exit node is enabled without local-network access.
Docker published ports will be routed to tailscale0 and become unavailable.

Fix once with:
  sudo tailscale set --exit-node-allow-lan-access=true
EOF
    exit 1
  fi
fi

if docker network inspect mtbank-ai-hiring_default >/dev/null 2>&1; then
  route="$(ip route get "${docker_probe_ip}" 2>/dev/null || true)"
  if grep -q 'dev tailscale0' <<<"${route}"; then
    echo "ERROR: Docker subnet is still routed through tailscale0: ${route}" >&2
    exit 1
  fi
fi

echo "Network preflight passed."
