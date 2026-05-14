#!/usr/bin/env bash
# Wire up Vault k8s auth for pf2e-mcp.
# Usage: VAULT_TOKEN=<token> bash vault-setup.sh
set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://localhost:18201}"

vput() {
  local path=$1; shift
  curl -sf -H "X-Vault-Token: ${VAULT_TOKEN}" \
    -H "Content-Type: application/json" \
    -X POST "${VAULT_ADDR}/v1/${path}" \
    -d "$@"
}

echo "==> Write policy"
vput sys/policies/acl/pf2e-mcp \
  '{"policy":"path \"secret/data/voyage\" { capabilities = [\"read\"] }\npath \"secret/data/qdrant\" { capabilities = [\"read\"] }"}'

echo "==> Write role"
vput auth/kubernetes/role/pf2e-mcp \
  '{"bound_service_account_names":"pf2e-mcp","bound_service_account_namespaces":"ttrpg","policies":"pf2e-mcp","ttl":"1h"}'

echo "==> Done. Vault configured for pf2e-mcp."
