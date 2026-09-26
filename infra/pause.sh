#!/usr/bin/env bash
# End-of-session cost guard (docs/AZURE-PLAN.md, "Cost control"). Disables
# ingress so nothing -- including internet scanners -- can wake the app, then
# lists everything billable in the subscription.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source azure.env; set +a
az containerapp ingress disable -g "$RG" -n "$APP" -o none && echo "ingress disabled"
az containerapp update -g "$RG" -n "$APP" --min-replicas 0 -o none && echo "min replicas 0"
echo "search tier: $(az search service show -g "$RG" -n "$SRCH" --query sku.name -o tsv) (basic bills hourly -- delete it if so)"
echo "--- everything in the subscription (anything unexpected is a mistake):"
az resource list --query "[].{name:name, type:type, rg:resourceGroup}" -o table
