#!/usr/bin/env bash
# Start of a session: re-open ingress, restricted to your CURRENT public IP
# (home IPs change -- update MYIP in azure.env first), then health-check.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source azure.env; set +a
az containerapp ingress enable -g "$RG" -n "$APP" --type external --target-port 8000 --transport auto -o none
az containerapp ingress access-restriction set -g "$RG" -n "$APP" --rule-name me \
  --ip-address "$MYIP/32" --action Allow -o none
FQDN=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
echo "https://$FQDN (allowed from $MYIP only)"
curl -s -m 120 "https://$FQDN/healthz"; echo
