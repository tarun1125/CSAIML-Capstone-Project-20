#!/usr/bin/env bash
# Deploys the RAG service to Azure Container Apps (docs/AZURE-PLAN.md, Phase 5).
# Safe to re-run: every create is guarded by a "does it exist" check, and a
# re-run with a new TAG builds and rolls out a new image.
#
#   ./infra/deploy.sh            # everything
#   TAG=v2 ./infra/deploy.sh     # rebuild + roll out a new image
#
# Reads resource NAMES from azure.env (never commits them to anything else).
# Needs, in azure.env: RG LOC SRCH AOAI ACR ENV APP KV LAW AI MYIP
# and, the first time only, a file ./atlas-ro-uri.txt holding the READ-ONLY
# Atlas URI (gitignored; stored in Key Vault, then you delete the file).
#
# What gets created (all in $RG, all deleted by `az group delete`):
#   Log Analytics (0.1 GB/day cap) + App Insights, Container Registry (Basic),
#   Key Vault (RBAC), Container Apps environment (consumption) + the app
#   (scale 0..1, system-assigned identity, ingress restricted to $MYIP).
# No keys go into environment variables: the app reaches Search, OpenAI and
# Key Vault with its managed identity; the Atlas URI and the service API key
# are Key Vault secret references.
#
# The Azure CLI changes more often than this script. If a flag is refused,
# check `az <command> --help` before anything else.

set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source azure.env; set +a
: "${RG:?}" "${LOC:?}" "${SRCH:?}" "${AOAI:?}" "${ACR:?}" "${ENV:?}" "${APP:?}" "${KV:?}" "${LAW:?}" "${AI:?}" "${MYIP:?}"
TAG="${TAG:-v1}"
IMAGE="$ACR.azurecr.io/capstone-rag:$TAG"
exists() { "$@" -o none >/dev/null 2>&1; }
say() { printf '\n== %s\n' "$*"; }

say "1. Logs: Log Analytics (daily cap) + App Insights"
exists az monitor log-analytics workspace show -g "$RG" -n "$LAW" ||
  az monitor log-analytics workspace create -g "$RG" -n "$LAW" -l "$LOC" --retention-time 30 -o none
az monitor log-analytics workspace update -g "$RG" -n "$LAW" --quota 0.1 -o none   # GB/day
LAW_ID=$(az monitor log-analytics workspace show -g "$RG" -n "$LAW" --query id -o tsv)
az extension add --name application-insights --upgrade --yes -o none 2>/dev/null || true
exists az monitor app-insights component show -g "$RG" -a "$AI" ||
  az monitor app-insights component create -g "$RG" -a "$AI" -l "$LOC" --workspace "$LAW_ID" -o none
AI_CONN=$(az monitor app-insights component show -g "$RG" -a "$AI" --query connectionString -o tsv)

say "2. Registry + image (cloud build, no local Docker)"
exists az acr show -g "$RG" -n "$ACR" || az acr create -g "$RG" -n "$ACR" --sku Basic -l "$LOC" -o none
az acr build -r "$ACR" -t "capstone-rag:$TAG" -f service/Dockerfile .

say "3. Key Vault (RBAC) + secrets"
exists az keyvault show -g "$RG" -n "$KV" ||
  az keyvault create -g "$RG" -n "$KV" -l "$LOC" --enable-rbac-authorization true -o none
KV_ID=$(az keyvault show -g "$RG" -n "$KV" --query id -o tsv)
ME=$(az ad signed-in-user show --query id -o tsv)
az role assignment create --assignee-object-id "$ME" --assignee-principal-type User \
  --role "Key Vault Secrets Officer" --scope "$KV_ID" -o none 2>/dev/null || true
if ! exists az keyvault secret show --vault-name "$KV" -n atlas-uri-ro; then
  [ -f atlas-ro-uri.txt ] || { echo "atlas-ro-uri.txt missing (READ-ONLY Atlas URI) -- create it first"; exit 1; }
  for i in 1 2 3 4 5 6; do   # the role assignment above can take a minute to apply
    az keyvault secret set --vault-name "$KV" -n atlas-uri-ro --file atlas-ro-uri.txt -o none && break; sleep 20
  done
  echo "   stored atlas-uri-ro -- now delete atlas-ro-uri.txt"
fi
if ! exists az keyvault secret show --vault-name "$KV" -n service-api-key; then
  tmp=$(mktemp); chmod 600 "$tmp"; python3 -c "import secrets;print(secrets.token_urlsafe(32),end='')" > "$tmp"
  az keyvault secret set --vault-name "$KV" -n service-api-key --file "$tmp" -o none; rm -f "$tmp"
  echo "   generated service-api-key (read it with: az keyvault secret show --vault-name $KV -n service-api-key --query value -o tsv)"
fi

say "4. Container Apps environment + app (scale 0..1)"
if ! exists az containerapp env show -g "$RG" -n "$ENV"; then
  LAW_CID=$(az monitor log-analytics workspace show -g "$RG" -n "$LAW" --query customerId -o tsv)
  LAW_KEY=$(az monitor log-analytics workspace get-shared-keys -g "$RG" -n "$LAW" --query primarySharedKey -o tsv)
  az containerapp env create -g "$RG" -n "$ENV" -l "$LOC" --logs-workspace-id "$LAW_CID" --logs-workspace-key "$LAW_KEY" -o none
fi
if ! exists az containerapp show -g "$RG" -n "$APP"; then
  # Starts without secrets and will fail its first revision -- step 6 fixes that
  # once the identity exists and has its roles.
  az containerapp create -g "$RG" -n "$APP" --environment "$ENV" --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-identity system --system-assigned \
    --min-replicas 0 --max-replicas 1 --cpu 1 --memory 2Gi \
    --ingress external --target-port 8000 -o none
fi
PID=$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)

say "5. Roles for the app's managed identity"
assign() { az role assignment create --assignee-object-id "$PID" --assignee-principal-type ServicePrincipal \
           --role "$1" --scope "$2" -o none 2>/dev/null || true; }
assign "AcrPull" "$(az acr show -g "$RG" -n "$ACR" --query id -o tsv)"
assign "Key Vault Secrets User" "$KV_ID"
assign "Search Index Data Reader" "$(az search service show -g "$RG" -n "$SRCH" --query id -o tsv)"
assign "Cognitive Services OpenAI User" "$(az cognitiveservices account show -g "$RG" -n "$AOAI" --query id -o tsv)"

say "6. Secrets (Key Vault references) + settings -> new revision"
KVURI=$(az keyvault show -g "$RG" -n "$KV" --query properties.vaultUri -o tsv)
az containerapp secret set -g "$RG" -n "$APP" -o none --secrets \
  "atlas-uri=keyvaultref:${KVURI}secrets/atlas-uri-ro,identityref:system" \
  "service-api-key=keyvaultref:${KVURI}secrets/service-api-key,identityref:system"
AOAI_ENDPOINT=$(az cognitiveservices account show -g "$RG" -n "$AOAI" --query properties.endpoint -o tsv)
az containerapp update -g "$RG" -n "$APP" --image "$IMAGE" -o none --set-env-vars \
  MONGODB_URI=secretref:atlas-uri SERVICE_API_KEY=secretref:service-api-key \
  RETRIEVER=azure-vector DB_POLICY=rank1 TOP_K=10 MAX_ROWS=50 MAX_QUESTION_CHARS=500 RATE_LIMIT_PER_MIN=10 \
  AZURE_SEARCH_ENDPOINT="https://$SRCH.search.windows.net" AZURE_SEARCH_INDEX=fewshot-exemplars \
  AZURE_OPENAI_ENDPOINT="$AOAI_ENDPOINT" AZURE_OPENAI_DEPLOYMENT=gpt-4o \
  APPLICATIONINSIGHTS_CONNECTION_STRING="$AI_CONN"

say "7. Lock down: ingress to $MYIP only; print outbound IPs for the Atlas access list"
az containerapp ingress access-restriction set -g "$RG" -n "$APP" --rule-name me \
  --ip-address "$MYIP/32" --action Allow -o none
FQDN=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
echo "   URL:          https://$FQDN"
echo "   Outbound IPs (add ONLY these to the Atlas IP access list):"
az containerapp show -g "$RG" -n "$APP" --query properties.outboundIpAddresses -o tsv | tr ' ' '\n' | sed 's/^/     /'
echo "   Health check: curl -s https://$FQDN/healthz"
