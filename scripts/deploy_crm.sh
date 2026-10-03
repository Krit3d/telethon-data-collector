#!/bin/bash
set -e

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST [COMPOSE_FILE] [MODE]"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"
COMPOSE_FILE="${3:-docker-compose.crm.yml}"
DEPLOY_MODE="${4:-run}"

if [ "$DEPLOY_MODE" != "run" ] && [ "$DEPLOY_MODE" != "build-only" ]; then
    echo "Error: MODE must be 'run' or 'build-only'"
    exit 1
fi

echo "Deploying CRM to $SSH_USER@$SSH_HOST (mode: $DEPLOY_MODE)..."

rsync -avz --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='sessions/' \
    --exclude='.env' \
    --exclude='backups/' \
    --exclude='certbot/' \
    --exclude='tests/' \
    --exclude='node_modules/' \
    --exclude='dist/' \
    --exclude='.vite/' \
    --exclude='.pytest_cache/' \
    --exclude='*.log' \
    --exclude='src/parser/' \
    --exclude='src/embeddings/' \
    --exclude='src/graph/' \
    --exclude='src/workers/' \
    --exclude='src/web/search/' \
    --exclude='src/api/' \
    --exclude='src/db/' \
    --exclude='src/utils/' \
    --exclude='src/config/' \
    --exclude='docker/api/' \
    --exclude='docker/scraper/' \
    --exclude='docker-compose.api.yml' \
    --exclude='docker-compose.scraper.yml' \
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-crm"

ssh "$SSH_USER@$SSH_HOST" bash -s -- "$COMPOSE_FILE" "$DEPLOY_MODE" <<'EOF'
set -e

COMPOSE_FILE="$1"
DEPLOY_MODE="$2"

cd /opt/telethon-crm

if [ "$DEPLOY_MODE" = "build-only" ]; then
    docker compose -p telethon-crm -f "$COMPOSE_FILE" build
    docker image prune -f
    exit 0
fi

docker compose -p telethon-crm -f "$COMPOSE_FILE" up -d

SERVICES=("twenty-db" "twenty-redis" "twenty-server" "twenty-worker" "crm-frontend")
MAX_WAIT=120
INTERVAL=5

for SERVICE in "${SERVICES[@]}"; do
    ELAPSED=0
    while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
        CONTAINER_ID=$(docker compose -p telethon-crm -f "$COMPOSE_FILE" ps -q "$SERVICE" 2>/dev/null)

        if [ -z "$CONTAINER_ID" ]; then
            echo "Container for $SERVICE not found, waiting..."
            sleep "$INTERVAL"
            ELAPSED=$((ELAPSED + INTERVAL))
            continue
        fi

        STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_ID" 2>/dev/null)
        if [ "$STATUS" != "running" ]; then
            echo "$SERVICE container is $STATUS, waiting..."
            sleep "$INTERVAL"
            ELAPSED=$((ELAPSED + INTERVAL))
            continue
        fi

        HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$CONTAINER_ID" 2>/dev/null)

        if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "no-healthcheck" ]; then
            echo "$SERVICE is ready!"
            break
        fi

        echo "Waiting for $SERVICE to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
        sleep "$INTERVAL"
        ELAPSED=$((ELAPSED + INTERVAL))
    done

    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "Error: Timeout waiting for $SERVICE to become ready."
        exit 1
    fi
done

docker compose -p telethon-crm -f "$COMPOSE_FILE" ps

docker image prune -f
EOF

echo "CRM deployment completed successfully!"