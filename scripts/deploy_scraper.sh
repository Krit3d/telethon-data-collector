#!/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"

echo "Deploying SCRAPER to $SSH_USER@$SSH_HOST..."

echo "Checking and configuring remote environment..."
ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
if [ ! -f ~/.bashrc ]; then
    touch ~/.bashrc
    echo "Created ~/.bashrc"
fi

HISTORY_SETTINGS='
export HISTSIZE=10000
export HISTCONTROL=ignoreboth:erasedups
export PROMPT_COMMAND="history -a; ${PROMPT_COMMAND:-}"
'

if ! grep -q "HISTSIZE=10000" ~/.bashrc 2>/dev/null; then
    echo "$HISTORY_SETTINGS" >> ~/.bashrc
    echo "History preservation settings added to ~/.bashrc"
else
    echo "History preservation settings already present in ~/.bashrc"
fi

CURRENT_SHELL=$(getent passwd $USER | cut -d: -f7)
if [ "$CURRENT_SHELL" != "/bin/bash" ]; then
    echo "Warning: Current shell is $CURRENT_SHELL. Consider changing to /bin/bash for better compatibility."
fi
EOF

echo "Syncing source files..."
rsync -avz --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' --exclude='backups/' \
    --exclude='*.pyc' --exclude='sessions/' --exclude='.env' \
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-scraper"

sleep 2

echo "Starting remote deployment steps..."
ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
set -e

cd /opt/telethon-scraper
COMPOSE_FILE="docker-compose.scraper.yml"

echo "Building crawler Docker image..."
docker compose -f $COMPOSE_FILE build crawler

echo "Starting database infrastructure container (db)..."
docker compose -f $COMPOSE_FILE up -d db

echo "Waiting for db service to become healthy (max 60 seconds)..."
SERVICE="db"
MAX_WAIT=60
INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    CONTAINER_ID=$(docker compose -f $COMPOSE_FILE ps -q $SERVICE 2>/dev/null)

    if [ -z "$CONTAINER_ID" ]; then
        echo "db container not found, waiting..."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi

    STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        echo "db container is $STATUS, waiting..."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi

    HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' $CONTAINER_ID 2>/dev/null)

    if [ "$HEALTH" = "healthy" ]; then
        echo "db service is healthy!"
        break
    fi

    echo "Waiting for db to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "Error: db service did not become healthy within $MAX_WAIT seconds."
    exit 1
fi

echo "Running database migration with Alembic..."
docker compose -f $COMPOSE_FILE run --rm parser alembic upgrade head

echo "Starting crawler service..."
docker compose -f $COMPOSE_FILE up -d --no-deps crawler

echo "Waiting 10 seconds for crawler to initialize..."
sleep 10

echo "Starting parser service..."
docker compose -f $COMPOSE_FILE up -d --no-deps parser

echo "Performing final service status check..."
SERVICES=("db" "crawler" "parser")
for SERVICE in "${SERVICES[@]}"; do
    CONTAINER_ID=$(docker compose -f $COMPOSE_FILE ps -q $SERVICE 2>/dev/null)
    if [ -z "$CONTAINER_ID" ]; then
        echo "Error: Service $SERVICE is not running (no container found)."
        exit 1
    fi

    STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        echo "Error: Service $SERVICE is not running (status: $STATUS)."
        exit 1
    fi
    echo "Service $SERVICE is running."
done

echo ""
echo "=== Service Status Summary ==="
docker compose -f $COMPOSE_FILE ps

echo ""
echo "=== Recent Logs (last 20 lines per service) ==="
docker compose -f $COMPOSE_FILE logs --tail=20 db crawler parser

echo "Performing post-deployment cleanup of unused Docker images and build cache..."
docker image prune -f || true
docker builder prune -f || true

EOF

echo "SCRAPER deployment completed successfully!"
