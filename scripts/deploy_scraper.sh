#!/bin/bash
set -e

# Check if required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"

echo "Deploying SCRAPER to $SSH_USER@$SSH_HOST..."

# Step 1: Remote environment check and setup (idempotent)
echo "Checking and configuring remote environment..."
ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
# Ensure ~/.bashrc exists
if [ ! -f ~/.bashrc ]; then
    touch ~/.bashrc
    echo "Created ~/.bashrc"
fi

# History-preserving settings to append if not present
HISTORY_SETTINGS='
# History preservation settings (added by deployment script)
export HISTSIZE=10000
export HISTCONTROL=ignoreboth:erasedups
export PROMPT_COMMAND="history -a; ${PROMPT_COMMAND:-}"
'

# Check if settings already exist in ~/.bashrc (idempotent check)
if ! grep -q "HISTSIZE=10000" ~/.bashrc 2>/dev/null; then
    echo "$HISTORY_SETTINGS" >> ~/.bashrc
    echo "History preservation settings added to ~/.bashrc"
else
    echo "History preservation settings already present in ~/.bashrc"
fi

# Check if remote shell is /bin/bash
CURRENT_SHELL=$(getent passwd $USER | cut -d: -f7)
if [ "$CURRENT_SHELL" != "/bin/bash" ]; then
    echo "Warning: Current shell is $CURRENT_SHELL. Consider changing to /bin/bash for better compatibility."
fi
EOF

# Step 2: Sync source files to remote server
echo "Syncing source files..."
rsync -avz --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' \
    --exclude='*.pyc' --exclude='sessions/' --exclude='.env' \
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-scraper"

# Brief delay to ensure file sync completion
sleep 2

# Step 3: Execute deployment steps on remote server
echo "Starting remote deployment steps..."
ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
set -e  # Exit remote script on error

cd /opt/telethon-scraper
COMPOSE_FILE="docker-compose.scraper.yml"

# Prune unused Docker resources to prevent disk space issues
echo "Pruning unused Docker resources..."
docker image prune -f || echo "Warning: Docker prune failed, continuing..."

# Step 3a: Build all images explicitly before starting any containers
echo "Building all Docker images..."
docker compose -f $COMPOSE_FILE build

# Step 3b: Start infrastructure containers (db, llm_proxy) without rebuilding
echo "Starting infrastructure containers (db, llm_proxy)..."
docker compose -f $COMPOSE_FILE up -d db llm_proxy

# Wait for db service to become healthy (max 60 seconds)
echo "Waiting for db service to become healthy (max 60 seconds)..."
SERVICE="db"
MAX_WAIT=60
INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Get container ID for the service
    CONTAINER_ID=$(docker compose -f $COMPOSE_FILE ps -q $SERVICE 2>/dev/null)
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "db container not found, waiting..."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi
    
    # Check if container is running
    STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        echo "db container is $STATUS, waiting..."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi
    
    # Check health status using docker inspect
    HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' $CONTAINER_ID 2>/dev/null)
    
    if [ "$HEALTH" = "healthy" ]; then
        echo "db service is healthy!"
        break
    fi
    
    echo "Waiting for db to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

# Check if timeout occurred
if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "Error: db service did not become healthy within $MAX_WAIT seconds."
    exit 1
fi

# Step 3c: Run database migration using Alembic
echo "Running database migration with Alembic..."
docker compose -f $COMPOSE_FILE run --rm parser alembic upgrade head

# Step 3d: Start crawler service with --no-deps to avoid triggering DB restarts
echo "Starting crawler service..."
docker compose -f $COMPOSE_FILE up -d --no-deps crawler

# Cool-down delay for crawler initialization
echo "Waiting 10 seconds for crawler to initialize..."
sleep 10

# Step 3e: Start parser service with --no-deps to avoid triggering DB restarts
echo "Starting parser service..."
docker compose -f $COMPOSE_FILE up -d --no-deps parser

# Final check that all services are running
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

# Print clean summary table of service statuses
echo ""
echo "=== Service Status Summary ==="
docker compose -f $COMPOSE_FILE ps

# Print recent logs for debugging
echo ""
echo "=== Recent Logs (last 20 lines per service) ==="
docker compose -f $COMPOSE_FILE logs --tail=20 db crawler parser

EOF

echo "SCRAPER deployment completed successfully!"
