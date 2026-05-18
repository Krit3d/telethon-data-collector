#!/bin/bash
set -e

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"

echo "Deploying API to $SSH_USER@$SSH_HOST..."

# Step 1: Remote environment check and setup
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

# Step 2: Sync files
echo "Syncing source files..."
rsync -avz --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' \
    --exclude='*.pyc' --exclude='avatars/' --exclude='sessions/' \
    --exclude='.env' \
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-api"

sleep 2

# Step 3: Prune unused Docker resources to prevent disk space issues
echo "Pruning unused Docker resources..."
ssh "$SSH_USER@$SSH_HOST" "docker image prune -f" || echo "Warning: Docker prune failed, continuing with deployment..."

# Step 4: Build and start infrastructure and API containers (qdrant, api)
# Note: extractor worker is NOT started automatically for safe deployment
echo "Building and starting API infrastructure and service (qdrant, api)..."
ssh "$SSH_USER@$SSH_HOST" "cd /opt/telethon-api && docker compose -f docker-compose.api.yml up -d --build qdrant api"

# Step 5: Wait for API service health check and display logs
echo "Waiting for API service to become ready..."

ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
cd /opt/telethon-api

COMPOSE_FILE="docker-compose.api.yml"
SERVICE_NAME="api"

# Wait for container to be running and healthy
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    CONTAINER_ID=$(docker compose -f $COMPOSE_FILE ps -q $SERVICE_NAME 2>/dev/null)
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "Container for $SERVICE_NAME not found, waiting..."
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        continue
    fi
    
    # Check container running status
    STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        echo "Container is $STATUS, waiting..."
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        continue
    fi
    
    # Check health status if health check is configured
    HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' $CONTAINER_ID 2>/dev/null)
    
    if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "no-healthcheck" ]; then
        echo "$SERVICE_NAME is ready!"
        break
    fi
    
    echo "Waiting for $SERVICE_NAME to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "Warning: Timeout waiting for $SERVICE_NAME to become ready."
fi

# Display last 20 lines of logs
echo ""
echo "=== Last 20 lines of $SERVICE_NAME logs ==="
docker compose -f $COMPOSE_FILE logs --tail=20 $SERVICE_NAME
EOF

# Safe Start helper message for extractor worker
echo ""
echo "=========================================="
echo "Infrastructure and API service updated successfully!"
echo ""
echo "To start the extractor worker manually, use the following command:"
echo ""
echo "  Start extractor worker:"
echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f docker-compose.api.yml up -d extractor && docker compose -f docker-compose.api.yml logs -f --tail=50 extractor'"
echo ""
echo "  Stop extractor worker:"
echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f docker-compose.api.yml stop extractor'"
echo "=========================================="

echo "API deployment completed successfully!"
