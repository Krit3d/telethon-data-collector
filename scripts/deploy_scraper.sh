#!/bin/bash
set -e

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"

echo "Deploying SCRAPER to $SSH_USER@$SSH_HOST..."

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
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-scraper"

sleep 2

# Step 3: Prune unused Docker resources to prevent disk space issues
echo "Pruning unused Docker resources..."
ssh "$SSH_USER@$SSH_HOST" "docker image prune -f" || echo "Warning: Docker prune failed, continuing with deployment..."

# Step 4: Build and start infrastructure containers only (db, llm_proxy)
# Note: parser and crawler workers are NOT started automatically for safe deployment
echo "Building and starting SCRAPER infrastructure (db, llm_proxy)..."
ssh "$SSH_USER@$SSH_HOST" "cd /opt/telethon-scraper && docker compose -f docker-compose.scraper.yml up -d --build db llm_proxy"

# Note: The following block for waiting on parser health check has been commented out
# because parser is no longer auto-started. Use the manual commands below to start workers.
# ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
# cd /opt/telethon-scraper
#
# COMPOSE_FILE="docker-compose.scraper.yml"
# SERVICE_NAME="parser"
#
# # Wait for container to be running and healthy
# MAX_WAIT=60
# ELAPSED=0
# while [ $ELAPSED -lt $MAX_WAIT ]; do
#     CONTAINER_ID=$(docker compose -f $COMPOSE_FILE ps -q $SERVICE_NAME 2>/dev/null)
#
#     if [ -z "$CONTAINER_ID" ]; then
#         echo "Container for $SERVICE_NAME not found, waiting..."
#         sleep 5
#         ELAPSED=$((ELAPSED + 5))
#         continue
#     fi
#
#     # Check container running status
#     STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null)
#     if [ "$STATUS" != "running" ]; then
#         echo "Container is $STATUS, waiting..."
#         sleep 5
#         ELAPSED=$((ELAPSED + 5))
#         continue
#     fi
#
#     # Check health status if health check is configured
#     HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' $CONTAINER_ID 2>/dev/null)
#
#     if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "no-healthcheck" ]; then
#         echo "$SERVICE_NAME is ready!"
#         break
#     fi
#
#     echo "Waiting for $SERVICE_NAME to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
#     sleep 5
#     ELAPSED=$((ELAPSED + 5))
# done
#
# if [ $ELAPSED -ge $MAX_WAIT ]; then
#     echo "Warning: Timeout waiting for $SERVICE_NAME to become ready."
# fi
#
# # Display last 20 lines of logs
# echo ""
# echo "=== Last 20 lines of $SERVICE_NAME logs ==="
# docker compose -f $COMPOSE_FILE logs --tail=20 $SERVICE_NAME
# EOF

# Safe Start helper message
echo ""
echo "=========================================="
echo "Infrastructure updated successfully!"
echo ""
echo "To start workers manually, use the following commands:"
echo ""
echo "  Start parser worker:"
echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-scraper && docker compose -f docker-compose.scraper.yml up -d parser && docker compose -f docker-compose.scraper.yml logs -f --tail=50 parser'"
echo ""
echo "  Start crawler worker:"
echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-scraper && docker compose -f docker-compose.scraper.yml up -d crawler && docker compose -f docker-compose.scraper.yml logs -f --tail=50 crawler'"
echo ""
echo "  Start both workers:"
echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-scraper && docker compose -f docker-compose.scraper.yml up -d parser crawler && docker compose -f docker-compose.scraper.yml logs -f --tail=50'"
echo "=========================================="

echo "SCRAPER deployment completed successfully!"
