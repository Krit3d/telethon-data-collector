#!/bin/bash
set -e

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST [COMPOSE_FILE] [MODE]"
    echo ""
    echo "  MODE can be:"
    echo "    run        (default) Build, start infrastructure & API, run health checks"
    echo "    build-only Build images only, then print manual start instructions"
    exit 1
fi

SSH_USER="$1"
SSH_HOST="$2"
COMPOSE_FILE="${3:-docker-compose.api.yml}"
DEPLOY_MODE="${4:-run}"

echo "Deploying API to $SSH_USER@$SSH_HOST (mode: $DEPLOY_MODE)..."

echo "Checking and configuring remote environment..."

ssh "$SSH_USER@$SSH_HOST" bash <<'EOF'
if [ ! -f ~/.bashrc ]; then
    touch ~/.bashrc
    echo "Created ~/.bashrc"
fi

HISTSIZE=10000
HISTCONTROL=ignoreboth:erasedups

if ! grep -q "HISTSIZE=10000" ~/.bashrc 2>/dev/null; then
    printf '\n# History preservation settings (added by deployment script)\nexport HISTSIZE=10000\nexport HISTCONTROL=ignoreboth:erasedups\nexport PROMPT_COMMAND="history -a; ${PROMPT_COMMAND:-}"\n' >> ~/.bashrc
    echo "History preservation settings added to ~/.bashrc"
else
    echo "History preservation settings already present in ~/.bashrc"
fi

CURRENT_SHELL=$(getent passwd "$USER" | cut -d: -f7)
if [ "$CURRENT_SHELL" != "/bin/bash" ]; then
    echo "Warning: Current shell is $CURRENT_SHELL. Consider changing to /bin/bash for better compatibility."
fi
EOF

echo "Syncing source files..."
rsync -avz --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' \
    --exclude='*.pyc' --exclude='sessions/' --exclude='.env' \
    ./ "$SSH_USER@$SSH_HOST:/opt/telethon-api"

sleep 2

echo "Pruning unused Docker resources..."
ssh "$SSH_USER@$SSH_HOST" "docker image prune -f" || echo "Warning: Docker prune failed, continuing with deployment..."

if [ "$DEPLOY_MODE" = "build-only" ]; then
    echo "Building Docker images (build-only mode)..."
    ssh "$SSH_USER@$SSH_HOST" "cd /opt/telethon-api && docker compose -f $COMPOSE_FILE build"

    echo ""
    echo "=========================================="
    echo "Build completed successfully!"
    echo ""
    echo "Containers were NOT started (build-only mode)."
    echo "Follow the steps below to start and verify each service manually:"
    echo ""
    echo "  Step 1 - Start Qdrant (vector database):"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d qdrant'"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs -f --tail=20 qdrant'"
    echo ""
    echo "  Step 2 - Verify Qdrant is healthy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'docker inspect --format=\"{{.State.Status}}\" \$(docker compose -f $COMPOSE_FILE ps -q qdrant)'"
    echo ""
    echo "  Step 3 - Start Nginx LLM proxy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d nginx_llm'"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs -f --tail=20 nginx_llm'"
    echo ""
    echo "  Step 4 - Verify Nginx LLM is healthy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'docker inspect --format=\"{{.State.Status}}\" \$(docker compose -f $COMPOSE_FILE ps -q nginx_llm)'"
    echo ""
    echo "  Step 5 - Start API service:"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d api'"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs -f --tail=50 api'"
    echo ""
    echo "  Step 6 - Verify API is healthy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'docker inspect --format=\"{{.State.Status}}\" \$(docker compose -f $COMPOSE_FILE ps -q api)'"
    echo ""
    echo "  Step 7 - Start Embedding Worker:"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d embedding_worker'"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs -f --tail=50 embedding_worker'"
    echo ""
    echo "  Step 8 - Verify Embedding Worker is healthy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'docker inspect --format=\"{{.State.Status}}\" \$(docker compose -f $COMPOSE_FILE ps -q embedding_worker)'"
    echo ""
    echo "  Step 9 - Start Graph Worker:"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d graph_worker'"
    echo "    ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs -f --tail=50 graph_worker'"
    echo ""
    echo "  Step 10 - Verify Graph Worker is healthy:"
    echo "    ssh $SSH_USER@$SSH_HOST 'docker inspect --format=\"{{.State.Status}}\" \$(docker compose -f $COMPOSE_FILE ps -q graph_worker)'"
    echo ""
    echo "  Useful commands:"
    echo "    Stop a service:     ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE stop <service>'"
    echo "    Stop all services:  ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE down'"
    echo "    List all services:  ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE ps'"
    echo "    All logs:           ssh $SSH_USER@$SSH_HOST 'cd /opt/telethon-api && docker compose -f $COMPOSE_FILE logs --tail=100'"
    echo "=========================================="
else
    echo "Starting core API infrastructure and services (qdrant, nginx_llm, api) without recreating running containers..."
    ssh "$SSH_USER@$SSH_HOST" "cd /opt/telethon-api && docker compose -f $COMPOSE_FILE up -d --no-recreate qdrant nginx_llm api"

    echo "Waiting for services to become ready..."

    ssh "$SSH_USER@$SSH_HOST" bash -c "'
    cd /opt/telethon-api

    SERVICES=\"api\"
    MAX_WAIT=120
    ELAPSED=0

    for SERVICE_NAME in \$SERVICES; do
        while [ \$ELAPSED -lt \$MAX_WAIT ]; do
            CONTAINER_ID=\$(docker compose -f $COMPOSE_FILE ps -q \$SERVICE_NAME 2>/dev/null)

            if [ -z \"\$CONTAINER_ID\" ]; then
                echo \"Container for \$SERVICE_NAME not found, waiting...\"
                sleep 5
                ELAPSED=\$((ELAPSED + 5))
                continue
            fi

            STATUS=\$(docker inspect --format=\"{{.State.Status}}\" \$CONTAINER_ID 2>/dev/null)
            if [ \"\$STATUS\" != \"running\" ]; then
                echo \"\$SERVICE_NAME container is \$STATUS, waiting...\"
                sleep 5
                ELAPSED=\$((ELAPSED + 5))
                continue
            fi

            HEALTH=\$(docker inspect --format=\"{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}\" \$CONTAINER_ID 2>/dev/null)

            if [ \"\$HEALTH\" = \"healthy\" ] || [ \"\$HEALTH\" = \"no-healthcheck\" ]; then
                echo \"\$SERVICE_NAME is ready!\"
                break
            fi

            echo \"Waiting for \$SERVICE_NAME to become healthy... (\$ELAPSED/\$MAX_WAIT seconds)\"
            sleep 5
            ELAPSED=\$((ELAPSED + 5))
        done

        if [ \$ELAPSED -ge \$MAX_WAIT ]; then
            echo \"Warning: Timeout waiting for \$SERVICE_NAME to become ready.\"
        fi

        echo \"\"
        echo \"=== Last 20 lines of \$SERVICE_NAME logs ===\"
        docker compose -f $COMPOSE_FILE logs --tail=20 \$SERVICE_NAME
    done
    '"

    echo ""
    echo "=========================================="
    echo "Infrastructure and all services updated successfully!"
    echo "=========================================="
fi

echo "API deployment completed successfully!"
