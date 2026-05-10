#!/bin/bash
set -e
if [ $# -ne 2 ]; then
    echo "Usage: $0 SSH_USER SSH_HOST"
    exit 1
fi
SSH_USER="$1"
SSH_HOST="$2"
REMOTE_DIR="/opt/telethon-api"

echo "Deploying API to $SSH_USER@$SSH_HOST..."
rsync -avz --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' \
    --exclude='*.pyc' --exclude='avatars/' --exclude='sessions/' \
	--exclude='.env' \
    ./ "$SSH_USER@$SSH_HOST:$REMOTE_DIR"

ssh "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR && docker compose -f docker-compose.api.yml up -d --build"
echo "API deployed!"
