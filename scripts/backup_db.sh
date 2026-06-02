#!/bin/bash

# Exit on error, undefined variable, and pipe failure for production robustness
set -euo pipefail

# Get the directory where this script is located to resolve relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is parent directory of scripts/
PROJECT_ROOT="$SCRIPT_DIR/.."
# Path to .env configuration file
ENV_FILE="$PROJECT_ROOT/.env"
# Directory to store backup files
BACKUPS_DIR="$PROJECT_ROOT/backups"

# Load environment variables from .env file
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Export all variables from .env to make them available to subprocesses
set -a
source "$ENV_FILE"
set +a

# Validate required PostgreSQL environment variables are set
REQUIRED_VARS=("POSTGRES_USER" "POSTGRES_DB" "POSTGRES_PASSWORD")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable $var is not set in .env"
        exit 1
    fi
done

# Create backups directory if it does not exist
mkdir -p "$BACKUPS_DIR"

# Generate timestamp in yyyy-mm-dd_HHMMSS format for unique backup filename
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
BACKUP_FILENAME="backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="$BACKUPS_DIR/$BACKUP_FILENAME"

echo "Starting PostgreSQL backup at $(date)"
echo "Target backup file: $BACKUP_PATH"

# Execute pg_dump inside the 'tg_parser_db' Docker container, pipe output to gzip for compression
# PGPASSWORD is passed as environment variable to the container for authentication
if docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" tg_parser_db pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_PATH"; then
    echo "Backup completed successfully: $BACKUP_PATH"
else
    echo "ERROR: Backup failed during pg_dump or compression"
    # Remove partial/invalid backup file if creation failed
    rm -f "$BACKUP_PATH"
    exit 1
fi

# Remove backups older than 7 days to retain only recent daily backups
echo "Cleaning up backups older than 7 days..."
find "$BACKUPS_DIR" -name "backup_*.sql.gz" -type f -mtime +7 -delete

echo "Backup process finished successfully at $(date)"
