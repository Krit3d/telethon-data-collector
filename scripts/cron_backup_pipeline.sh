#!/bin/bash

set -euo pipefail

# Orchestrator script to run database backup and verification pipeline
# Redirects all output to a rolling log file with timestamp prefixes per run

# Determine the absolute path to the project root directory
# Script is located in <project_root>/scripts/, so navigate one level up from script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths for log file and backup directory
BACKUP_DIR="$PROJECT_ROOT/backups"
LOG_FILE="$BACKUP_DIR/backup_pipeline.log"

# Ensure the backups directory exists (required for log file storage)
mkdir -p "$BACKUP_DIR"

# Add timestamp header for this pipeline run
TIMESTAMP=$(date "+%Y-%m-%dT%H:%M:%S%z")
echo "=== Backup Pipeline Run: $TIMESTAMP ===" >> "$LOG_FILE"

# Redirect all subsequent stdout and stderr to the log file (append mode)
exec >> "$LOG_FILE" 2>&1

# Step 1: Execute database backup script
echo "Starting database backup..."
if ! "$PROJECT_ROOT/scripts/backup_db.sh"; then
    echo "ERROR: Database backup failed. Exiting with code 1."
    exit 1
fi
echo "Database backup completed successfully."

# Step 2: Locate the most recently created backup file in the backups directory
echo "Locating most recent backup file in $BACKUP_DIR..."
# List files sorted by modification time (newest first), filter to regular files, take first result
BACKUP_FILE=$(ls -t "$BACKUP_DIR" | while read -r file; do
    if [ -f "$BACKUP_DIR/$file" ]; then
        echo "$file"
        break
    fi
done)

# Validate that a backup file was found
if [ -z "$BACKUP_FILE" ]; then
    echo "ERROR: No backup files found in $BACKUP_DIR after successful backup. Exiting with code 1."
    exit 1
fi

BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"
echo "Found most recent backup file: $BACKUP_PATH"

# Step 3: Execute backup verification script against the new backup
echo "Starting backup verification for $BACKUP_PATH..."
if ! "$PROJECT_ROOT/scripts/verify_backup.sh" "$BACKUP_PATH"; then
    echo "ERROR: Backup verification failed. Exiting with code 1."
    exit 1
fi
echo "Backup verification completed successfully."

# All steps completed successfully
echo "Backup pipeline completed successfully. Exiting with code 0."
exit 0
