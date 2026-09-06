#!/bin/bash

# =============================================================================
# verify_backup.sh - PostgreSQL Backup Verification Script
# =============================================================================
# This script verifies the integrity of a PostgreSQL backup file by restoring
# it in a temporary Docker container and running smoke tests.
#
# Usage: ./scripts/verify_backup.sh /path/to/backup.sql
# Exit codes: 0 = success, 1 = failure
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Lightweight PostgreSQL image for the ephemeral verification container
PG_IMAGE="${VERIFY_PG_IMAGE:-$(docker inspect --format '{{.Config.Image}}' tg_parser_db 2>/dev/null || true)}"
PG_IMAGE="${PG_IMAGE:-postgres:17-alpine}"

# Temporary container and database settings
TEMP_CONTAINER_NAME="backup_verify_$(date +%s)_$$"
TEMP_DB_NAME="${VERIFY_DB_NAME:-postgres}"
TEMP_DB_USER="${VERIFY_DB_USER:-postgres}"
TEMP_DB_PASSWORD="${VERIFY_DB_PASSWORD:-postgres}"

# Timeout settings
DB_READY_TIMEOUT="${VERIFY_DB_READY_TIMEOUT:-30}"

# =============================================================================
# Global Variables
# =============================================================================

BACKUP_FILE=""
STDERR_FILE=""

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# =============================================================================
# Cleanup Function (called via trap)
# =============================================================================

cleanup() {
    local exit_code=$?

    log_info "Starting cleanup process..."

    # Stop and remove the temporary container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${TEMP_CONTAINER_NAME}$"; then
        log_info "Stopping and removing temporary container: ${TEMP_CONTAINER_NAME}"
        docker stop "${TEMP_CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${TEMP_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    # Remove the temporary stderr capture file if it exists
    if [ -n "${STDERR_FILE}" ] && [ -f "${STDERR_FILE}" ]; then
        rm -f "${STDERR_FILE}"
    fi

    if [ $exit_code -eq 0 ]; then
        log_success "Cleanup completed successfully"
    else
        log_warn "Cleanup completed (script exited with code ${exit_code})"
    fi

    exit $exit_code
}

# Set trap to ensure cleanup runs on script exit (normal or error)
trap cleanup EXIT INT TERM

# =============================================================================
# File Integrity Checks
# =============================================================================

check_file_exists() {
    local file_path="$1"

    if [ ! -f "$file_path" ]; then
        log_error "Backup file does not exist: ${file_path}"
        return 1
    fi

    log_info "Backup file exists: ${file_path}"
    return 0
}

check_file_not_empty() {
    local file_path="$1"

    if [ ! -s "$file_path" ]; then
        log_error "Backup file is empty: ${file_path}"
        return 1
    fi

    local file_size
    file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo "unknown")
    log_info "Backup file size: ${file_size} bytes"
    return 0
}

detect_backup_format() {
    local file_path="$1"
    local file_ext="${file_path##*.}"
    local format="sql"

    # Check file extension first
    case "$file_ext" in
        sql|SQL)
            # Check if it's a plain text SQL file or compressed
            if file "$file_path" | grep -qi "gzip"; then
                format="sql_gz"
            fi
            ;;
        gz|GZ)
            format="sql_gz"
            ;;
        dump|DUMP)
            format="custom"
            ;;
        *)
            # Try to detect format by reading file header
            if [ -f "$file_path" ]; then
                # Check for PGDMP magic bytes (custom format)
                if head -c 5 "$file_path" 2>/dev/null | grep -q "PGDMP"; then
                    format="custom"
                # Check for gzip magic bytes (0x1f 0x8b)
                elif [ "$(od -An -tx1 -N2 "$file_path" 2>/dev/null | tr -d ' ')" = "1f8b" ]; then
                    format="sql_gz"
                fi
            fi
            ;;
    esac

    echo "$format"
}

verify_file_signature() {
    local file_path="$1"
    local format="$2"

    if [ "$format" = "custom" ]; then
        # Verify PGDMP signature for custom format
        if ! head -c 5 "$file_path" 2>/dev/null | grep -q "PGDMP"; then
            log_error "Invalid custom format backup: missing PGDMP signature"
            return 1
        fi

        log_info "Custom format signature verified (PGDMP)"
    elif [ "$format" = "sql_gz" ]; then
        # Verify gzip integrity
        if ! gzip -t "$file_path" 2>/dev/null; then
            log_error "Invalid gzip file: integrity check failed"
            return 1
        fi

        log_info "Gzip integrity verified"
    elif [ "$format" = "sql" ]; then
        # Check if it looks like a SQL file (contains SQL keywords)
        if ! head -20 "$file_path" | grep -qi "postgresql\|pg_dump\|create\|insert"; then
            log_warn "File may not be a valid SQL dump (no SQL keywords found in header)"
        fi

        log_info "SQL format detected"
    fi

    return 0
}

# =============================================================================
# Docker Environment Setup
# =============================================================================

start_temp_database() {
    log_info "Starting temporary PostgreSQL container..."
    log_info "Container name: ${TEMP_CONTAINER_NAME}"

    # Start the temporary container (ephemeral, no volume mount)
    # Optimized for fast import: fsync, synchronous_commit and full_page_writes disabled
    if ! docker run -d \
        --name "${TEMP_CONTAINER_NAME}" \
        -e POSTGRES_USER="${TEMP_DB_USER}" \
        -e POSTGRES_PASSWORD="${TEMP_DB_PASSWORD}" \
        -e POSTGRES_DB="${TEMP_DB_NAME}" \
        -e POSTGRES_INITDB_ARGS="--encoding=UTF-8" \
        "${PG_IMAGE}" \
        postgres -c listen_addresses='*' \
                -c fsync=off \
                -c synchronous_commit=off \
                -c full_page_writes=off \
                -c shared_buffers=128MB \
                -c work_mem=8MB \
                -c maintenance_work_mem=64MB \
                -c max_wal_size=1GB \
        >/dev/null 2>&1; then
        log_error "Failed to start temporary Docker container"
        return 1
    fi

    log_info "Temporary container started successfully"
    return 0
}

wait_for_database() {
    local timeout="${DB_READY_TIMEOUT}"
    local elapsed=0
    local interval=1

    log_info "Waiting for database to be ready (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        if docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -c "SELECT 1;" >/dev/null 2>&1; then
            log_info "Database is ready and accepting connections"
            return 0
        fi

        sleep $interval
        elapsed=$((elapsed + interval))

        if [ $((elapsed % 5)) -eq 0 ]; then
            log_info "Still waiting for database... (${elapsed}s elapsed)"
        fi
    done

    log_error "Database did not become ready within ${timeout} seconds"
    return 1
}

# =============================================================================
# Restore Operations
# =============================================================================

restore_backup() {
    local file_path="$1"
    local format="$2"
    local restore_exit_code=0

    # Create a temporary file to capture stderr for detailed error logging
    STDERR_FILE=$(mktemp /tmp/restore_stderr.XXXXXX) || {
        log_error "Failed to create temporary stderr file"
        return 1
    }

    log_info "Starting backup restoration (format: ${format})..."

    case "$format" in
        sql)
            # Restore from plain SQL file
            log_info "Restoring from plain SQL file..."
            cat "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" psql \
                -h 127.0.0.1 \
                -v ON_ERROR_STOP=1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                >/dev/null 2>"${STDERR_FILE}" || restore_exit_code=$?
            ;;
        sql_gz)
            # Restore from gzip-compressed SQL file
            log_info "Restoring from gzip-compressed SQL file..."
            gunzip -c "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" psql \
                -h 127.0.0.1 \
                -v ON_ERROR_STOP=1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                >/dev/null 2>"${STDERR_FILE}" || restore_exit_code=$?
            ;;
        custom)
            # Restore from custom format using pg_restore
            log_info "Restoring from custom format (pg_restore)..."

            # First, create a temporary file in the container
            if ! cat "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" tee /tmp/restore.dump >/dev/null; then
                log_error "Failed to copy dump file into container"
                return 1
            fi

            # Run pg_restore with --exit-on-error to fail on any error
            docker exec "${TEMP_CONTAINER_NAME}" pg_restore \
                -h 127.0.0.1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                --no-owner \
                --no-acl \
                --if-exists \
                --clean \
                --exit-on-error \
                /tmp/restore.dump \
                >/dev/null 2>"${STDERR_FILE}" || restore_exit_code=$?

            # Clean up temp file
            docker exec "${TEMP_CONTAINER_NAME}" rm -f /tmp/restore.dump >/dev/null 2>&1 || true
            ;;
        *)
            log_error "Unknown backup format: ${format}"
            return 1
            ;;
    esac

    # Check restore exit code
    if [ $restore_exit_code -eq 0 ]; then
        log_success "Backup restored successfully"
        return 0
    fi

    # Display the captured stderr for detailed error logging
    log_error "Restore utility exited with code ${restore_exit_code}"
    if [ -s "${STDERR_FILE}" ]; then
        log_error "Restore errors (last 50 lines):"
        echo "======================================================================" >&2
        tail -n 50 "${STDERR_FILE}" >&2
        echo "======================================================================" >&2
    fi

    return 1
}

# =============================================================================
# Data Validation (Smoke Tests)
# =============================================================================

run_smoke_tests() {
    log_info "Running smoke tests on restored database..."

    # Test 1: Check if we can connect and query
    log_info "Test 1: Database availability check"
    if ! docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SELECT version();" >/dev/null 2>&1; then
        log_error "Database availability check failed"
        return 1
    fi
    log_success "Database is available (SELECT version() passed)"

    # Test 2: Check the migrations version table exists
    log_info "Test 2: Checking alembic_version table"
    if ! docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SELECT version_num FROM alembic_version;" >/dev/null 2>&1; then
        log_error "alembic_version table not found"
        return 1
    fi
    log_success "alembic_version table exists"

    # Test 3: Count records in 'accounts' table (must be non-zero)
    log_info "Test 3: Checking accounts table records"
    local accounts_count
    accounts_count=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM accounts;" 2>/dev/null | tr -d '[:space:]' || echo "0")

    if [ -z "$accounts_count" ] || [ "$accounts_count" = "0" ]; then
        log_error "Accounts table is empty or does not exist (count: ${accounts_count:-0})"
        return 1
    fi
    log_success "Accounts table has ${accounts_count} records"

    # Test 4: Count records in 'content' table (must be non-zero)
    log_info "Test 4: Checking content table records"
    local content_count
    content_count=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM content;" 2>/dev/null | tr -d '[:space:]' || echo "0")

    if [ -z "$content_count" ] || [ "$content_count" = "0" ]; then
        log_error "Content table is empty or does not exist (count: ${content_count:-0})"
        return 1
    fi
    log_success "Content table has ${content_count} records"

    # Test 5: Check integrity of key columns in 'accounts' table
    log_info "Test 5: Checking key columns of accounts table"
    if ! docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SELECT id, status, is_author_blog, static_avg_er, category_id FROM accounts LIMIT 5;"; then
        log_error "Key columns check failed for accounts table"
        return 1
    fi
    log_success "Key columns of accounts table are intact"

    # Test 6: Check integrity of key columns in 'content' table
    log_info "Test 6: Checking key columns of content table"
    if ! docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SELECT id, account_id, reactions_count, views, is_enriched FROM content LIMIT 5;"; then
        log_error "Key columns check failed for content table"
        return 1
    fi
    log_success "Key columns of content table are intact"

    log_success "All smoke tests passed"
    return 0
}

# =============================================================================
# Main Script Logic
# =============================================================================

print_usage() {
    echo "Usage: $0 <backup_file_path>"
    echo ""
    echo "Arguments:"
    echo "  backup_file_path    Path to the backup file (.sql, .sql.gz, or .dump)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/backup.sql"
    echo "  $0 /path/to/backup.sql.gz"
    echo "  $0 /path/to/backup.dump"
    echo ""
    echo "Exit codes:"
    echo "  0    Backup verification successful"
    echo "  1    Verification failed"
}

main() {
    log_info "=========================================="
    log_info "PostgreSQL Backup Verification"
    log_info "=========================================="

    # Check if backup file argument is provided
    if [ $# -lt 1 ]; then
        log_error "No backup file path provided"
        print_usage
        exit 1
    fi

    BACKUP_FILE="$1"

    # Convert to absolute path if relative
    if [[ ! "$BACKUP_FILE" = /* ]]; then
        if ! BACKUP_FILE="$(cd "$(dirname "$BACKUP_FILE")" 2>/dev/null && pwd)/$(basename "$BACKUP_FILE")"; then
            log_error "Failed to resolve absolute path for backup file: ${BACKUP_FILE}"
            exit 1
        fi
    fi

    log_info "Backup file: ${BACKUP_FILE}"

    # =========================================================================
    # Step 1: File Integrity Checks
    # =========================================================================
    log_info "--- Step 1: File Integrity Checks ---"

    if ! check_file_exists "$BACKUP_FILE"; then
        exit 1
    fi

    if ! check_file_not_empty "$BACKUP_FILE"; then
        exit 1
    fi

    local detected_format
    detected_format=$(detect_backup_format "$BACKUP_FILE")
    log_info "Detected backup format: ${detected_format}"

    if ! verify_file_signature "$BACKUP_FILE" "$detected_format"; then
        exit 1
    fi

    log_success "File integrity checks passed"

    # =========================================================================
    # Step 2: Start Temporary Docker Environment
    # =========================================================================
    log_info "--- Step 2: Starting Temporary Docker Environment ---"

    if ! start_temp_database; then
        exit 1
    fi

    if ! wait_for_database; then
        exit 1
    fi

    log_success "Temporary Docker environment is ready"

    # =========================================================================
    # Step 3: Restore Operation
    # =========================================================================
    log_info "--- Step 3: Restore Operation ---"

    if ! restore_backup "$BACKUP_FILE" "$detected_format"; then
        log_error "Backup restoration failed"
        exit 1
    fi

    log_success "Backup restoration completed"

    # =========================================================================
    # Step 4: Data Validation (Smoke Tests)
    # =========================================================================
    log_info "--- Step 4: Data Validation (Smoke Tests) ---"

    if ! run_smoke_tests; then
        log_error "Smoke tests failed"
        exit 1
    fi

    log_success "All smoke tests passed"

    # =========================================================================
    # Summary
    # =========================================================================
    log_info "=========================================="
    log_success "Backup verification completed successfully!"
    log_info "Backup file: ${BACKUP_FILE}"
    log_info "Format: ${detected_format}"
    log_info "Status: VALID"
    log_info "=========================================="

    return 0
}

# =============================================================================
# Script Entry Point
# =============================================================================

main "$@"
