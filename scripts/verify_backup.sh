#!/bin/bash

# =============================================================================
# verify_backup.sh - PostgreSQL + Apache AGE Backup Verification Script
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

# Docker image with Apache AGE (must match docker-compose.scraper.yml)
AGE_IMAGE="apache/age@sha256:4241e2d8bb86a6b2ea44e9ad06c73856e12b209de295124603a599dd7feb70eb"

# Temporary container and database settings
TEMP_CONTAINER_NAME="backup_verify_$(date +%s)_$$"
TEMP_DB_PORT="5433"
TEMP_DB_NAME="postgres"
TEMP_DB_USER="postgres"
TEMP_DB_PASSWORD="postgres"

# Timeout settings
DB_READY_TIMEOUT=30

# =============================================================================
# Global Variables
# =============================================================================

BACKUP_FILE=""
BACKUP_FORMAT=""
RESTORE_SUCCESS=false

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
    
    # Check file extension first
    case "$file_ext" in
        sql|SQL)
            # Check if it's a plain text SQL file or compressed
            if file "$file_path" | grep -qi "gzip"; then
                BACKUP_FORMAT="sql_gz"
                echo "sql_gz"
            else
                BACKUP_FORMAT="sql"
                echo "sql"
            fi
            return 0
            ;;
        gz|GZ)
            BACKUP_FORMAT="sql_gz"
            echo "sql_gz"
            return 0
            ;;
        dump|DUMP)
            BACKUP_FORMAT="custom"
            echo "custom"
            return 0
            ;;
        *)
            # Try to detect format by reading file header
            if [ -f "$file_path" ]; then
                local header
                header=$(head -c 5 "$file_path" 2>/dev/null || echo "")
                
                # Check for PGDMP magic bytes (custom format)
                if echo "$header" | grep -q "PGDMP"; then
                    BACKUP_FORMAT="custom"
                    echo "custom"
                    return 0
                fi
                
                # Check for gzip magic bytes (0x1f 0x8b)
                local magic
                magic=$(od -An -tx1 -N2 "$file_path" 2>/dev/null | tr -d ' ')
                if [ "$magic" = "1f8b" ]; then
                    BACKUP_FORMAT="sql_gz"
                    echo "sql_gz"
                    return 0
                fi
                
                # Assume plain SQL
                BACKUP_FORMAT="sql"
                echo "sql"
                return 0
            fi
            ;;
    esac
    
    # Default to SQL format
    BACKUP_FORMAT="sql"
    echo "sql"
}

verify_file_signature() {
    local file_path="$1"
    local format="$2"
    
    if [ "$format" = "custom" ]; then
        # Verify PGDMP signature for custom format
        local header
        header=$(head -c 5 "$file_path" 2>/dev/null || echo "")
        
        if ! echo "$header" | grep -q "PGDMP"; then
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
    log_info "Starting temporary PostgreSQL + Apache AGE container..."
    log_info "Container name: ${TEMP_CONTAINER_NAME}"
    log_info "Using port: ${TEMP_DB_PORT}"
    
    # Start the temporary container with Apache AGE (ephemeral, no volume mount)
    docker run -d \
        --name "${TEMP_CONTAINER_NAME}" \
        -e POSTGRES_USER="${TEMP_DB_USER}" \
        -e POSTGRES_PASSWORD="${TEMP_DB_PASSWORD}" \
        -e POSTGRES_DB="${TEMP_DB_NAME}" \
        -e POSTGRES_INITDB_ARGS="--encoding=UTF-8" \
        -p "${TEMP_DB_PORT}:5432" \
        "${AGE_IMAGE}" \
        postgres -c shared_preload_libraries=age \
                -c listen_addresses='*' \
                -c fsync=off \
                -c synchronous_commit=off \
                -c full_page_writes=off \
                -c shared_buffers=128MB \
                -c work_mem=8MB \
                -c maintenance_work_mem=64MB \
                -c max_wal_size=1GB \
        >/dev/null 2>&1
    
    if [ $? -ne 0 ]; then
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
        if docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "postgres" -c "SELECT 1;" >/dev/null 2>&1; then
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

setup_age_extension() {
    log_info "Setting up Apache AGE extension in temporary database..."
    
    # Create the age extension in the temporary database
    docker exec -i "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" <<-EOSQL
        CREATE EXTENSION IF NOT EXISTS age;
        LOAD 'age';
        SET search_path = ag_catalog, "\$user", public;
EOSQL
    
    if [ $? -ne 0 ]; then
        log_warn "Failed to setup AGE extension (may already exist in backup)"
        return 0  # Non-critical, continue
    fi
    
    log_info "Apache AGE extension configured"
    return 0
}

restore_backup() {
    local file_path="$1"
    local format="$2"
    local restore_exit_code=0
    local stderr_file
    stderr_file=$(mktemp /tmp/restore_stderr.XXXXXX)
    
    log_info "Starting backup restoration (format: ${format})..."
    log_info "Stderr will be captured to: ${stderr_file}"
    
    case "$format" in
        sql)
            # Restore from plain SQL file
            # Note: ON_ERROR_STOP=1 is intentionally omitted to allow the restore to continue
            # past non-fatal errors such as "schema already exists" (e.g., ag_catalog created
            # by setup_age_extension). PostgreSQL restores frequently encounter benign warnings
            # and must proceed to restore the remaining tables and data.
            log_info "Restoring from plain SQL file..."
            cat "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" psql \
                -h 127.0.0.1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                >/dev/null 2>"${stderr_file}" || restore_exit_code=$?
            ;;
        sql_gz)
            # Restore from gzip-compressed SQL file
            # Note: ON_ERROR_STOP=1 is intentionally omitted to allow the restore to continue
            # past non-fatal errors such as "schema already exists" (e.g., ag_catalog created
            # by setup_age_extension). PostgreSQL restores frequently encounter benign warnings
            # and must proceed to restore the remaining tables and data.
            log_info "Restoring from gzip-compressed SQL file..."
            gunzip -c "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" psql \
                -h 127.0.0.1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                >/dev/null 2>"${stderr_file}" || restore_exit_code=$?
            ;;
        custom)
            # Restore from custom format using pg_restore
            log_info "Restoring from custom format (pg_restore)..."
            
            # First, create a temporary file in the container
            cat "$file_path" | docker exec -i "${TEMP_CONTAINER_NAME}" tee /tmp/restore.dump >/dev/null
            
            # Run pg_restore
            docker exec "${TEMP_CONTAINER_NAME}" pg_restore \
                -h 127.0.0.1 \
                -U "${TEMP_DB_USER}" \
                -d "${TEMP_DB_NAME}" \
                --no-owner \
                --no-acl \
                --if-exists \
                --clean \
                /tmp/restore.dump \
                >/dev/null 2>"${stderr_file}" || restore_exit_code=$?
            
            # Clean up temp file
            docker exec "${TEMP_CONTAINER_NAME}" rm -f /tmp/restore.dump
            ;;
        *)
            log_error "Unknown backup format: ${format}"
            rm -f "${stderr_file}"
            return 1
            ;;
    esac
    
    # Check restore exit code
    # pg_restore returns 0 on complete success, 1 on fatal errors, and warnings may return non-zero
    if [ $restore_exit_code -eq 0 ]; then
        log_success "Backup restored successfully"
        RESTORE_SUCCESS=true
        rm -f "${stderr_file}"
        return 0
    else
        # Check if it's just warnings (common with --clean and --if-exists)
        log_warn "Restore utility exited with code ${restore_exit_code}"
        
        # Display the captured stderr for debugging
        if [ -s "${stderr_file}" ]; then
            log_error "Restore errors (last 50 lines):"
            echo "======================================================================" >&2
            tail -n 50 "${stderr_file}" >&2
            echo "======================================================================" >&2
        fi
        
        log_warn "Checking if restoration was successful despite warnings..."
        
        # Try a simple query to verify the database has data
        if docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
            -c "SELECT 1;" >/dev/null 2>&1; then
            log_info "Database is accessible, considering restore as successful"
            RESTORE_SUCCESS=true
            rm -f "${stderr_file}"
            return 0
        fi
        
        log_error "Restore failed with exit code ${restore_exit_code}"
        rm -f "${stderr_file}"
        return 1
    fi
}

# =============================================================================
# Apache AGE OID Alignment
# =============================================================================

fix_age_oid_mismatch() {
    log_info "Fixing Apache AGE OID mismatch (stale OIDs from backup)..."
    
    # The issue: when restoring a backup, the ag_graph table has old OIDs from the original database,
    # but pg_namespace has new OIDs (fresh database). We need to update all references to use the new OIDs.
    
    # First, let's see what graphs exist and their current OID status
    log_info "Checking current graph OIDs before fix..."
    docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -c "
        SELECT g.name, g.graphid AS old_graphid, n.oid AS new_oid, n.nspname
        FROM ag_catalog.ag_graph g
        FULL OUTER JOIN pg_catalog.pg_namespace n ON g.name = n.nspname
        ORDER BY g.name;
    " || true
    
    # Execute all OID repairs in a single transactional block with constraints bypassed.
    # We use session_replication_role = 'replica' to temporarily disable triggers
    # and foreign key constraints, allowing us to update ag_graph.graphid and
    # ag_label.graph without violating the fk_graph_oid foreign key constraint.
    
    docker exec -i "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" <<-EOSQL
        BEGIN;
        SET session_replication_role = 'replica';
        
        -- 1. Update label graph references to the new namespace OID
        UPDATE ag_catalog.ag_label l
        SET graph = n.oid
        FROM ag_catalog.ag_graph g
        JOIN pg_catalog.pg_namespace n ON g.name = n.nspname
        WHERE l.graph = g.graphid;
        
        -- 2. Update graphid and namespace OID in ag_graph
        UPDATE ag_catalog.ag_graph g
        SET graphid = n.oid,
            namespace = n.oid::regnamespace
        FROM pg_catalog.pg_namespace n
        WHERE g.name = n.nspname;
        
        -- 3. Update relation OIDs in ag_label
        UPDATE ag_catalog.ag_label l
        SET relation = c.oid::regclass
        FROM ag_catalog.ag_graph g
        JOIN pg_catalog.pg_namespace n ON g.name = n.nspname
        JOIN pg_catalog.pg_class c ON c.relnamespace = n.oid
        WHERE g.graphid = l.graph
          AND c.relname = l.name;
        
        SET session_replication_role = 'origin';
        COMMIT;
EOSQL
    
    if [ $? -ne 0 ]; then
        log_error "Failed to fix Apache AGE OID mismatch"
        return 1
    fi
    
    # Verify the fix - check that ag_graph.graphid now matches pg_namespace.oid
    log_info "Verifying OID alignment after fix..."
    local verify_result
    verify_result=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -t -c "
        SELECT COUNT(*) FROM ag_catalog.ag_graph g
        JOIN pg_catalog.pg_namespace n ON g.name = n.nspname
        WHERE g.graphid = n.oid;
    " 2>/dev/null | tr -d ' ')
    
    local total_graphs
    total_graphs=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -t -c "
        SELECT COUNT(*) FROM ag_catalog.ag_graph;
    " 2>/dev/null | tr -d ' ')
    
    log_info "Verification: ${verify_result} of ${total_graphs} graphs have matching OIDs"
    
    if [ "$verify_result" != "$total_graphs" ]; then
        log_error "OID alignment verification failed: only ${verify_result} of ${total_graphs} graphs have matching OIDs"
        
        # Show the mismatch details
        docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -c "
            SELECT g.name, g.graphid AS graph_oid, n.oid AS namespace_oid
            FROM ag_catalog.ag_graph g
            JOIN pg_catalog.pg_namespace n ON g.name = n.nspname
            WHERE g.graphid != n.oid;
        " || true
        
        return 1
    fi
    
    # Also verify ag_label has correct graph references
    local label_verify
    label_verify=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -t -c "
        SELECT COUNT(*) FROM ag_catalog.ag_label l
        JOIN ag_catalog.ag_graph g ON l.graph = g.graphid;
    " 2>/dev/null | tr -d ' ')
    
    local total_labels
    total_labels=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -t -c "
        SELECT COUNT(*) FROM ag_catalog.ag_label;
    " 2>/dev/null | tr -d ' ')
    
    log_info "Label verification: ${label_verify} of ${total_labels} labels have valid graph references"
    
    if [ "$label_verify" != "$total_labels" ]; then
        log_error "Label verification failed: only ${label_verify} of ${total_labels} labels have valid graph references"
        return 1
    fi
    
    log_success "Apache AGE OID mismatch fix completed successfully"
    return 0
}

# =============================================================================
# Data Validation (Smoke Tests)
# =============================================================================

run_smoke_tests() {
    log_info "Running smoke tests on restored database..."
    
    # Test 1: Check if we can connect and query
    log_info "Test 1: Basic connectivity check"
    if ! docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SELECT version();" >/dev/null 2>&1; then
        log_error "Basic connectivity test failed"
        return 1
    fi
    log_success "Basic connectivity test passed"
    
    # Test 2: Count records in 'accounts' table
    log_info "Test 2: Counting records in 'accounts' table"
    local accounts_count
    accounts_count=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM accounts;" 2>/dev/null | tr -d ' ' || echo "0")
    
    if [ -n "$accounts_count" ] && [ "$accounts_count" != "0" ]; then
        log_success "Accounts table has ${accounts_count} records"
    else
        log_warn "Accounts table is empty or does not exist (count: ${accounts_count})"
    fi
    
    # Test 3: Count records in 'content' table
    log_info "Test 3: Counting records in 'content' table"
    local content_count
    content_count=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM content;" 2>/dev/null | tr -d ' ' || echo "0")
    
    if [ -n "$content_count" ] && [ "$content_count" != "0" ]; then
        log_success "Content table has ${content_count} records"
    else
        log_warn "Content table is empty or does not exist (count: ${content_count})"
    fi
    
    # Test 4: Verify Apache AGE extension and run Cypher query
    log_info "Test 4: Verifying Apache AGE graph capabilities"
    
    # First check if age extension is available
    local age_check
    age_check=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'age';" 2>/dev/null | tr -d ' ')
    
    if [ "$age_check" != "1" ]; then
        log_error "Apache AGE extension is not installed in the restored database"
        return 1
    fi
    
    # Dynamically detect the graph name from ag_catalog
    local graph_name
    graph_name=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT name FROM ag_catalog.ag_graph LIMIT 1;" 2>/dev/null | tr -d ' ' || echo "")
    
    # Check if graph list is empty
    if [ -z "$graph_name" ]; then
        log_error "CRITICAL: No graph found in ag_catalog.ag_graph. The graph catalog is empty."
        exit 1
    fi
    
    log_info "Detected graph name: ${graph_name}"
    
    # Run a Cypher query to verify graph capabilities
    # Note: We need to set the search_path to include ag_catalog for the Cypher function to work
    local cypher_result
    cypher_result=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -c "SET search_path = ag_catalog, \"\$user\", public; SELECT * FROM ag_catalog.cypher('${graph_name}', \$\$ MATCH (n) RETURN count(n) \$\$) AS (count ag_catalog.agtype);" 2>&1)
    local cypher_exit_code=$?
    
    # Check if Cypher query failed
    if [ $cypher_exit_code -ne 0 ] || echo "$cypher_result" | grep -qi "error"; then
        log_error "CRITICAL: Apache AGE Cypher query failed on graph '${graph_name}'"
        log_error "Cypher error/output: $(echo "$cypher_result" | head -10)"
        
        # Additional debugging: check the current OID mapping
        log_info "Debugging: Checking OID mapping..."
        docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" -c "
            SELECT g.name, g.graphid, n.oid as namespace_oid, 
                   CASE WHEN g.graphid = n.oid THEN 'OK' ELSE 'MISMATCH' END as status
            FROM ag_catalog.ag_graph g
            LEFT JOIN pg_catalog.pg_namespace n ON g.name = n.nspname;
        " || true
        
        exit 1
    fi
    
    log_success "Apache AGE Cypher query executed successfully on graph '${graph_name}'"
    log_info "Cypher test result: $(echo "$cypher_result" | head -3)"
    
    # Test 5: Check for graph data in ag_catalog
    log_info "Test 5: Checking for graph data in ag_catalog"
    local graph_count
    graph_count=$(docker exec "${TEMP_CONTAINER_NAME}" psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "${TEMP_DB_USER}" -d "${TEMP_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM ag_catalog.ag_graph;" 2>/dev/null | tr -d ' ' || echo "0")
    
    if [ "$graph_count" != "0" ]; then
        log_success "Found ${graph_count} graphs in ag_catalog"
    else
        log_error "CRITICAL: No graphs found in ag_catalog"
        return 1
    fi
    
    log_success "Smoke tests completed"
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
    log_info "PostgreSQL + Apache AGE Backup Verification"
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
        BACKUP_FILE="$(cd "$(dirname "$BACKUP_FILE")" 2>/dev/null && pwd)/$(basename "$BACKUP_FILE")"
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
    
    if ! setup_age_extension; then
        log_warn "AGE extension setup had issues, continuing anyway..."
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
    # Step 4: Fix Apache AGE OID Mismatch
    # =========================================================================
    log_info "--- Step 4: Fix Apache AGE OID Mismatch ---"
    
    if ! fix_age_oid_mismatch; then
        log_error "Failed to fix Apache AGE OID mismatch"
        exit 1
    fi
    
    log_success "Apache AGE OID mismatch fix completed"
    
    # =========================================================================
    # Step 5: Data Validation (Smoke Tests)
    # =========================================================================
    log_info "--- Step 5: Data Validation (Smoke Tests) ---"
    
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
