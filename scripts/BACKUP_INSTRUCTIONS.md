# PostgreSQL Backup Script - Setup Instructions

## Overview
The `backup_db.sh` script performs automated daily backups of the PostgreSQL database running in the `tg_parser_db` Docker container.

## Prerequisites
- Docker and Docker Compose installed on the server
- PostgreSQL container named `tg_parser_db` (as defined in `docker-compose.scraper.yml`)
- `.env` file with required variables: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`

## Script Features
1. Creates compressed database dumps (`backup_YYYY-MM-DD_HHMMSS.sql.gz`)
2. Stores backups in `backups/` directory (created automatically)
3. Retains only the last 7 daily backups (older ones are deleted)
4. Uses environment variables from `.env` for authentication
5. Production-ready with error handling (`set -euo pipefail`)

## Setup on Linux Server

### 1. Make the script executable
```bash
cd /path/to/telethon-scraper-task
chmod +x scripts/backup_db.sh
```

### 2. Test the script manually
```bash
cd /path/to/telethon-scraper-task
./scripts/backup_db.sh
```

Verify the backup was created:
```bash
ls -la backups/
```

### 3. Add to crontab for daily execution at 02:00 AM
Open the crontab editor:
```bash
crontab -e
```

Add the following line:
```cron
0 2 * * * cd /path/to/telethon-scraper-task && /path/to/telethon-scraper-task/scripts/backup_db.sh >> /path/to/telethon-scraper-task/logs/backup.log 2>&1
```

**Important:** Replace `/path/to/telethon-scraper-task` with the actual absolute path to your project directory.

### 4. Create logs directory (optional but recommended)
```bash
mkdir -p /path/to/telethon-scraper-task/logs
```

### 5. Verify cron job is installed
```bash
crontab -l
```

## Backup File Format
- Filename: `backup_YYYY-MM-DD_HHMMSS.sql.gz`
- Example: `backup_2026-06-02_020000.sql.gz`
- Compression: gzip

## Restore from Backup
To restore a backup:
```bash
gunzip -c backups/backup_YYYY-MM-DD_HHMMSS.sql.gz | docker exec -i tg_parser_db psql -U $POSTGRES_USER -d $POSTGRES_DB
```

## Troubleshooting
- Check `logs/backup.log` for execution output
- Ensure Docker container `tg_parser_db` is running: `docker ps`
- Verify `.env` file exists and contains required variables
- Test `pg_dump` manually: `docker exec tg_parser_db pg_dump -U postgres tg_parser --help`
