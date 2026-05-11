-- Initialize Apache AGE extension in the database
-- This script runs automatically when the PostgreSQL container starts for the first time

-- Create the age extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS age;

-- Set the search path to include age for convenience
ALTER DATABASE "${POSTGRES_DB}" SET search_path = public, age;

-- Optional: Verify extension is loaded (for logging purposes)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'age') THEN
        RAISE NOTICE 'Apache AGE extension successfully loaded';
    END IF;
END $$;