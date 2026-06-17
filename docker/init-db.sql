CREATE EXTENSION IF NOT EXISTS age;

ALTER DATABASE "${POSTGRES_DB}" SET search_path = public, ag_catalog;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'age') THEN
        RAISE NOTICE 'Apache AGE extension successfully loaded';
    END IF;
END $$;

SELECT ag_catalog.create_graph('social_graph')
WHERE NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'social_graph');
