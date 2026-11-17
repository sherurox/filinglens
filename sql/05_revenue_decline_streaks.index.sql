-- Partial index: only annual USD rows are indexed (smaller, cheaper to scan). Postgres uses
-- it only when the query's WHERE implies the index's WHERE (duration_quarters = 4, unit = 'USD').
create index ix_fct_annual_usd_tag_cik_date on marts.fct_financials (tag, cik, value_date)
    include (value)
    where duration_quarters = 4 and unit = 'USD';
