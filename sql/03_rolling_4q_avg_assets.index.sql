-- Covering index: every column the query reads is in it (INCLUDE adds unit and value as
-- payload), so Postgres can answer from the index alone, already ordered by cik, value_date.
create index ix_fct_tag_duration_cik_date on marts.fct_financials
    (tag, duration_quarters, cik, value_date) include (unit, value);
