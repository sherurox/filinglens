-- Covering version of query 02's index: cik, unit and value ride along in the index, so the
-- scan can be index-only (no table visits). Compare with 02's plain index scan.
create index ix_fct_tag_duration_date_covering on marts.fct_financials
    (tag, duration_quarters, value_date) include (cik, unit, value);
