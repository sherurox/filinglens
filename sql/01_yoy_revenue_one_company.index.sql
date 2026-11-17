-- Equality on all three columns; cik first so any one-company query can use the prefix.
create index ix_fct_cik_tag_duration on marts.fct_financials (cik, tag, duration_quarters);
