-- (tag, cik, value_date): for one tag, rows come grouped by cik with the earliest date first,
-- so min() per company is read straight from the index (index-only scan, no sort).
create index ix_fct_tag_cik_date on marts.fct_financials (tag, cik, value_date);
