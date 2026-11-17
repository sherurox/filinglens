-- Serves the inner side of the nested-loop join: one index range lookup per peer company
-- (cik, tag, duration_quarters equal; value_date in a range), covering unit and value.
create index ix_fct_cik_tag_duration_date on marts.fct_financials
    (cik, tag, duration_quarters, value_date) include (unit, value);
