-- Equality columns first (tag, duration_quarters), the range column (value_date) last:
-- a B-tree can only use a range on the last column it searches.
create index ix_fct_tag_duration_date on marts.fct_financials (tag, duration_quarters, value_date);
