-- Expression index on exactly the expression the query filters on, so the non-sargable
-- extract(year from value_date) = 2024 can still use an index.
create index ix_fct_value_year_duration on marts.fct_financials
    ((extract(year from value_date)), duration_quarters);
