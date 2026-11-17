{% macro period_key(date_column, quarters_column) -%}
{#- Integer key yyyymmdd * 10 + quarters, e.g. 2024-12-31 annual -> 202412314.
    extract() works the same in DuckDB and BigQuery. -#}
(extract(year from {{ date_column }}) * 100000
    + extract(month from {{ date_column }}) * 1000
    + extract(day from {{ date_column }}) * 10
    + {{ quarters_column }})
{%- endmacro %}
