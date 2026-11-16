{% test unique_key_by_hash(model, combination_of_columns) %}
-- Uniqueness for very large tables: compare the row count with the number of distinct
-- 64-bit hashes of the key (one small value per row instead of grouping by every column).
-- Real duplicates always hash alike, so this never misses one; a rare hash collision can
-- only make it fail (a false alarm), never pass wrongly. Confirm a failure with
-- unique_combination_of_columns. DuckDB's hash(); other engines need their own function.
with counts as (
    select
        count(*) as n_rows,
        count(distinct hash({{ combination_of_columns | join(', ') }})) as n_distinct_keys
    from {{ model }}
)
select * from counts where n_rows <> n_distinct_keys
{% endtest %}
