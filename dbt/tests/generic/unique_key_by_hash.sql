{% test unique_key_by_hash(model, combination_of_columns) %}
-- Uniqueness for very large tables: compare the row count with the number of distinct
-- 64-bit hashes of the key (one small value per row instead of grouping by every column).
-- Real duplicates always hash alike, so this never misses one; a rare hash collision can
-- only make it fail (a false alarm), never pass wrongly. Confirm a failure with
-- unique_combination_of_columns. The hash function per engine is in macros/key_hash.sql.
with counts as (
    select
        count(*) as n_rows,
        count(distinct {{ key_hash(combination_of_columns) }}) as n_distinct_keys
    from {{ model }}
)
select * from counts where n_rows <> n_distinct_keys
{% endtest %}
