{% test unique_combination_of_columns(model, combination_of_columns) %}
-- Fails with one row per key that appears more than once (nulls compare equal, like GROUP BY).
select {{ combination_of_columns | join(', ') }}, count(*) as n_rows
from {{ model }}
group by {{ combination_of_columns | join(', ') }}
having count(*) > 1
{% endtest %}
