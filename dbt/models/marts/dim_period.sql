-- One row per (value_date, duration_quarters) used by the facts. Calendar-based on purpose:
-- fiscal years differ by company (fiscal_year != year(period end) for ~9% of filings).
with periods as (
    select distinct value_date, duration_quarters
    from {{ ref('int_fsds__company_facts') }}
)

select
    {{ period_key('value_date', 'duration_quarters') }} as period_key,
    value_date,
    duration_quarters,
    case duration_quarters
        when 0 then 'instant'
        when 1 then 'quarter'
        when 2 then 'half_year'
        when 3 then 'nine_months'
        when 4 then 'annual'
    end as period_type,
    extract(year from value_date) as calendar_year,
    extract(quarter from value_date) as calendar_quarter
from periods
