-- One row per company x tag x period (value_date, duration_quarters) x unit.
-- Later filings repeat earlier periods as comparatives, sometimes with a changed value.
-- value is "as last reported"; first_reported_value is kept so restatements can be queried.
with ranked as (
    select
        *,
        row_number() over (
            partition by cik, tag, value_date, duration_quarters, unit
            order by filed_date desc, accepted_at desc, accession_number desc
        ) as rn_latest,
        row_number() over (
            partition by cik, tag, value_date, duration_quarters, unit
            order by filed_date asc, accepted_at asc, accession_number asc
        ) as rn_earliest
    from {{ ref('int_fsds__company_facts') }}
),

collapsed as (
    select
        cik,
        tag,
        value_date,
        duration_quarters,
        unit,
        max(case when rn_latest = 1 then value end) as value,
        max(case when rn_latest = 1 then tag_version end) as tag_version,
        max(case when rn_latest = 1 then accession_number end) as last_accession_number,
        max(case when rn_latest = 1 then filed_date end) as last_filed_date,
        max(case when rn_earliest = 1 then value end) as first_reported_value,
        max(case when rn_earliest = 1 then filed_date end) as first_filed_date,
        count(*) as times_reported
    from ranked
    group by cik, tag, value_date, duration_quarters, unit
)

select
    cik,
    {{ period_key('value_date', 'duration_quarters') }} as period_key,
    tag || '|' || tag_version as tag_key,
    tag,
    tag_version,
    value_date,
    duration_quarters,
    unit,
    value,
    first_reported_value,
    value <> first_reported_value as was_restated,
    times_reported,
    first_filed_date,
    last_filed_date,
    last_accession_number
from collapsed
