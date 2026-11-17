-- Every whole-company number those filings reported: one row per (filing, fact).
-- Rules, measured in bench/fsds_profile_t8.txt:
--   whole company only: segments and coregistrant both null
--   value not null: nulls are empty in SEC's file (2024q1: 122,967 empty, 0 unparseable)
--   duration_quarters 0-4 (instant to annual); longer spans are rare cumulative values
select
    f.cik,
    n.tag,
    n.tag_version,
    n.value_date,
    n.duration_quarters,
    n.unit,
    n.value,
    f.accession_number,
    f.form,
    f.filed_date,
    f.accepted_at
from {{ ref('stg_fsds__numbers') }} as n
inner join {{ ref('int_fsds__periodic_filings') }} as f
    on n.accession_number = f.accession_number
where n.segments is null
    and n.coregistrant is null
    and n.value is not null
    and n.duration_quarters between 0 and 4
