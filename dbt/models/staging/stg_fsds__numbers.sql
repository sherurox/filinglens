-- One row per reported number. segments/coregistrant null = whole-company value.
select
    adsh as accession_number,
    tag,
    version as tag_version,
    ddate as value_date,
    qtrs as duration_quarters,
    uom as unit,
    segments,
    coreg as coregistrant,
    value,
    footnote,
    fy as fiscal_year,
    source_quarter
from {{ source('fsds', 'num') }}
