-- Annual and quarterly reports (and their amendments) of the 300 universe companies.
-- Other forms (S-1, 8-K, 11-K, ...) hold almost all blank fiscal years and no full statements.
select
    s.accession_number,
    s.cik,
    s.form,
    s.fiscal_year,
    s.fiscal_period,
    s.period_end_date,
    s.filed_date,
    s.accepted_at
from {{ ref('stg_fsds__submissions') }} as s
inner join {{ ref('universe') }} as u
    on s.cik = u.cik
where s.form in ('10-K', '10-K/A', '10-Q', '10-Q/A')
