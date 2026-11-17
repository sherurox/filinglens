-- The universe is 300 companies; a different count means the seed or a join changed.
select count(*) as n_companies
from {{ ref('dim_company') }}
having count(*) <> 300
