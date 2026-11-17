-- 04 The first period each company reported OperatingLeaseRightOfUseAsset (the ASC 842
-- leases standard, effective for public companies in fiscal years starting after
-- 15 Dec 2018), then the number of companies per first year. The right-of-use asset is the
-- most widely used lease tag here (174 companies; OperatingLeaseLiability: 10).
with first_seen as (
    select cik, min(value_date) as first_value_date
    from marts.fct_financials
    where tag = 'OperatingLeaseRightOfUseAsset'
    group by cik
)
select extract(year from first_value_date)::int as first_year, count(*) as companies
from first_seen
group by first_year
order by first_year;
