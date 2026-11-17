-- 07 Size peers: the 10 companies in Apple's sector whose annual revenue (a period ending
-- within 183 days of Apple's latest fiscal year end) is closest to Apple's, by |ln(peer/Apple)|.
-- Self-join of the fact table: Apple's latest annual revenue row against its peers' rows.
with apple as (
    select f.value_date, f.value as revenue, c.sector
    from marts.fct_financials as f
    inner join marts.dim_company as c on c.cik = f.cik
    where f.cik = 320193
        and f.tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        and f.duration_quarters = 4
        and f.unit = 'USD'
    order by f.value_date desc
    limit 1
),
peers as (
    select distinct on (p.cik)
        p.ticker,
        b.value_date,
        b.value as revenue,
        abs(ln(b.value / a.revenue)) as log_size_gap
    from apple as a
    inner join marts.dim_company as p on p.sector = a.sector and p.cik <> 320193
    inner join marts.fct_financials as b
        on b.cik = p.cik
        and b.tag in (
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet'
        )
        and b.duration_quarters = 4
        and b.unit = 'USD'
        and b.value_date between a.value_date - 183 and a.value_date + 183
    where b.value > 0
    order by p.cik, abs(ln(b.value / a.revenue)), b.value_date
)
select ticker, value_date, revenue, round(log_size_gap, 3) as log_size_gap
from peers
order by log_size_gap, ticker
limit 10;
