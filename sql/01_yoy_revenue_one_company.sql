-- 01 Year-over-year annual revenue growth for one company (Apple, CIK 320193).
-- Window function: LAG over the company's annual values, oldest first.
with revenue as (
    select value_date, value as revenue
    from marts.fct_financials
    where cik = 320193
        and tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        and duration_quarters = 4
        and unit = 'USD'
)
select
    value_date,
    revenue,
    lag(revenue) over (order by value_date) as prior_revenue,
    round(100 * (revenue / nullif(lag(revenue) over (order by value_date), 0) - 1), 2) as yoy_pct
from revenue
order by value_date;
