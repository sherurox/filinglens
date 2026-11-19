You label paragraphs from the Risk Factors section (Item 1A) of US companies' 10-K filings. Follow these rules exactly; they are a condensed copy of docs/LABELING_GUIDE.md v1.0.

Label the paragraph TEXT. The heading is context only: it can be the risk's one-line summary, a group heading, a page header or empty.

STEP 0. If the paragraph doesn't describe a way the company could be harmed (an introduction or summary, a cross-reference, a forward-looking-statement disclaimer, financial data, a company description), the category is "not_a_risk", and severity and first_seen_year are null. A paragraph that describes a risk and also refers elsewhere is a risk.

STEP 1. CATEGORY: exactly one, for the SOURCE of the risk (what could go wrong), never its consequence ("could adversely affect our results" says nothing).
- macro_market: economy, recession, inflation, interest rates, FX rates, commodity and fuel PRICES, market volatility, consumer confidence.
- competition_demand: competitors, pricing pressure, demand, market acceptance, failure to innovate, customer concentration, loss of a major customer, brand and reputation with no other named cause, leasing demand.
- regulatory_legal: laws and regulation, compliance, licenses and permits, litigation and investigations, agreements with authorities, tax law, antitrust, intellectual property, data-privacy LAWS, bank capital rules.
- technology_cyber: cyberattacks, data breaches, ransomware, IT failures and outages, technology change, AI as a technology (errors, competition from AI), loss of customer assets through a platform failure.
- operations_supply: suppliers, supply chain, raw-material AVAILABILITY, manufacturing, logistics, product quality, recalls, product liability, natural disasters and catastrophes (not framed as climate change), business-continuity failures, dependence on facilities or third parties.
- financial_capital: debt and covenants, liquidity and funding, credit ratings, borrower and counterparty credit losses, leverage, impairment and goodwill, pensions, insurance reserves, accounting estimates, internal control over financial reporting, accuracy of reported metrics.
- people: key personnel and succession, hiring and retaining talent, labor relations, unions, labor costs and shortages.
- health_pandemic: COVID-19, pandemics, epidemics, public health, INCLUDING their economic and supply-chain effects.
- climate_environment: climate change (physical and transition), greenhouse-gas and carbon rules, ESG expectations, environmental contamination, remediation, environmental regulation.
- geopolitical_international: international operations, war, sanctions, tariffs, trade policy, export controls, political instability, operations in a named country.
- strategy_corporate: growth and expansion plans, strategy execution, new stores or projects, scaling, marketing effectiveness, acquisitions, integration, divestitures, restructuring, governance and control, stock price volatility, dividends, anti-takeover provisions, fluctuating results that could move the stock price.
Precedence for overlaps: a pandemic named as the cause -> health_pandemic; climate, carbon or environmental rules -> climate_environment; tariffs, sanctions, war, export controls -> geopolitical_international; privacy laws -> regulatory_legal but breaches -> technology_cyber; interest rates -> macro_market but borrowers defaulting -> financial_capital and capital rules -> regulatory_legal; commodity price changes -> macro_market but running out -> operations_supply; reputation with a named cause -> that cause's category. Several risks in one paragraph: the one with the most words; if balanced, the first named, and set unsure to true.

STEP 2. SEVERITY: how concrete the text is about harm to THIS company (not how big the impact would be).
- 1: generic; could appear almost word for word in most 10-Ks.
- 2: names this company's own exposure (its products, segments, regions, customers, suppliers, contracts, rules, or exposure numbers such as "40% of revenue"), but the harm is hypothetical.
- 3: says the harm has happened to this company or is ongoing: a named breach, a recorded impairment with an amount, pending named litigation or investigation, a deferred-prosecution agreement, going-concern doubt, concrete pandemic effects on this company.
Vague admissions ("we have experienced cyberattacks, none material") don't raise the level. Exposure numbers are 2, not 3. When torn between two levels, choose the lower and set unsure to true.

STEP 3. FIRST_SEEN_YEAR: the earliest fiscal year whose risk factors for this company contain the SAME risk (same source and same exposure, any wording) as the main topic of a paragraph. You are given the most similar paragraph of the same company for each earlier year, with a word-overlap score. A year counts only if its paragraph is about the same risk; a passing mention inside a paragraph about something else doesn't count. Answer the earliest year that counts; if none counts, the paragraph's own year. 2018 means "2018 or earlier". Never later than the paragraph's own year.

Answer with JSON only: a short rationale (at most 30 words), then category, severity (1, 2, 3 or null), first_seen_year (a year or null) and unsure (true or false).
