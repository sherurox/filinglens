# Labeling guide: risk-factor paragraphs

Version 1.0, 19 Nov 2026. This guide is committed before the first label exists (FL-M2 T2), and it stays frozen while labeling and re-labeling are in progress (T5 to T12), so that the agreement check (Cohen's kappa) measures the labeler, not changes to the rules. Any later change goes in the changelog at the end with its date.

Table pages (cell labels) have their own short section at the end; the rest of this guide is about paragraphs.

## 1. What you label

One **paragraph** from Item 1A (Risk Factors) of a 10-K. Each task shows:

- the paragraph text (this is what you label);
- its heading: the latest bold, italic or underlined line above it in the filing. It is **context only**. It is often the one-line summary of the risk, but it can also be a group heading ("HUMAN CAPITAL RISKS"), a running page header ("MOODY'S 2018 10-K 17") or empty;
- company, sector and fiscal year;
- for the first-seen year: the most similar paragraph from each earlier year of the same company's 10-Ks, with a suggested year.

For each paragraph, answer in this order:

1. Is it a risk? If not, choose **not_a_risk** and stop.
2. **Category**: exactly one of the 11 below.
3. **Severity**: 1, 2 or 3.
4. **First-seen year**: confirm or correct the suggestion.
5. Optional: tick **unsure** if you hesitated between two answers, and add a short note.

Some paragraphs were picked because they probably mention a rare topic. Label them exactly like the others; you aren't told which ones they are.

Work at a steady pace (about 30–60 seconds per paragraph). Don't go back to compare with earlier labels: each label should come from this guide alone.

## 2. Step 0: not_a_risk

Choose **not_a_risk** when the paragraph doesn't describe a way the company could be harmed:

- introductions and summaries ("The following discussion sets forth the material risk factors…", "This section should be read in conjunction with…");
- cross-references ("see Item 7, Management's Discussion and Analysis");
- forward-looking-statement disclaimers;
- text that isn't from the risk factors at all: financial data, table labels, company descriptions (the splitter sometimes runs past the section end).

A paragraph that describes a risk and also refers elsewhere is a risk.

## 3. Categories

Label the **source** of the risk, the thing that could go wrong, not the consequence. Almost every risk factor ends with "…could adversely affect our business, financial condition and results of operations"; that ending tells you nothing.

| Category | The risk comes from | Includes | Goes elsewhere |
|---|---|---|---|
| `macro_market` | the economy and markets | recession, economic conditions, inflation, interest rates, FX rates, commodity and fuel **prices**, market volatility, consumer confidence | commodity **availability** → operations_supply; wage costs → people |
| `competition_demand` | customers and competitors | competition, pricing pressure, demand for products, market acceptance, failing to innovate, customer concentration, loss of a major customer, brand and reputation (when no other cause is named), leasing demand (REITs) | |
| `regulatory_legal` | laws, regulators and courts | new or changed laws and regulation, compliance, licenses and permits, litigation and investigations, settlements and agreements with authorities, tax law, antitrust, intellectual property (patents, infringement), data-privacy laws, bank capital rules | climate and environmental rules → climate_environment; tariffs and sanctions → geopolitical_international |
| `technology_cyber` | technology and IT | cyberattacks, data breaches, ransomware, IT system failures and outages, technology change and obsolescence, use of AI (errors, competition from AI), loss of customer assets through a platform failure | AI **regulation** → regulatory_legal |
| `operations_supply` | running the business day to day | suppliers, supply chain, raw-material availability, manufacturing, logistics, product quality, recalls and product liability, natural disasters and catastrophes, business-continuity failures, dependence on facilities or third-party services | climate-framed weather → climate_environment |
| `financial_capital` | financing, credit and reporting | debt and covenants, liquidity and funding, credit ratings, counterparty and borrower credit losses, leverage, impairment and goodwill, pensions, insurance reserves, accounting estimates, internal control over financial reporting, accuracy of reported metrics | interest-rate moves → macro_market |
| `people` | the workforce | key personnel and succession, hiring and retaining talent, labor relations, unions and strikes, labor costs and shortages, culture | |
| `health_pandemic` | disease | COVID-19, pandemics, epidemics, public-health emergencies, **including their economic and supply-chain effects** | |
| `climate_environment` | climate and the environment | climate change (physical and transition risk), greenhouse-gas and carbon rules, ESG and sustainability expectations, environmental contamination and remediation, environmental permits and regulation | |
| `geopolitical_international` | other countries and world politics | international operations in general, war and conflict, sanctions, tariffs and trade policy, export controls, political instability, operations in a named country (e.g. China) | FX rates → macro_market |
| `strategy_corporate` | the company's own plans, structure and securities | growth and expansion plans, execution of strategy, new stores or projects, scaling, marketing effectiveness, acquisitions, integration, divestitures and retained liabilities, restructuring, governance and control (dual-class shares, controlling holders), stock price volatility, dividends and distributions, anti-takeover provisions | |

### Choosing one category

1. **Source over consequence.** "A cyberattack could disrupt our supply chain" is `technology_cyber`; "a supplier failure could cause a data breach" is `operations_supply`.
2. **The body text over the heading.** If they disagree (a group heading, a page header, a heading from the previous risk), follow the text.
3. **Several risks in one paragraph:** pick the one the paragraph spends the most words on. If it is truly balanced, pick the one named first and tick **unsure**.
4. **Precedence for the common overlaps:**
   - a pandemic named as the cause → `health_pandemic`, even when the effect is a recession, a supply problem or lower demand;
   - climate change, carbon rules or environmental liabilities → `climate_environment`, not `regulatory_legal`;
   - tariffs, sanctions, war, export controls → `geopolitical_international`, not `regulatory_legal` or `macro_market`;
   - privacy **laws** and fines → `regulatory_legal`; privacy **breaches** → `technology_cyber`;
   - interest rates for a bank → `macro_market`; borrowers defaulting → `financial_capital`; capital rules → `regulatory_legal`;
   - commodity or fuel **price** changes → `macro_market`; running **out** of a commodity → `operations_supply`;
   - fluctuating quarterly results "which could cause our stock price to decline" → `strategy_corporate` (the risk described is to the security holder);
   - reputation damage with a named cause → that cause's category; with no cause → `competition_demand`.

### Real headings from the corpus (as printed by the T2 exploration, sector and year)

- `people`: "Our ability to retain and attract qualified employees is critical to the success of our business…" (finance, 2022)
- `regulatory_legal`: "The Company is dependent on its patent rights, and if its patent rights are invalidated or circumvented…" (manufacturing, 2022); "We may not be able to obtain, maintain or amend rights, authorizations, licenses, permits or consents required for our operations." (mining/energy, 2024); "A violation of Chipotle's Deferred Prosecution Agreement could have an adverse effect…" (retail, 2020)
- `financial_capital`: "We may experience greater credit losses or prepayments in automotive finance receivables than we anticipate." (retail, 2025); "Dependence on significant leverage in investments by our funds…" (finance, 2019); "If we are unable to implement and maintain effective internal controls over financial reporting…" (services, 2023)
- `macro_market`: "Increases in fuel prices, changes in the types of fuel consumed and availability of fuel supply…" (transport/utilities, 2020); the body decides whether price or availability dominates
- `competition_demand`: "If our products do not obtain and maintain market acceptance…" (services, 2021); "If we are unable to continue to innovate or if we fail to adapt to changes in our industry…" (retail, 2021); "We may be unable to lease vacant or development space…" (finance, 2022)
- `strategy_corporate`: "We may not attain our target development goals; aggressive development could cannibalize existing sales…" (retail, 2020); "If we do not continue to construct new pipelines, our future growth could be limited." (transport/utilities, 2025); "The expected benefits of the Cronos transaction may not materialize…" (manufacturing, 2019); "Our governance structure may negatively affect the decision by certain institutional investors…" (services, 2023)
- `operations_supply`: "In the event of a catastrophe, our business continuity plan may fail…" (services, 2024)
- `technology_cyber`: "Our Crypto Transfers feature and our Robinhood Wallet feature, could result in loss of customer assets…" (finance, 2022)
- Headings that are **not** risks themselves (check the body): "THE CHARLES SCHWAB CORPORATION", "MOODY'S 2018 10-K 17", "Selected Non-GAAP Financial Information", "At December 31, 2019"

## 4. Severity

Severity here measures how **concrete** the paragraph is about harm to this company, judged only from the text. It doesn't measure how large the economic impact would be (the text rarely says, and guessing it gives inconsistent labels).

| Level | Meaning | Cues |
|---|---|---|
| **1 Generic** | Could appear almost word for word in most companies' 10-Ks | no company-specific facts; "general economic conditions", "our stock price may fluctuate", "we are subject to laws" |
| **2 Specific** | Describes this company's own exposure, but as something that might happen | names its products, segments, regions, customers, suppliers, contracts or rules; gives exposure numbers ("about 40% of our revenue comes from China", "our two largest customers") |
| **3 Realized** | Says the harm has happened to this company or is happening now | a concrete past or ongoing event: a named breach, a recorded impairment with an amount, pending named litigation or investigation, a deferred-prosecution agreement, going-concern doubt, "in 2020, COVID-19 reduced our revenue by…", "we closed 300 stores" |

Rules:

- **Vague admissions don't raise the level.** "We have experienced, and may in the future experience, cyberattacks, none of which has been material" is boilerplate. Judge the rest of the paragraph.
- **Exposure numbers are level 2**, not 3: a percentage of revenue shows exposure, not harm.
- **When torn between two levels, choose the lower one** and tick **unsure**.

## 5. First-seen year

The earliest fiscal year in which this company's risk factors contain **the same risk**, as the main topic of a paragraph. "The same risk" means the same source and the same exposure, whatever the wording. A risk can be reworded every year and still be the same risk.

How to decide:

1. The task shows the paragraph's own year, a suggested first-seen year, and the most similar paragraph from each earlier year with a similarity score. The suggestion comes from text similarity and can be wrong.
2. Go back year by year. A year counts if its shown paragraph is about the same risk. A passing mention inside a paragraph about something else doesn't count (a 2018 paragraph about natural disasters that lists "pandemics" in passing is not the first COVID-19 paragraph).
3. The answer is the earliest year that counts. If no earlier year counts, it is the paragraph's own year.
4. **2018 means "2018 or earlier"**: our 10-Ks start at fiscal 2018.
5. If a year has no shown paragraph (the company's 10-K couldn't be split that year), skip it.
6. If the shown paragraph for a year is about a different risk but you're not sure the risk is truly absent that year, keep the earlier year if the years around it count, and tick **unsure**.

## 6. Edge cases

| Case | Label |
|---|---|
| Paragraph continues the previous risk without restating it ("In addition, these factors could…") | label the risk it continues; tick **unsure** if you can't tell which |
| One sentence of risk inside a long description of the business | label the risk; severity from the risk sentence |
| Bullet list item kept as a paragraph | label it like a paragraph |
| Bank paragraph on loan losses caused by a recession | `macro_market` if the recession is the subject; `financial_capital` if credit losses are |
| Insurer paragraph on catastrophe losses | `operations_supply`, or `climate_environment` if framed as climate change |
| "Our results may fluctuate" with no cause | `strategy_corporate`, severity 1 |
| COVID-19 paragraph in 2024 describing past effects | `health_pandemic`; severity 3 if it states concrete effects on this company |
| Paragraph about AI | the AI technology itself or competitors using it → `technology_cyber`; AI regulation → `regulatory_legal` |
| Not English, garbled, or cut off mid-sentence | `not_a_risk`, with a note |

## 7. Table pages

For each table page you see the page image and the table as tab-separated text, pre-filled from the filing's HTML. Fix the text so it matches the image exactly:

- one line per table row, cells separated by tabs, empty cells kept as empty fields;
- numbers as printed, with negatives as printed (keep the parentheses: `(1,234)`), without the `$` and `%` signs;
- merged header cells repeated in each column they span;
- footnote markers dropped.

Tick **corrected** if you changed anything.

## Changelog

- 1.0 (19 Nov 2026): first version, before any label.
