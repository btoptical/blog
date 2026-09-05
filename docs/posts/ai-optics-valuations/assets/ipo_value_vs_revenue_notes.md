# IPO equity valuation versus trailing revenue

Both axes use July 2026 purchasing power. X is trailing-12-month GAAP revenue, Y is equity capitalization at the IPO offer price. AAOI, Acacia and Infinera are plotted: the other companies in the companion charts are private and have no IPO observations.

| Company | Nominal TTM revenue | Revenue at IPO-date purchasing power, restated to July 2026 | IPO equity value, July 2026 dollars | Nominal multiple |
|---|---:|---:|---:|---:|
| AAOI | $69.191M | $98.673M | $179.749M | 1.822x |
| Acacia | $276.301M | $384.058M | $1,139.932M | 2.968x |
| Infinera | $104.775M | $167.919M | $1,732.120M | 10.315x |

## Inputs and sources

[Acacia final IPO prospectus](https://www.sec.gov/Archives/edgar/data/1651235/000119312516589514/d46988d424b4.htm): revenue $239.056M for 2015, $47.244M for Q1 2015, and $84.489M for Q1 2016. Thus TTM = 239.056 − 47.244 + 84.489 = $276.301M. Offer valuation = $23 × 35,656,350 basic post-offering shares = $820.09605M.

[Infinera final IPO prospectus](https://www.sec.gov/Archives/edgar/data/1138639/000119312507131232/d424b4.htm): revenue $58.236M for 2006, $2.653M for Q1 2006, and $49.192M for Q1 2007. Thus TTM = 58.236 − 2.653 + 49.192 = $104.775M. Offer valuation = $13 × 83,136,638 basic post-offering shares = $1,080.776294M. Infinera recognized much of its bundled-product sales over time; GAAP revenue is not shipment value.

[BLS CPI-U historical table](https://www.bls.gov/regions/northeast/data/consumerpriceindex_us_table.htm): US city average, all items, not seasonally adjusted, 1982–84=100. July 2026 target = 333.918. Acacia IPO-month CPI (May 2016) = 240.229; Infinera IPO-month CPI (June 2007) = 208.352. The mean of the twelve monthly observations in each trailing revenue period is 237.65125 for April 2015–March 2016 and 202.79725 for April 2006–March 2007.

Both chart coordinates use nominal amount × 333.918 / IPO-month CPI. A common factor preserves the original nominal valuation-to-revenue ratio. This displays the revenue amount in IPO-date purchasing power before restating it, rather than adjusting each month of earned revenue separately.

Historical equity uses basic post-offering shares, excluding unissued awards and the additional-share option. Private financing valuation definitions are not standardized to those basic counts. The article gives an explicit dilution sensitivity and explains why undisclosed private cash prevents a complete enterprise-value comparison.

The lines show the observed AAOI, Acacia and Infinera nominal multiples, not a regression. The companion startup_required_revenue chart divides each startup valuation by each multiple. Both axes on each chart use the same units. The companion assumes no future revenue date, cash adjustment or return hurdle; the article separately models a five-year 20% return scenario.

Research snapshot: September 4, 2026. Run `python3 plot_ipo_value_vs_revenue.py` with Python 3.9+ and Matplotlib 3.8+ to regenerate PNG (300 dpi), SVG (editable text), PDF (vector) and CSV. All numeric inputs are embedded; execution needs no network access.

## AAOI addition

Source: [final IPO prospectus, September 25, 2013](https://appliedoptoelectronics.gcs-web.com/node/7266/html).

TTM revenue = FY2012 63.421 - H1 2012 28.144 + H1 2013 33.914 = $69.191M.
TTM gross profit = 18.929 - 8.813 + 10.032 = $20.148M; gross margin = 29.1194%.
Offer equity value = $10 x 12,604,334 post-offering basic shares = $126.04334M.
The overallotment shares were secondary shares and do not increase shares outstanding.
TTM CPI average (July 2012 through June 2013) = 231.3523333; IPO-month CPI September 2013 = 234.149.
2026 revenue at the common IPO-month conversion = $98.672727M; equity value = $179.749390M; nominal and plotted multiple = 1.821672472x.
Datacenter TTM revenue = 5.293 - 1.317 + 10.260 = $14.236M, or 20.575% of total.
No cumulative private-funding estimate is asserted for AAOI. Prospectus total consideration paid by existing shareholders is not assumed to equal cash venture funding.
The CSV uses prior_interim/latest_interim: Q1 for Acacia and Infinera, H1 for AAOI.
