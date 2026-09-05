#!/usr/bin/env python3
"""IPO value vs trailing-12-month GAAP revenue. Python 3.9+, matplotlib>=3.8.
Offline snapshot; run this file to regenerate PNG/SVG/PDF/CSV.
Source links and methodology: ipo_value_vs_revenue_notes.md.
"""
from pathlib import Path
import os,tempfile,csv
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'optical-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
TARGET=333.918
DATA=[dict(company='Acacia Communications',ipo='May 2016',ttm_end='March 31, 2016',fy_revenue=239.056,prior_interim=47.244,latest_interim=84.489,ipo_value=23*35656350/1e6,ipo_cpi=240.229,ttm_cpi=sum([236.599,237.805,238.638,238.654,238.316,237.945,237.838,237.336,236.525,236.916,237.111,238.132])/12),
 dict(company='Infinera',ipo='June 2007',ttm_end='March 31, 2007',fy_revenue=58.236,prior_interim=2.653,latest_interim=49.192,ipo_value=13*83136638/1e6,ipo_cpi=208.352,ttm_cpi=sum([201.5,202.5,202.9,203.5,203.9,202.9,201.8,201.5,201.8,202.416,203.499,205.352])/12)]
DATA.append(dict(company='AAOI',ipo='September 2013',ttm_end='June 30, 2013',fy_revenue=63.421,prior_interim=28.144,latest_interim=33.914,ipo_value=10*12604334/1e6,ipo_cpi=234.149,ttm_cpi=sum([229.104,230.379,231.407,231.317,230.221,229.601,230.280,232.166,232.773,232.531,232.945,233.504])/12))
for r in DATA:
 r['ttm_nominal']=r['fy_revenue']-r['prior_interim']+r['latest_interim']
 r['revenue_2026']=r['ttm_nominal']*TARGET/r['ipo_cpi']
 r['value_2026']=r['ipo_value']*TARGET/r['ipo_cpi']
 r['adjusted_multiple']=r['value_2026']/r['revenue_2026']
 r['nominal_multiple']=r['ipo_value']/r['ttm_nominal']
plt.rcParams.update({'font.family':'DejaVu Sans','text.parse_math':False,'text.color':'#172b3a','axes.labelcolor':'#526273','xtick.color':'#526273','ytick.color':'#526273','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(12,8),facecolor='white')
ax=fig.add_axes([.10,.28,.84,.52])
fig.text(.07,.937,'IPO valuation versus revenue',fontsize=25,weight='bold')
fig.text(.07,.886,'AAOI, Acacia and Infinera · both axes in July 2026 dollars',fontsize=13,color='#526273')
ax.set(xlim=(0,500),ylim=(0,2250),xlabel='Trailing 12-month GAAP revenue (US$ millions)',ylabel='IPO equity valuation (US$ millions)')
ax.xaxis.labelpad=12;ax.yaxis.labelpad=12
ax.grid(color='#e4e9ed',linewidth=.8)
for s in ax.spines.values():s.set_visible(False)
ax.tick_params(length=0,pad=7)
# Reference slopes are ratios, not a regression or prediction.
for multiple in sorted(r['nominal_multiple'] for r in DATA):
 end=min(480,2150/multiple)
 ax.plot([0,end],[0,end*multiple],ls=(0,(4,5)),lw=1,color='#bdc8d0',zorder=1)
 ax.text(end+3,end*multiple,f'{multiple:.2f}×',fontsize=10,color='#83929e',va='center')
for r,color,offset in zip(DATA,['#187f78','#527d96','#a06437'],[(-14,-46),(14,-5),(14,24)]):
 x=r['revenue_2026'];y=r['value_2026']
 ax.scatter(x,y,s=155,c=color,edgecolors='white',linewidths=1.5,zorder=5)
 label=f"{r['company']} · {r['ipo']} IPO\n${x:.1f}M revenue / ${y:,.0f}M valuation\n{r['nominal_multiple']:.2f}× revenue"
 ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=10.5,linespacing=1.6,ha='right' if r['company'].startswith('Acacia') else 'left',va='center',color=color,zorder=6,bbox=dict(facecolor="white",edgecolor="none",alpha=.95,pad=3))
notes=[
 'Revenue: latest disclosed trailing 12 months before IPO; full year minus prior interim plus current interim.',
 'Valuation: IPO offer price × post-offering basic shares; excludes the underwriters’ additional-share option and unissued awards.',
 'Both coordinates use IPO-month CPI, preserving nominal revenue multiples; July 2026 CPI-U = 333.918.',
 'Infinera deferred substantial bundled-product revenue. Its GAAP-revenue multiple does not measure its shipment-value multiple.',
 'Dashed lines show constant valuation/revenue ratios, not a fitted relationship. Three observations do not establish a valuation model.',
 'Sources: final SEC IPO prospectuses and BLS CPI-U · Research snapshot: September 4, 2026 · Companion source: plot_ipo_value_vs_revenue.py.'
]
for i,line in enumerate(notes):fig.text(.07,.18-i*.026,line,fontsize=8.1,color='#596a79')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'ipo_value_vs_revenue_2026.{ext}',dpi=300,facecolor='white')
plt.close(fig)
with (OUT/'ipo_value_vs_revenue_data.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(DATA[0]));w.writeheader();w.writerows(DATA)
for r in DATA:print(r)

# Companion chart: the operating revenue implied by each historical multiple.
fig, ax = plt.subplots(figsize=(11,7))
fig.subplots_adjust(left=.10,right=.95,bottom=.23,top=.79)
fig.text(.10,.93,'Revenue required at historical IPO multiples',fontsize=22,weight='bold')
fig.text(.10,.875,'Disclosed startup equity valuations · annual revenue sensitivity',fontsize=12,color='#526273')
for row,color in zip(sorted(DATA,key=lambda r:r['nominal_multiple']),['#a06437','#187f78','#527d96']):
 m=row['nominal_multiple']
 ax.plot([0,6.3/m],[0,6.3],color=color,lw=2,label=f"{row['company']}: {m:.2f}×")
for name,v in [('Ayar Labs',3.75),('Lightmatter',4.4),('Lumilens',5.51)]:
 ax.axhline(v,color='#aab5be',lw=.8,ls='--')
 ax.text(3.45,v+.07,f'{name}  ${v:.2f}B',ha='right',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=2))
 for row in DATA:ax.scatter(v/row['nominal_multiple'],v,s=35,color='#253d4b',zorder=5)
ax.set(xlim=(0,3.5),ylim=(0,6.3),xlabel='Required annual revenue (US$ billions)',ylabel='Equity valuation (US$ billions)')
ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.10,.13,'Revenue = equity valuation / nominal IPO equity-to-TTM-revenue multiple. No regression is fitted.',fontsize=9)
fig.text(.10,.10,'Historical values use basic post-offering shares. Private financing terms and diluted share counts are not standardized.',fontsize=9)
fig.text(.10,.07,'No cash adjustment, return hurdle, or future revenue date is assumed. Sources and sensitivities accompany the article.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'startup_required_revenue.{ext}',dpi=300,facecolor='white')
plt.close(fig)
