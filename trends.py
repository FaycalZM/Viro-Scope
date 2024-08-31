from pytrends.request import TrendReq
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


ptrends = TrendReq(hl='en-US')
kw_list = ['flu', 'fever', 'cough', 'pandemic', 'symptoms']

# cat=0 => All categories
ptrends.build_payload(
    kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')

# df = ptrends.interest_by_region().sort_values('mpox', ascending=False).head(20)

df = ptrends.interest_over_time()
print(df)
if 'isPartial' in df.columns:
    df = df.drop(columns='isPartial')

df.plot(figsize=(14, 7))
plt.show()
