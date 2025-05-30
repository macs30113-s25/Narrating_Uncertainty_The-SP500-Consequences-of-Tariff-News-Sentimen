import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Scraping the S&P 500 companies’s GICS data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)

df = tables[0]

df_sector = df[['Symbol', 'Security', 'GICS Sector']].copy()

# Code the GICS Sector using LabelEncoder
encoder = LabelEncoder()
df_sector['GICS Sector Code'] = encoder.fit_transform(df_sector['GICS Sector'])

print(df_sector.head())

sector_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nGICS Sector Encodding Map：")
for sector, code in sector_mapping.items():
    print(f"{code}: {sector}")

df_sector.to_csv("sp500_with_sector_codes1.csv", index=False)