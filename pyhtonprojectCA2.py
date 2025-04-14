import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv(r"C:\Users\yadav\Desktop\Electric_Vehicle_Population_Data (1).csv")

print("Top 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include='all'))

missing_values = df.isnull().sum()
print("\nMissing Values Per Column:")
print(missing_values)

# Clean and inspect
df['Make'] = df['Make'].str.strip()
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].str.strip()

# Growth trend by year
yearly_growth = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=yearly_growth.index, y=yearly_growth.values, marker='o', color='dodgerblue')
plt.title('Growth Trend of Electric Vehicles Over the Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.grid(True)
plt.tight_layout()
plt.show()

# Top 10 makes
top_10_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_makes.index, y=top_10_makes.values, palette="cubehelix")
plt.title('Top 10 Electric Vehicle Makes')
plt.xlabel('Make')
plt.ylabel('Number of Vehicles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(top_10_makes)

# Box plot of electric range by type
df_filtered = df[df['Electric Range'] > 0]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered, x='Electric Vehicle Type', y='Electric Range', palette='Set2')
plt.title('Box Plot of Electric Range by Vehicle Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Electric Range (miles)')
plt.tight_layout()
plt.show()

# Stacked bar: Vehicle type by year
df_grouped = df[['Model Year', 'Electric Vehicle Type']].dropna()
vehicle_type_by_year = df_grouped.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack(fill_value=0)
vehicle_type_by_year.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Accent')
plt.title('Electric vs. Hybrid Market Share per Year')
plt.xlabel('Model Year')
plt.ylabel('Number of Vehicles')
plt.legend(title='Vehicle Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart: Tesla vs Others
tesla_count = df[df['Make'] == 'TESLA'].shape[0]
other_count = df[df['Make'] != 'TESLA'].shape[0]
labels = ['Tesla', 'Other Brands']
sizes = [tesla_count, other_count]
colors = ['#0F4C81', '#A2C523']
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0))
plt.title('Tesla vs. Other EV Brands Market Share')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Scatter plot: Model Year vs Electric Range
df_filtered = df[['Model Year', 'Electric Range']].dropna()
df_filtered['Model Year'] = pd.to_numeric(df_filtered['Model Year'], errors='coerce')
df_filtered['Electric Range'] = pd.to_numeric(df_filtered['Electric Range'], errors='coerce')
df_filtered.dropna(inplace=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='Model Year', y='Electric Range', alpha=0.6, color='mediumseagreen')
plt.title('Relationship Between Model Year and Electric Range')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
plt.grid(True)
plt.tight_layout()
plt.show()
