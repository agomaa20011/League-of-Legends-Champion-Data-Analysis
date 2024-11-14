import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#loading the data 
file_path = r'D:\DATA SCIENCE\LOL\lol_champions2.csv'
lol_df = pd.read_csv(file_path, encoding='ISO-8859-1', sep=';')

#cleaning the data
lol_df = lol_df.dropna(axis=1, how='all')
lol_df['role'] = lol_df['role'].apply(lambda x: x.split(',')[0])

#deleting the columns withe more than 10% missing and droping critical ones
lol_df = lol_df.drop(columns=['ulti_cooldown,,,,,,,,,,,,,,,,,,'])
lol_df = lol_df.dropna(thresh=len(lol_df) * 0.9, axis=1)
lol_df = lol_df.dropna(subset=['name', 'key'])

#filling the missing data
lol_df['difficulty'] = lol_df['difficulty'].fillna('Unknown')
lol_df['tags'] = lol_df['tags'].fillna('Unknown')
lol_df['partype'] = lol_df['partype'].fillna('Unknown')


lol_df['hp'] = lol_df['hp'].fillna(lol_df['hp'].mean())
lol_df['mp'] = lol_df['mp'].fillna(lol_df['mp'].mean())
lol_df['armor'] = lol_df['armor'].fillna(lol_df['armor'].mean())

lol_df['spell3_name'] = lol_df['spell3_name'].fillna('Unknown')
lol_df['spell3_description'] = lol_df['spell3_description'].fillna('Unknown')
lol_df['spell3_cost'] = lol_df['spell3_cost'].fillna(0)  # Assuming 0 cost if missing
lol_df['spell3_cooldown'] = lol_df['spell3_cooldown'].fillna(0)  # Assuming 0 cooldown if missing

lol_df['ulti_name'] = lol_df['ulti_name'].fillna('Unknown')
lol_df['ulti_description'] = lol_df['ulti_description'].fillna('Unknown')
lol_df['ulti_cost'] = lol_df['ulti_cost'].fillna(0)  # Assuming 0 cost if missing

numerical_cols = ['hpperlevel', 'mpperlevel', 'movespeed', 'armorperlevel', 
                  'spellblock', 'spellblockperlevel', 'attackrange', 'hpregen', 
                  'hpregenperlevel', 'mpregen', 'mpregenperlevel', 'attackdamage', 
                  'attackdamageperlevel', 'attackspeed', 'attackspeedperlevel']

lol_df[numerical_cols] = lol_df[numerical_cols].fillna(lol_df[numerical_cols].median())

categorical_cols = ['spell1_name', 'spell1_description', 'spell1_cost', 'spell1_cooldown', 
                    'spell2_name', 'spell2_description', 'spell2_cost', 'spell2_cooldown', 
                    'spell3_name', 'spell3_description', 'spell3_cost', 'spell3_cooldown', 
                    'ulti_name', 'ulti_description', 'ulti_cost']

lol_df[categorical_cols] = lol_df[categorical_cols].fillna(lol_df[categorical_cols].mode().iloc[0])

#saving the cleaned dataset
lol_df.to_csv('d:/DATA SCIENCE/LOL/cleaned_lol_champions.csv', index=False)

#correcting formats
lol_df['releasedate'] = pd.to_datetime(lol_df['releasedate'], errors='coerce')
lol_df['release_year'] = lol_df['releasedate'].dt.year

# Remove the percentage signs and convert to float
lol_df['winrate'] = lol_df['winrate'].replace('%', '', regex=True).astype(float) / 100
lol_df['banrate'] = lol_df['banrate'].replace('%', '', regex=True).astype(float) / 100
lol_df['popularity'] = lol_df['popularity'].replace('%', '', regex=True).astype(float) / 100

# 1.Time Series Analysis of Champion Releases

lol_df['release_year'] = lol_df['releasedate'].dt.year
releases_per_year = lol_df.groupby('release_year').size()

plt.figure(figsize=(14, 6))
releases_per_year.plot(kind='line', marker='o', color='darkcyan')
plt.title('Champion Releases Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Champions Released')
plt.grid()
plt.show()

# 2.Understanding the relation between Win Rate, Ban Rate, and Popularity

correlation_matrix = lol_df[['winrate', 'banrate', 'popularity']].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='PuBuGn', fmt='.2%', cbar_kws={'label': 'Correlation'}, annot_kws={'size': 12})
plt.title('The relation between: Win Rate, Ban Rate, Popularity')
plt.show()

# 3.Role-Based Win Rate and Ban Rate Comparison

plt.subplot(1, 2, 1)
sns.boxplot(x='role', y='winrate', data=lol_df, palette='BuGn')
plt.title('Win Rate by Champion Role', fontsize=16)
plt.xlabel('Champion Role', fontsize=14)
plt.ylabel('Win Rate (%)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  

plt.subplot(1, 2, 2)
sns.boxplot(x='role', y='banrate', data=lol_df, palette='GnBu')
plt.title('Ban Rate by Champion Role', fontsize=16)
plt.xlabel('Champion Role', fontsize=14)
plt.ylabel('Ban Rate (%)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# 4.Top 10 Most Popular Champions and Their Win Rates

top_picks = lol_df[['name', 'popularity', 'winrate']].nlargest(10, 'popularity')

bar_width = 0.4
x = np.arange(len(top_picks))

plt.figure(figsize=(12, 8))


plt.bar(x - bar_width/2, top_picks['popularity'], width=bar_width, color='#3c4e4b', label='Popularity')

plt.bar(x + bar_width/2, top_picks['winrate'], width=bar_width, color='#466964', label='Win Rate')

plt.xticks(x, top_picks['name'], rotation=45)

plt.title('Top picks and Their Win Rates')
plt.ylabel('percentages for popularity and win rate')
plt.xlabel('champions')
plt.legend(loc='lower right')
plt.show()


# 5.Predicting Top Ban Champions Using Win Rate and Popularity

plt.figure(figsize=(10, 6))
#sns.boxplot(x='role', y='banrate', data=lol_df, hue='role', palette='BuGn')
sns.boxplot(x='role', y='banrate', data=lol_df, hue='role', palette='GnBu', legend=False)
plt.title('Ban Rate by Role')
plt.show()

X = lol_df[['winrate', 'popularity']]
y = lol_df['banrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='darkcyan')
plt.title('Actual vs. Predicted Ban Rate')
plt.xlabel('Actual Ban Rate')
plt.ylabel('Predicted Ban Rate')
plt.show()
