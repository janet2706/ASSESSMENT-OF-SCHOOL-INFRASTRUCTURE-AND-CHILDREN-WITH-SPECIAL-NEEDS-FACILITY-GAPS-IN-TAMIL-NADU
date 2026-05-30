# Tamil Nadu School — CWSN (Children With Special Needs) Facility Analysis
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 2. CONFIGURE FILE PATHS AND STYLING ---
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FILE_FACILITIES = os.path.join(SCRIPT_FOLDER, "33_fac.csv")
FILE_PROFILE    = os.path.join(SCRIPT_FOLDER, "33_PROFILE1_with_direction.csv")
print("Reading files from:", SCRIPT_FOLDER)
print("  Facilities File :", FILE_FACILITIES)
print("  Profile File    :", FILE_PROFILE)

# Set global visual theme for all charts
sns.set_theme(style="whitegrid", palette="Blues_d")
CHART_SAVE_SETTINGS = dict(dpi=150, bbox_inches='tight', facecolor='white')

# STEP 1: DATA LOADING & MERGING
print("\n" + "="*55)
print("  STEP 1: DATA LOADING & MERGING")
print("="*55)
data_facilities = pd.read_csv(FILE_FACILITIES)
data_profile    = pd.read_csv(FILE_PROFILE)
# Merge both datasets using the unique school identifier 'pseudocode'
df = data_profile.merge(data_facilities, on='pseudocode', how='left')
# Output dataset schema and missing values (As documented in the project report)
df.info()
print("\nMissing Values per Column:")
print(df.isnull().sum())
print(f"\nDataset Summary: {len(df):,} schools | {df['district'].nunique()} districts | {df['Direction'].nunique()} zones")

# STEP 2: CWSN FACILITY OVERVIEW
print("\n" + "="*55)
print("  STEP 2: CWSN FACILITY OVERVIEW")
print("="*55)

# Define boolean masks to check if a facility exists 
# (Assuming 1 = Available, >0 = At least one unit exists)
facility_checks = {
    'Ramps':(df['availability_ramps'] == 1),
    'Boys CWSN WC': (df['total_boys_cwsn_toilet'] > 0),
    'Girls CWSN WC':(df['total_girls_cwsn_toilet'] > 0),
    'Handrails':    (df['availability_of_handrails'] == 1),
    'Spl. Educator':(df['spl_educator_yn'] == 1),
}

print("\nCWSN Facility Summary:")
for facility_name, has_facility_mask in facility_checks.items():
    count = has_facility_mask.sum()
    percentage = has_facility_mask.mean() * 100
    print(f"  {facility_name:<18}: {count:,} schools  ({percentage:.1f}%)")

# STEP 3: K-MEANS CLUSTERING (Unsupervised Learning)
print("\n" + "="*55)
print("  STEP 3: K-MEANS CLUSTERING")
print("="*55)
# Select features related to CWSN infrastructure for clustering
clustering_features = ['total_boys_cwsn_toilet', 'total_girls_cwsn_toilet','func_boys_cwsn_friendly', 'func_girls_cwsn_friendly',
    'availability_ramps', 'availability_of_handrails']

# Standardize features (K-Means requires scaled data as it is distance-based)
scaled_features = StandardScaler().fit_transform(df[clustering_features])

# Apply K-Means to group schools into 3 distinct infrastructure levels
df['CWSN_Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled_features)

# Map the generated cluster IDs (0, 1, 2) to meaningful text labels based on average toilets
cluster_means = df.groupby('CWSN_Cluster')['total_boys_cwsn_toilet'].mean()
sorted_clusters = cluster_means.sort_values().index

cluster_label_mapping = {sorted_clusters[0]: 'Poor', sorted_clusters[1]: 'Moderate', sorted_clusters[2]: 'Good'}
df['CWSN_Level'] = df['CWSN_Cluster'].map(cluster_label_mapping)

print("\nCluster Distribution (Poor / Moderate / Good):")
print(df['CWSN_Level'].value_counts().to_string())

# Calculate district-wise scores based on CWSN facilities
district_summary = df.groupby(['district', 'Direction']).agg(
    Avg_Boys_WC  = ('total_boys_cwsn_toilet',  'mean'),
    Avg_Girls_WC = ('total_girls_cwsn_toilet', 'mean'),
    Pct_Ramps    = ('availability_ramps', lambda x: (x == 1).mean() * 100)
).reset_index()

# Create a composite score for ranking
district_summary['Composite_Score'] = district_summary['Avg_Boys_WC'] + district_summary['Avg_Girls_WC'] + (district_summary['Pct_Ramps'] / 100)
district_summary = district_summary.sort_values('Composite_Score', ascending=False)

print("\nTop 5 CWSN Districts:")
print(district_summary[['district', 'Direction', 'Composite_Score']].head().to_string(index=False))
print("\nBottom 5 CWSN Districts:")
print(district_summary[['district', 'Direction', 'Composite_Score']].tail().to_string(index=False))

# STEP 4: RANDOM FOREST CLASSIFIER (Supervised Learning)
print("\n" + "="*55)
print("  STEP 4: RANDOM FOREST CLASSIFIER")
print("="*55)

# Define general infrastructure features to predict CWSN availability
infrastructure_features = ['rural_urban', 'school_type', 'managment', 'total_class_rooms','building_status', 'electricity_availability', 'library_availability',
    'internet', 'playground_available', 'furniture_availability','medical_checkups', 'handwash_near_toilet', 'separate_room_for_hm', 'boundary_wall']
# Define Target Variable: 1 if school has at least one boys' CWSN toilet, else 0
df['Has_CWSN_Toilet'] = (df['total_boys_cwsn_toilet'] > 0).astype(int)

features_X = df[infrastructure_features]
target_y   = df['Has_CWSN_Toilet']
X_train, X_test, y_train, y_test = train_test_split(features_X, target_y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_predictions = rf_model.predict(X_test)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf_predictions):.1%}")
print(classification_report(y_test, rf_predictions, target_names=['No CWSN Toilet', 'Has CWSN Toilet'], zero_division=0))

# Extract Feature Importances to see which infrastructure factors matter most
feature_importances = pd.Series(rf_model.feature_importances_, index=infrastructure_features).sort_values(ascending=False)

# Identify districts with the highest need for CWSN facilities
district_needs = df.groupby('district').agg(
    Total_Schools = ('pseudocode', 'count'),
    Schools_With  = ('Has_CWSN_Toilet', 'sum')
).reset_index()

district_needs['Schools_Needing'] = district_needs['Total_Schools'] - district_needs['Schools_With']
district_needs['Need_Percentage']  = (district_needs['Schools_Needing'] / district_needs['Total_Schools'] * 100).round(1)

print("\nTop 10 Districts Needing CWSN Improvement:")
print(district_needs.sort_values('Need_Percentage', ascending=False).head(10)[['district', 'Total_Schools', 'Schools_Needing', 'Need_Percentage']].to_string(index=False))


# =============================================================================
# STEP 5: DECISION TREE CLASSIFIER (Interpretability & Rule Extraction)
# =============================================================================
print("\n" + "="*55)
print("  STEP 5: DECISION TREE CLASSIFIER")
print("="*55)

# Initialize Decision Tree with constraints to prevent overfitting and keep the tree readable
# max_depth=4 ensures the printed rules are short enough for a report
dt_model = DecisionTreeClassifier(max_depth=4, min_samples_split=500, random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, dt_predictions):.1%}")
print("\nExtracted Decision Rules:")
print(export_text(dt_model, feature_names=infrastructure_features))

# Calculate Infrastructure Gap Table per district
# Assuming '2' indicates 'Not Available' in the dataset based on earlier '1' logic
gap_analysis = df.groupby('district').agg(
    Total_Schools    = ('pseudocode','count'),
    Missing_Ramps    = ('availability_ramps',        lambda x: (x == 2).sum()),
    Missing_Handrails= ('availability_of_handrails', lambda x: (x == 2).sum()),
    Missing_Boys_WC  = ('total_boys_cwsn_toilet',   lambda x: (x == 0).sum()),
    Missing_Girls_WC = ('total_girls_cwsn_toilet',  lambda x: (x == 0).sum()),
    Missing_Educator = ('spl_educator_yn',           lambda x: (x == 2).sum()),
).reset_index()

# Convert raw counts to percentages
gap_metrics = ['Missing_Ramps', 'Missing_Handrails', 'Missing_Boys_WC', 'Missing_Girls_WC', 'Missing_Educator']
for metric in gap_metrics:
    gap_analysis[f'Pct_{metric}'] = (gap_analysis[metric] / gap_analysis['Total_Schools'] * 100).round(1)

# Calculate an overall average gap score
gap_analysis['Avg_Gap_Score'] = (
    gap_analysis['Pct_Missing_Ramps'] + 
    gap_analysis['Pct_Missing_Boys_WC'] + 
    gap_analysis['Pct_Missing_Girls_WC'] + 
    gap_analysis['Pct_Missing_Educator'])/4
gap_analysis = gap_analysis.sort_values('Avg_Gap_Score', ascending=False)


# =============================================================================
# STEP 6: RECOMMENDATIONS SUMMARY
# =============================================================================
print("\n" + "="*55)
print("  STEP 6: RECOMMENDATIONS SUMMARY")
print("="*55)

for facility_name, has_facility_mask in facility_checks.items():
    schools_missing = (~has_facility_mask).sum()
    pct_missing = schools_missing / len(df) * 100
    print(f"  • {facility_name}: {schools_missing:,} schools ({pct_missing:.1f}%) require immediate improvement")


# =============================================================================
# STEP 7: DATA VISUALIZATION (GENERATING 7 CHARTS)
# =============================================================================
print("\n" + "="*55)
print("  STEP 7: GENERATING 7 VISUAL CHARTS")
print("="*55)


# ── CHART 1: CWSN Facility Coverage ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))

facility_names = list(facility_checks.keys())
pct_available  = [mask.mean() * 100 for mask in facility_checks.values()]
pct_missing    = [100 - pct for pct in pct_available]
chart_colors   = sns.color_palette("Blues_d", 2)

ax1.barh(facility_names, pct_available, color=chart_colors[0], label='Has facility', height=0.5)
ax1.barh(facility_names, pct_missing, left=pct_available, color=chart_colors[1], label='Missing facility', height=0.5, alpha=0.4)

# Add percentage labels inside the bars
for i, (avail, missing) in enumerate(zip(pct_available, pct_missing)):
    ax1.text(avail / 2, i, f'{avail:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax1.text(avail + missing / 2, i, f'{missing:.1f}%', ha='center', va='center', fontsize=9, color='#7B0000', fontweight='bold')

ax1.set(xlim=(0, 100), xlabel='% of Schools', title='Chart 1 — CWSN Facility Coverage across Tamil Nadu Schools')
ax1.legend()
plt.tight_layout()
chart1_path = os.path.join(SCRIPT_FOLDER, 'chart1_cwsn_facility_coverage.png')
fig1.savefig(chart1_path, **CHART_SAVE_SETTINGS)
plt.close(fig1)
print(f"Chart 1 saved → {chart1_path}")


# ── CHART 2: Top 5 Worst Districts ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

top5_worst_districts = gap_analysis.nlargest(5, 'Avg_Gap_Score')
bar_colors = sns.color_palette("Reds_d", 5)

ax2.barh(range(5), top5_worst_districts['Avg_Gap_Score'], color=bar_colors, height=0.5, edgecolor='white')

# Add detailed text labels for each bar
for i, (_, row) in enumerate(top5_worst_districts.iterrows()):
    ax2.text(row['Avg_Gap_Score'] + 0.3, i, f"{row['Avg_Gap_Score']:.1f}%", va='center', fontsize=10, fontweight='bold')
    ax2.text(0.5, i - 0.28, 
             f"No Ramps: {row['Pct_Missing_Ramps']:.0f}%  |  No Boys WC: {row['Pct_Missing_Boys_WC']:.0f}%  |  No Girls WC: {row['Pct_Missing_Girls_WC']:.0f}%  |  No Educator: {row['Pct_Missing_Educator']:.0f}%", 
             fontsize=7.5, color='#595959')

ax2.set_yticks(range(5))
ax2.set_yticklabels(top5_worst_districts['district'], fontsize=10)
ax2.set(xlabel='Average Gap Score (%)', title='Chart 2 — Top 5 Districts with Worst CWSN Infrastructure')
plt.tight_layout()
chart2_path = os.path.join(SCRIPT_FOLDER, 'chart2_top5_worst_districts.png')
fig2.savefig(chart2_path, **CHART_SAVE_SETTINGS)
plt.close(fig2)
print(f"Chart 2 saved → {chart2_path}")


# ── CHART 3: Rural vs Urban Comparison ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))

# Filter data based on location type
rural_schools = df[df['rural_urban'] == 1]
urban_schools = df[df['rural_urban'] == 2]

metrics_to_compare = list(facility_checks.keys())
rural_percentages = [mask[rural_schools.index].mean() * 100 for mask in facility_checks.values()]
urban_percentages = [mask[urban_schools.index].mean() * 100 for mask in facility_checks.values()]

x_indices = np.arange(len(metrics_to_compare))
bar_width = 0.35
comparison_colors = sns.color_palette("Set2", 2)

rural_bars = ax3.bar(x_indices - bar_width/2, rural_percentages, bar_width, color=comparison_colors[0], label=f'Rural ({len(rural_schools):,})', edgecolor='white')
urban_bars = ax3.bar(x_indices + bar_width/2, urban_percentages, bar_width, color=comparison_colors[1], label=f'Urban ({len(urban_schools):,})', edgecolor='white')

for bars, values in [(rural_bars, rural_percentages), (urban_bars, urban_percentages)]:
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.set_xticks(x_indices)
ax3.set_xticklabels(metrics_to_compare)
ax3.set(ylim=(0, 115), ylabel='% Schools With Facility', title='Chart 3 — Rural vs Urban CWSN Facility Comparison')
ax3.legend()
plt.tight_layout()
chart3_path = os.path.join(SCRIPT_FOLDER, 'chart3_rural_vs_urban.png')
fig3.savefig(chart3_path, **CHART_SAVE_SETTINGS)
plt.close(fig3)
print(f"Chart 3 saved → {chart3_path}")


# ── CHART 4: Functional vs Total Toilets ────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5))
toilet_colors = sns.color_palette("Set1", 3)

chart_configs = [
    ('Boys', 'total_boys_cwsn_toilet', 'func_boys_cwsn_friendly'),
    ('Girls', 'total_girls_cwsn_toilet', 'func_girls_cwsn_friendly')
]

for ax, gender, total_col, func_col in zip(axes4, *zip(*chart_configs)):
    total_count = df[total_col].sum()
    func_count  = df[func_col].sum()
    nonfunc_count = total_count - func_count

    ax.bar(['Total', 'Functional', 'Non-functional'], [total_count, func_count, nonfunc_count], color=toilet_colors, edgecolor='white', width=0.5)

    for val, pct, xi in zip([total_count, func_count, nonfunc_count], [100, func_count/total_count*100, nonfunc_count/total_count*100], [0, 1, 2]):
        ax.text(xi, val + total_count * 0.01, f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

    ax.set(title=f'{gender} CWSN Toilets', ylim=(0, total_count * 1.28))
    if nonfunc_count > 0:
        ax.text(0.5, 0.93, f'{nonfunc_count/total_count*100:.1f}% non-functional', ha='center', transform=ax.transAxes, fontsize=9, color='#C00000', fontweight='bold')

fig4.suptitle('Chart 4 — Functional vs Total CWSN Toilets', fontsize=12, fontweight='bold')
plt.tight_layout()
chart4_path = os.path.join(SCRIPT_FOLDER, 'chart4_functional_vs_total_toilets.png')
fig4.savefig(chart4_path, **CHART_SAVE_SETTINGS)
plt.close(fig4)
print(f"Chart 4 saved → {chart4_path}")


# ── CHART 5: School Category Analysis ───────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))

# Map numeric school categories to readable text
category_mapping = {1: 'Primary', 2: 'Upper Pri', 3: 'Pri+Up Pri', 5: 'Secondary', 7: 'Higher Sec'}
df['School_Category'] = df['school_category'].map(category_mapping).fillna('Other')
category_colors = sns.color_palette("tab10", df['School_Category'].nunique())

# Subplot A: Count of schools per category
df['School_Category'].value_counts().plot(kind='bar', ax=axes5[0], color=category_colors, edgecolor='white')
axes5[0].set(title='Schools by Category', xlabel='', ylabel='Count')
axes5[0].tick_params(axis='x', rotation=15)
for patch in axes5[0].patches:
    axes5[0].text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 50, f'{int(patch.get_height()):,}', ha='center', fontsize=8, fontweight='bold')

# Subplot B: Percentage of schools WITH CWSN toilets per category
cwsn_pct_by_category = (df.groupby('School_Category').apply(lambda grp: (grp['total_boys_cwsn_toilet'] > 0).mean() * 100, include_groups=False).sort_values())
cwsn_pct_by_category.plot(kind='barh', ax=axes5[1], color=category_colors[:len(cwsn_pct_by_category)], edgecolor='white')
axes5[1].set(title='% With CWSN Toilet by Category', xlabel='%')
for patch in axes5[1].patches:
    axes5[1].text(patch.get_width() + 0.3, patch.get_y() + patch.get_height() / 2, f'{patch.get_width():.1f}%', va='center', fontsize=9, fontweight='bold')

fig5.suptitle('Chart 5 — School Category Distribution & CWSN Coverage', fontsize=12, fontweight='bold')
plt.tight_layout()
chart5_path = os.path.join(SCRIPT_FOLDER, 'chart5_school_category.png')
fig5.savefig(chart5_path, **CHART_SAVE_SETTINGS)
plt.close(fig5)
print(f"Chart 5 saved → {chart5_path}")


# ── CHART 6: Infrastructure Gap Heatmap ─────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(12, 9))

top15_gaps = gap_analysis.nlargest(15, 'Avg_Gap_Score').set_index('district')

# Select and rename columns for clear heatmap labels
heatmap_data = top15_gaps[['Pct_Missing_Ramps', 'Pct_Missing_Handrails', 'Pct_Missing_Boys_WC', 'Pct_Missing_Girls_WC', 'Pct_Missing_Educator']].rename(columns={
    'Pct_Missing_Ramps':     'No Ramps %',
    'Pct_Missing_Handrails': 'No Handrails %',
    'Pct_Missing_Boys_WC':   'No Boys WC %',
    'Pct_Missing_Girls_WC':  'No Girls WC %',
    'Pct_Missing_Educator':  'No Educator %'
})

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5, linecolor='white', ax=ax6, vmin=0, vmax=100, annot_kws={'size': 9, 'weight': 'bold'}, cbar_kws={'label': '% Schools Missing (Red=Critical)'})

ax6.set(title='Chart 6 — CWSN Infrastructure Gap Heatmap\nTop 15 Worst Districts  (Red = critical, Green = better)')
ax6.tick_params(axis='x', rotation=0)
ax6.tick_params(axis='y', rotation=0)
plt.tight_layout()
chart6_path = os.path.join(SCRIPT_FOLDER, 'chart6_infrastructure_gap_heatmap.png')
fig6.savefig(chart6_path, **CHART_SAVE_SETTINGS)
plt.close(fig6)
print(f"Chart 6 saved → {chart6_path}")


# ── CHART 7: Zone / Direction Comparison ────────────────────────────────────
# Aggregate data by geographical zone
zone_summary = df.groupby('Direction').agg(
    Total_Schools    = ('pseudocode',               'count'),
    Pct_Ramps        = ('availability_ramps',        lambda x: (x == 1).mean() * 100),
    Pct_Boys_WC      = ('total_boys_cwsn_toilet',    lambda x: (x > 0).mean() * 100),
    Pct_Girls_WC     = ('total_girls_cwsn_toilet',   lambda x: (x > 0).mean() * 100),
    Pct_Educator     = ('spl_educator_yn',           lambda x: (x == 1).mean() * 100),
    Pct_Electricity  = ('electricity_availability',  lambda x: (x == 1).mean() * 100),
    Pct_Library      = ('library_availability',      lambda x: (x == 1).mean() * 100),
    Pct_Internet     = ('internet',                  lambda x: (x == 1).mean() * 100),
    Pct_Poor_CWSN    = ('CWSN_Level',               lambda x: (x == 'Poor').mean() * 100),
).reset_index()

zone_names = zone_summary['Direction'].tolist()
zone_colors = sns.color_palette("Set2", len(zone_summary))
x_pos = np.arange(len(zone_summary))
bar_w = 0.2

fig7, axes7 = plt.subplots(2, 2, figsize=(13, 9))

# Subplot A: CWSN Facilities by Zone
ax_cwsn_facilities = axes7[0, 0]
cwsn_metrics = [('Pct_Ramps', 'Ramps'), ('Pct_Boys_WC', 'Boys WC'), ('Pct_Girls_WC', 'Girls WC'), ('Pct_Educator', 'Educator')]
for idx, (col, label) in enumerate(cwsn_metrics):
    ax_cwsn_facilities.bar(x_pos + idx * bar_w - 1.5 * bar_w, zone_summary[col], bar_w, label=label, edgecolor='white', color=sns.color_palette("tab10")[idx])
ax_cwsn_facilities.set_xticks(x_pos)
ax_cwsn_facilities.set_xticklabels(zone_names)
ax_cwsn_facilities.set(ylim=(0, 115), ylabel='% Schools', title='CWSN Facilities by Zone')
ax_cwsn_facilities.legend(fontsize=7, ncol=2)

# Subplot B: % of Schools in 'Poor' CWSN Cluster by Zone
ax_poor_cluster = axes7[0, 1]
ax_poor_cluster.bar(zone_names, zone_summary['Pct_Poor_CWSN'], color=zone_colors, edgecolor='white', width=0.5)
for patch in ax_poor_cluster.patches:
    ax_poor_cluster.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.5, f'{patch.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
ax_poor_cluster.set(ylim=(0, 115), ylabel='% in Poor CWSN Cluster', title='% Schools — Poor CWSN by Zone')

# Subplot C: General Infrastructure by Zone
ax_general_infra = axes7[1, 0]
infra_metrics = [('Pct_Electricity', 'Electricity'), ('Pct_Library', 'Library'), ('Pct_Internet', 'Internet')]
for idx, (col, label) in enumerate(infra_metrics):
    ax_general_infra.bar(x_pos + idx * bar_w - bar_w, zone_summary[col], bar_w * 2, label=label, edgecolor='white', color=sns.color_palette("Paired")[idx * 2])
ax_general_infra.set_xticks(x_pos)
ax_general_infra.set_xticklabels(zone_names)
ax_general_infra.set(ylim=(0, 115), ylabel='% Schools', title='General Infrastructure by Zone')
ax_general_infra.legend(fontsize=8)

# Subplot D: Pie Chart for School Distribution
ax_pie_distribution = axes7[1, 1]
ax_pie_distribution.pie(zone_summary['Total_Schools'], labels=zone_names, autopct='%1.1f%%', colors=zone_colors, startangle=90, pctdistance=0.75, textprops={'fontsize': 8})
ax_pie_distribution.set_title('Schools Distribution by Zone')

fig7.suptitle('Chart 7 — Zone / Direction Wise Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
chart7_path = os.path.join(SCRIPT_FOLDER, 'chart7_zone_direction_comparison.png')
fig7.savefig(chart7_path, **CHART_SAVE_SETTINGS)
plt.close(fig7)
print(f"Chart 7 saved → {chart7_path}")


# ── FINAL SCRIPT COMPLETION MESSAGE ─────────────────────────────────────────
print("\n" + "="*55)
print(" ALL 7 CHARTS SAVED SUCCESSFULLY")
print(f"Folder: {SCRIPT_FOLDER}")
print("Files: chart1_*.png to chart7_*.png")
print("="*55)