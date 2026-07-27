import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/mnt/user-data/uploads/tesco_groceries_dataset.csv')

print("=" * 80)
print("TESCO GROCERIES - TRANSACTION ENCODER & ASSOCIATION LEARNING")
print("=" * 80)

# ============================================================================
# STEP 1: CREATE TRANSACTIONS FROM DATASET
# ============================================================================
print("\nSTEP 1: CREATING TRANSACTIONS FROM DATASET")
print("-" * 80)

# Extract main category and subcategory from breadcrumbs
df['main_category'] = df['breadcrumbs'].str.split('~').str[0].fillna('Unknown')
df['sub_category'] = df['breadcrumbs'].str.split('~').str[1].fillna('Unknown')

# Create price bins for more meaningful associations
df['price_range'] = pd.cut(df['price'], 
                            bins=[0, 1, 2, 5, 10, 100], 
                            labels=['<£1', '£1-2', '£2-5', '£5-10', '>£10'])

# Create rating categories
df['rating_category'] = pd.cut(df['avg_rating'], 
                               bins=[0, 2, 3, 4, 5], 
                               labels=['Low (1-2)', 'Medium (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
df['rating_category'] = df['rating_category'].astype(str)
df['rating_category'] = df['rating_category'].replace('nan', 'Not Rated')

# Create availability status
df['status'] = df['availability'].apply(lambda x: 'Available' if x == 'InStock' else 'Out of Stock')

print(f"Total Products: {len(df)}")
print(f"Main Categories: {df['main_category'].nunique()}")
print(f"Sub Categories: {df['sub_category'].nunique()}")
print(f"Price Ranges: {df['price_range'].nunique()}")
print(f"Rating Categories: {df['rating_category'].nunique()}")

# ============================================================================
# STEP 2: BUILD TRANSACTION LIST (Approach 1: Item-Level Transactions)
# ============================================================================
print("\nSTEP 2: CREATING TRANSACTION LIST")
print("-" * 80)

# For each product, create a transaction with its attributes
transactions = []
for idx, row in df.iterrows():
    transaction = []
    
    # Add brand
    if pd.notna(row['brand']):
        transaction.append(f"Brand: {row['brand']}")
    
    # Add main category
    if pd.notna(row['main_category']):
        transaction.append(f"Category: {row['main_category']}")
    
    # Add sub category
    if pd.notna(row['sub_category']) and row['sub_category'] != 'Unknown':
        transaction.append(f"SubCat: {row['sub_category']}")
    
    # Add price range
    if pd.notna(row['price_range']):
        transaction.append(f"Price: {row['price_range']}")
    
    # Add availability
    transaction.append(f"Status: {row['status']}")
    
    # Add rating category
    if row['rating_category'] != 'Not Rated':
        transaction.append(f"Rating: {row['rating_category']}")
    
    if len(transaction) > 0:
        transactions.append(transaction)

print(f"Total Transactions Created: {len(transactions)}")
print(f"\nExample Transactions (first 5):")
for i, trans in enumerate(transactions[:5], 1):
    print(f"  Transaction {i}: {trans}")

# ============================================================================
# STEP 3: APPLY TRANSACTION ENCODER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: APPLYING TRANSACTION ENCODER")
print("-" * 80)

# Create TransactionEncoder object
te = TransactionEncoder()

# Fit and transform the transactions
te_ary = te.fit_transform(transactions)

# Convert to DataFrame
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"\nEncoded DataFrame Shape: {df_encoded.shape}")
print(f"Number of Transactions (rows): {df_encoded.shape[0]}")
print(f"Number of Unique Items (columns): {df_encoded.shape[1]}")
print(f"\nEncoded Columns ({len(df_encoded.columns)} items):")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "-" * 80)
print("One-Hot Encoded DataFrame (first 10 rows):")
print("-" * 80)
print(df_encoded.head(10).to_string())

print(f"\nData Type: {df_encoded.dtypes.iloc[0]}")
print(f"Value Range: {df_encoded.iloc[:, 0].unique()}")

# ============================================================================
# STEP 4: FIND FREQUENT ITEMSETS USING APRIORI
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FINDING FREQUENT ITEMSETS (Apriori Algorithm)")
print("-" * 80)

# Support threshold: items that appear in at least 10% of transactions
min_support = 0.10

frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

print(f"\nMinimum Support Threshold: {min_support * 100:.1f}%")
print(f"Total Frequent Itemsets Found: {len(frequent_itemsets)}")
print(f"\nTop 20 Frequent Itemsets:")
print("-" * 80)
print(frequent_itemsets.head(20).to_string(index=False))

# Save frequent itemsets
frequent_itemsets.to_csv('/mnt/user-data/outputs/frequent_itemsets.csv', index=False)
print("\n✓ Saved: frequent_itemsets.csv")

# ============================================================================
# STEP 5: GENERATE ASSOCIATION RULES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: GENERATING ASSOCIATION RULES")
print("-" * 80)

if len(frequent_itemsets) > 1:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    if len(rules) > 0:
        # Calculate additional metrics
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Sort by lift (how much more likely items are bought together)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"Total Association Rules Generated: {len(rules)}")
        print(f"\nMetrics:")
        print(f"  - Support: Probability of X and Y occurring together")
        print(f"  - Confidence: Probability of Y given X (P(Y|X))")
        print(f"  - Lift: How many times more likely X and Y are bought together")
        print(f"\nTop 15 Association Rules (by Lift):")
        print("-" * 80)
        
        display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).copy()
        print(display_rules.to_string(index=False))
        
        # Save rules
        rules.to_csv('/mnt/user-data/outputs/association_rules.csv', index=False)
        print("\n✓ Saved: association_rules.csv")
    else:
        print("No association rules found with current thresholds.")
else:
    print("Not enough frequent itemsets to generate rules.")

# ============================================================================
# STEP 6: SAVE ENCODED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: SAVING OUTPUTS")
print("-" * 80)

df_encoded.to_csv('/mnt/user-data/outputs/tesco_encoded_transactions.csv', index=False)
print("✓ Saved: tesco_encoded_transactions.csv")

# Save original data with new features
df_processed = df[['name', 'brand', 'main_category', 'sub_category', 'price', 
                   'price_range', 'availability', 'status', 'avg_rating', 
                   'rating_category', 'reviews_count']].copy()
df_processed.to_csv('/mnt/user-data/outputs/tesco_processed_data.csv', index=False)
print("✓ Saved: tesco_processed_data.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & INSIGHTS")
print("=" * 80)

print(f"\nDataset Summary:")
print(f"  • Original Products: {len(df)}")
print(f"  • Transactions Created: {len(transactions)}")
print(f"  • Unique Items: {df_encoded.shape[1]}")
print(f"  • Items per Transaction (avg): {df_encoded.sum(axis=1).mean():.2f}")
print(f"  • Sparsity: {(1 - df_encoded.sum().sum() / (len(df_encoded) * df_encoded.shape[1])) * 100:.2f}%")

print(f"\nFrequent Itemsets:")
print(f"  • Total Found: {len(frequent_itemsets)}")
print(f"  • Single Items: {len(frequent_itemsets[frequent_itemsets['length'] == 1])}")
print(f"  • Pairs: {len(frequent_itemsets[frequent_itemsets['length'] == 2])}")
print(f"  • Triples: {len(frequent_itemsets[frequent_itemsets['length'] == 3])}")

if len(rules) > 0:
    print(f"\nAssociation Rules:")
    print(f"  • Total Rules: {len(rules)}")
    print(f"  • Avg Confidence: {rules['confidence'].mean():.3f}")
    print(f"  • Avg Lift: {rules['lift'].mean():.3f}")
    print(f"  • Max Lift: {rules['lift'].max():.3f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
