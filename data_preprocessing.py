import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# Load transactions data
transactions_df = pd.read_csv('/content/gdrive/MyDrive/SASRec.pytorch/data/transactions_train.csv')

# Filter transactions to include only those from the last month
filtered_df = transactions_df[(transactions_df['t_dat'] >= '2020-07-23') & (transactions_df['t_dat'] <= '2020-09-22')]

# Filter customers with at least 20 interactions and at most 150 interactions
customer_transaction_counts = filtered_df['customer_id'].value_counts()
active_customers = customer_transaction_counts[(customer_transaction_counts >= 20) & (customer_transaction_counts <= 200)].index
filtered_df = filtered_df[filtered_df['customer_id'].isin(active_customers)]

# Map Alphanumeric User IDs to Numeric IDs
user_id_map = {}
item_id_map = {}
usernum = 0
itemnum = 0
User = defaultdict(list)

# Assign numeric IDs and build user interaction histories
for _, row in filtered_df.iterrows():
    user_id = row['customer_id']
    item_id = row['article_id']
    timestamp = pd.to_datetime(row['t_dat']).timestamp()

    if user_id not in user_id_map:
        usernum += 1
        user_id_map[user_id] = usernum

    if item_id not in item_id_map:
        itemnum += 1
        item_id_map[item_id] = itemnum

    User[user_id_map[user_id]].append([timestamp, item_id_map[item_id]])

# Sort each user's interactions by time
for user_id in User.keys():
    User[user_id].sort(key=lambda x: x[0])

# Save preprocessed data
with open('/content/gdrive/MyDrive/SASRec.pytorch/data/preprocessed_hm_data.txt', 'w') as f:
    for user_id in User.keys():
        for interaction in User[user_id]:
            f.write(f'{user_id} {interaction[1]}\n')

print(f'Number of users: {usernum}')
print(f'Number of items: {itemnum}')

# Calculate the number of interactions per user
interaction_counts = [len(User[user_id]) for user_id in User.keys()]

# Plot the distribution of interactions per user
plt.figure(figsize=(10, 6))
plt.hist(interaction_counts, bins=range(5, max(interaction_counts) + 10), edgecolor='black')
plt.title('Distribution of Interactions per User')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Users')

# Save and show the plot
plt.savefig('/content/gdrive/MyDrive/SASRec.pytorch/data/interactions_plot.png')
plt.show()

# Print interaction statistics
print("Minimum interactions per user:", min(interaction_counts))
print("Maximum interactions per user:", max(interaction_counts))
print("Average interactions per user:", sum(interaction_counts) / len(interaction_counts))
