import pandas as pd

# Load the dataset
df = pd.read_csv('datasetNew.csv')

# # Group by Category, Subcategory, and Action and count occurrences
# counts = df.groupby(['Category', 'Subcategory', 'Action']).size().reset_index(name='Count')

# # Display the counts
# print(counts)

# # Optionally save the result to a new CSV
# counts.to_csv('category_subcategory_action_counts.csv', index=False)

# Find rows where Action is None or NaN
#none_actions = df[df['Action']=='none']

# Replace 'none' with 'no_action' in the 'Action' column
# df['Action'] = df['Action'].replace('none', 'no_action')


# Drop 'Category', 'Action', and 'Subcategory' columns
df = df.drop(columns=['Category','Action','Subcategory'])
df.to_csv("datasetNew.csv", index=False)


# # Save the updated DataFrame back to the CSV
# df.to_csv("datasetBalanced1.csv", index=False)

#nan_rows = df[df.isna().any(axis=1)]

# Print the rows with NaNs
#print(nan_rows)





