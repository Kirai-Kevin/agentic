import pandas as pd
import sqlite3

# Load the fictional Online Retail dataset
data = {
    "Customer_ID": [1, 2, 3, 4, 5],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Gender": ["Female", "Male", "Male", "Male", "Female"],
    "Age": [25, 30, 35, 40, 45],
    "Country": ["USA", "USA", "Canada", "Canada", "USA"],
    "State": ["CA", "NY", "BC", "ON", "TX"],
    "City": ["Los Angeles", "New York", "Vancouver", "Toronto", "Houston"],
    "Zip_Code": ["90001", "10001", "V5K0A1", "M5H2N2", "77001"],
    "Product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
    "Category": ["Electronics", "Electronics", "Electronics", "Electronics", "Accessories"],
    "Price": [1200, 800, 600, 300, 100],
    "Purchase_Date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-22"],
    "Quantity": [1, 2, 3, 1, 4],
    "Total_Spent": [1200, 1600, 1800, 300, 400]
}

df = pd.DataFrame(data)

# Save to SQLite database
conn = sqlite3.connect('retail.db')
df.to_sql('Retail', conn, if_exists='replace', index=False)
conn.close()

def query_db(query):
    conn = sqlite3.connect('retail.db')
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()
