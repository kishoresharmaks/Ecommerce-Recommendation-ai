import sqlite3
import pandas as pd
from user_profiling import UserProfilingAgent
from product_analysis import ProductAnalysisAgent
from recommendation import RecommendationAgent
from utils import load_data, preprocess_data, evaluate_recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import ast 

# Initialize SQLite database
def init_database():
    conn = sqlite3.connect('ecommerce_recommendations.db')
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS users")
    c.execute("DROP TABLE IF EXISTS products")
    c.execute("DROP TABLE IF EXISTS recommendations")
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id TEXT PRIMARY KEY, preferences TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (product_id INTEGER PRIMARY KEY, details TEXT, price REAL, rating REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (user_id TEXT, product_id INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(user_id),
                  FOREIGN KEY (product_id) REFERENCES products(product_id))''')
    conn.commit()
    return conn

# Save data to database
def save_to_database(conn, user_id, preferences, product_data):
    c = conn.cursor()
    print(f"Inserting user_id: {user_id}, preferences: {preferences}")
    try:
        c.execute("INSERT OR REPLACE INTO users (user_id, preferences) VALUES (?, ?)",
                  (user_id, preferences))
    except sqlite3.IntegrityError as e:
        print(f"IntegrityError: {e}. Check user_id: {user_id} and preferences: {preferences}")
        raise
    for _, product in product_data.iterrows():
        try:
            product_id_str = str(product['Product_ID'])
            if product_id_str.startswith('P'):
                product_id = int(product_id_str.replace('P', ''))
            else:
                product_id = int(product_id_str)
        except (ValueError, TypeError):
            print(f"Warning: Invalid Product_ID {product['Product_ID']} for row. Skipping.")
            continue
        c.execute("INSERT OR REPLACE INTO products (product_id, details, price, rating) VALUES (?, ?, ?, ?)",
                  (product_id, f"{product['Brand']} {product['Subcategory']} in {product['Category']}",
                   product['Price'] if pd.notna(product['Price']) else None,
                   product['Product_Rating'] if pd.notna(product['Product_Rating']) else None))
    conn.commit()

def main():
    conn = init_database()

    customer_data_path = 'customer_data_collection.csv'
    product_data_path = 'product_recommendation_data.csv'

    customer_data = load_data(customer_data_path)
    product_data = load_data(product_data_path)

    if customer_data is None or product_data is None:
        print(f"Error: Could not load data. Check {customer_data_path} and {product_data_path}.")
        conn.close()
        return

    customer_data = preprocess_data(customer_data)
    product_data = preprocess_data(product_data)

    # Ensure all required columns are present
    required_customer_cols = ['Customer_ID', 'Age', 'Gender', 'Location', 'Browsing_History', 
                             'Purchase_History', 'Customer_Segment', 'Avg_Order_Value', 'Holiday', 'Season']
    missing_cols = [col for col in required_customer_cols if col not in customer_data.columns]
    if missing_cols:
        print(f"Warning: Missing columns in customer_data: {missing_cols}")
        customer_data = customer_data.reindex(columns=required_customer_cols, fill_value='N/A')

    # Parse history columns as lists
    customer_data['Browsing_History'] = customer_data['Browsing_History'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    customer_data['Purchase_History'] = customer_data['Purchase_History'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    print("Available Customer_IDs:", customer_data['Customer_ID'].unique())
    print("Customer Data Sample:\n", customer_data.head())
    print("Available Product_IDs:", product_data['Product_ID'].unique())
    print("Mapped Product IDs:", [int(str(pid).replace('P', '')) for pid in product_data['Product_ID']])
    print("Product Data Sample:\n", product_data.head())

    all_text = pd.concat([customer_data['Purchase_History'].apply(' '.join).fillna('') + ' ' + customer_data['Browsing_History'].apply(' '.join).fillna(''),
                          product_data['Category'].fillna('') + ' ' + product_data['Subcategory'].fillna('') + ' ' + product_data['Brand'].fillna('')])
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_text)

    user_agent = UserProfilingAgent(customer_data_path, vectorizer)
    product_agent = ProductAnalysisAgent(product_data_path, vectorizer)

    user_profiles = user_agent.get_user_profiles()
    product_features = product_agent.get_product_features()

    customer_ids = customer_data['Customer_ID'].tolist()
    # Replace the RecommendationAgent instantiation in main()
    rec_agent = RecommendationAgent(user_profiles, product_features, customer_ids, product_data, customer_data)

    user_id = 'C1001'
    top_k = 5
    recommendations = rec_agent.get_recommendations(user_id, top_k)

    print(f"Recommendations: {recommendations}")

    filtered_data = customer_data[customer_data['Customer_ID'] == user_id]
    if filtered_data.empty:
        print(f"Warning: No data found for User ID {user_id}. Using default values.")
        user_details = {
            'Customer_ID': user_id, 'Age': 'N/A', 'Gender': 'N/A', 'Location': 'N/A',
            'Browsing_History': [], 'Purchase_History': [], 'Customer_Segment': 'N/A',
            'Avg_Order_Value': 'N/A', 'Holiday': 'N/A', 'Season': 'N/A'
        }
        preferences = "No history available"
        username = "Unknown User"
    else:
        user_details = filtered_data.iloc[0].to_dict()
        preferences = ' '.join(filtered_data['Purchase_History'].iloc[0]) + " " + ' '.join(filtered_data['Browsing_History'].iloc[0])
        username = user_details.get('Username', f"User_{user_id}")
        print(f"User Details from Data: {user_details}")

    # Analyze preferences for color scheme and chart data
    purchase_words = user_details['Purchase_History'] if user_details['Purchase_History'] else []
    browsing_words = user_details['Browsing_History'] if user_details['Browsing_History'] else []
    all_words = purchase_words + browsing_words
    word_freq = Counter(all_words)
    top_categories = word_freq.most_common(5)
    color = '#4CAF50' if 'Books' in all_words else '#2196F3' if 'Fashion' in all_words else '#FF9800'

    recommended_product_ids = recommendations
    recommended_products = product_data[product_data['Product_ID'].str.replace('P', '').astype(int).isin(recommended_product_ids)].copy()

    if recommended_products.empty:
        print(f"Warning: No matching products found for recommendations {recommendations}. Using top {top_k} from product_data.")
        recommended_products = product_data.head(top_k).copy()

    recommended_products['Description'] = recommended_products.apply(
        lambda row: f"{row['Brand']} {row['Subcategory']} in {row['Category']}", axis=1
    )
    save_to_database(conn, user_id, preferences, recommended_products)

    c = conn.cursor()
    for product_id in recommended_product_ids:
        c.execute("INSERT INTO recommendations (user_id, product_id) VALUES (?, ?)", (user_id, product_id))
    conn.commit()

    # Generate HTML content with user details table and chart
    chart_data = {
        'labels': [cat[0] for cat in top_categories],
        'data': [cat[1] for cat in top_categories]
    }
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recommendations for {username}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f4;
                color: {color};
            }}
            h1 {{
                text-align: center;
                color: {color};
            }}
            .user-profile, .recommendations-table {{
                background-color: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            table {{
                width: 80%;
                margin: 20px auto;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: {color};
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            .recommended-badge {{
                background-color: #ff4444;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.8em;
            }}
            canvas {{
                max-width: 600px;
                margin: 20px auto;
                display: block;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Recommendations for {username} (ID: {user_id})</h1>
        <div class="user-profile">
            <h2>User Profile</h2>
            <table>
                <tr><th>Customer ID</th><td>{user_details['Customer_ID']}</td></tr>
                <tr><th>Age</th><td>{user_details['Age']}</td></tr>
                <tr><th>Gender</th><td>{user_details['Gender']}</td></tr>
                <tr><th>Location</th><td>{user_details['Location']}</td></tr>
                <tr><th>Browsing History</th><td>{', '.join(user_details['Browsing_History']) if user_details['Browsing_History'] else 'None'}</td></tr>
                <tr><th>Purchase History</th><td>{', '.join(user_details['Purchase_History']) if user_details['Purchase_History'] else 'None'}</td></tr>
                <tr><th>Customer Segment</th><td>{user_details['Customer_Segment']}</td></tr>
                <tr><th>Avg Order Value</th><td>${user_details['Avg_Order_Value']}</td></tr>
                <tr><th>Holiday</th><td>{user_details['Holiday']}</td></tr>
                <tr><th>Season</th><td>{user_details['Season']}</td></tr>
            </table>
        </div>
        <canvas id="interestChart"></canvas>
        <div class="recommendations-table">
            <h2>Recommended Products</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Product ID</th>
                    <th>Description</th>
                    <th>Price</th>
                    <th>Rating</th>
                </tr>
    """

    for i, (_, product) in enumerate(recommended_products.iterrows(), 1):
        product_id = int(str(product['Product_ID']).replace('P', ''))
        description = product['Description']
        price = product['Price'] if pd.notna(product['Price']) else 'N/A'
        rating = product['Product_Rating'] if pd.notna(product['Product_Rating']) else 'N/A'
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{product_id}</td>
                <td>{description} <span class="recommended-badge">Recommended for You</span></td>
                <td>${price}</td>
                <td>{rating}</td>
            </tr>
        """
        print(f"Recommended Product Check: ID={product_id}, Category={product['Category']}, Price=${price}, Rating={rating}")

    html_content += """
            </table>
            <p style="text-align: center;">Stored in SQLite Database for future use.</p>
        </div>
        <script>
            const ctx = document.getElementById('interestChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: """ + str(chart_data['labels']) + """,
                    datasets: [{
                        label: 'Interest Frequency',
                        data: """ + str(chart_data['data']) + """,
                        backgroundColor: 'rgba(76, 175, 80, 0.6)',
                        borderColor: 'rgba(76, 175, 80, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """

    with open('recommendations.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("HTML file 'recommendations.html' has been generated. Open it in a web browser to view the recommendations.")
    print("Expected Technical Output: Multiagent framework and SQLite Database for long term memory")
    print("----------------------------------------------------------------")

    conn.close()

if __name__ == "__main__":
    main()