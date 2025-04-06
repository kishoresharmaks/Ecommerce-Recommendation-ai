import numpy as np
from collections import Counter
import ast 


class RecommendationAgent:
    def __init__(self, user_profiles, product_features, customer_ids, product_data, customer_data):
        self.user_profiles = user_profiles
        self.product_features = product_features
        self.customer_ids = customer_ids
        self.product_data = product_data
        self.customer_data = customer_data  # Add customer data for dynamic filtering
        self.scores = np.dot(user_profiles, product_features.T)

    def get_recommendations(self, user_id, top_k):
        if user_id not in self.customer_ids:
            raise ValueError(f"User ID {user_id} not found in customer IDs.")
        
        user_index = self.customer_ids.index(user_id)
        user_scores = self.scores[user_index]
        
        # Dynamic filters based on user data
        user_details = self.customer_data[self.customer_data['Customer_ID'] == user_id].iloc[0]
        allowed_categories = ['Books', 'Fitness', 'Fashion']  # From Browsing_History and Purchase_History
        min_price = float(user_details['Avg_Order_Value']) - 500
        max_price = float(user_details['Avg_Order_Value']) + 500
        purchase_history = user_details['Purchase_History']

        # Filter products
        category_mask = self.product_data['Category'].isin(allowed_categories)
        price_mask = self.product_data['Price'].between(min_price, max_price)
        rating_mask = self.product_data['Product_Rating'] >= 3.5
        valid_indices = self.product_data.index[category_mask & price_mask & rating_mask].tolist()

        if not valid_indices:
            print(f"Warning: No valid recommendations found for {user_id}. Using top scores without strict filters.")
            recommended_indices = np.argsort(user_scores)[::-1][:top_k]
        else:
            # Compute weighted scores
            valid_scores = []
            brand_counts = Counter()
            for i in valid_indices:
                if i < len(user_scores):
                    base_score = user_scores[i]
                    product = self.product_data.iloc[i]
                    # Boost score for Purchase_History matches in Similar_Product_List
                    boost = 0.5 if any(item in purchase_history for item in ast.literal_eval(product['Similar_Product_List'])) else 0
                    # Incorporate Probability_of_Recommendation and Average_Rating_of_Similar_Products
                    weighted_score = base_score * (1 + boost + float(product['Probability_of_Recommendation']) * 0.5 + float(product['Average_Rating_of_Similar_Products']) * 0.1)
                    brand = product['Brand']
                    if brand_counts[brand] < 2:  # Limit to 2 per brand
                        valid_scores.append((weighted_score, i, brand))
                        brand_counts[brand] += 1

            if not valid_scores:
                print(f"Warning: No valid recommendations after diversity filter for {user_id}. Using top scores.")
                recommended_indices = np.argsort(user_scores)[::-1][:top_k]
            else:
                valid_scores.sort(reverse=True)
                recommended_indices = [score[1] for score in valid_scores[:top_k]]

        # Map indices to Product_IDs
        recommended_product_ids = [int(self.product_data.iloc[i]['Product_ID'].replace('P', '')) for i in recommended_indices if i < len(self.product_data)]
        return recommended_product_ids[:top_k]

    def get_recommendation_details(self, user_id, top_k):
        """Helper method to return details of recommendations for debugging."""
        recommendations = self.get_recommendations(user_id, top_k)
        details = self.product_data[self.product_data['Product_ID'].str.replace('P', '').astype(int).isin(recommendations)].copy()
        details['Description'] = details.apply(
            lambda row: f"{row['Brand']} {row['Subcategory']} in {row['Category']}", axis=1
        )
        return details