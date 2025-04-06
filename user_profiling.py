import pandas as pd
from utils import load_data

class UserProfilingAgent:
    def __init__(self, customer_data_path, vectorizer):
        self.customer_data = load_data(customer_data_path)
        if self.customer_data is None:
            raise ValueError("Customer data could not be loaded.")
        self.vectorizer = vectorizer

    def generate_preference_vectors(self):
        if 'Purchase_History' not in self.customer_data.columns or 'Browsing_History' not in self.customer_data.columns:
            raise KeyError("Columns 'Purchase_History' and 'Browsing_History' not found in customer data.")
        preference_data = self.customer_data['Purchase_History'].fillna('') + ' ' + \
                         self.customer_data['Browsing_History'].fillna('')
        preference_vectors = self.vectorizer.transform(preference_data)  # Use pre-fitted vectorizer
        return preference_vectors

    def get_user_profiles(self):
        preference_vectors = self.generate_preference_vectors()
        user_profiles = pd.DataFrame(preference_vectors.toarray(), 
                                   index=self.customer_data['Customer_ID'])
        return user_profiles

if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    agent = UserProfilingAgent('customer_data_collection.csv', vectorizer)
    profiles = agent.get_user_profiles()
    print("User Profiles Shape:", profiles.shape)