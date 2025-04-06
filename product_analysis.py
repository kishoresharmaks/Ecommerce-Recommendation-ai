import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data

class ProductAnalysisAgent:
    def __init__(self, product_data_path, vectorizer):
        self.product_data = load_data(product_data_path)
        if self.product_data is None:
            raise ValueError("Product data could not be loaded.")
        self.vectorizer = vectorizer

    def generate_feature_vectors(self):
        if 'Category' not in self.product_data.columns or 'Subcategory' not in self.product_data.columns or 'Brand' not in self.product_data.columns:
            raise KeyError("Columns 'Category', 'Subcategory', and 'Brand' not found in product data.")
        feature_data = self.product_data['Category'].fillna('') + ' ' + \
                       self.product_data['Subcategory'].fillna('') + ' ' + \
                       self.product_data['Brand'].fillna('')
        feature_vectors = self.vectorizer.transform(feature_data)  # Use pre-fitted vectorizer
        return feature_vectors

    def compute_similarity_matrix(self):
        feature_vectors = self.generate_feature_vectors()
        similarity_matrix = cosine_similarity(feature_vectors)
        return similarity_matrix

    def get_product_features(self):
        feature_vectors = self.generate_feature_vectors()
        product_features = pd.DataFrame(feature_vectors.toarray(), 
                                      index=self.product_data['Product_ID'])
        return product_features

if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    agent = ProductAnalysisAgent('product_recommendation_data.csv', vectorizer)
    features = agent.get_product_features()
    similarity = agent.compute_similarity_matrix()
    print("Product Features Shape:", features.shape)
    print("Similarity Matrix Shape:", similarity.shape)