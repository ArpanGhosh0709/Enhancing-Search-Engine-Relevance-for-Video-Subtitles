import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time
import torch 
import pickle

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DataFrame containing subtitles
df_subtitle = pd.read_csv(r"C:\Users\Arpan Ghosh\OneDrive\Desktop\Search Engine\final_subtitles.csv")

# Load the SentenceTransformer model with GPU support
model = SentenceTransformer("bert-base-nli-mean-tokens", device=device)
with open(r"C:\Users\Arpan Ghosh\OneDrive\Desktop\Search Engine\subtitle_embds.pkl", "rb") as file:
    subtitle_embds = pickle.load(file)
    
# Load or create the index
try:
    # Try to load the index if it exists
    index = faiss.read_index(r"C:\Users\Arpan Ghosh\OneDrive\Desktop\Search Engine\final_subtitles.csv")
except:
    # If the index doesn't exist, create a new one
    index = faiss.IndexFlatL2(subtitle_embds.shape[1])
    index.add(subtitle_embds)
    # Save the index
    faiss.write_index(index, r"C:\Users\Arpan Ghosh\OneDrive\Desktop\Search Engine")
    # Load the index again
    index = faiss.read_index(r"C:\Users\Arpan Ghosh\OneDrive\Desktop\Search Engine")

# Function to perform search
def search(query):
    # Encode the query using the SentenceTransformer model
    query_vector = model.encode([query])[0]  # Get the first (and only) element from the list
    
    # Debugging output: Print dimensions of query vector and index
    print("Query Vector Dimensions:", query_vector.shape)
    print("Index Dimensions:", index.d)
    
    # Check if the query_vector has the same dimensionality as the index
    if query_vector.shape[0] != index.d:
        print("Adjusting the dimensionality of the query vector.")
        query_vector = query_vector[:index.d]  # Adjust the dimensionality
        
    # Perform the search
    k = 5  # Number of nearest neighbors to retrieve
    distances, top_k_indices = index.search(query_vector.reshape(1, -1), k)
    
    # Retrieve the document names corresponding to the top_k indices
    top_k_ids = top_k_indices[0].tolist()
    document_names = [df_subtitle.loc[_id, 'name'] for _id in top_k_ids]
    
    return document_names

# Streamlit UI
def main():
    st.title("Subtitle Search Engine")
    
    # Input field for the search query
    query = st.text_input("Enter your search query:")
    
    # Search button
    if st.button("Search"):
        if query:
            # Perform the search
            results = search(query)
            
            # Display the search results
            st.write("Search Results:")
            for idx, result in enumerate(results, start=1):
                st.write(f"{idx}. {result}")
        else:
            st.write("Please enter a search query.")

if __name__ == "_main_":
    main()