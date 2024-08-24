from fastapi import FastAPI, HTTPException, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can specify specific origins instead.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.). You can specify specific methods instead.
    allow_headers=["*"],  # Allows all headers. You can specify specific headers instead.
)


# Load the CSV file once when the application starts
try:
    df = pd.read_excel('Travel Data.xlsx')
    df.replace([np.inf, -np.inf, np.nan], None, inplace=True)
except Exception as e:
    raise RuntimeError(f"Failed to load CSV file: {e}")

# Combine relevant text columns into a single column for TF-IDF processing
def combine_columns(row):
    return ' '.join(row.astype(str))

df['combined_text'] = df.apply(combine_columns, axis=1)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

@app.get("/get-places/")
async def get_places(
    max_distance: float = Query(None, title="Maximum Distance", description="Maximum distance in kilometers"),
    max_duration: int = Query(None, title="Maximum Duration", description="Maximum duration in minutes"),
    max_cost: float = Query(None, title="Maximum Hotel Cost", description="Maximum hotel cost in thousands"),
    min_cost: float = Query(None, title="Minimum Hotel Cost", description="Minimum hotel cost in thousands"),
    rating: int = Query(None, title="Exact Rating", description="Exact rating"),
    description: str = Query(None, title="Description", description="Description to match with features")
):
    try:
        print(max_distance, max_duration, min_cost, max_cost, rating, description)
        filtered_df = df.copy()
        
        if max_distance is not None:
            filtered_df = filtered_df[filtered_df['distance (km)'] <= max_distance]
        
        if max_duration is not None:
            filtered_df = filtered_df[filtered_df['Duration (min)'] <= max_duration]

        if max_cost is not None:
            filtered_df = filtered_df[filtered_df['Maximum hotel cost (thousands/per night)'] <= max_cost]
        
        if min_cost is not None:
            filtered_df = filtered_df[filtered_df['Minimum hotel cost (thousands/per night)'] >= min_cost]

        if rating is not None:
            filtered_df = filtered_df[filtered_df['Ratting'] == rating]

        if description is not None and not filtered_df.empty:
            # Fit the TF-IDF model on the filtered combined text
            tfidf_matrix = vectorizer.fit_transform(filtered_df['combined_text'].fillna(""))
            # Compute the TF-IDF vector for the input description
            input_vector = vectorizer.transform([description])
            # Compute cosine similarity between the input vector and the filtered dataset
            similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
            filtered_df['Similarity Score'] = similarity_scores
            # Filter by similarity score if desired
            filtered_df = filtered_df[filtered_df['Similarity Score'] > 0]
            # Sort by similarity score
            filtered_df = filtered_df.sort_values(by='Similarity Score', ascending=False)

        return filtered_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
