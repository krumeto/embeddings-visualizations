import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer

from streamlit_plotly_events import plotly_events
from umap import UMAP

# -------------------------------------------------------------------
# Caching helpers
# -------------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str):
    """Load (and cache) a SentenceTransformer model."""
    return SentenceTransformer(model_name)

@st.cache_data
def compute_embeddings(texts, model_name: str):
    """Compute (and cache) embeddings for a list of texts given a model."""
    model = load_model(model_name)
    if model_name == "intfloat/multilingual-e5-small":
        texts = ["passage: " + t for t in texts]
    return model.encode(texts, show_progress_bar=True, trust_remote_code=True)

@st.cache_data
def reduce_to_2d(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Reduce embeddings to 2D space via UMAP (cached)."""
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state
    )
    coords_2d = umap_model.fit_transform(embeddings)
    return coords_2d

# -------------------------------------------------------------------
# Sidebar components
# -------------------------------------------------------------------
st.sidebar.title("Configuration")

# Let the user select a pre-trained Sentence-Transformers model
model_choices = [
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "intfloat/multilingual-e5-small", # passage: 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # very slow
    "Snowflake/snowflake-arctic-embed-m-v2.0"
    
]
model_name = st.sidebar.selectbox("Select Sentence-Transformers model:", model_choices)
if model_name == "Snowflake/snowflake-arctic-embed-m-v2.0":
    st.write(f"Warning! {model_name} is a very big model and runs slowly on a CPU. This might take a long time!")
    
if model_name == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
    st.write(f"Warning! {model_name} is a relatively big model and runs slowly on a CPU. This might take a long time!")
    
# Allow the user to upload a CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV with a 'text' column", type=["csv"])

# UMAP parameters (optional, you can hide these if you prefer)
st.sidebar.subheader("UMAP Parameters")
n_neighbors = st.sidebar.slider("n_neighbors", 2, 50, 15, 1)
min_dist = st.sidebar.slider("min_dist", 0.0, 0.99, 0.1, 0.01)

# Let the user input a search term
search_term = st.sidebar.text_input("Enter search term for highlighting (optional)")

st.title("Document Embedding Visualizer (UMAP Edition)")

# -------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------
if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    if df.shape[0] > 500:
        st.write("This is a relatively big dataset and embedding it might take awhile.")
        
    # Check for the required column
    if "text" not in df.columns:
        st.error("Error: The uploaded CSV does not contain a 'text' column.")
    else:
        # Remove rows that might have NaN or empty text
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]

        # -------------------------------------------------------------------
        # 1. Compute the embeddings with SentenceTransformer
        # -------------------------------------------------------------------
        with st.spinner(f"Computing embeddings using {model_name}..."):
            embeddings = compute_embeddings(df["text"].tolist(), model_name)

        # -------------------------------------------------------------------
        # 2. Reduce dimensionality to 2D (using UMAP)
        # -------------------------------------------------------------------
        with st.spinner("Reducing embeddings to 2D space with UMAP..."):
            coords_2d = reduce_to_2d(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
        df["x"] = coords_2d[:, 0]
        df["y"] = coords_2d[:, 1]

        # -------------------------------------------------------------------
        # 3. Handle highlighting based on search term
        # -------------------------------------------------------------------
        if search_term:
            # We consider a document highlighted if it contains the search term (case-insensitive)
            df["highlight"] = df["text"].apply(
                lambda x: "highlight" if search_term.lower() in x.lower() else "normal"
            )
        else:
            df["highlight"] = "normal"

        # -------------------------------------------------------------------
        # 4. Build Plotly scatter plot
        # -------------------------------------------------------------------
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="highlight",  # color by "highlight" or "normal"
            hover_data={"text": True, "x": False, "y": False, "highlight": False},
            width=800,
            height=600
        )
        fig.update_traces(mode="markers")

        st.subheader("2D Projection of Documents (UMAP)")
        st.markdown(
            "Use **box select** or **lasso select** in the toolbar (top-right of the chart) to select points."
        )

        # -------------------------------------------------------------------
        # 5. Display the chart and capture selected points
        # -------------------------------------------------------------------
        selected_points = plotly_events(
            fig,
            select_event=True,   # capture box/lasso select
            override_height=600,
            override_width="100%"
        )

        # -------------------------------------------------------------------
        # 6. Show selected documents in sidebar
        # -------------------------------------------------------------------
        st.sidebar.subheader("Selected Documents")
        if selected_points:
            indices = [p["pointIndex"] for p in selected_points]
            for i in indices:
                text_snippet = df["text"].iloc[i][:100]
                st.sidebar.write(f"**Doc {i}**: {text_snippet}...")
        else:
            st.sidebar.info("No documents selected yet.")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
