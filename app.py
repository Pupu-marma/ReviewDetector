# ============================================================
# REVIEWDETECTOR | FAKE REVIEW DETECTION SYSTEM
# Complete Streamlit Application
# ============================================================

import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ReviewDetector | AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROJECT PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "fake_review_dataset_5000.csv"
MODEL_PATH = BASE_DIR / "model_5000_reviews.pkl"


# =====================================================
# 3. MODERN MINIMAL BLUE CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F9FC;
    color: #172B4D;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #123B70;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    padding: 12px 0 5px 0;
}

.sidebar-subtitle {
    color: #C8D8EE;
    font-size: 11px;
    letter-spacing: 1px;
}

/* HEADER */

.brand {
    font-size: 29px;
    font-weight: 800;
    color: #173B67;
}

.brand span {
    color: #2563EB;
}

.tagline {
    color: #718096;
    font-size: 13px;
    margin-top: 3px;
}

/* HERO */

.hero {
    background: linear-gradient(120deg, #123B70, #2563EB);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin: 20px 0;
}

.hero h1 {
    color: white;
    font-size: 27px !important;
    font-weight: 800;
}

.hero p {
    color: #E5EEFF;
    font-size: 14px;
}

/* SECTION */

.section-title {
    color: #173B67;
    font-size: 21px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #718096;
    font-size: 13px;
    margin-bottom: 18px;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(20, 50, 90, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #718096;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #173B67;
    font-size: 25px !important;
    font-weight: 800;
}

/* ALGORITHM CARD */

.algorithm-card {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    padding: 18px;
    min-height: 110px;
}

.algorithm-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

.algorithm-value {
    color: #2563EB;
    font-size: 22px;
    font-weight: 800;
    margin-top: 8px;
}

/* RESULT CARD */

.result-card {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    padding: 22px;
    margin: 15px 0;
}

.fake-result {
    color: #DC3545;
    font-size: 23px;
    font-weight: 800;
}

.original-result {
    color: #15966A;
    font-size: 23px;
    font-weight: 800;
}

/* INPUT */

.stTextArea textarea {
    border: 1px solid #DCE4EF;
    border-radius: 9px;
    background: #FFFFFF;
}

/* BUTTON */

.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: 1px solid #2563EB;
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #1D4ED8;
    color: #FFFFFF;
    border: 1px solid #1D4ED8;
}

/* DOWNLOAD */

.stDownloadButton > button {
    background: #123B70;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #E5EAF2;
    border-radius: 10px;
}

/* SELECTBOX */

div[data-baseweb="select"] {
    border-radius: 8px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    padding: 30px 0 10px;
}

hr {
    border-color: #E5EAF2;
}

@media (max-width: 768px) {
    .brand {
        font-size: 23px;
    }

    .hero {
        padding: 20px;
    }

    .hero h1 {
        font-size: 21px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("Linear SVM Model could not be loaded.")
    st.warning("Check models/model_5000_reviews.pkl")
    st.exception(e)
    st.stop()


# =====================================================
# 5. LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "category",
        "rating",
        "label",
        "text_"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing dataset column: {col}")

    df = df.copy()

    df["text_"] = df["text_"].fillna("").astype(str)

    df["label"] = df["label"].astype(str).str.strip()

    df["category"] = (
        df["category"].fillna("Unknown").astype(str)
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    return df


try:
    amazon_df = load_dataset()

except Exception as e:
    st.error("Dataset could not be loaded.")
    st.warning(
        "Check dataset/fake_review_dataset_5000.csv"
    )
    st.exception(e)
    st.stop()


# =====================================================
# 6. DATASET STATISTICS
# =====================================================

total_reviews = len(amazon_df)

fake_reviews = int(
    (amazon_df["label"] == "CG").sum()
)

original_reviews = int(
    (amazon_df["label"] == "OR").sum()
)

total_categories = amazon_df["category"].nunique()

avg_rating = amazon_df["rating"].mean()


# =====================================================
# 7. TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =====================================================
# 8. AUTO DETECT REVIEW COLUMN
# =====================================================

def detect_review_column(df):

    aliases = [
        "text_",
        "review",
        "review_text",
        "review text",
        "text",
        "content",
        "review_body",
        "review body",
        "body",
        "reviews",
        "review_content",
        "review content"
    ]

    normalized_columns = {}

    for col in df.columns:
        normalized = str(col).strip().lower()
        normalized_columns[normalized] = col

    for alias in aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    for normalized, original in normalized_columns.items():
        if "review" in normalized:
            return original

    return None


# =====================================================
# 9. PREDICTION FUNCTION
# =====================================================

def predict_review(review):

    cleaned_review = clean_text(review)

    if not cleaned_review:
        return "EMPTY", "Empty Review"

    prediction = model.predict([cleaned_review])[0]

    if prediction == "CG":
        return "CG", "Fake Review"

    elif prediction == "OR":
        return "OR", "Original Review"

    return "UNKNOWN", "Unknown"


# =====================================================
# 10. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔍 ReviewDetector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI POWERED REVIEW ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Single Review",
            "Bulk Analysis",
            "Dataset Analysis",
            "Model Evaluation",
            "About Project"
        ]
    )

    st.divider()

    st.caption("Machine Learning Project")
    st.caption("Computer Science & Engineering")


# =====================================================
# 11. HEADER
# =====================================================

left, right = st.columns([4, 1])

with left:

    st.markdown(
        '<div class="brand">Review<span>Detector</span> 🔍</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">'
        'Machine Learning Based Fake Review Detection System'
        '</div>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        "<div style='text-align:right;padding-top:10px;"
        "color:#2563EB;font-weight:700;font-size:12px;'>"
        "CSE PROJECT</div>",
        unsafe_allow_html=True
    )


# =====================================================
# 12. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Discover the Truth Behind Every Review</h1>
        <p>
        Harness the power of Machine Learning to detect fake
        product reviews and build trust in online shopping.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Size", f"{total_reviews:,}")
    c2.metric("Fake Reviews", f"{fake_reviews:,}")
    c3.metric("Original Reviews", f"{original_reviews:,}")

    with c4:
        st.markdown("""
        <div class="algorithm-card">
            <div class="algorithm-label">Algorithm</div>
            <div class="algorithm-value">Linear SVM</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">System Features</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 📝 Single Detection")
        st.write("Analyze individual product reviews.")

    with b:
        st.markdown("### 📊 Bulk Analysis")
        st.write("Analyze multiple reviews using CSV upload.")

    with c:
        st.markdown("### 📈 Visual Analytics")
        st.write("Explore statistics and model performance.")

    st.markdown(
        '<div class="section-title">Dataset Snapshot</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    d1.metric("Product Categories", total_categories)

    d2.metric(
        "Average Rating",
        f"{avg_rating:.2f}/5"
        if pd.notna(avg_rating) else "N/A"
    )

    d3.metric(
        "Dataset Balance",
        "Balanced"
        if fake_reviews == original_reviews
        else "Imbalanced"
    )


# =====================================================
# 13. SINGLE REVIEW
# =====================================================

elif page == "Single Review":

    st.markdown(
        '<div class="section-title">'
        'Single Review Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter a product review to analyze its authenticity.'
        '</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "Product Review",
        placeholder="Type or paste your review here...",
        height=180
    )

    if st.button("🔎 Analyze Review", use_container_width=True):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:

            prediction, label = predict_review(review)

            if prediction == "CG":

                st.markdown("""
                <div class="result-card">
                    <div class="fake-result">
                        ⚠️ Fake Review Detected
                    </div>
                    <p style="color:#718096;">
                    The model classified this review as Fake.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            elif prediction == "OR":

                st.markdown("""
                <div class="result-card">
                    <div class="original-result">
                        ✓ Original Review
                    </div>
                    <p style="color:#718096;">
                    The model classified this review as Original.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("Review could not be classified.")


# =====================================================
# 14. BULK REVIEW ANALYSIS
# =====================================================

elif page == "Bulk Analysis":

    st.markdown(
        '<div class="section-title">Bulk Review Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a CSV file containing product reviews.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Supported review columns: text_, review, review_text, "
        "text, content, review_body and other similar names."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="bulk_csv"
    )

    if uploaded_file is not None:

        try:

            try:
                df = pd.read_csv(
                    uploaded_file,
                    encoding="utf-8"
                )

            except UnicodeDecodeError:
                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="latin-1"
                )

            if df.empty:

                st.warning("The uploaded CSV file is empty.")

            else:

                st.success("CSV file uploaded successfully!")

                st.markdown("### Uploaded Dataset Preview")

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                review_column = detect_review_column(df)

                if review_column is None:

                    st.error("No review text column was detected.")

                    st.write("Available CSV columns:")
                    st.write(list(df.columns))

                    st.warning(
                        "Rename your review text column to "
                        "'review' or 'text_' and upload again."
                    )

                else:

                    st.success(
                        f"Detected Review Column: {review_column}"
                    )

                    if st.button(
                        "🚀 Analyze All Reviews",
                        use_container_width=True
                    ):

                        result_df = df.copy()

                        reviews = (
                            result_df[review_column]
                            .fillna("")
                            .astype(str)
                        )

                        valid_mask = reviews.str.strip() != ""

                        result_df["Predicted Label"] = "Empty Review"

                        if valid_mask.any():

                            valid_reviews = reviews[
                                valid_mask
                            ].tolist()

                            cleaned_reviews = [
                                clean_text(review)
                                for review in valid_reviews
                            ]

                            predictions = model.predict(
                                cleaned_reviews
                            )

                            readable_labels = [
                                "Fake Review"
                                if prediction == "CG"
                                else "Original Review"
                                if prediction == "OR"
                                else "Unknown"
                                for prediction in predictions
                            ]

                            result_df.loc[
                                valid_mask,
                                "Predicted Label"
                            ] = readable_labels

                        st.session_state["bulk_results"] = result_df
                        st.session_state["bulk_source"] = uploaded_file.name

                    if (
                        "bulk_results" in st.session_state
                        and st.session_state.get("bulk_source")
                        == uploaded_file.name
                    ):

                        result_df = st.session_state["bulk_results"]

                        fake_count = int(
                            (
                                result_df["Predicted Label"]
                                == "Fake Review"
                            ).sum()
                        )

                        original_count = int(
                            (
                                result_df["Predicted Label"]
                                == "Original Review"
                            ).sum()
                        )

                        empty_count = int(
                            (
                                result_df["Predicted Label"]
                                == "Empty Review"
                            ).sum()
                        )

                        unknown_count = int(
                            (
                                result_df["Predicted Label"]
                                == "Unknown"
                            ).sum()
                        )

                        total = len(result_df)

                        analyzed = fake_count + original_count

                        fake_rate = (
                            fake_count / analyzed * 100
                            if analyzed else 0
                        )

                        original_rate = (
                            original_count / analyzed * 100
                            if analyzed else 0
                        )

                        st.success(
                            "Review analysis completed successfully!"
                        )

                        st.markdown(
                            '<div class="section-title">'
                            'Analysis Summary'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric("Total Reviews", f"{total:,}")
                        c2.metric("Fake Reviews", f"{fake_count:,}")
                        c3.metric("Original Reviews", f"{original_count:,}")
                        c4.metric(
                            "Empty / Unknown",
                            empty_count + unknown_count
                        )

                        c1, c2 = st.columns(2)

                        c1.metric("Fake Rate", f"{fake_rate:.2f}%")
                        c2.metric(
                            "Original Rate",
                            f"{original_rate:.2f}%"
                        )

                        st.markdown(
                            '<div class="section-title">'
                            'Visual Analytics'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        chart_data = pd.DataFrame({
                            "Review Type": [
                                "Fake Review",
                                "Original Review"
                            ],
                            "Count": [
                                fake_count,
                                original_count
                            ]
                        })

                        left, right = st.columns(2)

                        with left:

                            st.markdown("#### Review Comparison")

                            fig_bar = px.bar(
                                chart_data,
                                x="Review Type",
                                y="Count",
                                color="Review Type",
                                text="Count",
                                color_discrete_map={
                                    "Fake Review": "#DC3545",
                                    "Original Review": "#15966A"
                                }
                            )

                            fig_bar.update_layout(
                                showlegend=False,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                font=dict(
                                    family="Inter",
                                    color="#173B67"
                                )
                            )

                            st.plotly_chart(
                                fig_bar,
                                use_container_width=True
                            )

                        with right:

                            st.markdown("#### Review Distribution")

                            if analyzed > 0:

                                fig_pie = px.pie(
                                    chart_data,
                                    names="Review Type",
                                    values="Count",
                                    hole=0.55,
                                    color="Review Type",
                                    color_discrete_map={
                                        "Fake Review": "#DC3545",
                                        "Original Review": "#15966A"
                                    }
                                )

                                fig_pie.update_layout(
                                    paper_bgcolor="white",
                                    font=dict(
                                        family="Inter",
                                        color="#173B67"
                                    )
                                )

                                st.plotly_chart(
                                    fig_pie,
                                    use_container_width=True
                                )

                            else:
                                st.info(
                                    "No valid reviews to display."
                                )

                        st.markdown(
                            '<div class="section-title">'
                            'Detailed Prediction Results'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            "⬇️ Download Analysis Results",
                            data=csv_data,
                            file_name="review_detection_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                "An error occurred while processing the CSV."
            )

            st.exception(e)


# =====================================================
# 15. DATASET ANALYSIS
# =====================================================

elif page == "Dataset Analysis":

    st.markdown(
        '<div class="section-title">Amazon Dataset Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore review distribution, product categories and dataset patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total_reviews:,}")
    c2.metric("Fake Reviews", f"{fake_reviews:,}")
    c3.metric("Original Reviews", f"{original_reviews:,}")
    c4.metric("Categories", total_categories)

    # REVIEW LABEL DISTRIBUTION

    st.markdown("---")
    st.markdown("### Review Label Distribution")

    label_counts = (
        amazon_df["label"]
        .value_counts()
        .rename_axis("Label")
        .reset_index(name="Count")
    )

    label_counts["Review Type"] = (
        label_counts["Label"]
        .map({
            "CG": "Fake Review",
            "OR": "Original Review"
        })
        .fillna("Other")
    )

    left, right = st.columns(2)

    with left:

        fig_label = px.bar(
            label_counts,
            x="Review Type",
            y="Count",
            color="Review Type",
            text="Count",
            title="Fake vs Original Reviews",
            color_discrete_map={
                "Fake Review": "#DC3545",
                "Original Review": "#15966A"
            }
        )

        fig_label.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_label,
            use_container_width=True
        )

    with right:

        fig_pie = px.pie(
            label_counts,
            names="Review Type",
            values="Count",
            hole=0.5,
            title="Review Distribution",
            color="Review Type",
            color_discrete_map={
                "Fake Review": "#DC3545",
                "Original Review": "#15966A"
            }
        )

        fig_pie.update_layout(
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    # PRODUCT CATEGORY ANALYSIS

    st.markdown("---")
    st.markdown("### Product Category Analysis")

    category_counts = (
        amazon_df["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Review Count"
    ]

    fig_category = px.bar(
        category_counts,
        x="Category",
        y="Review Count",
        text="Review Count",
        title="Reviews Across Product Categories"
    )

    fig_category.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    # RATING ANALYSIS

    st.markdown("---")
    st.markdown("### Rating Analysis")

    rating_counts = (
        amazon_df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_counts.columns = [
        "Rating",
        "Count"
    ]

    fig_rating = px.bar(
        rating_counts,
        x="Rating",
        y="Count",
        text="Count",
        title="Product Rating Distribution"
    )

    fig_rating.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )

    # REVIEW LENGTH ANALYSIS

    st.markdown("---")
    st.markdown("### Review Length Analysis")

    review_lengths = (
        amazon_df["text_"]
        .str.split()
        .str.len()
    )

    length_df = pd.DataFrame({
        "Review Length": review_lengths,
        "Label": amazon_df["label"].map({
            "CG": "Fake Review",
            "OR": "Original Review"
        })
    })

    fig_length = px.histogram(
        length_df,
        x="Review Length",
        color="Label",
        barmode="overlay",
        opacity=0.7,
        nbins=40,
        title="Review Length Distribution",
        color_discrete_map={
            "Fake Review": "#DC3545",
            "Original Review": "#15966A"
        }
    )

    fig_length.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

    # PCA VISUALIZATION

    st.markdown("---")
    st.markdown("### PCA Visualization")

    st.write(
        "Principal Component Analysis (PCA) reduces "
        "high-dimensional TF-IDF features into two dimensions "
        "for visualizing review patterns."
    )

    if st.button(
        "Generate PCA Visualization",
        key="generate_pca"
    ):

        try:

            with st.spinner("Generating PCA visualization..."):

                pca_df = amazon_df[
                    amazon_df["label"].isin(["CG", "OR"])
                ].copy()

                if len(pca_df) > 2000:
                    pca_df = pca_df.sample(
                        n=2000,
                        random_state=42
                    )

                vectorizer = TfidfVectorizer(
                    max_features=800,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2
                )

                tfidf_matrix = vectorizer.fit_transform(
                    pca_df["text_"].apply(clean_text)
                )

                if tfidf_matrix.shape[1] < 2:

                    st.warning(
                        "Not enough text features for PCA."
                    )

                else:

                    dense_matrix = tfidf_matrix.toarray()

                    pca = PCA(
                        n_components=2,
                        random_state=42
                    )

                    pca_result = pca.fit_transform(
                        dense_matrix
                    )

                    pca_plot_df = pd.DataFrame({
                        "Principal Component 1": pca_result[:, 0],
                        "Principal Component 2": pca_result[:, 1],
                        "Review Type": pca_df["label"].map({
                            "CG": "Fake Review",
                            "OR": "Original Review"
                        }).values
                    })

                    variance = (
                        pca.explained_variance_ratio_ * 100
                    )

                    fig_pca = px.scatter(
                        pca_plot_df,
                        x="Principal Component 1",
                        y="Principal Component 2",
                        color="Review Type",
                        color_discrete_map={
                            "Fake Review": "#DC3545",
                            "Original Review": "#15966A"
                        },
                        opacity=0.65,
                        title="PCA: Fake vs Original Reviews"
                    )

                    fig_pca.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=550
                    )

                    st.plotly_chart(
                        fig_pca,
                        use_container_width=True
                    )

                    v1, v2 = st.columns(2)

                    v1.metric(
                        "PC1 Explained Variance",
                        f"{variance[0]:.2f}%"
                    )

                    v2.metric(
                        "PC2 Explained Variance",
                        f"{variance[1]:.2f}%"
                    )

                    st.caption(
                        "PCA is used here for visualization. "
                        "It does not retrain or replace the Linear SVM model."
                    )

        except Exception as e:

            st.error("PCA visualization could not be generated.")
            st.exception(e)

    # WORD CLOUD

    st.markdown("---")
    st.markdown("### Word Cloud Visualization")

    st.write(
        "Word clouds display frequently occurring words "
        "in Fake Reviews and Original Reviews."
    )

    if not WORDCLOUD_AVAILABLE:

        st.warning(
            "WordCloud package is not installed. "
            "Install using: pip install wordcloud matplotlib"
        )

    else:

        if st.button(
            "Generate Word Clouds",
            key="generate_wordcloud"
        ):

            try:

                fake_text = " ".join(
                    amazon_df.loc[
                        amazon_df["label"] == "CG",
                        "text_"
                    ].astype(str)
                )

                original_text = " ".join(
                    amazon_df.loc[
                        amazon_df["label"] == "OR",
                        "text_"
                    ].astype(str)
                )

                fake_text = clean_text(fake_text)
                original_text = clean_text(original_text)

                if not fake_text.strip() or not original_text.strip():

                    st.warning(
                        "Not enough review text to generate word clouds."
                    )

                else:

                    left, right = st.columns(2)

                    with left:

                        st.markdown("#### Fake Review Word Cloud")

                        fake_wc = WordCloud(
                            width=900,
                            height=450,
                            background_color="white",
                            colormap="Reds",
                            max_words=100,
                            collocations=False
                        ).generate(fake_text)

                        fig_fake, ax_fake = plt.subplots(
                            figsize=(10, 5)
                        )

                        ax_fake.imshow(
                            fake_wc,
                            interpolation="bilinear"
                        )

                        ax_fake.axis("off")

                        st.pyplot(
                            fig_fake,
                            use_container_width=True
                        )

                        plt.close(fig_fake)

                    with right:

                        st.markdown(
                            "#### Original Review Word Cloud"
                        )

                        original_wc = WordCloud(
                            width=900,
                            height=450,
                            background_color="white",
                            colormap="Greens",
                            max_words=100,
                            collocations=False
                        ).generate(original_text)

                        fig_original, ax_original = plt.subplots(
                            figsize=(10, 5)
                        )

                        ax_original.imshow(
                            original_wc,
                            interpolation="bilinear"
                        )

                        ax_original.axis("off")

                        st.pyplot(
                            fig_original,
                            use_container_width=True
                        )

                        plt.close(fig_original)

            except Exception as e:

                st.error("Word Cloud could not be generated.")
                st.exception(e)

    # CORRELATION HEATMAP

    st.markdown("---")
    st.markdown("### Correlation Heatmap")

    st.write(
        "The correlation heatmap shows relationships among "
        "numerical dataset features."
    )

    if st.button(
        "Generate Correlation Heatmap",
        key="generate_correlation"
    ):

        try:

            correlation_df = pd.DataFrame()

            correlation_df["Rating"] = amazon_df["rating"]

            correlation_df["Review Length"] = (
                amazon_df["text_"].str.len()
            )

            correlation_df["Word Count"] = (
                amazon_df["text_"].str.split().str.len()
            )

            correlation_df["Label"] = (
                amazon_df["label"].map({
                    "CG": 1,
                    "OR": 0
                })
            )

            correlation_df = correlation_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            correlation_matrix = correlation_df.corr(
                method="pearson"
            )

            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title="Correlation Heatmap of Dataset Features",
                labels={"color": "Correlation"}
            )

            fig_corr.update_layout(
                height=550,
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

            st.caption(
                "Label encoding: CG (Fake Review) = 1, "
                "OR (Original Review) = 0. "
                "Correlation indicates association, not causation."
            )

        except Exception as e:

            st.error(
                "Correlation heatmap could not be generated."
            )

            st.exception(e)

    # DATASET PREVIEW

    st.markdown("---")
    st.markdown("### Dataset Preview")

    st.dataframe(
        amazon_df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 16. MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Performance comparison of three Machine Learning algorithms.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Evaluation Results: 5,000 Amazon Reviews | "
        "4,000 Training Reviews | 1,000 Testing Reviews"
    )

    # MODEL PERFORMANCE DATA

    algorithm_df = pd.DataFrame({

        "Algorithm": [
            "Logistic Regression",
            "Linear SVM",
            "Naive Bayes"
        ],

        "CV Accuracy": [
            88.15,
            88.25,
            85.82
        ],

        "CV Std": [
            0.90,
            0.77,
            1.42
        ],

        "Test Accuracy": [
            88.60,
            88.80,
            83.70
        ],

        "Correct Predictions": [
            886,
            888,
            837
        ],

        "Wrong Predictions": [
            114,
            112,
            163
        ]

    })

    st.markdown("---")
    st.markdown("### Overall Model Performance")

    p1, p2, p3 = st.columns(3)

    p1.metric("Logistic Regression", "88.60%")
    p2.metric("Linear SVM", "88.80%")
    p3.metric("Naive Bayes", "83.70%")

    st.dataframe(
        algorithm_df.style.format({
            "CV Accuracy": "{:.2f}%",
            "CV Std": "{:.2f}%",
            "Test Accuracy": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # CONFUSION MATRIX DATA

    confusion_matrices = {

        "Logistic Regression": np.array([
            [450, 50],
            [64, 436]
        ]),

        "Linear SVM": np.array([
            [449, 51],
            [61, 439]
        ]),

        "Naive Bayes": np.array([
            [418, 82],
            [81, 419]
        ])

    }

    # CONFUSION MATRIX HEATMAPS

    st.markdown("---")

    st.markdown(
        "### Confusion Matrix Heatmap Comparison"
    )

    st.caption(
        "Rows represent Actual Labels and columns represent "
        "Predicted Labels."
    )

    model_names = [
        "Logistic Regression",
        "Linear SVM",
        "Naive Bayes"
    ]

    for model_name in model_names:

        cm = confusion_matrices[model_name]

        st.markdown(
            f"#### {model_name}"
        )

        fig_heatmap = px.imshow(
            cm,
            x=["Fake (CG)", "Original (OR)"],
            y=["Fake (CG)", "Original (OR)"],
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="equal",
            labels={
                "x": "Predicted Label",
                "y": "Actual Label",
                "color": "Reviews"
            }
        )

        fig_heatmap.update_layout(
            height=430,
            margin=dict(
                l=60,
                r=40,
                t=30,
                b=60
            ),
            font=dict(
                size=13
            ),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        fig_heatmap.update_xaxes(
            side="bottom"
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

        accuracy = (
            np.trace(cm) / np.sum(cm)
        ) * 100

        st.metric(
            "Test Accuracy",
            f"{accuracy:.2f}%"
        )

        st.markdown("---")


    # CONFUSION MATRIX INTERPRETATION

    st.markdown("---")
    st.markdown("### Confusion Matrix Interpretation")

    interpretation = pd.DataFrame({

        "Algorithm": [
            "Logistic Regression",
            "Linear SVM",
            "Naive Bayes"
        ],

        "Correct Fake": [
            450,
            449,
            418
        ],

        "Incorrect Fake": [
            50,
            51,
            82
        ],

        "Incorrect Original": [
            64,
            61,
            81
        ],

        "Correct Original": [
            436,
            439,
            419
        ]

    })

    st.dataframe(
        interpretation,
        use_container_width=True,
        hide_index=True
    )

    # ACCURACY COMPARISON

    st.markdown("---")
    st.markdown("### Algorithm Accuracy Comparison")

    fig_algorithm = px.bar(
        algorithm_df,
        x="Algorithm",
        y="Test Accuracy",
        color="Algorithm",
        text="Test Accuracy",
        color_discrete_map={
            "Logistic Regression": "#2563EB",
            "Linear SVM": "#15966A",
            "Naive Bayes": "#E8A44B"
        }
    )

    fig_algorithm.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=430
    )

    fig_algorithm.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_algorithm,
        use_container_width=True
    )

    # CLASSIFICATION REPORT

    st.markdown("---")
    st.markdown("### Detailed Classification Report")

    selected_model = st.selectbox(
        "Select Algorithm",
        [
            "Logistic Regression",
            "Linear SVM",
            "Naive Bayes"
        ]
    )

    cm = confusion_matrices[selected_model]

    y_true = (
        ["CG"] * 500 +
        ["OR"] * 500
    )

    y_pred = (
        ["CG"] * cm[0, 0] +
        ["OR"] * cm[0, 1] +
        ["CG"] * cm[1, 0] +
        ["OR"] * cm[1, 1]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=["CG", "OR"],
        target_names=[
            "Fake Review (CG)",
            "Original Review (OR)"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    # BEST MODEL

    st.markdown("---")
    st.markdown("### Best Performing Model")

    best_model = algorithm_df.loc[
        algorithm_df["Test Accuracy"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Algorithm']} | "
        f"Test Accuracy: {best_model['Test Accuracy']:.2f}%"
    )

    st.write(
        "Linear SVM achieved the highest test accuracy "
        "among the three evaluated algorithms."
    )

    st.caption(
        "All three confusion matrices are based on the "
        "same 1,000-review test dataset."
    )


# =====================================================
# 17. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About ReviewDetector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Learn about the project, machine learning algorithm and dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    ReviewDetector is a Machine Learning-based Fake Review
    Detection System developed as an academic CSE project.

    The system analyzes Amazon product reviews and classifies
    reviews into two categories: Fake Review and Original Review.
    """)

    st.markdown("### Machine Learning Algorithm")

    st.info("Linear Support Vector Machine (Linear SVM)")

    st.markdown("### Feature Extraction")

    st.write("""
    - Word-level TF-IDF
    - Character-level TF-IDF
    - FeatureUnion
    """)

    st.markdown("### Dataset Information")

    st.write(f"""
    - Dataset: Amazon Product Reviews
    - Total Reviews: {total_reviews:,}
    - Fake Reviews (CG): {fake_reviews:,}
    - Original Reviews (OR): {original_reviews:,}
    - Product Categories: {total_categories}
    - Training Reviews: 4,000
    - Testing Reviews: 1,000
    """)

    st.markdown("### Model Performance")

    st.metric(
        "Test Accuracy",
        "88.80%"
    )

    st.warning("""
    Project Scope: This model distinguishes CG reviews from
    OR reviews. It does not establish whether every review is
    paid, fraudulent, or manipulated.
    """)


# =====================================================
# 18. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <b style="color:#173B67;">ReviewDetector 🔍</b><br>
    Fake Review Detection Using Machine Learning<br>
    Academic Project | Computer Science and Engineering
</div>
""", unsafe_allow_html=True)
