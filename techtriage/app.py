"""Run: streamlit run app.py"""
import csv
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from classifier import train_from_csv
from retrieval import load_articles, Retriever
from answer import answer_question, safe_question

load_dotenv()
st.set_page_config(page_title="TechTriage (demo)", page_icon="🛠️")
st.title("🛠️ TechTriage")
st.caption("CS221 Impact Track • synthetic demonstration data, not official Stanford IT guidance")
st.warning("DEMO: Do not use these instructions for real support cases yet.")
st.info("Do not enter student names, SUNet IDs, device serial numbers, emails, or ticket text.")

@st.cache_resource
def get_classifier():
    return train_from_csv()

@st.cache_resource
def get_retriever(use_embeddings):
    return Retriever(load_articles(), use_embeddings=use_embeddings)

method = st.selectbox("Retrieval method", ["BM25", "Embedding", "Hybrid"])
use_embeddings = method != "BM25"
question = st.text_area("Describe the technical issue (no personal information)",
                        placeholder="My Xbox says connected, but nothing loads")
if st.button("Find troubleshooting guidance", type="primary"):
    if not question.strip():
        st.warning("Enter an issue first.")
    else:
        try:
            safe_question(question)
            preds = get_classifier().predict(question)
            category, confidence = preds[0]
            st.write(f"**Predicted category:** {category} ({confidence:.0%} on synthetic training examples)")
            st.caption("These are uncalibrated classifier outputs, not verified confidence estimates.")
            # Fallback limits the damage of uncertain classifications.
            search_category = category if confidence >= 0.70 else None
            st.write("**Search scope:** " + (search_category or "All categories (classifier uncertain)"))
            with st.spinner("Searching documents..."):
                results = get_retriever(use_embeddings).search(
                    question, category=search_category, method=method
                )
            st.subheader("Top matching articles")
            for i, r in enumerate(results, 1):
                st.markdown(f"**[{i}] {r['title']}** · {r['category']}")
                st.caption(f"Retrieval score: {r['score']:.3f} | Demo ID: {r['id']}")
                st.write(r["text"])
            text, generated = answer_question(question, results)
            st.subheader("Suggested response" if generated else "Evidence / escalation")
            st.write(text)
            if generated:
                st.warning("Verify every recommendation against its cited source before use.")
            st.session_state['last_meta'] = {
                'category': category, 'method': method,
                'generated': generated, 'escalated': text.lower().startswith('escalate')
            }
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Unable to run this configuration: {error}")
if 'last_meta' in st.session_state:
    st.write("Was this result useful? (stores metadata only; not the question)")
    c1, c2 = st.columns(2)
    for col, feedback in [(c1, "up"), (c2, "down")]:
        if col.button("👍" if feedback == "up" else "👎"):
            file = Path(__file__).parent / "feedback.csv"
            new = not file.exists()
            with file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["utc_time", "feedback", "category", "method", "generated", "escalated"])
                if new:
                    writer.writeheader()
                writer.writerow({"utc_time": datetime.now(timezone.utc).isoformat(),
                                 "feedback": feedback, **st.session_state['last_meta']})
            st.success("Feedback saved without question text.")
