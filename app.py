import io
import os
import re
from typing import List, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import streamlit as st
from groq import Groq
from PIL import Image
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="AI Job Cover Letter Generator",
    page_icon="✉️",
    layout="wide",
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Current Groq production recommendation for high-quality generation.
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K_CV = 7
TOP_K_JOB = 7
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. Add it in your Hugging Face Space "
            "under Settings → Secrets, then restart the Space."
        )
        st.stop()
    return Groq(api_key=api_key)


# -----------------------------
# File extraction
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    return pytesseract.image_to_string(image)


def extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a normal PDF; OCR scanned pages when necessary."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    total_direct_chars = 0

    for page in doc:
        text = page.get_text("text") or ""
        text = clean_text(text)
        pages.append(text)
        total_direct_chars += len(text)

    # If the PDF has little/no selectable text, OCR rendered pages.
    if total_direct_chars < 100:
        ocr_pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_pages.append(clean_text(ocr_image(image)))
        pages = ocr_pages

    doc.close()
    return clean_text("\n\n".join(p for p in pages if p))


def extract_text(uploaded_file) -> str:
    """Extract text from PDF, JPG/JPEG or PNG."""
    data = uploaded_file.getvalue()
    file_type = uploaded_file.type or ""
    name = uploaded_file.name.lower()

    if file_type == "application/pdf" or name.endswith(".pdf"):
        return extract_from_pdf(data)

    if file_type.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(io.BytesIO(data))
        return clean_text(ocr_image(image))

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


# -----------------------------
# RAG utilities
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size // 5, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        step_words = max((chunk_size - overlap) // 5, 1)
        start += step_words

    return chunks


def build_faiss_index(chunks: List[str], model: SentenceTransformer):
    if not chunks:
        return None, np.empty((0, 384), dtype="float32")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(
    query: str,
    chunks: List[str],
    index,
    model: SentenceTransformer,
    top_k: int,
) -> List[Tuple[str, float]]:
    if index is None or not chunks:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append((chunks[idx], float(score)))
    return results


def format_context(results: List[Tuple[str, float]]) -> str:
    if not results:
        return "No relevant context was retrieved."
    return "\n\n".join(
        f"[Retrieved passage {i + 1}]\n{chunk}" for i, (chunk, _score) in enumerate(results)
    )


# -----------------------------
# Cover letter generation
# -----------------------------
def generate_cover_letter(
    client: Groq,
    cv_context: str,
    job_context: str,
) -> str:
    system_prompt = """
You are an expert professional cover-letter writer and recruitment assistant.
Create a tailored, truthful cover letter using ONLY the information present in the
retrieved CV and job-ad context. Never invent employers, job titles, years,
qualifications, certifications, achievements, skills, salary, locations, or other facts.

Requirements:
- Match the candidate's strongest relevant experience to the job requirements.
- Prioritize concrete skills, responsibilities, projects, certifications, and achievements.
- Keep the letter professional and natural, not robotic or keyword-stuffed.
- Do not mention RAG, embeddings, retrieval, AI, OCR, or the internal process.
- Do not repeat the full job advertisement.
- Avoid generic claims that are not supported by the CV.
- Use a clean business-letter structure.
- Aim for approximately 350-500 words.
- If the employer or hiring manager name is not available, use a neutral greeting such as “Dear Hiring Manager,”.
- End with a professional closing using the candidate's name only if it appears clearly in the CV.
""".strip()

    user_prompt = f"""
Create a tailored cover letter from the following retrieved information.

===== RETRIEVED CV INFORMATION =====
{cv_context}

===== RETRIEVED JOB AD INFORMATION =====
{job_context}

Before writing, identify the strongest overlaps between the candidate and the role,
then write only the final polished cover letter. Do not include an analysis section.
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35,
        max_tokens=1800,
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# UI
# -----------------------------
st.title("✉️ AI Job Cover Letter Generator")
st.caption("RAG + FAISS + Sentence Transformers + Groq")

with st.sidebar:
    st.subheader("Supported files")
    st.write("CV: PDF, JPG, JPEG, PNG")
    st.write("Job ad: PDF, JPG, JPEG, PNG")
    st.divider()
    st.write("Your GROQ API key is read from the GROQ_API_KEY environment variable / Hugging Face Secret.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload CV")
    cv_file = st.file_uploader(
        "Upload your CV",
        type=["pdf", "jpg", "jpeg", "png"],
        key="cv",
    )

with col2:
    st.subheader("2. Upload Job Ad")
    job_ad_file = st.file_uploader(
        "Upload the job advertisement",
        type=["pdf", "jpg", "jpeg", "png"],
        key="job_ad",
    )

if cv_file and job_ad_file:
    try:
        with st.spinner("Extracting text from your files..."):
            cv_text = extract_text(cv_file)
            job_ad_text = extract_text(job_ad_file)

        if len(cv_text) < 80:
            st.error("Could not extract enough text from the CV. Try a clearer PDF/image.")
            st.stop()

        if len(job_ad_text) < 80:
            st.error("Could not extract enough text from the job ad. Try a clearer PDF/image.")
            st.stop()

        tab1, tab2 = st.tabs(["Extracted CV", "Extracted Job Ad"])
        with tab1:
            st.text_area("CV text", cv_text, height=280)
        with tab2:
            st.text_area("Job ad text", job_ad_text, height=280)

        with st.spinner("Building the RAG knowledge base..."):
            embedding_model = load_embedding_model()
            cv_chunks = chunk_text(cv_text)
            job_chunks = chunk_text(job_ad_text)
            cv_index, _ = build_faiss_index(cv_chunks, embedding_model)
            job_index, _ = build_faiss_index(job_chunks, embedding_model)

        # Use the complete job ad as the retrieval query for CV evidence.
        cv_results = retrieve(
            job_ad_text,
            cv_chunks,
            cv_index,
            embedding_model,
            TOP_K_CV,
        )

        # Use the candidate/CV as the retrieval query for the most relevant job requirements.
        job_results = retrieve(
            cv_text,
            job_chunks,
            job_index,
            embedding_model,
            TOP_K_JOB,
        )

        with st.expander("View retrieved RAG context"):
            st.markdown("**Relevant CV passages**")
            st.text(format_context(cv_results))
            st.markdown("**Relevant job-ad passages**")
            st.text(format_context(job_results))

        if st.button("✨ Generate Cover Letter", type="primary", use_container_width=True):
            client = get_groq_client()
            with st.spinner("Generating your tailored cover letter..."):
                cover_letter = generate_cover_letter(
                    client,
                    format_context(cv_results),
                    format_context(job_results),
                )

            st.subheader("Generated Cover Letter")
            st.text_area("Cover letter", cover_letter, height=500)
            st.download_button(
                "Download Cover Letter (.txt)",
                data=cover_letter,
                file_name="tailored_cover_letter.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.success("Cover letter generated successfully.")

    except Exception as exc:
        st.error("Something went wrong while processing the files.")
        st.exception(exc)
else:
    st.info("Upload both your CV and the job advertisement to begin.")
