
import embedding as em
import streamlit as st
import PyPDF2

st.set_page_config(page_title="RAG Book Chat", page_icon="📚", layout="wide")

st.title("📚 Retrieval Augmented Generation")
st.markdown("#### Powered by :green[ChromaDB] + :blue[Gemini] ✨")
st.write("Upload your book :green-background[(PDF or TXT)] and ask it anything!")

st.divider()

#-----------------------**************----------------------------------
#                          Sidebar code

st.sidebar.title("⚙️ Menu")
st.sidebar.markdown("Configure how answers are generated.")
n_results = st.sidebar.number_input(
    "🔍 Number of results to retrieve",
    min_value=1, max_value=5, value=1
)
st.sidebar.divider()
st.sidebar.info("💡 Tip: fewer results = faster, more focused answers.")

#----------------------------************--------------------------------
#                            main page code

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ Upload your document")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    # Extract file content as text
    if uploaded_file.name.endswith(".txt"):
        # For text files
        file_content = uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):

        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_content = " ".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("🚫 Unsupported file type. Please upload a .txt or .pdf file.")
        st.stop()

    with col1:
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")

        with st.spinner("🧠 Generating embeddings and indexing your document..."):
            file_name = uploaded_file.name
            collection = em.create_embeddings(file_content, file_name)
            print(collection)

        st.toast("Document indexed and ready!", icon="🎉")

    with col2:
        st.subheader("2️⃣ Ask a question")
        user_input = st.text_input("💬 Ask me something about the document:")
        if user_input:
            if st.button("🚀 Get Answers", type="primary"):
                with st.spinner("🤔 Thinking..."):
                    ans = em.get_final_answer(user_input, file_content, collection, n_results, file_name)
                st.markdown("##### 📝 Answer")
                st.markdown(f"> {ans}")
else:
    with col2:
        st.subheader("2️⃣ Ask a question")
        st.warning("⬅️ Upload a document first to start asking questions.")

