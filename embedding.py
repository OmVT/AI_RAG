
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

gemini_key=os.getenv("gemini_api_key")
print(f"[DEBUG] gemini_api_key loaded: {'yes, length ' + str(len(gemini_key)) if gemini_key else 'NO - environment variable not set'}")

def create_embeddings(file_content,file_name):
    print(f"[DEBUG] create_embeddings called for file: {file_name}")
    try:
        embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path="./chroma_db")
        collection=client.get_or_create_collection(name=f"collection_{file_name}",metadata={"hnsw:space":"cosine"})
        print(f"[DEBUG] ChromaDB collection ready: collection_{file_name}")
    except Exception as e:
        print(f"[ERROR] Failed to set up ChromaDB client/collection: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! "],  # List of characters to split on
    chunk_size=2000,  # The maximum size of your chunks
    chunk_overlap=400,  # The maximum overlap between chunks
    )

    text={}
    text['chunk']=text_splitter.create_documents([file_content])
    print(f"[DEBUG] Document split into {len(text['chunk'])} chunks")

    try:
        for chunk_id,chunk in enumerate(text["chunk"]):
            collection.add(
                documents=[chunk.page_content],
                ids=[f"chunk_id"],
                metadatas={"title":file_name}
                )
        print(f"[DEBUG] All chunks added to collection successfully")
    except Exception as e:
        print(f"[ERROR] Failed to add chunk {chunk_id} to ChromaDB: {e}")
        raise

    return collection




def get_final_answer(user_query,file_content,collection,n_results,file_name):
    print(f"[DEBUG] get_final_answer called with query: {user_query!r}, n_results={n_results}")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-3.5-flash-lite")

    try:
        search_results = collection.query(query_texts=[user_query], n_results=n_results)
        print(f"[DEBUG] ChromaDB returned {len(search_results['documents'][0])} matching chunk(s)")
    except Exception as e:
        print(f"[ERROR] ChromaDB query failed: {e}")
        raise

    result_str = ""
    response=""
    for result in search_results["documents"][0]:

        final_prompt= f"""
                        Instructions:
                        Answer the user question mentioned below based on search results provided

                        At the end of your answer, cite the URL of the search result your answer draws from. Use the following format:
                        <Your answer here>. Source: <URL of the search result your answer comes from here>. Output only the answer part and reference URL

                        User question: <{user_query}>

                        Search Results: <{result}>

                        Source:<{file_name}>
                        Your answer:
                        """

        try:
            response += model.generate_content(final_prompt).text
            print(f"[DEBUG] Gemini API call succeeded for one chunk")
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            raise
    return response
