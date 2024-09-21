import streamlit as st
from src.helper import load_documents, create_faiss_index, get_retrieval_chain

st.title("Interactive QA Bot with LangChain")

uploaded_file = st.file_uploader("Upload your document", type=["pdf"])

if uploaded_file is not None:
    st.write("Document uploaded successfully.")
    
    # Save the uploaded file temporarily
    file_path = f"temp_{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Write the uploaded file to a temporary location
    
    # Load and process the document
    documents = load_documents(file_path)  # Pass the file path directly
    vectorstore = create_faiss_index(documents)
    qa_chain = get_retrieval_chain(vectorstore)
    
    # Query section
    query = st.text_input("Ask a question related to the document:")
    
    if query:
        answer = qa_chain.run(query)
        st.write(f"Answer: {answer}")

    # Optionally, provide an option to delete the temporary file
    import os
    os.remove(file_path)  # Clean up the temporary file after processing
