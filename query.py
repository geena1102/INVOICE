from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma  
from langchain_ollama import OllamaLLM

CHROMA_DB_PATH = "chroma_storage"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}


---

Answer the question based on the above context: {question}
"""

def main():
    db = Chroma(persist_directory=CHROMA_DB_PATH,
                collection_name="image_texts", 
                embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                )

    query_text = input("Enter query : ")
    results = db.similarity_search_with_score(query_text, k=5)
    
    print(results)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Prompting..")
    model = OllamaLLM(
        model="llama3.2",
        system="""
            You are a helpful AI Assistant.
            You will be given data degive seller name buyers name and items name of each tected from invoice images. 
            The data will have the order in which parameters such as Product name, quantity, units etc.. 
            The data will appear in the mentioned order.

            You should answer the questions by carefully examining the order in which the parameters are mentioned in the data.
        """
        )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

main()