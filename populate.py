
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import cv2

def extract_text_from_images(folder_path):
    """
    Extract text from all images in the specified folder using EasyOCR.
    """
    
    extracted_texts = {}

    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is an image
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing: {file_name}")
            try:

                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[{
                        'role': 'user',
                        'content': """ 
                            This is a real invoice for some purchase. Extract the content in the given image. 
                            The invoice will have tables. Make sure tables are extracted as it is including all the columns and values/empty cells.
                            Preserve empty columns as it is. Do not remove or adjust them.
                            There may be size wise billing such as S, M, L, XL etc... They should be preserved as they are.
                            Make sure they are extracted correctly.
                        """,
                        'images': [file_path]
                    }]
                )

                return response['message']['content']

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return extracted_texts

def chunk_text(text, chunk_size=1000, overlap=400):
    """
    Chunk text into smaller pieces using a recursive character splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

def store_embeddings_in_chromadb(text_chunks, image_name, chroma_client):
    """ 
    Store embeddings of text chunks in ChromaDB using Ollama embeddings.
    """
    metadata = {"source": image_name}

    for chunk in text_chunks:
        chroma_client.add_texts(
            texts=[chunk],
            metadatas=[metadata],
            ids=[f"{image_name}-{hash(chunk)}"]
        )

def main():
    folder_path = "test"  # Folder containing images
    extracted_texts = extract_text_from_images(folder_path)
    # extracted_texts = {'image_15.jpg': "PURCHASE ORDER # 15879/22 Order Date Apr 22,2022 WHOLESALE GIANTS Payment Terms NET 30 Shipping Method FOB clearly DIFFereng Promised Date Apr 25,2022 BILL TO: WHOLESALE GIANTS 56 Giants Avenue Shopingtown; NY 24556 Phone: (555) 1235 66789 Fax: (555) 1254 88587 giantswholesale123@gmail.com VENDOR: SHIP TO: Oceanic Traders Ltd: EuroMarket Gmb 456 Port Street; Mumbai, India 23 Commerce Lane; Berlin, Gemany info@oceanictraders in procurement@euromarket de 9122765410 30 333 4567 STYLE / ITEM COLOUR DESCRIPTION SIZE QTY UNIT PRICE DISCOUNT LINE TOTAL 55258-42562 Orange Women's Snowan Cotton Socks 37-39 $4.99 S20.00 S229.50 55258-00562 Dope Soul Shirt S7.89 51.20 S38.25 55887-52663 Yellow Lace Cross Tee XL S8.99 S8.99 NOTES SUBTOTAL S276.74 The amount of the Purchase Order is the agreed fixed price and SALES TAX 10% shall not be exceeded without advanced written consent  FREIGHT S1.45 TOTAL COST USD 305.86 Grey"}
    
    if not extracted_texts:
        print("No text extracted from images.")
        return
    
    chroma_client = Chroma(collection_name="image_texts", persist_directory="chroma_storage", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

    for image_name, text in extracted_texts.items():
        print(f"\nText extracted from {image_name}:\n{text}\n")
        text_chunks = chunk_text(text)
        store_embeddings_in_chromadb(text_chunks, image_name, chroma_client)

main()