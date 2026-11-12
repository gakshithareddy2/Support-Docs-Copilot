from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("documents")


def load_documents():
    """
    Load all supported documents from the documents folder.
    Supported formats: TXT and PDF.
    """

    DOCUMENTS_DIR.mkdir(exist_ok=True)

    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        try:
            if extension == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )
                loaded_docs = loader.load()

            elif extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded_docs = loader.load()

            else:
                continue

            for document in loaded_docs:
                document.metadata["source"] = file_path.name

            documents.extend(loaded_docs)

        except Exception as error:
            print(f"Could not load {file_path.name}: {error}")

    return documents


def split_documents(documents):
    """
    Split documents into smaller overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\nChunk {index + 1}")
        print("Source:", chunk.metadata.get("source"))
        print("Content:", chunk.page_content[:200])