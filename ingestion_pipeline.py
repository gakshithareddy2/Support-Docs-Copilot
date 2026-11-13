
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("documents")


def load_documents(selected_files):
    """Load only the files selected in the Streamlit app."""

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    documents = []

    for filename in selected_files:
        file_path = DOCUMENTS_DIR / Path(filename).name

        if not file_path.is_file():
            print(f"File not found: {filename}")
            continue

        try:
            extension = file_path.suffix.lower()

            if extension == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            elif extension == ".pdf":
                loader = PyPDFLoader(str(file_path))

            else:
                continue

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_path.name

            documents.extend(loaded_docs)

        except Exception as error:
            print(f"Could not load {filename}: {error}")

    return documents


def split_documents(documents):
    """Split loaded documents into overlapping text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        length_function=len
    )

    return splitter.split_documents(documents)
