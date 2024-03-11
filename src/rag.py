import json
import zipfile
from typing import Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARCHIVE_FILE = "archive.zip"


def load_documents(zip_file_path: str, limit: Optional[int] = None) -> list[str]:
    """Load documents from a ZIP file containing JSON files.

    This function extracts JSON files from a ZIP archive from inner folder
    'document_parses/pdf_json' and optionally limits the number of documents processed.
    It reads the 'abstract' and 'body_text' fields of each JSON, accumulating each text
    into a list of strings.

    Args:
        zip_file_path:  The path to the ZIP file containing the document JSON files.
        limit:          An optional integer specifying the maximum number
                        of JSON files to process.

    Returns:
        A list of strings, where each string is the text from the 'abstract' or
        'body_text' field of the JSON files.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        json_files = [
            f for f in zip_ref.namelist()
            if f.endswith(".json") and f.startswith("document_parses/pdf_json")
        ]
        if limit is not None:
            json_files = json_files[:limit]

        all_documents = []
        for json_file in json_files:
            with zip_ref.open(json_file, "r") as file:
                content = file.read()

                data = json.loads(content.decode("utf-8"))

                for abs_data in data["abstract"]:
                    all_documents.append(abs_data["text"])
                for abs_data in data["body_text"]:
                    all_documents.append(abs_data["text"])

    return all_documents


def create_chroma_instance(
    zip_file_path: str,
    save_folder: str = CHROMA_DIRECTORY,
    embedding_model: str = EMBEDDING_MODEL,
    document_limit: Optional[int] = None,
) -> None:
    """Create a Chroma instance from documents loaded from a ZIP file.

    This function loads documents, creates `Document` instances for each text,
    splits the documents into chunks using `CharacterTextSplitter`, and finally
    creates a `Chroma` vector storage instance from the processed documents.
    Vector

    Args:
        zip_file_path:      The path to the ZIP file containing the document JSON
                            files.
        save_folder:        The directory where the Chroma database should be saved.
        embedding_model:    Embedding model used to create embeddings from texts.
        document_limit:     An optional integer specifying the maximum number of
                            documents to process.

    """
    documents = load_documents(zip_file_path, document_limit)
    documents = [Document(page_content=doc) for doc in documents]

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
    _ = Chroma.from_documents(docs, embedding_function, persist_directory=save_folder)


def load_chroma_instance(
    save_folder: str = CHROMA_DIRECTORY,
    embedding_model: str = EMBEDDING_MODEL
) -> Chroma:
    """Load a Chroma instance with a specified embedding model from a given folder.

    Args:
        save_folder:        The directory from which the Chroma database
                            should be loaded.
        embedding_model:    The name of the embedding model to be used for
                            generating text embeddings.

    Returns:
        A Chroma instance configured with the specified embedding model and
        data directory.
    """
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=save_folder,
        embedding_function=embedding_function,
    )


def main() -> None:
    create_chroma_instance(ARCHIVE_FILE, document_limit=None)


if __name__ == '__main__':
    main()
