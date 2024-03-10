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
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        json_files = [
            f for f in zip_ref.namelist()
            if f.endswith('.json') and f.startswith("document_parses/pdf_json")
        ]
        if limit is not None:
            json_files = json_files[:limit]

        all_documents = []
        for json_file in json_files:
            with zip_ref.open(json_file, 'r') as file:
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
    document_limit: Optional[int] = None,
) -> None:
    documents = load_documents(zip_file_path, document_limit)
    documents = [Document(page_content=doc) for doc in documents]

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    _ = Chroma.from_documents(docs, embedding_function, persist_directory=save_folder)


def load_chroma_instance(save_folder: str = CHROMA_DIRECTORY) -> Chroma:
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=save_folder,
        embedding_function=embedding_function,
    )


def main() -> None:
    create_chroma_instance(ARCHIVE_FILE, document_limit=None)


if __name__ == '__main__':
    main()
