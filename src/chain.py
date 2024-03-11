from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable

from src.llm import get_llm
from src.rag import load_chroma_instance


def get_rag_conversation_chain() -> RunnableSerializable:
    """Initialize and return a RAG (Retrieval-Augmented Generation) conversation chain.

    This function creates a conversation chain by integrating a language model with
    a document retriever. It uses a language model to generate responses based on
    the prompts provided and augmented with information retrieved from a set of
    documents.

    Returns:
        A `RunnableSerializable` instance that represents the assembled RAG
        conversation chain.
    """
    llm = get_llm()
    retriever = load_chroma_instance().as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
