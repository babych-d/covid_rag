from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable

from src.llm import get_llm
from src.rag import load_chroma_instance


def get_rag_conversation_chain() -> RunnableSerializable:
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
