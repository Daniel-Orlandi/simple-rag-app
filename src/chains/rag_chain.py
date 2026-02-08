"""
RAG Chain module for creating retrieval-augmented generation pipelines.

This module provides functionality to create RAG chains that combine document retrieval
with answer generation using language models.
"""
import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

def create_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    system_prompt: Optional[str] = None
) -> Runnable:
    """
    Create a RAG chain that retrieves documents and generates answers.
    
    The chain follows this pipeline:
    1. Retrieves relevant documents based on the query
    2. Formats documents into context
    3. Constructs a prompt with context and question
    4. Generates answer using the language model
    5. Parses and returns the answer as a string
    
    Args:
        llm: Language model instance for answer generation.
        retriever: Document retriever for finding relevant documents.
        system_prompt: Optional custom system prompt. If None, uses default
                      Portuguese legal assistant prompt.
    
    Returns:
        Runnable: A LangChain runnable chain that takes a question (str) and returns an answer (str).
    
    Example:
        >>> from langchain_community.chat_models import ChatMaritalk
        >>> llm = ChatMaritalk(api_key="...", model="sabia-3")
        >>> chain = create_rag_chain(llm, retriever)
        >>> answer = chain.invoke("O que é alienação fiduciária?")
    """
    logger.info("Creating RAG chain")
    
    if system_prompt is None:
        system_prompt = """Você é um assistente de leitura de documentos de maquinas industriais.
                            Use as seguintes informações do contexto para responder à pergunta. Se a resposta não estiver no contexto, 
                            diga que não tem informações suficientes nos documentos fornecidos."""
        logger.debug("Using default system prompt")
    else:
        logger.debug("Using custom system prompt")
    
    template = f"""{system_prompt}

                    Contexto:
                    {{context}}

                    Pergunta: {{question}}

                    Resposta:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain created successfully")
    return chain

