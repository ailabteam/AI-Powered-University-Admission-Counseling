# src/pipeline.py
from src.retriever import FaissRetriever
from src.generator import LLMGenerator
from src.config import PROMPT_TEMPLATE, TOP_K_RETRIEVER
from typing import Dict, Any, List

class QAPipeline:
    """
    The main Question-Answering pipeline that integrates Retriever and Generator.
    """
    def __init__(self):
        print("Initializing QA Pipeline...")
        self.retriever = FaissRetriever()
        self.generator = LLMGenerator()
        print("QA Pipeline initialized successfully.")

    def _format_context(self, contexts: List[str]) -> str:
        """Formats the retrieved contexts into a single string for the prompt."""
        return "\n\n".join([f"--- Nguồn tham khảo {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)])

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Asks a question and returns a dictionary with the answer and debug info.
        
        Args:
            question (str): The user's question.
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer, retrieved contexts, and the final prompt.
        """
        # 1. Retrieve relevant contexts
        retrieved_contexts = self.retriever.search(question, top_k=TOP_K_RETRIEVER)
        
        # 2. Format the contexts for the prompt
        formatted_context = self._format_context(retrieved_contexts)
        
        # 3. Create the final prompt
        final_prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=question)
        
        # 4. Generate the answer
        answer = self.generator.generate(final_prompt)
        
        return {
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "prompt": final_prompt
        }
