# src/pipeline.py
from src.retriever import FaissRetriever
from src.generator import LLMGenerator
from src.config import PROMPT_TEMPLATE, TOP_K_RETRIEVER

class QAPipeline:
    """
    The main Question-Answering pipeline that integrates Retriever and Generator.
    """
    def __init__(self):
        print("Initializing QA Pipeline...")
        self.retriever = FaissRetriever()
        self.generator = LLMGenerator()
        print("QA Pipeline initialized successfully.")

    def _format_context(self, contexts: list) -> str:
        """Formats the retrieved contexts into a single string for the prompt."""
        return "\n\n".join([f"Nguồn {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

    def ask(self, question: str) -> str:
        """
        The main method to ask a question and get an answer.
        
        Args:
            question (str): The user's question.
            
        Returns:
            str: The final answer.
        """
        # 1. Retrieve relevant contexts
        contexts = self.retriever.search(question, top_k=TOP_K_RETRIEVER)
        
        # 2. Format the contexts for the prompt
        formatted_context = self._format_context(contexts)
        
        # 3. Create the final prompt
        prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=question)
        print("\n--- Final Prompt to LLM ---")
        print(prompt)
        print("---------------------------\n")
        
        # 4. Generate the answer
        answer = self.generator.generate(prompt)
        
        return answer
