# test_pipeline.py
import argparse
import textwrap
from src.pipeline import QAPipeline

def main():
    """
    Command-line interface to test the QAPipeline.
    Provides detailed debug output for analysis.
    """
    parser = argparse.ArgumentParser(
        description="Test the Mentor AI QA Pipeline from the command line.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("question", type=str, help="The question you want to ask the chatbot.")
    args = parser.parse_args()
    
    print("--- Initializing Pipeline (this may take a moment on first run) ---")
    try:
        pipeline = QAPipeline()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize pipeline: {e}")
        return

    print(f"\n--- Processing question: '{args.question}' ---")
    try:
        result = pipeline.ask(args.question)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during pipeline execution: {e}")
        return

    # --- Display Debugging Information ---
    print("\n" + "="*80)
    print(" " * 28 + "PIPELINE DEBUG OUTPUT")
    print("="*80)

    # 1. Retrieved Contexts
    print("\n[🔍] Step 1: Retrieved Contexts\n" + "-"*40)
    if result.get("retrieved_contexts"):
        for i, ctx in enumerate(result["retrieved_contexts"]):
            print(f"--- Context #{i+1} ---\n{textwrap.indent(ctx, '  ')}")
    else:
        print("  -> No contexts were retrieved.")

    # 2. Final Prompt
    print("\n[📝] Step 2: Final Prompt sent to LLM\n" + "-"*40)
    print(textwrap.indent(result.get("prompt", "No prompt generated."), '  '))

    # 3. Final Answer
    print("\n[🤖] Step 3: Final Generated Answer\n" + "-"*40)
    print(textwrap.indent(result.get("answer", "No answer generated."), '  '))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
