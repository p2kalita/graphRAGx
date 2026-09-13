"""
Interactive CLI for querying the CORD-19 knowledge graph.

    python build_graph.py                 # index once (see that file for options)
    python main.py                        # then ask questions
    python main.py --question "..." --mode local|global|auto
"""
import argparse

from graphrag_pipeline import GraphRAGPipeline


def print_result(question: str, result: dict):
    print(f"\nQ: {question}")
    print(f"[search_type: {result.get('search_type')}]")
    if result.get("matched_nodes"):
        print(f"[matched entities: {', '.join(result['matched_nodes'])}]")
    if result.get("community_context"):
        titles = [c["title"] for c in result["community_context"]]
        print(f"[communities used: {', '.join(titles)}]")
    print(f"\nA: {result.get('answer', '(no answer)')}")
    if result.get("sources"):
        print(f"\nSources: {', '.join(result['sources'])}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Query the CORD-19 GraphRAG knowledge graph.")
    parser.add_argument("--question", type=str, default=None,
                         help="Ask a single question and exit (otherwise starts a REPL).")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "local", "global"])
    args = parser.parse_args()

    pipeline = GraphRAGPipeline()
    print(f"Loaded graph: {pipeline.kg.stats()}, "
          f"{len(pipeline.communities)} community summaries.")

    if args.question:
        result = pipeline.ask(args.question, mode=args.mode)
        print_result(args.question, result)
        return

    print("Type a question (or 'exit'). Prefix with 'local:' / 'global:' to force a mode.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break

        mode = "auto"
        if q.lower().startswith("local:"):
            mode, q = "local", q[6:].strip()
        elif q.lower().startswith("global:"):
            mode, q = "global", q[7:].strip()

        result = pipeline.ask(q, mode=mode)
        print_result(q, result)


if __name__ == "__main__":
    main()
