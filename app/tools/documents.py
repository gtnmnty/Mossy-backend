# Feature 6: Tool - search the user's documents
# Lets Claude pull facts from the user's own files instead of guessing. Claude decides when
# to call this, same as the time and marine tools; the system prompt's <documents_guidance>
# tells it when. Flow: run hybrid search (rag.retrieve) -> wrap each passage in XML tags with
# its source file -> return that text as the tool result, which Claude reads and answers from.

from app.services.rag import retrieve

SCHEMA = {
    "name": "search_documents",
    "description": (
        "Search the user's own documents and notes. Use it for any question about their "
        "files or things only their documents would say. Returns the most relevant "
        "passages with the file name each came from."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "description": "A focused search query using the key terms of the question"}},
        "required": ["query"],
    },
}


async def run(query: str) -> str:
    chunks = await retrieve(query)
    if not chunks:
        return "No documents are indexed yet, or nothing matched."
    return "\n\n".join(
        f'<document index="{n}"><source>{c["source"]}</source>'
        f'<document_content>{c["text"]}</document_content></document>'
        for n, c in enumerate(chunks, 1)
    )