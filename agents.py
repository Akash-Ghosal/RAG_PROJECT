def agent_router(query):
    query_lower = query.lower()
    if "calculate" in query_lower:
        return "calculator"
    elif "define" in query_lower:
        return "dictionary"
    else:
        return "rag"
