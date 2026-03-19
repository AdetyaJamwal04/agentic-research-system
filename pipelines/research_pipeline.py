"""
Research Pipeline — Callable function that runs the full research workflow.
Returns structured results for both CLI and API use.
"""

import time
from agents.query_analyzer import analyse_query
from agents.premise_validator import validate_premise
from agents.planner import create_plan
from agents.query_generator import generate_queries
from agents.reasoning_agent import generate_answer
from agents.retriever_agent import retrieve_documents
from agents.relevance_filter import filter_documents
from retrieval.reranker import rerank_documents
from retrieval.context_compressor import compress_context
from agents.critic_agent import critique_answer
from agents.evidence_agent import synthesize_evidence, format_synthesis_for_reasoning
from evidence.claim_extractor import extract_claims
from planning.dynamic_planner import revise_plan
from memory.evidence_pool import EvidencePool
from memory.research_history import ResearchHistory
from memory.long_term_memory import LongTermMemory


def run_research(query: str, max_iterations: int = 3, use_memory: bool = True, stream_callback=None, conversation_context: str = "", local_only: bool = False) -> dict:
    """
    Run the full research pipeline and return structured results.

    Returns:
        {
            "query": str,
            "effective_query": str,
            "premise": {...},
            "analysis": {...},
            "plan": {...},
            "claims": [...],
            "answer": str,
            "sources": [...],
            "critique": {...},
            "iterations": int,
            "steps": [{"step": str, "status": str, "detail": str}, ...]
        }
    """
    result = {
        "query": query,
        "effective_query": query,
        "premise": None,
        "analysis": None,
        "plan": None,
        "claims": [],
        "answer": "",
        "sources": [],
        "critique": None,
        "iterations": 0,
        "steps": [],
    }

    _step_timer = [time.time()]  # mutable container for closure

    def log(step, status, detail=""):
        now = time.time()
        elapsed = now - _step_timer[0]
        _step_timer[0] = now
        timing = f"{elapsed:.1f}s" if elapsed >= 0.1 else ""
        if timing and detail:
            detail = f"{detail} • {timing}"
        elif timing:
            detail = timing
        step_dict = {"step": step, "status": status, "detail": detail}
        result["steps"].append(step_dict)
        if stream_callback:
            stream_callback({"type": "step", **step_dict})

    # ---- Step 0: Memory (query-scoped + long-term) ----
    pool = EvidencePool()  # Fresh per query — no cross-query contamination
    history = ResearchHistory()
    ltm = LongTermMemory()  # Persistent verified claims
    ltm_context = ""

    # Load relevant long-term claims for this query
    relevant_ltm = ltm.retrieve_relevant(query)
    if relevant_ltm:
        ltm_context = ltm.format_for_context(relevant_ltm)
        log("Long-Term Memory", "loaded", f"{len(relevant_ltm)} relevant claims")
    else:
        log("Long-Term Memory", "empty", "No relevant prior research")

    # ---- Step 1: Query Analysis ----
    try:
        analysis = analyse_query(query)
        result["analysis"] = analysis
        log("Query Analysis", "done", analysis.get("refined_query", query))
    except Exception as e:
        analysis = {"original_query": query, "refined_query": query}
        result["analysis"] = analysis
        log("Query Analysis", "error", str(e))

    # ---- Step 1.5: Premise Validation ----
    effective_query = query
    premise_correction = None

    try:
        premise = validate_premise(analysis)
        result["premise"] = premise

        if premise.get("premise_status") == "incorrect":
            premise_correction = premise.get("correction", "")
            effective_query = premise.get("corrected_query", query)
            result["effective_query"] = effective_query
            analysis["original_query"] = effective_query
            analysis["refined_query"] = effective_query
            analysis["premise_correction"] = premise_correction
            log("Premise Validation", "false_premise", premise_correction)
        elif premise.get("premise_status") == "uncertain":
            log("Premise Validation", "uncertain", "Proceeding with caution")
        else:
            log("Premise Validation", "valid")
    except Exception as e:
        log("Premise Validation", "error", str(e))

    # ---- Step 2: Planning ----
    try:
        plan = create_plan(analysis)
        result["plan"] = plan
        log("Planning", "done", f"{len(plan.get('tasks', []))} tasks")
    except Exception as e:
        plan = {"tasks": [{"task_id": 1, "description": f"Retrieve: {effective_query}"}]}
        result["plan"] = plan
        log("Planning", "error", str(e))

    all_tasks = list(plan.get("tasks", []))

    # ---- Step 3: Query Generation ----
    try:
        queries_generated = generate_queries(plan)
        total_q = sum(len(tq.get("queries", [])) for tq in queries_generated.get("task_queries", []))
        log("Query Generation", "done", f"{total_q} queries")
    except Exception as e:
        queries_generated = {"task_queries": [{"task_id": 1, "queries": [effective_query]}]}
        log("Query Generation", "error", str(e))

    # ---- Step 4: Retrieval ----
    try:
        all_tagged_docs = retrieve_documents(queries_generated, local_only=local_only)
        log("Retrieval", "done", f"{len(all_tagged_docs)} documents")
    except Exception as e:
        all_tagged_docs = []
        log("Retrieval", "error", str(e))

    if not all_tagged_docs:
        result["answer"] = "No documents retrieved. Cannot generate an answer."
        log("Pipeline", "stopped", "No documents")
        return result

    # ---- Step 5: Early Domain Filter (before pool, before rerank) ----
    try:
        doc_contents_raw = [d["content"] for d in all_tagged_docs]
        filtered_contents = filter_documents(effective_query, doc_contents_raw)
        if filtered_contents:
            # Keep only tagged docs whose content passed the filter
            matching_contents = set(c[:200] for c in filtered_contents)
            all_tagged_docs = [d for d in all_tagged_docs if d["content"][:200] in matching_contents]
        log("Domain Filter", "done", f"{len(doc_contents_raw)} → {len(all_tagged_docs)} docs")
    except Exception as e:
        log("Domain Filter", "error", str(e))

    # ---- Step 6: Evidence Pool (query-scoped) ----
    pool.add(all_tagged_docs)
    doc_contents = pool.get_contents()

    # ---- Step 7: Rerank ----
    try:
        docs_for_reasoning = rerank_documents(effective_query, doc_contents)
        log("Reranker", "done", f"→ {len(docs_for_reasoning)} docs")
    except Exception as e:
        docs_for_reasoning = doc_contents[:5]
        log("Reranker", "error", str(e))

    # ---- Step 8: Compress ----
    try:
        compressed_docs = compress_context(effective_query, docs_for_reasoning)
    except Exception as e:
        compressed_docs = [doc[:800] for doc in docs_for_reasoning]

    # ---- Step 9: Claim Extraction ----
    try:
        numbered_evidence = pool.format_for_reasoning()
        claims = extract_claims(effective_query, numbered_evidence)
        pool.add_claims(claims)
        result["claims"] = claims
        log("Claim Extraction", "done", f"{len(claims)} claims")
    except Exception as e:
        claims = []
        log("Claim Extraction", "error", str(e))

    # ---- Step 10: Evidence Synthesis ----
    try:
        if pool.get_claims():
            evidence_text = pool.format_claims_for_reasoning()
        else:
            evidence_text = numbered_evidence
        synthesis = synthesize_evidence(effective_query, evidence_text)
        log("Evidence Synthesis", "done", f"{len(synthesis.get('evidence_brief', []))} themes")
    except Exception as e:
        synthesis = None
        evidence_text = numbered_evidence if numbered_evidence else ""
        log("Evidence Synthesis", "error", str(e))

    # ---- Step 11: Reasoning + Critic Loop ----
    evidence_for_reasoning = evidence_text

    # Prepend long-term memory context if available
    if ltm_context:
        evidence_for_reasoning = f"{ltm_context}\n\n{evidence_for_reasoning}"

    if premise_correction:
        evidence_for_reasoning = f"IMPORTANT PREMISE CORRECTION: {premise_correction}\n\n{evidence_for_reasoning}"

    iteration = 0
    critique = {}

    while iteration < max_iterations:
        try:
            answer = generate_answer(effective_query, evidence_for_reasoning, conversation_context=conversation_context)
            result["answer"] = answer
        except Exception as e:
            log("Reasoning", "error", str(e))
            break

        try:
            critique = critique_answer(effective_query, answer)
            result["critique"] = critique
        except Exception as e:
            log("Critic", "error", str(e))
            break

        if critique.get("verdict") == "sufficient":
            log("Critic", "sufficient", f"Iteration {iteration + 1}")
            break

        # ---- Dynamic Planning ----
        missing = critique.get("missing_aspects", [])
        if not missing:
            iteration += 1
            continue

        pool.next_iteration()

        try:
            revised = revise_plan(effective_query, missing, all_tasks)
            new_tasks = revised.get("tasks", [])
            all_tasks.extend(new_tasks)

            new_queries = generate_queries({"tasks": new_tasks})
            new_docs = retrieve_documents(new_queries, local_only=local_only)
            pool.add(new_docs)

            new_numbered = pool.format_for_reasoning()
            new_claims = extract_claims(effective_query, new_numbered)
            pool.add_claims(new_claims)

            evidence_for_reasoning = pool.format_claims_for_reasoning()
            if ltm_context:
                evidence_for_reasoning = f"{ltm_context}\n\n{evidence_for_reasoning}"
            if premise_correction:
                evidence_for_reasoning = f"IMPORTANT PREMISE CORRECTION: {premise_correction}\n\n{evidence_for_reasoning}"

            log("Dynamic Planning", "done", f"{len(new_tasks)} new tasks, {len(new_claims)} new claims")
        except Exception as e:
            log("Dynamic Planning", "error", str(e))

        iteration += 1

    result["iterations"] = iteration + 1
    result["sources"] = pool.get_sources()

    # ---- Step 12: Persist ----
    if use_memory:
        history.log_query(query, len(pool.get_claims()), critique.get("verdict", "unknown"))

        # Quality-gated save to long-term memory
        saved = ltm.save_verified_claims(
            claims=pool.get_claims(),
            query=query,
            verdict=critique.get("verdict", "")
        )
        if saved:
            log("Long-Term Memory", "saved", f"{saved} verified claims persisted")
            
    if stream_callback:
        stream_callback({"type": "complete", "result": result})

    return result

import asyncio

async def run_research_stream(query: str, max_iterations: int = 3, use_memory: bool = True, conversation_context: str = "", local_only: bool = False):
    """
    Async generator that yields status updates and the final result.
    It runs the synchronous run_research in a separate thread.
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(event):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    future = loop.run_in_executor(
        None,
        lambda: run_research(query, max_iterations, use_memory, stream_callback=callback, conversation_context=conversation_context, local_only=local_only)
    )

    while True:
        event = await queue.get()
        yield event
        if event.get("type") in ("complete", "error"):
            break

    await future
