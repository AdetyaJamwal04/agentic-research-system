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
from evidence.claim_extractor import extract_claims, format_claims_for_display
from planning.dynamic_planner import revise_plan
from memory.evidence_pool import EvidencePool
from memory.research_history import ResearchHistory
from memory.long_term_memory import LongTermMemory
import json


query = "Why does gradient descent zigzag?"

# ============================================================
# Step 0: Initialize (query-scoped evidence + long-term memory)
# ============================================================
pool = EvidencePool()  # Fresh per query — no cross-query contamination
history = ResearchHistory()
ltm = LongTermMemory()  # Persistent verified claims
ltm_context = ""

# Load relevant long-term claims
relevant_ltm = ltm.retrieve_relevant(query)
if relevant_ltm:
    ltm_context = ltm.format_for_context(relevant_ltm)
    print("=" * 60)
    print(f"LONG-TERM MEMORY: {len(relevant_ltm)} relevant claims loaded")
    print("=" * 60)
    print(ltm_context)

# ============================================================
# Step 1: Analyze the query
# ============================================================
try:
    analysis = analyse_query(query)
    print("\n" + "=" * 60)
    print("STEP 1: QUERY ANALYSIS")
    print("=" * 60)
    print(json.dumps(analysis, indent=2))
except Exception as e:
    print(f"Error in query analysis: {e}")
    analysis = {"original_query": query, "refined_query": query}

# ============================================================
# Step 1.5: Premise Validation
# ============================================================
effective_query = query
premise_correction = None

try:
    premise = validate_premise(analysis)
    print("\n" + "=" * 60)
    print("STEP 1.5: PREMISE VALIDATION")
    print("=" * 60)
    print(json.dumps(premise, indent=2))

    if premise.get("premise_status") == "incorrect":
        premise_correction = premise.get("correction", "")
        effective_query = premise.get("corrected_query", query)
        print(f"\n  ⚠ FALSE PREMISE DETECTED")
        print(f"  Correction: {premise_correction}")
        print(f"  Using corrected query: {effective_query}")

        # Update analysis with corrected query
        analysis["original_query"] = effective_query
        analysis["refined_query"] = effective_query
        analysis["premise_correction"] = premise_correction

    elif premise.get("premise_status") == "uncertain":
        print(f"\n  ⚡ UNCERTAIN PREMISE — proceeding with caution")
    else:
        print(f"\n  ✓ Premise is valid")
except Exception as e:
    print(f"Error in premise validation: {e}")

# ============================================================
# Step 2: Create a structured retrieval plan
# ============================================================
try:
    plan = create_plan(analysis)
    print("\n" + "=" * 60)
    print("STEP 2: RETRIEVAL PLAN")
    print("=" * 60)
    print(json.dumps(plan, indent=2))
except Exception as e:
    print(f"Error in planning: {e}")
    plan = {"tasks": [{"task_id": 1, "description": f"Retrieve information about: {effective_query}"}]}

# Track all tasks for dynamic planner
all_tasks = list(plan.get("tasks", []))

# ============================================================
# Step 3: Generate task-mapped search queries
# ============================================================
try:
    queries_generated = generate_queries(plan)
    print("\n" + "=" * 60)
    print("STEP 3: GENERATED QUERIES")
    print("=" * 60)
    print(json.dumps(queries_generated, indent=2))

    for tq in queries_generated.get("task_queries", []):
        print(f"  Task {tq['task_id']} → {tq['queries']}")
except Exception as e:
    print(f"Error generating queries: {e}")
    queries_generated = {"task_queries": [{"task_id": 1, "queries": [effective_query]}]}

# ============================================================
# Step 4: Retrieve documents (hybrid: web + local)
# ============================================================
try:
    all_tagged_docs = retrieve_documents(queries_generated)
    print("\n" + "=" * 60)
    print("STEP 4: RETRIEVAL (hybrid)")
    print("=" * 60)
    print(f"Retrieved {len(all_tagged_docs)} documents")

    for doc in all_tagged_docs:
        src = doc.get("source_type", "?")
        print(f"  [{src}] Task {doc['task_id']} | {doc['query'][:40]}... → {doc['content'][:60]}...")
except Exception as e:
    print(f"Error retrieving documents: {e}")
    all_tagged_docs = []

if not all_tagged_docs:
    print("\nNo documents retrieved. Cannot generate an answer.")
else:
    critique = {}  # Initialize before use in Step 12
    # ============================================================
    # Step 5: Early Domain Filter (before pool, before rerank)
    # ============================================================
    try:
        doc_contents_raw = [d["content"] for d in all_tagged_docs]
        filtered_contents = filter_documents(effective_query, doc_contents_raw)
        if filtered_contents:
            matching = set(c[:200] for c in filtered_contents)
            all_tagged_docs = [d for d in all_tagged_docs if d["content"][:200] in matching]
        print(f"\n  Domain filter: {len(doc_contents_raw)} → {len(all_tagged_docs)} docs")
    except Exception as e:
        print(f"Error in domain filter: {e}")

    # ============================================================
    # Step 6: Add filtered docs to Evidence Pool
    # ============================================================
    pool.add(all_tagged_docs)
    print(f"\n{pool.summary()}")

    doc_contents = pool.get_contents()

    # ============================================================
    # Step 7: Rerank documents
    # ============================================================
    try:
        docs_for_reasoning = rerank_documents(effective_query, doc_contents)
        print(f"  Reranker: {len(doc_contents)} → {len(docs_for_reasoning)} docs")
    except Exception as e:
        print(f"Error reranking: {e}")
        docs_for_reasoning = doc_contents[:5]

    # ============================================================
    # Step 8: Compress context
    # ============================================================
    try:
        compressed_docs = compress_context(effective_query, docs_for_reasoning)
    except Exception as e:
        print(f"Error compressing context: {e}")
        compressed_docs = [doc[:800] for doc in docs_for_reasoning]

    # ============================================================
    # Step 9: Claim Extraction (atomic evidence)
    # ============================================================
    numbered_evidence = pool.format_for_reasoning()  # Always available for fallback
    try:
        claims = extract_claims(effective_query, numbered_evidence)
        claims_added = pool.add_claims(claims)

        print("\n" + "=" * 60)
        print(f"STEP 9: CLAIM EXTRACTION ({len(claims)} claims)")
        print("=" * 60)
        print(format_claims_for_display(claims))
    except Exception as e:
        print(f"Error in claim extraction: {e}")
        claims = []

    # ============================================================
    # Step 10: Evidence Synthesis (over claims)
    # ============================================================
    try:
        # Use claim-level evidence if available, else fall back to doc-level
        if pool.get_claims():
            evidence_text = pool.format_claims_for_reasoning()
        else:
            evidence_text = numbered_evidence

        synthesis = synthesize_evidence(effective_query, evidence_text)
        synthesis_text = format_synthesis_for_reasoning(synthesis)
        print("\n" + "=" * 60)
        print("STEP 10: EVIDENCE SYNTHESIS")
        print("=" * 60)
        print(synthesis_text)

        gaps = synthesis.get("coverage_gaps", [])
        if gaps:
            print(f"\n  Coverage gaps identified: {gaps}")
    except Exception as e:
        print(f"Error in evidence synthesis: {e}")
        synthesis = None
        synthesis_text = None
        evidence_text = numbered_evidence if numbered_evidence else ""

    # ============================================================
    # Step 11: Generate answer + Critic loop (adaptive)
    # ============================================================
    max_iterations = 3
    iteration = 0

    evidence_for_reasoning = evidence_text

    # Prepend long-term memory context if available
    if ltm_context:
        evidence_for_reasoning = f"{ltm_context}\n\n{evidence_for_reasoning}"

    # If premise was incorrect, prepend correction to evidence
    if premise_correction:
        evidence_for_reasoning = f"IMPORTANT PREMISE CORRECTION: {premise_correction}\n\n{evidence_for_reasoning}"

    while iteration < max_iterations:

        try:
            answer = generate_answer(effective_query, evidence_for_reasoning)
        except Exception as e:
            print(f"Error generating answer: {e}")
            break

        print("\n" + "=" * 60)
        print(f"STEP 11: ANSWER (iteration {iteration + 1})")
        print("=" * 60)
        print(answer)

        print("\n" + "-" * 40)
        print("SOURCES")
        print("-" * 40)
        print(pool.format_sources_list())

        try:
            critique = critique_answer(effective_query, answer)
        except Exception as e:
            print(f"Error in critic: {e}")
            break

        print("\n" + "-" * 40)
        print("CRITIC EVALUATION")
        print("-" * 40)
        print(json.dumps(critique, indent=2))

        if critique.get("verdict") == "sufficient":
            print("\n✓ Answer accepted by critic")
            break

        # ---- Dynamic Planning: create new tasks from gaps ----
        missing = critique.get("missing_aspects", [])
        print(f"\n✗ Insufficient — missing: {missing}")

        if not missing:
            iteration += 1
            continue

        pool.next_iteration()

        try:
            # Dynamic planner creates proper tasks (not ad-hoc queries)
            revised = revise_plan(effective_query, missing, all_tasks)
            new_tasks = revised.get("tasks", [])

            print(f"\n  Dynamic planner created {len(new_tasks)} new tasks:")
            for t in new_tasks:
                print(f"    Task {t.get('task_id', '?')}: {t.get('description', '')[:80]}")

            all_tasks.extend(new_tasks)

            # Generate queries from new tasks (full pipeline)
            new_plan = {"tasks": new_tasks}
            new_queries = generate_queries(new_plan)
            print(f"  Generated queries: {json.dumps(new_queries, indent=2)}")

            # Retrieve new documents
            new_docs = retrieve_documents(new_queries)
            added = pool.add(new_docs)
            print(f"  Added {added} new documents to evidence pool")

            # Extract claims from new evidence
            new_numbered = pool.format_for_reasoning()
            new_claims = extract_claims(effective_query, new_numbered)
            pool.add_claims(new_claims)
            print(f"  Extracted {len(new_claims)} new claims")
            print(f"  {pool.summary()}")

            # Re-synthesize with enriched claims
            evidence_for_reasoning = pool.format_claims_for_reasoning()
            new_synthesis = synthesize_evidence(effective_query, evidence_for_reasoning)
            if new_synthesis.get("evidence_brief"):
                synthesis_text = format_synthesis_for_reasoning(new_synthesis)
                print(f"  Re-synthesized ({len(new_synthesis['evidence_brief'])} themes)")

            if ltm_context:
                evidence_for_reasoning = f"{ltm_context}\n\n{evidence_for_reasoning}"
            if premise_correction:
                evidence_for_reasoning = f"IMPORTANT PREMISE CORRECTION: {premise_correction}\n\n{evidence_for_reasoning}"

        except Exception as e:
            print(f"  Error in adaptive retrieval: {e}")

        iteration += 1

    # ============================================================
    # Step 12: Save to long-term memory & log research
    # ============================================================
    history.log_query(
        query=query,
        claims_found=len(pool.get_claims()),
        verdict=critique.get("verdict", "unknown")
    )

    # Quality-gated save: only critic-approved, high-confidence claims
    saved_count = ltm.save_verified_claims(
        claims=pool.get_claims(),
        query=query,
        verdict=critique.get("verdict", "")
    )

    print("\n" + "=" * 60)
    print("STEP 12: MEMORY")
    print("=" * 60)
    print(f"  {history.summary()}")
    if saved_count:
        print(f"  ✓ Saved {saved_count} verified claims to long-term memory")
    else:
        print(f"  — No claims met quality threshold for long-term storage")
    print(f"  {ltm.summary()}")
