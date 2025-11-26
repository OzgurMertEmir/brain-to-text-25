import heapq
from collections import defaultdict
from math import inf
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

# --- shortest path on LatticeExport with (graph, acoustic) total cost ---
def best_weight_from_lattice(lat):
    """
    lat: lm_decoder.LatticeExport

    Returns:
        best_g, best_a
      where:
        best_g = sum of graph costs (Value1) along best path
        best_a = sum of acoustic costs (Value2) along best path

    Best path = path that minimizes total_cost = best_g + best_a
    (Kaldi LatticeWeight behavior).
    """
    # adjacency: src -> [(dst, g_cost, a_cost)]
    adj = defaultdict(list)
    states = set()

    for src, dst, ilab, olab, g_cost, a_cost in lat.arcs:
        g = float(g_cost)
        a = float(a_cost)
        adj[src].append((dst, g, a))
        states.add(src)
        states.add(dst)

    finals = {}
    for s, g_f, a_f in lat.finals:
        finals[s] = (float(g_f), float(a_f))
        states.add(s)

    if not states:
        raise RuntimeError("Empty lattice")

    start = lat.start_state
    states.add(start)

    # dist_total[state] = best (g+a) seen so far
    dist_total = {s: inf for s in states}
    dist_g = {s: inf for s in states}
    dist_a = {s: inf for s in states}
    dist_total[start] = 0.0
    dist_g[start] = 0.0
    dist_a[start] = 0.0

    # parent pointers if you ever want to reconstruct labels
    # parent[state] = (prev_state, g_arc, a_arc, olabel)
    parent = {}

    pq = [(0.0, start)]  # (total_cost, state)

    while pq:
        total, s = heapq.heappop(pq)
        if total != dist_total[s]:
            continue
        for dst, g_arc, a_arc in adj.get(s, ()):
            new_g = dist_g[s] + g_arc
            new_a = dist_a[s] + a_arc
            new_total = new_g + new_a
            if new_total < dist_total[dst]:
                dist_total[dst] = new_total
                dist_g[dst] = new_g
                dist_a[dst] = new_a
                parent[dst] = (s, g_arc, a_arc)
                heapq.heappush(pq, (new_total, dst))

    # add final weights and pick best final by total cost
    best_state = None
    best_total = inf
    best_g = None
    best_a = None

    for s, (g_f, a_f) in finals.items():
        if dist_total[s] == inf:
            continue
        g0 = dist_g[s]
        a0 = dist_a[s]
        g_tot = g0 + g_f
        a_tot = a0 + a_f
        total = g_tot + a_tot
        if total < best_total:
            best_total = total
            best_g = g_tot
            best_a = a_tot
            best_state = s

    if best_state is None:
        raise RuntimeError("No path reaches a final state")

    return best_g, best_a

@dataclass
class LatticeHypothesis:
    state: int
    g: float          # accumulated graph/LM cost (Value1)
    a: float          # accumulated acoustic cost (Value2)
    llm: float        # accumulated LLM log-prob of the word prefix
    words: Tuple[str, ...]
    score: float      # combined search score (for pruning)


def load_word_symbol_table(words_txt_path: str) -> Dict[int, str]:
    """
    Parse Kaldi-style words.txt into {id -> symbol}.
    """
    id2sym: Dict[int, str] = {}
    with open(words_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            sym, id_str = parts
            try:
                idx = int(id_str)
            except ValueError:
                continue
            id2sym[idx] = sym
    return id2sym


def llm_lattice_rescore(
    lat,
    id2word: Dict[int, str],
    llm_scorer,
    acoustic_scale: float,
    alpha: float,
    beta: float,
    gamma: float,
    beam_size: int = 8,
    nbest: int = 20,
    max_batch_size: int = 1024,
):
    """
    Lattice + (optional) LLM-guided beam search.

    Returns:
        nbest_list: List of (sentence, ac_score, lm_score, llm_score),
                    sorted by combined search score:
                        search_score = alpha*ac_score + beta*lm_score + gamma*llm_score
    """
    # Build adjacency: src -> list of (dst, g_cost, a_cost, olabel)
    adj = defaultdict(list)
    states = set()

    for src, dst, ilab, olab, g_cost, a_cost in lat.arcs:
        g = float(g_cost)
        a = float(a_cost)
        adj[src].append((dst, g, a, int(olab)))
        states.add(src)
        states.add(dst)

    finals = {}
    for s, g_f, a_f in lat.finals:
        finals[s] = (float(g_f), float(a_f))
        states.add(s)

    if not states:
        return []  # empty lattice → no hyps

    start = lat.start_state
    states.add(start)

    def compute_components(g, a, llm_logprob):
        ac_score = -a / acoustic_scale
        lm_score = -g
        return ac_score, lm_score, llm_logprob

    def combined_score(g, a, llm_logprob):
        ac_score, lm_score, llm_score = compute_components(g, a, llm_logprob)
        return alpha * ac_score + beta * lm_score + gamma * llm_logprob

    # LLM cache: prefix tuple -> log P_LLM(prefix)
    llm_cache: Dict[Tuple[str, ...], float] = {}

    h0 = LatticeHypothesis(
        state=start,
        g=0.0,
        a=0.0,
        llm=0.0,
        words=tuple(),
        score=0.0,
    )
    beam: List[LatticeHypothesis] = [h0]
    final_hyps: List[LatticeHypothesis] = []

    visited: Dict[Tuple[int, Tuple[str, ...]], float] = {}

    steps = 0
    max_steps = 1000

    while beam and steps < max_steps:
        steps += 1

        expansions = []  # (dst, g2, a2, words2, needs_llm, llm2)
        unseen_prefixes: Dict[Tuple[str, ...], int] = {}
        texts_to_score: List[str] = []

        # 1) Collect expansions and all new prefixes
        for hyp in beam:
            key = (hyp.state, hyp.words)
            if visited.get(key, float("-inf")) >= hyp.score:
                continue
            visited[key] = hyp.score

            # If this state has a final weight, we can end here
            if hyp.state in finals:
                g_f, a_f = finals[hyp.state]
                g_tot = hyp.g + g_f
                a_tot = hyp.a + a_f
                llm_tot = hyp.llm
                score_tot = combined_score(g_tot, a_tot, llm_tot)

                final_hyps.append(
                    LatticeHypothesis(
                        state=hyp.state,
                        g=g_tot,
                        a=a_tot,
                        llm=llm_tot,
                        words=hyp.words,
                        score=score_tot,
                    )
                )

            for dst, g_arc, a_arc, olabel in adj.get(hyp.state, ()):
                g2 = hyp.g + g_arc
                a2 = hyp.a + a_arc
                base_words = hyp.words
                base_llm = hyp.llm

                if olabel == 0:
                    # epsilon arc
                    expansions.append((dst, g2, a2, base_words, False, base_llm))
                else:
                    word = id2word.get(olabel)
                    if word is None:
                        continue

                    if word in {"<eps>", "<blk>", "<blank>", "<sil>"}:
                        expansions.append((dst, g2, a2, base_words, False, base_llm))
                    else:
                        new_words = base_words + (word.lower(),)
                        if new_words in llm_cache:
                            llm2 = llm_cache[new_words]
                            expansions.append((dst, g2, a2, new_words, False, llm2))
                        else:
                            # mark for batch scoring
                            if new_words not in unseen_prefixes:
                                unseen_prefixes[new_words] = len(texts_to_score)
                                texts_to_score.append(" ".join(new_words))
                            expansions.append((dst, g2, a2, new_words, True, 0.0))

        # 2) Batch LLM scoring for unseen prefixes in CHUNKS
        if texts_to_score:
            # We will fill this list in place:
            lps = [0.0] * len(texts_to_score)
            # Chunk over texts_to_score to avoid huge batches
            for start_idx in range(0, len(texts_to_score), max_batch_size):
                end_idx = start_idx + max_batch_size
                batch_texts = texts_to_score[start_idx:end_idx]
                batch_lps = llm_scorer.sentence_logprob(batch_texts)
                for j, lp in enumerate(batch_lps):
                    lps[start_idx + j] = float(lp)

            # Write all results into cache
            for prefix, idx in unseen_prefixes.items():
                llm_cache[prefix] = lps[idx]

        # 3) Build new beam using cached LLM scores
        new_beam: List[LatticeHypothesis] = []
        for dst, g2, a2, words2, needs_llm, llm2 in expansions:
            if needs_llm:
                llm2 = llm_cache[words2]
            score2 = combined_score(g2, a2, llm2)

            new_beam.append(
                LatticeHypothesis(
                    state=dst,
                    g=g2,
                    a=a2,
                    llm=llm2,
                    words=words2,
                    score=score2,
                )
            )

        if not new_beam:
            break

        new_beam.sort(key=lambda h: h.score, reverse=True)
        beam = new_beam[:beam_size]

    candidates = final_hyps if final_hyps else beam

    if not candidates:
        return []

    candidates.sort(key=lambda h: h.score, reverse=True)
    out = []
    for h in candidates[:nbest]:
        ac_score, lm_score, llm_score = compute_components(h.g, h.a, h.llm)
        sentence = " ".join(h.words)
        out.append((sentence, ac_score, lm_score, llm_score))

    return out
