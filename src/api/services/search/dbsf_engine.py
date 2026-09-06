import math

from src.api.schemas import AuthorVectorAggregate, GraphAuthorEvidence, DbsfScoredCandidate
from src.utils.languages import canonicalize_language, canonicalize_languages


class DbsfRankingEngine:

    @staticmethod
    def aggregate_author_vector_score(scores: list[float], base_threshold: float = 0.40, decay_lambda: float = 0.70) -> float:
        if not scores:
            return 0.0
        scores_sorted = sorted(scores, reverse=True)
        total = scores_sorted[0]
        for i in range(1, len(scores_sorted)):
            total += max(0.0, scores_sorted[i] - base_threshold) / ((i + 1) ** decay_lambda)
        return total

    @staticmethod
    def calculate_raw_graph_score(
        evidence: GraphAuthorEvidence | None,
        target_tone: str | None = None,
        target_hormones: list[str] | None = None,
    ) -> float:
        if evidence is None or evidence.is_spam_or_gambling:
            return 0.0
        if evidence.topic_coverage_weight == 0.0 and evidence.direct_mentions_count == 0 and evidence.matched_topics_count == 0:
            return 0.0
        mentions_signal = min(1.0, evidence.direct_mentions_count * 0.25 + evidence.matched_entities_count * 0.15)
        tone_match = 0.15 if evidence.primary_tone and target_tone and evidence.primary_tone.lower() == target_tone.lower() else 0.0
        hormone_match = 0.15 if evidence.primary_hormone and target_hormones and any(h and evidence.primary_hormone.lower() == h.lower() for h in target_hormones) else 0.0
        ontological_priors = (
            (0.35 if evidence.has_role_relation else 0.0)
            + (0.30 if evidence.has_tech_relation else 0.0)
            + (0.25 if evidence.is_creator else 0.0)
            - (0.15 if evidence.is_promoter and not (evidence.has_role_relation or evidence.is_creator) else 0.0)
            + tone_match
            + hormone_match
        )
        priors_signal = min(1.0, max(0.0, ontological_priors))
        raw_score = 0.50 * evidence.topic_coverage_weight + 0.35 * mentions_signal + 0.15 * priors_signal
        return min(1.0, max(0.0, raw_score))

    @staticmethod
    def _normalize_distribution(values: list[float], tau: float = 1.8) -> list[float]:
        if not values:
            return []
        mu = sum(values) / len(values)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))
        if sigma < 1e-6:
            return [min(1.0, max(0.0, v)) for v in values]
        result = []
        for v in values:
            if v == 0.0:
                result.append(0.0)
            else:
                z = (v - mu) / sigma
                result.append(1.0 / (1.0 + math.exp(-tau * z)))
        return result

    @staticmethod
    def _matches_language(author_lang: str | None, target_languages: set[str]) -> bool:
        canonical_author_lang = canonicalize_language(author_lang)
        if canonical_author_lang is None:
            return False
        return canonical_author_lang in target_languages

    @staticmethod
    def rank_candidates(
        vector_aggregates: dict[int, AuthorVectorAggregate],
        graph_evidences: dict[int, GraphAuthorEvidence],
        vector_weight: float = 0.65,
        graph_weight: float = 0.35,
        target_languages: list[str] | None = None,
        target_tone: str | None = None,
        target_hormones: list[str] | None = None,
        affinity_mode: bool = False,
        affinity_reason: str | None = None,
        match_type: str = "direct",
    ) -> list[DbsfScoredCandidate]:
        all_ids = list(set(vector_aggregates.keys()) | set(graph_evidences.keys()))

        raw_vector_scores: dict[int, float] = {}
        raw_graph_scores: dict[int, float] = {}

        for account_id in all_ids:
            evidence = graph_evidences.get(account_id)
            if evidence is not None and (evidence.is_spam_or_gambling or getattr(evidence, "has_negative_match", False)):
                continue

            agg = vector_aggregates.get(account_id)
            if agg is not None:
                raw_vector = DbsfRankingEngine.aggregate_author_vector_score(agg.post_scores)
            else:
                raw_vector = 0.0
            raw_vector_scores[account_id] = raw_vector

            raw_graph = DbsfRankingEngine.calculate_raw_graph_score(evidence, target_tone=target_tone, target_hormones=target_hormones)
            raw_graph_scores[account_id] = raw_graph

        clean_ids = list(raw_vector_scores.keys())

        canonical_targets = canonicalize_languages(target_languages)

        vec_values = [raw_vector_scores[aid] for aid in clean_ids]
        graph_values = [raw_graph_scores[aid] for aid in clean_ids]

        normalized_vec = DbsfRankingEngine._normalize_distribution(vec_values)
        normalized_graph = DbsfRankingEngine._normalize_distribution(graph_values)

        norm_vec_map: dict[int, float] = dict(zip(clean_ids, normalized_vec))
        norm_graph_map: dict[int, float] = dict(zip(clean_ids, normalized_graph))

        candidates: list[DbsfScoredCandidate] = []
        for account_id in clean_ids:
            evidence = graph_evidences.get(account_id)
            nv = norm_vec_map[account_id]
            ng = norm_graph_map[account_id]
            raw_graph = raw_graph_scores[account_id]

            if evidence is not None and raw_graph > 0.0:
                base_score = vector_weight * nv + graph_weight * ng
            elif evidence is not None and evidence.total_topics_count > 0 and evidence.matched_topics_count == 0 and evidence.direct_mentions_count == 0:
                base_score = nv * 0.20
            else:
                base_score = nv * 0.60

            final_score = min(1.0, max(0.0, base_score))

            if affinity_mode and final_score < 0.35:
                continue

            candidates.append(
                DbsfScoredCandidate(
                    account_id=account_id,
                    raw_vector_score=raw_vector_scores[account_id],
                    raw_graph_score=raw_graph,
                    normalized_vector_score=nv,
                    normalized_graph_score=ng,
                    final_score=final_score,
                    match_type=match_type,
                    affinity_reason=affinity_reason if match_type == "affinity" else None,
                )
            )

        if canonical_targets:
            target_lang_candidates: list[DbsfScoredCandidate] = []
            fallback_candidates: list[DbsfScoredCandidate] = []
            for candidate in candidates:
                evidence = graph_evidences.get(candidate.account_id)
                author_lang = evidence.primary_language if evidence else None
                if DbsfRankingEngine._matches_language(author_lang, canonical_targets):
                    target_lang_candidates.append(candidate)
                else:
                    fallback_candidates.append(candidate)
            target_lang_candidates.sort(key=lambda c: -c.final_score)
            fallback_candidates.sort(key=lambda c: -c.final_score)
            return target_lang_candidates + fallback_candidates
        candidates.sort(key=lambda c: -c.final_score)
        return candidates

    @staticmethod
    def interleave_candidates(
        direct_candidates: list[DbsfScoredCandidate],
        affinity_candidates: list[DbsfScoredCandidate],
    ) -> list[DbsfScoredCandidate]:
        seen_ids: set[int] = set()
        merged: list[DbsfScoredCandidate] = []
        for candidate in direct_candidates:
            if candidate.account_id not in seen_ids:
                seen_ids.add(candidate.account_id)
                merged.append(candidate)
        for candidate in affinity_candidates:
            if candidate.account_id not in seen_ids:
                seen_ids.add(candidate.account_id)
                merged.append(candidate)
        return merged

    @staticmethod
    def combine_multi_cluster_candidates(
        audience_cluster_pools: list[list[DbsfScoredCandidate]],
        direct_account_ids: set[int],
    ) -> list[DbsfScoredCandidate]:
        active_pools: list[list[DbsfScoredCandidate]] = []
        for pool in audience_cluster_pools:
            if not pool:
                continue
            sorted_pool = sorted(pool, key=lambda c: -c.final_score)
            active_pools.append(sorted_pool)

        if not active_pools:
            return []

        seen_ids: set[int] = set(direct_account_ids)
        merged: list[DbsfScoredCandidate] = []
        pool_index = 0
        while True:
            pool = active_pools[pool_index]
            while pool:
                candidate = pool.pop(0)
                if candidate.account_id not in seen_ids:
                    seen_ids.add(candidate.account_id)
                    merged.append(candidate)
                    break
            pool_index += 1
            if pool_index >= len(active_pools):
                pool_index = 0
            if all(not p for p in active_pools):
                break
        return merged