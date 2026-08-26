import math

from src.api.schemas import AuthorVectorAggregate, GraphAuthorEvidence, DbsfScoredCandidate


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
    def calculate_raw_graph_score(evidence: GraphAuthorEvidence | None) -> float:
        if evidence is None or evidence.is_spam_or_gambling:
            return 0.0
        if evidence.topic_coverage_weight == 0.0 and evidence.direct_mentions_count == 0 and evidence.matched_entities_count == 0:
            return 0.0
        mentions_signal = min(1.0, evidence.direct_mentions_count * 0.25 + evidence.matched_entities_count * 0.15)
        ontological_priors = (
            (0.30 if evidence.has_role_relation else 0.0)
            + (0.25 if evidence.has_tech_relation else 0.0)
            + (0.20 if evidence.is_creator else 0.0)
            - (0.15 if evidence.is_promoter and not (evidence.has_role_relation or evidence.is_creator) else 0.0)
        )
        priors_signal = min(1.0, max(0.0, ontological_priors))
        raw_score = 0.45 * evidence.topic_coverage_weight + 0.35 * mentions_signal + 0.20 * priors_signal
        return min(1.0, max(0.0, raw_score))

    @staticmethod
    def _normalize_distribution(values: list[float], tau: float = 1.8) -> list[float]:
        if not values:
            return []
        if max(values) == 0.0:
            return [0.0] * len(values)
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
    def rank_candidates(
        vector_aggregates: dict[int, AuthorVectorAggregate],
        graph_evidences: dict[int, GraphAuthorEvidence],
        vector_weight: float = 0.60,
        graph_weight: float = 0.40,
    ) -> list[DbsfScoredCandidate]:
        all_ids = list(set(vector_aggregates.keys()) | set(graph_evidences.keys()))

        raw_vector_scores: dict[int, float] = {}
        raw_graph_scores: dict[int, float] = {}

        for account_id in all_ids:
            evidence = graph_evidences.get(account_id)
            if evidence is not None and evidence.is_spam_or_gambling:
                continue

            agg = vector_aggregates.get(account_id)
            if agg is not None:
                raw_vector = DbsfRankingEngine.aggregate_author_vector_score(agg.post_scores)
            else:
                raw_vector = 0.0
            raw_vector_scores[account_id] = raw_vector

            raw_graph = DbsfRankingEngine.calculate_raw_graph_score(evidence)
            raw_graph_scores[account_id] = raw_graph

        clean_ids = list(raw_vector_scores.keys())

        vec_values = [raw_vector_scores[aid] for aid in clean_ids]
        graph_values = [raw_graph_scores[aid] for aid in clean_ids]

        normalized_vec = DbsfRankingEngine._normalize_distribution(vec_values)
        normalized_graph = DbsfRankingEngine._normalize_distribution(graph_values)

        norm_vec_map: dict[int, float] = dict(zip(clean_ids, normalized_vec))
        norm_graph_map: dict[int, float] = dict(zip(clean_ids, normalized_graph))

        candidates: list[DbsfScoredCandidate] = []
        for account_id in clean_ids:
            nv = norm_vec_map[account_id]
            ng = norm_graph_map[account_id]
            if nv == 0.0 and ng == 0.0:
                final_score = 0.0
            else:
                final_score = vector_weight * nv + graph_weight * ng
            candidates.append(
                DbsfScoredCandidate(
                    account_id=account_id,
                    raw_vector_score=raw_vector_scores[account_id],
                    raw_graph_score=raw_graph_scores[account_id],
                    normalized_vector_score=nv,
                    normalized_graph_score=ng,
                    final_score=final_score,
                )
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates