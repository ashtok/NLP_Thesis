import numpy as np

class SimpleCTCDecoder:
    """
    Proper CTC prefix beam search decoder (no external LM).
    - Maintains beam of prefixes (collapsed CTC sequences).
    - At each timestep, tracks two probabilities per prefix:
      * p_nb: probability that prefix ends with a non-blank
      * p_b: probability that prefix ends with a blank
    - Merges hypotheses that produce the same prefix during decoding.
    - After the last frame, returns the top-N prefixes.

    This implementation properly handles CTC path merging and is more
    accurate than naive beam search that only collapses at the end.
    """

    def __init__(
        self,
        blank_id: int = 0,
        beam_size: int = 10,
        top_k_per_timestep: int = 20,
        length_norm: bool = False,
    ):
        self.blank_id = int(blank_id)
        self.beam_size = int(max(1, beam_size))
        self.top_k = int(max(1, top_k_per_timestep))
        self.length_norm = bool(length_norm)

    def decode(self, emissions: np.ndarray, n_best: int = 1) -> list[list[int]]:
        """
        Args:
            emissions: [T, V] array of LOG probabilities (log-softmaxed).
            n_best: how many best collapsed sequences to return.

        Returns:
            List of length <= nbest; each is a List[int] of token ids (after CTC collapse).
        """
        if emissions.ndim != 2:
            raise ValueError("emissions must be [T, V] log-probabilities")
        T, V = emissions.shape

        # Precompute top-k ids per frame for speed
        k = min(self.top_k, V)
        if k < V:
            topk_ids = np.argpartition(emissions, -k, axis=1)[:, -k:]
            # Reorder top-k by score (descending)
            idx = np.take_along_axis(emissions, topk_ids, axis=1).argsort(axis=1)[
                :, ::-1
            ]
            topk_ids = np.take_along_axis(topk_ids, idx, axis=1)
        else:
            topk_ids = emissions.argsort(axis=1)[:, ::-1]

        # Beam state: dict mapping prefix (tuple of ints) to (p_nb, p_b)
        # p_nb: log probability that this prefix ends with a non-blank
        # p_b: log probability that this prefix ends with a blank
        # Initialize with empty prefix
        NEG_INF = float('-inf')
        beam: dict[tuple[int, ...], tuple[float, float]] = {
            (): (NEG_INF, 0.0)  # empty prefix: p_nb=-inf, p_b=0.0
        }

        for t in range(T):
            frame_scores = emissions[t]  # [V], log-probs
            cand_tokens = topk_ids[t]  # [k]

            next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

            for prefix, (p_nb, p_b) in beam.items():
                # Total probability of this prefix (log-sum-exp of p_nb and p_b)
                p_total = np.logaddexp(p_nb, p_b)

                # 1) Extend with blank
                blank_score = p_total + float(frame_scores[self.blank_id])
                if prefix in next_beam:
                    old_p_nb, old_p_b = next_beam[prefix]
                    next_beam[prefix] = (old_p_nb, np.logaddexp(old_p_b, blank_score))
                else:
                    next_beam[prefix] = (NEG_INF, blank_score)

                # 2) Extend with non-blank tokens
                for tok in cand_tokens:
                    tok = int(tok)
                    if tok == self.blank_id:
                        continue

                    token_score = float(frame_scores[tok])
                    new_prefix = prefix + (tok,)

                    if len(prefix) > 0 and prefix[-1] == tok:
                        # Repeating the last token: only p_b path can extend
                        new_p_nb = p_b + token_score
                    else:
                        # Different token: both paths can extend
                        new_p_nb = p_total + token_score

                    if new_prefix in next_beam:
                        old_p_nb, old_p_b = next_beam[new_prefix]
                        next_beam[new_prefix] = (
                            np.logaddexp(old_p_nb, new_p_nb),
                            old_p_b
                        )
                    else:
                        next_beam[new_prefix] = (new_p_nb, NEG_INF)

            # Prune beam to top beam_size prefixes by total probability
            if len(next_beam) > self.beam_size:
                # Calculate total probability for each prefix
                scored = [
                    (prefix, np.logaddexp(p_nb, p_b))
                    for prefix, (p_nb, p_b) in next_beam.items()
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                beam = {
                    prefix: next_beam[prefix]
                    for prefix, _ in scored[:self.beam_size]
                }
            else:
                beam = next_beam

        # Final ranking by total probability
        finals = []
        for prefix, (p_nb, p_b) in beam.items():
            score = np.logaddexp(p_nb, p_b)
            if self.length_norm and len(prefix) > 0:
                score = score / len(prefix)
            finals.append((list(prefix), float(score)))

        finals.sort(key=lambda x: x[1], reverse=True)
        return [prefix for prefix, _ in finals[:n_best]]

