"""Module for aligning audio emissions with text tokens using forced alignment"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
from line_profiler import profile


@dataclass
class Point:
    """Represents a point in the alignment path"""

    token_position: int  # Position in the token sequence
    frame_index: int  # Index in the time/emission sequence
    confidence: float  # Confidence score (exp of log-probability)


@profile
class ForcedAligner:
    """Aligns audio emissions with text tokens using dynamic programming"""

    @staticmethod
    def align(
        emission_log_probs: "torch.Tensor",
        token_ids: list[int],
        blank_token_id: int = 0,
    ):
        """
        Aligns log-probabilities with token sequence using forced alignment.

        Args:
            emission_log_probs: Log-probabilities from ASR model (frames × vocab_size)
            token_ids: Sequence of token IDs to align
            blank_token_id: ID used for the blank token

        Returns:
            List of Point indicating matched token positions and time frames
        """
        trellis = ForcedAligner._compute_trellis(
            emission_log_probs, token_ids, blank_token_id
        )
        alignment_path = ForcedAligner._backtrack_path(
            trellis, emission_log_probs, token_ids, blank_token_id
        )
        return alignment_path

    @staticmethod
    def _compute_trellis(
        emission_log_probs: "torch.Tensor", token_ids: list[int], blank_token_id: int
    ):
        """
        Construct the trellis (DP matrix) used for alignment.

        Args:
            emission_log_probs: Log-probabilities from ASR model
            token_ids: Sequence of token IDs to align
            blank_token_id: ID used for the blank token

        Returns:
            Trellis matrix for alignment
        """
        import torch

        num_frames = emission_log_probs.size(0)
        num_tokens = len(token_ids)
        trellis = torch.empty((num_frames + 1, num_tokens + 1))

        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission_log_probs[:, blank_token_id], dim=0)
        trellis[0, 1:] = -float("inf")

        for frame in range(num_frames):
            stay_scores = trellis[frame, 1:] + emission_log_probs[frame, blank_token_id]
            change_scores = trellis[frame, :-1] + emission_log_probs[frame, token_ids]
            trellis[frame + 1, 1:] = torch.maximum(stay_scores, change_scores)

        return trellis

    @staticmethod
    def _backtrack_path(
        trellis: "torch.Tensor",
        emission_log_probs: "torch.Tensor",
        token_ids: list[int],
        blank_token_id: int,
    ):
        """
        Backtrack through trellis to find the best alignment path.

        Args:
            trellis: Trellis matrix from _compute_trellis
            emission_log_probs: Log-probabilities from ASR model
            token_ids: Sequence of token IDs
            blank_token_id: ID used for the blank token

        Returns:
            List of Point in chronological order
        """
        import torch

        token_pos = trellis.size(1) - 1
        frame_start = torch.argmax(trellis[:, token_pos]).item()

        alignment = []
        for frame in range(frame_start, 0, -1):
            stay_score = (
                trellis[frame - 1, token_pos]
                + emission_log_probs[frame - 1, blank_token_id]
            )
            change_score = (
                trellis[frame - 1, token_pos - 1]
                + emission_log_probs[frame - 1, token_ids[token_pos - 1]]
            )

            chose_change = change_score > stay_score
            chosen_token = token_ids[token_pos - 1] if chose_change else blank_token_id
            confidence = emission_log_probs[frame - 1, chosen_token].exp().item()

            alignment.append(
                Point(
                    token_position=token_pos - 1,
                    frame_index=frame - 1,
                    confidence=confidence,
                )
            )

            if chose_change:
                token_pos -= 1
                if token_pos == 0:
                    break
        else:
            raise ValueError("Failed to construct a valid alignment path")

        return alignment[::-1]
