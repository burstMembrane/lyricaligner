"""Module for aligning audio with text using force alignment"""
from dataclasses import dataclass

import torch


@dataclass
class Point:
    """Represents a point in the alignment path"""
    token_index: int  # Index in the token sequence
    time_index: int   # Index in the time sequence
    prob: float       # Probability of this alignment point


class Aligner:
    """Aligns audio logits with text tokens using force alignment"""
    
    @staticmethod
    def align(emission, tokens, blank_id=0):
        """Align emission logits with tokens using force alignment
        
        Args:
            emission: Log probabilities from ASR model
            tokens: Token IDs to align with
            blank_id: ID of the blank token
            
        Returns:
            List of alignment points
        """
        trellis = Aligner.get_trellis(emission, tokens, blank_id=blank_id)
        path = Aligner.backtrack(trellis, emission, tokens, blank_id=blank_id)
        return path

    @staticmethod
    def get_trellis(emission, tokens, blank_id=0):
        """Build trellis matrix for alignment using dynamic programming
        
        Args:
            emission: Log probabilities from ASR model
            tokens: Token IDs to align with
            blank_id: ID of the blank token
            
        Returns:
            Trellis matrix for alignment
        """
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    @staticmethod
    def backtrack(trellis, emission, tokens, blank_id=0):
        """Backtrack through trellis to find alignment path
        
        Args:
            trellis: Trellis matrix from get_trellis
            emission: Log probabilities from ASR model
            tokens: Token IDs to align with
            blank_id: ID of the blank token
            
        Returns:
            List of alignment points
        """
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
            prob = (emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item())
            
            path.append(Point(j - 1, t - 1, prob))
            
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to find a valid alignment path")
            
        return path[::-1]
