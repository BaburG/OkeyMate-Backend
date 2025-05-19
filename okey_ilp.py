import itertools
from typing import List, Dict, Optional, Tuple
import pulp


class Piece:
    """Represents an Okey tile (real or fake joker, or normal piece)."""
    def __init__(self, color: str, number: Optional[int], okey_color: Optional[str] = None, okey_number: Optional[int] = None):
        # Determine type
        self.type = (
            "okey" if color == "okey" else
            "fake_okey" if color == "fake_okey" else
            "piece"
        )
        # Assign attributes based on type
        if self.type == "okey":
            # Real joker: get actual color/number from game context
            self.color = okey_color
            self.number = okey_number
        elif self.type == "fake_okey":
            # Fake joker: acts as next number of real okey
            self.color = okey_color
            self.number = (okey_number % 13) + 1 if okey_number is not None else None
        else:
            # Normal piece
            self.color = color
            self.number = number

    def __repr__(self):
        return f"Piece(color={self.color!r}, number={self.number!r}, type={self.type!r})"

    def __str__(self):
        # Always show real joker as "JOKER"
        if self.type == "okey":
            return "JOKER"
        if self.type == "fake_okey":
            return f"FAKE_OKEY({self.color} {self.number})"
        return f"{self.color} {self.number}"


class OkeyILPSolver:
    """
    ILP solver for Okey melding using PuLP.
    Input:
      - pieces: List[Dict{"color": str, "number": Optional[int]}]
      - okey_color: str
      - okey_number: int
    Output: List of melds, each meld is List[Piece]
    """
    def __init__(self,
                 pieces: List[Dict],
                 okey_color: Optional[str] = None,
                 okey_number: Optional[int] = None):
        # Build Piece objects, passing okey context
        self.tiles: List[Piece] = []
        for p in pieces:
            pc = Piece(
                p['color'], p['number'],
                okey_color=okey_color, okey_number=okey_number
            )
            self.tiles.append(pc)

        # Precompute joker indices
        self.joker_indices = [i for i, t in enumerate(self.tiles) if t.type == 'okey']
        self.non_joker_indices = [i for i, t in enumerate(self.tiles) if t.type != 'okey']

    def _enumerate_melds(self):
        melds = []
        W = len(self.joker_indices)

        # Sets: same number, unique colors, size >=3 (including jokers)
        for num in range(1, 14):
            real_idxs = [i for i in self.non_joker_indices if self.tiles[i].number == num]
            max_size = min(4, len(real_idxs) + W)
            for size in range(3, max_size + 1):
                for r in range(max(1, size - W), min(size, len(real_idxs)) + 1):
                    k = size - r
                    for real_comb in itertools.combinations(real_idxs, r):
                        for joker_comb in itertools.combinations(self.joker_indices, k):
                            idxs = list(real_comb) + list(joker_comb)
                            weight = num * size
                            melds.append((idxs, weight))

        # Runs: same color, consecutive numbers, length >=3 (including jokers)
        colors = set(t.color for t in self.tiles if t.type != 'okey')
        for color in colors:
            num_to_idx = {t.number: i for i, t in enumerate(self.tiles)
                          if t.type != 'okey' and t.color == color}
            for length in range(3, 14):
                for start in range(1, 14 - length + 1):
                    run_nums = list(range(start, start + length))
                    real_idxs = [num_to_idx[n] for n in run_nums if n in num_to_idx]
                    missing = length - len(real_idxs)
                    if missing <= W and len(real_idxs) >= 1:
                        for joker_comb in itertools.combinations(self.joker_indices, missing):
                            idxs = sorted(real_idxs + list(joker_comb))
                            weight = sum(run_nums)
                            melds.append((idxs, weight))

        return melds

    def solve(self) -> List[Tuple[List[Piece], int]]:
        melds = self._enumerate_melds()
        prob = pulp.LpProblem("okey_max_meld", pulp.LpMaximize)

        # Create binary variables
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(melds))]

        # Objective
        prob += pulp.lpSum(melds[i][1] * x[i] for i in range(len(melds)))

        # Constraints: each tile can be in at most one meld
        for tile_idx in range(len(self.tiles)):
            prob += (
                pulp.lpSum(x[i] for i, (idxs, _) in enumerate(melds) if tile_idx in idxs) <= 1,
                f"tile_{tile_idx}_once"
            )

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract solution melds
        solution: List[Tuple[List[Piece], int]] = []
        for i, var in enumerate(x):
            if pulp.value(var) == 1:
                idxs, weight = melds[i]
                meld_pieces = [self.tiles[idx] for idx in idxs]
                solution.append((meld_pieces, weight))
        return solution
