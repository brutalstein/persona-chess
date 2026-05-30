from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MoveTendencies:
    capture_rate: float
    check_rate: float
    kingside_castle_rate: float
    queenside_castle_rate: float
    promotion_rate: float

    @classmethod
    def from_counts(
        cls,
        *,
        moves: int,
        captures: int,
        checks: int,
        kingside_castles: int,
        queenside_castles: int,
        promotions: int,
    ) -> "MoveTendencies":
        denominator = max(moves, 1)
        return cls(
            capture_rate=captures / denominator,
            check_rate=checks / denominator,
            kingside_castle_rate=kingside_castles / denominator,
            queenside_castle_rate=queenside_castles / denominator,
            promotion_rate=promotions / denominator,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveTendencies":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PersonaProfile:
    player: str
    games: int
    white_games: int
    black_games: int
    target_moves: int
    tendencies: MoveTendencies
    result_distribution: dict[str, int]
    first_move_distribution: dict[str, int]
    opening_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tendencies"] = self.tendencies.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaProfile":
        payload = dict(data)
        payload["tendencies"] = MoveTendencies.from_dict(payload["tendencies"])
        return cls(**payload)
