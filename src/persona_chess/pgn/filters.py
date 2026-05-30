from dataclasses import dataclass
from typing import Literal

PlayerColor = Literal["white", "black", "both"]


@dataclass(frozen=True, slots=True)
class GameFilter:
    player: str
    color: PlayerColor = "both"
    max_games: int | None = None
    include_variants: bool = False

    def normalized_player(self) -> str:
        return self.player.casefold().strip()
