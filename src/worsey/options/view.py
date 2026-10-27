"""The view object (SPEC §4.1): parse-validated fields, a diff for "make it 2 weeks", and
the user's distribution derived from it.

Claude turns the user's words into these fields; this module validates them (strict,
with messages that go back to the model for one retry, SPEC §1b) and does the math.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date

from .pricer import MoveDistribution

DIRECTIONS = ("up", "down", "neutral")
VOL_VIEWS = ("expand", "contract", "none")
CONVICTIONS = ("low", "med", "high")
# log-sd = realized vol·√T × this factor
CONVICTION_SD = {"high": 0.75, "med": 1.0, "low": 1.25}


class ViewError(ValueError):
    pass


@dataclass
class View:
    underlying: str
    direction: str
    horizon: date
    target_move_pct: float | None = None   # signed: +8 = up 8%; None for neutral/vol-only views
    vol_view: str = "none"
    conviction: str = "med"
    event: str | None = None               # e.g. "earnings 2026-11-19"
    constraints: dict = field(default_factory=dict)  # defined_risk_only, max_premium_usd

    def __post_init__(self):
        self.underlying = self.underlying.upper()
        if isinstance(self.horizon, str):
            self.horizon = date.fromisoformat(self.horizon)
        self.validate()

    def validate(self, today: date | None = None):
        errs = []
        if self.direction not in DIRECTIONS:
            errs.append(f"direction must be one of {DIRECTIONS}")
        if self.vol_view not in VOL_VIEWS:
            errs.append(f"vol_view must be one of {VOL_VIEWS}")
        if self.conviction not in CONVICTIONS:
            errs.append(f"conviction must be one of {CONVICTIONS}")
        if self.target_move_pct is not None:
            if not -80 <= self.target_move_pct <= 300:
                errs.append("target_move_pct must be in [-80, 300]")
            if self.direction == "up" and self.target_move_pct < 0 or self.direction == "down" and self.target_move_pct > 0:
                errs.append("target_move_pct sign must match direction")
        elif self.direction != "neutral":
            errs.append("directional views need target_move_pct")
        if today and self.horizon <= today:
            errs.append("horizon must be in the future")
        bad = set(self.constraints) - {"defined_risk_only", "max_premium_usd", "risk_budget_usd"}
        if bad:
            errs.append(f"unknown constraints {sorted(bad)}")
        if errs:
            raise ViewError("; ".join(errs))

    def T(self, today: date) -> float:
        return max((self.horizon - today).days, 1) / 365

    def distribution(self, today: date, realized_vol: float) -> MoveDistribution:
        T = self.T(today)
        sd = realized_vol * math.sqrt(T) * CONVICTION_SD[self.conviction]
        return MoveDistribution((self.target_move_pct or 0.0) / 100, sd, T)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["horizon"] = self.horizon.isoformat()
        return d

    def update(self, **changes) -> tuple["View", dict]:
        """Apply a change ("make it 2 weeks") and return (new view, diff)."""
        new = replace(self, **changes)
        a, b = self.to_dict(), new.to_dict()
        return new, {k: {"from": a[k], "to": b[k]} for k in a if a[k] != b[k]}
