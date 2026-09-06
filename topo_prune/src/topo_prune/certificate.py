"""The certification check and its falsification audit (plan §3.3-§3.4).

We implement the *check*, not the theorem. Given

- ``D`` -- the superlevel diagram of the unpruned class-logit map ``f_c^(i)``,
- ``tau`` -- the operating point, in logit units, ``tau = logit(p_conf)``,
- ``epsilon`` -- a bound on ``||f_c^(i) - f_c^(i),M||_inf`` for pruning mask ``M``
  (from ``lipschitz.epsilon_certified`` or ``.epsilon_empirical``),

the image/class/level triple is **certified** when no bar endpoint can cross
``tau`` under a move of size ``epsilon`` and no straddling bar is short enough to
be squeezed off ``tau``. Bottleneck stability (Cohen-Steiner-Edelsbrunner-Harer,
Lipschitz constant 1) then guarantees the set of bars alive at ``tau`` -- i.e.
the Betti numbers of ``{f_c^(i) >= tau}`` -- is unchanged by the pruning.

``audit`` checks that guarantee against the Betti numbers actually produced by
the pruned model. On a certified triple they must match; a mismatch means the
bound, the filtration convention, or the code is wrong -- find the bug, do not
tune ``tau`` (``CLAUDE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np

from .filtrations import Diagram, betti_numbers_at


def margin(diagram: Diagram, tau: float) -> float:
    """Smallest distance from any bar endpoint to ``tau`` (``m(tau, D)``, §3.3).

    Essential bars (``death == -inf``) contribute only their birth. An empty
    diagram returns ``+inf`` -- nothing can change, so the margin is unbounded.
    """
    best = np.inf
    for pairs in diagram.by_dim.values():
        if pairs.size == 0:
            continue
        births = pairs[:, 0]
        deaths = pairs[:, 1]
        best = min(best, float(np.min(np.abs(births - tau))))
        finite = deaths[np.isfinite(deaths)]
        if finite.size:
            best = min(best, float(np.min(np.abs(finite - tau))))
    return best


def straddling_min_lifetime(diagram: Diagram, tau: float) -> float:
    """Shortest lifetime among bars alive at ``tau`` (``ell_min``, §3.3).

    "Alive at ``tau``" means ``birth >= tau > death``. Returns ``+inf`` when no
    bar straddles ``tau``.
    """
    best = np.inf
    for pairs in diagram.by_dim.values():
        if pairs.size == 0:
            continue
        births = pairs[:, 0]
        deaths = pairs[:, 1]
        straddle = (births >= tau) & (deaths < tau)
        if not np.any(straddle):
            continue
        # death may be -inf for an essential class -> infinite lifetime.
        lifetimes = births[straddle] - deaths[straddle]
        best = min(best, float(np.min(lifetimes)))
    return best


@dataclass(frozen=True)
class CertificationResult:
    certified: bool
    tau: float
    epsilon: float
    margin: float
    straddling_min_lifetime: float
    betti: dict[int, int]

    def as_dict(self) -> dict:
        return {
            "certified": self.certified,
            "tau": self.tau,
            "epsilon": self.epsilon,
            "margin": self.margin,
            "straddling_min_lifetime": self.straddling_min_lifetime,
            "betti": {str(k): v for k, v in self.betti.items()},
        }


def certify(diagram: Diagram, tau: float, epsilon: float) -> CertificationResult:
    """Run the §3.3 check on one (unpruned diagram, tau, epsilon) triple."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")
    m = margin(diagram, tau)
    ell = straddling_min_lifetime(diagram, tau)
    certified = bool(m > epsilon and ell > 2.0 * epsilon)
    return CertificationResult(
        certified=certified,
        tau=float(tau),
        epsilon=float(epsilon),
        margin=m,
        straddling_min_lifetime=ell,
        betti=betti_numbers_at(diagram, tau),
    )


@dataclass(frozen=True)
class Violation:
    key: str
    tau: float
    epsilon: float
    betti_reference: dict[int, int]
    betti_pruned: dict[int, int]
    margin: float
    straddling_min_lifetime: float


@dataclass
class AuditReport:
    """Falsification harness output (§3.4). ``violations`` should be empty."""

    certified_count: int = 0
    uncertified_count: int = 0
    violations: list[Violation] = _dc_field(default_factory=list)
    # Betti agreement among *uncertified* triples, for context only.
    uncertified_betti_matches: int = 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def sound(self) -> bool:
        return self.violation_count == 0

    def as_dict(self) -> dict:
        return {
            "certified_count": self.certified_count,
            "uncertified_count": self.uncertified_count,
            "violation_count": self.violation_count,
            "sound": self.sound,
            "uncertified_betti_matches": self.uncertified_betti_matches,
            "violations": [
                {
                    "key": v.key,
                    "tau": v.tau,
                    "epsilon": v.epsilon,
                    "betti_reference": {str(k): n for k, n in v.betti_reference.items()},
                    "betti_pruned": {str(k): n for k, n in v.betti_pruned.items()},
                    "margin": v.margin,
                    "straddling_min_lifetime": v.straddling_min_lifetime,
                }
                for v in self.violations
            ],
        }


def _betti_equal(a: dict[int, int], b: dict[int, int]) -> bool:
    keys = set(a) | set(b)
    return all(a.get(k, 0) == b.get(k, 0) for k in keys)


def audit(
    triples: list[tuple[str, Diagram, Diagram, float, float]],
) -> AuditReport:
    """Audit ``(key, reference_diagram, pruned_diagram, tau, epsilon)`` triples.

    ``reference_diagram`` is from the unpruned model (drives certification);
    ``pruned_diagram`` is from the model actually pruned by ``M`` (ground truth
    for the audit). For every certified triple we require
    ``beta_k({f >= tau})`` to agree between the two.
    """
    report = AuditReport()
    for key, reference, pruned, tau, epsilon in triples:
        result = certify(reference, tau, epsilon)
        betti_ref = result.betti
        betti_pruned = betti_numbers_at(pruned, tau)
        if result.certified:
            report.certified_count += 1
            if not _betti_equal(betti_ref, betti_pruned):
                report.violations.append(
                    Violation(
                        key=key,
                        tau=float(tau),
                        epsilon=float(epsilon),
                        betti_reference=betti_ref,
                        betti_pruned=betti_pruned,
                        margin=result.margin,
                        straddling_min_lifetime=result.straddling_min_lifetime,
                    )
                )
        else:
            report.uncertified_count += 1
            if _betti_equal(betti_ref, betti_pruned):
                report.uncertified_betti_matches += 1
    return report


def coverage(results: list[CertificationResult]) -> float:
    """Certified fraction over a set of triples (§3.4 'certified coverage')."""
    if not results:
        return 0.0
    return float(np.mean([r.certified for r in results]))
