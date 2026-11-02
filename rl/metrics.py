#!/usr/bin/env python3
"""The evaluation metrics of proposal S4.4.3.

    python3 -m rl.metrics --selftest

Equations 4.1-4.5, verbatim from the proposal:

    4.1  S_i = 1 if d(p_final_i, g_i) <= delta and C_i == 0 and t_i <= T_max, else 0
    4.2  SR  = (1/N) * sum_i S_i
    4.3  SPL = (1/N) * sum_i S_i * l*_i / max(l_i, l*_i)
    4.4  dM  = M_sim - M_real,  for M in {SR, SPL}
    4.5  CR  = sum_i c_i / sum_i l_i          (collisions per metre travelled)

Why this is a module and not a function in the evaluator
-------------------------------------------------------
SR, SPL and CR must be computed identically for the photogrammetric and standard policies,
or the comparison in Sub-question 2 measures the scoring code.

A note on the SPL denominator
-----------------------------
`max(l_i, l*_i)` is in the proposal and in Savva et al., and it is doing real work: a robot
that cuts a corner the planner would not can traverse *less* than l*, and without the max
its SPL contribution would exceed 1 and a policy could score above 1.0 overall. It also
absorbs the case where l* is a slight over-estimate because the planner works on a grid.

A note on what l* must be
-------------------------
l* is the shortest *feasible* path, not the straight-line distance. In a room with
furniture those differ, and using Euclidean distance inflates SPL for every route that has
to go around something -- unequally across the two branches, because the two branches have
different geometry. See `rl/env/routes.py`, which computes it once per room and caches it.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class RouteResult:
    """One evaluation route.

    Every field is something the evaluator must record at the time.
    """

    route_id: str
    reached_goal: bool          # final position within delta of the goal
    final_distance_m: float     # d(p_final, g), Eq. 4.1
    collisions: int             # c_i, Eq. 4.5. Zero or more; C_i in Eq. 4.1 is (c_i > 0)
    path_length_m: float        # l_i, actually traversed
    optimal_length_m: float     # l*_i, shortest feasible, from routes.py
    duration_s: float           # t_i
    timed_out: bool = False     # t_i > T_max
    notes: str = ""

    def success(self, goal_tolerance_m):
        """S_i, Eq. 4.1. All three conditions, not just the first."""
        return bool(
            self.final_distance_m <= goal_tolerance_m
            and self.collisions == 0
            and not self.timed_out
        )

    def spl_term(self, goal_tolerance_m):
        """S_i * l*_i / max(l_i, l*_i), one term of Eq. 4.3."""
        if not self.success(goal_tolerance_m):
            return 0.0
        denom = max(self.path_length_m, self.optimal_length_m)
        if denom <= 0.0:
            # Start and goal coincide: degenerate, and a route set should not contain one.
            # Scoring it 1.0 rather than dividing by zero, and routes.py rejects them.
            return 1.0
        return self.optimal_length_m / denom


@dataclass
class Evaluation:
    """The aggregate of one policy over one route set."""

    branch: str                 # "photogrammetry" | "standard"
    source: str                 # "sim" | "real"
    goal_tolerance_m: float
    max_duration_s: float
    routes: list = field(default_factory=list)

    # -- Eq. 4.2 ------------------------------------------------------------------------
    @property
    def sr(self):
        if not self.routes:
            return float("nan")
        return sum(r.success(self.goal_tolerance_m) for r in self.routes) / len(self.routes)

    # -- Eq. 4.3 ------------------------------------------------------------------------
    @property
    def spl(self):
        if not self.routes:
            return float("nan")
        return sum(r.spl_term(self.goal_tolerance_m) for r in self.routes) / len(self.routes)

    # -- Eq. 4.5 ------------------------------------------------------------------------
    @property
    def cr(self):
        """Collisions per metre travelled, over ALL routes -- failed ones included.

        The denominator is sum(l_i) across the whole route set, not across successes.
        Restricting it to successes would divide the collisions of the failures by a
        distance that excludes them, and CR exists precisely to characterise how a policy
        fails.
        """
        total_length = sum(r.path_length_m for r in self.routes)
        if total_length <= 0.0:
            return float("nan")
        return sum(r.collisions for r in self.routes) / total_length

    def summary(self):
        return {
            "branch": self.branch,
            "source": self.source,
            "n_routes": len(self.routes),
            "SR": self.sr,
            "SPL": self.spl,
            "CR_per_m": self.cr,
            "goal_tolerance_m": self.goal_tolerance_m,
            "max_duration_s": self.max_duration_s,
        }

    def to_json(self, path):
        """Write aggregates AND every per-route record.

        Per-route records are not optional. Task 7 asks how much of the real success rate
        survives a perturbation, which is a per-route question, and re-running an
        evaluation to recover numbers that were computed once already is the exact rework
        that logging from the first run is meant to avoid.
        """
        payload = {"summary": self.summary(),
                   "routes": [asdict(r) for r in self.routes]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def from_json(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = d["summary"]
        ev = Evaluation(branch=s["branch"], source=s["source"],
                        goal_tolerance_m=s["goal_tolerance_m"],
                        max_duration_s=s["max_duration_s"])
        ev.routes = [RouteResult(**r) for r in d["routes"]]
        return ev


# --------------------------------------------------------------------------------------
# Eq. 4.4 -- the sim-to-real gap
# --------------------------------------------------------------------------------------

def sim_to_real_gap(sim, real):
    """dSR and dSPL for one branch. Computed separately per branch, per S4.4.3.

    The hypothesis predicts d_photo < d_standard for both. Note the sign: d is
    sim - real, so it is normally positive (simulation flatters the policy) and *smaller
    is better*. A negative gap means the policy did better on hardware than in the
    simulator it trained in, which is worth investigating rather than celebrating.
    """
    if sim.branch != real.branch:
        raise ValueError(
            f"gap is defined per branch, but got sim={sim.branch!r} real={real.branch!r}")
    if sim.source != "sim" or real.source != "real":
        raise ValueError(f"expected sim/real, got {sim.source!r}/{real.source!r}")
    sim_ids = [r.route_id for r in sim.routes]
    real_ids = [r.route_id for r in real.routes]
    # Sets, not sorted lists: either side may pool several sessions over the one set.
    if set(sim_ids) != set(real_ids):
        # "over the same route set" is in the definition of Eq. 4.4. Differing route sets
        # produce a number that looks exactly like a valid gap.
        raise ValueError(
            "sim and real evaluations cover different routes; Eq. 4.4 is defined over the "
            f"same route set. sim-only={sorted(set(sim_ids) - set(real_ids))}, "
            f"real-only={sorted(set(real_ids) - set(sim_ids))}")
    return {"branch": sim.branch,
            "delta_SR": sim.sr - real.sr,
            "delta_SPL": sim.spl - real.spl}


def retention(real_nominal, real_perturbed):
    """Sub-question 3: the proportion of real success retained under perturbation.

    Task 7 changes routes and displaces furniture and asks which policy holds up. The
    proposal states the comparison as "a greater proportion of its real-world success
    rate", so this is a ratio and not a difference: a branch that starts at 0.20 and keeps
    0.15 retains more than one that starts at 0.60 and keeps 0.30, even though the second
    is still the better policy in absolute terms. Report retention alongside both raw
    rates, never on its own.
    """
    if real_nominal.branch != real_perturbed.branch:
        raise ValueError("retention is defined per branch")
    if real_nominal.sr <= 0.0:
        return float("nan")   # nothing to retain; do not report 0/0 as a result
    return real_perturbed.sr / real_nominal.sr


def compare_branches(photo, standard):
    """Sub-question 2, at one source: higher SR, higher SPL, lower CR."""
    if photo.source != standard.source:
        raise ValueError(
            f"comparing {photo.source!r} against {standard.source!r}; the branch "
            f"comparison is defined within one source")
    return {
        "source": photo.source,
        "SR":  {"photogrammetry": photo.sr,  "standard": standard.sr,
                "photo_better": photo.sr > standard.sr},
        "SPL": {"photogrammetry": photo.spl, "standard": standard.spl,
                "photo_better": photo.spl > standard.spl},
        "CR":  {"photogrammetry": photo.cr,  "standard": standard.cr,
                "photo_better": photo.cr < standard.cr},   # lower is better
        "significance": {
            "SR": sr_significance(photo, standard),
            "SPL": spl_significance(photo, standard),
        },
    }


# --------------------------------------------------------------------------------------
# S4.4.3 significance testing
#
# "Where sample sizes permit, differences are tested for statistical significance using
# an appropriate test (e.g. a two-proportion z-test for SR, or a Mann-Whitney U test for
# SPL distributions), at a significance level of alpha = 0.05." Named in the proposal,
# implemented nowhere until this section. Both live next to the estimators they test, so
# a change to how SR or SPL is computed cannot drift out of step with how it is tested.
# --------------------------------------------------------------------------------------

ALPHA = 0.05


def sr_significance(a, b, alpha=ALPHA):
    """Two-proportion z-test comparing SR between two Evaluations.

    H0: the two branches/sources have the same true success rate. Uses the pooled
    proportion under H0 for the standard error, which is the textbook two-proportion
    z-test (not Welch's, which would use the two sample proportions separately -- that
    version is for a difference-of-means test, not a shared-proportion test).

    "Where sample sizes permit" (S4.4.3) is not a formality: with N=20 routes and a low
    SR, the normal approximation the z-test relies on is shaky. `min_expected` reports
    whether both groups clear the conventional n*p >= 5 and n*(1-p) >= 5 threshold, and
    the result says so explicitly rather than silently returning a number nobody should
    trust.
    """
    from scipy.stats import norm

    n1, n2 = len(a.routes), len(b.routes)
    if n1 == 0 or n2 == 0:
        return {"z": None, "p_value": None, "significant": None, "alpha": alpha,
                "note": "at least one evaluation has zero routes"}
    x1 = sum(r.success(a.goal_tolerance_m) for r in a.routes)
    x2 = sum(r.success(b.goal_tolerance_m) for r in b.routes)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    sample_ok = all(n * p >= 5 and n * (1 - p) >= 5
                    for n, p in ((n1, p1), (n2, p2)))
    if se == 0.0:
        # Both proportions identical (often both 0 or both 1): no evidence of a
        # difference, and dividing by zero would otherwise look like an error rather
        # than the correct answer of "no difference detected".
        return {"z": 0.0, "p_value": 1.0, "significant": False, "alpha": alpha,
                "n1": n1, "n2": n2, "p1": p1, "p2": p2, "sample_size_adequate": sample_ok}
    z = (p1 - p2) / se
    p_value = 2.0 * norm.sf(abs(z))
    return {"z": z, "p_value": p_value, "significant": bool(p_value < alpha),
            "alpha": alpha, "n1": n1, "n2": n2, "p1": p1, "p2": p2,
            "sample_size_adequate": sample_ok}


def spl_significance(a, b, alpha=ALPHA):
    """Mann-Whitney U test comparing the per-route SPL-term distributions.

    "SPL distributions" (S4.4.3) means the per-route values whose mean IS SPL (Eq. 4.3),
    each in [0, 1] with a large mass at exactly 0 for failed routes -- not a quantity with
    a sensible normal approximation, which is exactly the case Mann-Whitney (a
    distribution-free rank test) is for.
    """
    from scipy.stats import mannwhitneyu

    if a.goal_tolerance_m != b.goal_tolerance_m:
        raise ValueError(
            f"SPL is compared at one goal tolerance; got {a.goal_tolerance_m} and "
            f"{b.goal_tolerance_m}. Both evaluations must share rl/config/env.yaml.")
    xa = [r.spl_term(a.goal_tolerance_m) for r in a.routes]
    xb = [r.spl_term(b.goal_tolerance_m) for r in b.routes]
    if not xa or not xb:
        return {"u": None, "p_value": None, "significant": None, "alpha": alpha,
                "note": "at least one evaluation has zero routes"}
    u, p_value = mannwhitneyu(xa, xb, alternative="two-sided")
    return {"u": float(u), "p_value": float(p_value),
            "significant": bool(p_value < alpha), "alpha": alpha,
            "n1": len(xa), "n2": len(xb)}


# --------------------------------------------------------------------------------------
# S4.4.3 multiple evaluation sessions
#
# "Each policy is evaluated on the same fixed set of N start/goal routes, repeated over
# multiple independent sessions to obtain a mean and variability." Aggregates a list of
# per-session Evaluations of the SAME branch/source/route-set into the mean +- sd that
# statement asks for.
# --------------------------------------------------------------------------------------

def session_summary(evaluations):
    """Mean and sample standard deviation of SR, SPL and CR across independent sessions.

    A pooled Evaluation (one flat list of every route from every session) is deliberately
    NOT computed here: pooling would answer "what is the overall rate", which is a
    different question from "how much does it vary session to session", and S4.4.3 asks
    for the second. `sr_significance`/`spl_significance` take two single Evaluations, not
    a list of sessions -- comparing two conditions' session-level variability, rather
    than their pooled rates, needs a different test (e.g. Welch's t-test on the per-
    session means) and is not implemented here.
    """
    if not evaluations:
        raise ValueError("need at least one session")
    branches = {e.branch for e in evaluations}
    sources = {e.source for e in evaluations}
    if len(branches) > 1 or len(sources) > 1:
        raise ValueError(
            f"sessions must all be the same branch/source; got branches={branches}, "
            f"sources={sources}")

    def stats(values):
        n = len(values)
        mean = sum(values) / n
        sd = (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
              if n > 1 else 0.0)
        return {"mean": mean, "sd": sd, "n_sessions": n, "values": values}

    return {
        "branch": evaluations[0].branch,
        "source": evaluations[0].source,
        "n_sessions": len(evaluations),
        "SR": stats([e.sr for e in evaluations]),
        "SPL": stats([e.spl for e in evaluations]),
        "CR_per_m": stats([e.cr for e in evaluations]),
    }


def load_sessions(path):
    """Every session an evaluation file stands for, as a list of Evaluations.

    An evaluation file has one of two shapes: a single-session file (`summary` + `routes`),
    or a session_summary() file whose `session_files` name the per-session files (paths
    relative to the repo root). Either loads here, so results with different session counts
    can be set against each other without anyone picking a session by hand.
    """
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if "summary" in d:
        return [Evaluation.from_json(p)]
    if "session_files" in d:
        repo = Path(__file__).resolve().parents[1]
        return [Evaluation.from_json(repo / f) for f in d["session_files"]]
    raise ValueError(f"{p} is neither a per-session evaluation nor a session summary")


def pooled(evaluations):
    """One Evaluation holding every route of every session.

    With equal sessions over one route set, its SR and SPL equal the mean of the session
    values exactly; its CR is total collisions over total distance, the ratio Eq. 4.5
    defines. These are the point estimates the comparisons use. The pooled routes are NOT
    independent samples (each route recurs once per session), so a test that treats them
    as independent overstates the evidence -- see paired_route_test for the one that does
    not.
    """
    if not evaluations:
        raise ValueError("need at least one session")
    first = evaluations[0]
    for e in evaluations[1:]:
        if (e.branch, e.source, e.goal_tolerance_m, e.max_duration_s) != (
                first.branch, first.source, first.goal_tolerance_m, first.max_duration_s):
            raise ValueError("sessions differ in branch, source or task settings")
        if sorted(r.route_id for r in e.routes) != sorted(r.route_id for r in first.routes):
            raise ValueError("sessions cover different route sets")
    ev = Evaluation(branch=first.branch, source=first.source,
                    goal_tolerance_m=first.goal_tolerance_m,
                    max_duration_s=first.max_duration_s)
    ev.routes = [r for e in evaluations for r in e.routes]
    return ev


def per_route_rates(evaluations):
    """{route_id: (success rate, mean SPL term)} across sessions."""
    tol = evaluations[0].goal_tolerance_m
    acc = {}
    for e in evaluations:
        for r in e.routes:
            acc.setdefault(r.route_id, []).append((float(r.success(tol)), r.spl_term(tol)))
    return {rid: (sum(s for s, _ in v) / len(v), sum(t for _, t in v) / len(v))
            for rid, v in acc.items()}


def paired_route_test(a_sessions, b_sessions, alpha=ALPHA):
    """Wilcoxon signed-rank test over routes, pairing each route with itself.

    Both sides of every comparison here run the SAME route set (S4.4.3), and a route's
    difficulty is shared by both sides, so the honest unit of analysis is the route: its
    success rate (and mean SPL term) across sessions on side a, against the same route on
    side b. This respects the repeated sessions instead of counting them as extra
    independent routes, and it is what the pooled two-proportion z-test cannot do. With
    20 routes it has modest power; report it next to the pooled tests, not instead.
    """
    from scipy.stats import wilcoxon

    ra, rb = per_route_rates(a_sessions), per_route_rates(b_sessions)
    if sorted(ra) != sorted(rb):
        raise ValueError("paired test needs the same route set on both sides")
    ids = sorted(ra)
    out = {"n_routes": len(ids), "alpha": alpha,
           "n_sessions": (len(a_sessions), len(b_sessions))}
    for name, idx in (("SR", 0), ("SPL", 1)):
        diffs = [ra[i][idx] - rb[i][idx] for i in ids]
        if all(d == 0 for d in diffs):
            out[name] = {"statistic": None, "p_value": 1.0, "significant": False,
                         "note": "identical on every route"}
            continue
        # zero_method="wilcox" drops tied routes, the textbook signed-rank convention.
        # With ties present SciPy falls back to the normal approximation and warns about
        # it; n_untied is recorded instead, so the reader can judge the sample.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stat, p_value = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        out[name] = {"statistic": float(stat), "p_value": float(p_value),
                     "significant": bool(p_value < alpha),
                     "n_untied": sum(d != 0 for d in diffs),
                     "routes_a_better": sum(d > 0 for d in diffs),
                     "routes_b_better": sum(d < 0 for d in diffs)}
    return out


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def _selftest():
    fails = []

    def check(label, got, want, tol=1e-9):
        ok = (want is None and got is None) or (
            isinstance(got, float) and isinstance(want, float) and
            (math.isnan(got) and math.isnan(want) or abs(got - want) <= tol)
        ) or got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<56} got={got!r}")
        if not ok:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    def route(**kw):
        base = dict(route_id="r", reached_goal=True, final_distance_m=0.1, collisions=0,
                    path_length_m=10.0, optimal_length_m=10.0, duration_s=30.0,
                    timed_out=False)
        base.update(kw)
        return RouteResult(**base)

    DELTA = 0.25

    # -- Eq. 4.1: all three conditions, independently --------------------------------
    check("S=1 when inside delta, no collision, in time",
          route().success(DELTA), True)
    check("S=0 when outside delta",
          route(final_distance_m=0.30).success(DELTA), False)
    check("S=0 on collision even though the goal was reached",
          route(collisions=1).success(DELTA), False)
    check("S=0 on timeout even though the goal was reached",
          route(timed_out=True).success(DELTA), False)

    # -- Eq. 4.3: the max() in the denominator ----------------------------------------
    check("SPL term = 1 on a perfectly efficient route",
          route(path_length_m=10.0, optimal_length_m=10.0).spl_term(DELTA), 1.0)
    check("SPL term = 0.5 on a route twice as long as optimal",
          route(path_length_m=20.0, optimal_length_m=10.0).spl_term(DELTA), 0.5)
    check("SPL term capped at 1 when the robot beats l* (corner-cutting)",
          route(path_length_m=8.0, optimal_length_m=10.0).spl_term(DELTA), 1.0)
    check("SPL term = 0 for a failure regardless of efficiency",
          route(collisions=1, path_length_m=10.0, optimal_length_m=10.0).spl_term(DELTA), 0.0)

    # -- Eqs. 4.2, 4.3, 4.5 over a route set ------------------------------------------
    ev = Evaluation(branch="photogrammetry", source="sim",
                    goal_tolerance_m=DELTA, max_duration_s=120.0, routes=[
        route(route_id="a"),                                          # success, SPL 1.0
        route(route_id="b", path_length_m=20.0),                      # success, SPL 0.5
        route(route_id="c", collisions=2, path_length_m=5.0),         # failure
        route(route_id="d", timed_out=True, path_length_m=15.0),      # failure
    ])
    check("SR over 2 successes in 4", ev.sr, 0.5)
    check("SPL over the same set", ev.spl, (1.0 + 0.5) / 4.0)
    # CR: 2 collisions over 10+20+5+15 = 50 m travelled, failures included.
    check("CR counts distance from failed routes too", ev.cr, 2.0 / 50.0)

    # -- Eq. 4.4 ------------------------------------------------------------------------
    real = Evaluation(branch="photogrammetry", source="real",
                      goal_tolerance_m=DELTA, max_duration_s=120.0, routes=[
        route(route_id="a"), route(route_id="b", collisions=1),
        route(route_id="c", collisions=1), route(route_id="d", timed_out=True)])
    gap = sim_to_real_gap(ev, real)
    check("delta_SR = SR_sim - SR_real", gap["delta_SR"], 0.5 - 0.25)

    mismatched = Evaluation(branch="photogrammetry", source="real",
                            goal_tolerance_m=DELTA, max_duration_s=120.0,
                            routes=[route(route_id="a")])
    try:
        sim_to_real_gap(ev, mismatched)
        check("gap rejects a mismatched route set", False, True)
    except ValueError:
        check("gap rejects a mismatched route set", True, True)

    # -- round trip ---------------------------------------------------------------------
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ev.json")
        ev.to_json(p)
        back = Evaluation.from_json(p)
        check("JSON round trip preserves SR", back.sr, ev.sr)
        check("JSON round trip preserves SPL", back.spl, ev.spl)
        check("JSON round trip preserves every route", len(back.routes), len(ev.routes))

    # -- S4.4.3 significance testing ------------------------------------------------------
    # A large, obviously-different pair: 18/20 successes vs 2/20. The z-test should
    # reject "no difference" outright.
    strong_a = Evaluation(branch="photogrammetry", source="sim", goal_tolerance_m=DELTA,
                          max_duration_s=120.0,
                          routes=[route(route_id=str(i)) for i in range(18)]
                                + [route(route_id=str(i), collisions=1) for i in range(18, 20)])
    strong_b = Evaluation(branch="standard", source="sim", goal_tolerance_m=DELTA,
                          max_duration_s=120.0,
                          routes=[route(route_id=str(i)) for i in range(2)]
                                + [route(route_id=str(i), collisions=1) for i in range(2, 20)])
    sig = sr_significance(strong_a, strong_b)
    check("z-test detects an obvious SR difference (p < 0.05)",
          sig["significant"], True)
    # n=20 at p=0.9 gives n*(1-p)=2 < 5, so the rule-of-thumb threshold correctly says
    # NOT to trust the normal approximation here -- this is the check earning its keep,
    # not a bug: it is exactly the "where sample sizes permit" qualifier from S4.4.3.
    check("z-test correctly flags 18/20 as too skewed for the normal approximation",
          sig["sample_size_adequate"], False)

    identical = Evaluation(branch="standard", source="sim", goal_tolerance_m=DELTA,
                           max_duration_s=120.0,
                           routes=[route(route_id=str(i)) for i in range(18)]
                                 + [route(route_id=str(i), collisions=1) for i in range(18, 20)])
    sig_same = sr_significance(strong_a, identical)
    check("z-test does not flag two identical SRs as significant",
          sig_same["significant"], False)

    zero_se = Evaluation(branch="standard", source="sim", goal_tolerance_m=DELTA,
                         max_duration_s=120.0,
                         routes=[route(route_id=str(i)) for i in range(20)])
    zero_se_b = Evaluation(branch="photogrammetry", source="sim", goal_tolerance_m=DELTA,
                           max_duration_s=120.0,
                           routes=[route(route_id=str(i)) for i in range(20)])
    sig_zero = sr_significance(zero_se, zero_se_b)
    check("z-test handles two 100% SRs (zero standard error) without a crash",
          (sig_zero["significant"], sig_zero["p_value"]), (False, 1.0))

    mwu = spl_significance(strong_a, strong_b)
    check("Mann-Whitney U runs on an obvious SPL difference and finds it",
          mwu["significant"], True)

    # -- S4.4.3 multiple sessions ---------------------------------------------------------
    sessions = [
        Evaluation(branch="photogrammetry", source="real", goal_tolerance_m=DELTA,
                  max_duration_s=120.0, routes=[route(route_id="a")]),
        Evaluation(branch="photogrammetry", source="real", goal_tolerance_m=DELTA,
                  max_duration_s=120.0, routes=[route(route_id="a", collisions=1)]),
        Evaluation(branch="photogrammetry", source="real", goal_tolerance_m=DELTA,
                  max_duration_s=120.0, routes=[route(route_id="a")]),
    ]
    summ = session_summary(sessions)
    check("session_summary averages SR across sessions",
          summ["SR"]["mean"], 2.0 / 3.0)
    check("session_summary reports session count", summ["n_sessions"], 3)
    check("session_summary reports nonzero spread when sessions differ",
          summ["SR"]["sd"] > 0.0, True)

    mismatched_sessions = sessions + [
        Evaluation(branch="standard", source="real", goal_tolerance_m=DELTA,
                  max_duration_s=120.0, routes=[route(route_id="a")])
    ]
    try:
        session_summary(mismatched_sessions)
        check("session_summary rejects mixed branches", False, True)
    except ValueError:
        check("session_summary rejects mixed branches", True, True)

    # -- pooling and the paired per-route test ----------------------------------------
    def ev_of(source, outcomes):
        # outcomes: one bool per route r00..; False = stopped 2 m short
        return Evaluation(branch="standard", source=source, goal_tolerance_m=DELTA,
                          max_duration_s=120.0,
                          routes=[route(route_id=f"r{i:02d}", reached_goal=ok,
                                        final_distance_m=0.1 if ok else 2.0)
                                  for i, ok in enumerate(outcomes)])
    s1 = ev_of("sim", [True] * 8 + [False] * 2)
    s2 = ev_of("sim", [True] * 6 + [False] * 4)
    pool = pooled([s1, s2])
    check("pooled SR equals the mean of equal-sized sessions", pool.sr, 0.7)
    check("pooled CR is total collisions over total distance", pool.cr, 0.0)
    try:
        pooled([s1, ev_of("sim", [True] * 9)])
        check("pooled rejects sessions over different route sets", False, True)
    except ValueError:
        check("pooled rejects sessions over different route sets", True, True)

    real_bad = ev_of("real", [False] * 10)
    pr = paired_route_test([s1, s2], [real_bad])
    check("paired test counts the routes each side wins",
          (pr["SR"]["routes_a_better"], pr["SR"]["routes_b_better"]), (8, 0))
    check("paired test pairs every route once", pr["n_routes"], 10)
    same = paired_route_test([s1], [s1])
    check("paired test of a set against itself finds nothing",
          (same["SR"]["significant"], same["SR"]["p_value"]), (False, 1.0))

    print()
    if fails:
        print(f"{len(fails)} FAILED:")
        for f in fails:
            print("   " + f)
        return 1
    print("all metric checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("nothing to do; pass --selftest (this module is a library otherwise)")
    sys.exit(_selftest())
