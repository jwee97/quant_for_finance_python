"""What an evaluation says, in words.

The answer is built from the numbers of ``evaluate`` and nothing else, at the horizon the person chose. It has two parts because they have different answers in practice: how well a method describes the *spread* of
what might happen (volatility-aware methods usually beat the random walk there), and whether it forecasts the *average*, the direction (almost none does, beside the historical average and always saying up).
"""

from __future__ import annotations


ICON = {"good": "✓", "mixed": "!", "bad": "✗"}
ALPHA = 0.05                                                       # a method beats the baseline when its Holm-adjusted p-value is below this and its skill is positive
MIN_SKILL = 0.01                                                   # and by at least this share of the baseline's CRPS


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%"


def _label(result: dict, mid: str) -> str:
    """The short name of a method (the long one, with what it assumes, is on the page beside the results)."""
    return next((m.get("short") or m["label"] for m in result["methods"] if m["id"] == mid), mid)


def judge(result: dict, horizon: int | None = None) -> dict:
    """``{"level", "headline", "reasons"}`` for the horizon (the evaluation's headline horizon by default)."""
    hz = str(int(horizon or result["headline"]))
    base = result["baseline"]
    o = result["origins"]
    rows = []
    for m in result["methods"]:
        s = m["scores"].get(hz)
        if m["id"] == base or not s or s.get("skill") is None:
            continue
        rows.append((m, s))
    reasons = [f"Tested on {o['count']} forecasts made between {o['first']} and {o['last']}, one every {o['step_days']} trading days, each made with only the prices up to its day and scored on the {hz} trading days that followed."
               f" (About {max(1, round(o['count'] * o['step_days'] / max(int(hz), 1)))} of them are independent of each other: forecasts closer together than the horizon share most of their outcome, and the tests allow for it.)"]
    if not rows:
        return {"level": "bad", "headline": "Nothing could be scored", "reasons": reasons + ["No method produced a forecast that could be compared with the random walk."]}
    ranked = sorted(rows, key=lambda t: -(t[1]["skill"] or -9))
    winners = [(m, s) for m, s in ranked if s.get("beats_baseline") and (s["skill"] or 0) >= MIN_SKILL]
    best_m, best_s = ranked[0]
    base_row = next(m for m in result["methods"] if m["id"] == base)["scores"][hz]
    reasons.append(f"The yardstick is the random walk with the spread of the last year ({_label(result, base)}). Its average CRPS, the score of a whole forecast distribution (lower is better), was {base_row['crps'] * 1000:.2f} per 1,000.")
    if winners:
        level = "good"
        names = ", ".join(f"{m.get('short') or m['label']} ({pct(s['skill'], 1)})" for m, s in winners[:4]) + (f" and {len(winners) - 4} more" if len(winners) > 4 else "")
        top_m, top_s = winners[0]
        headline = f"{len(winners)} method{'s' if len(winners) != 1 else ''} described the spread of {hz}-day returns better than the random walk, the best by {abs(top_s['skill']) * 100:.0f}%"
        reasons.append(f"Better than the random walk by more than luck allows, after allowing for the number of methods compared (Holm-adjusted p under {ALPHA:.0%}): {names}.")
        reasons.append(f"The best was {top_m.get('short') or top_m['label']}: its forecasts scored {pct(top_s['skill'], 1)} against the random walk's (adjusted p = {top_s.get('p_adjusted', float('nan')):.3f}).")
    elif (best_s["skill"] or 0) > 0:
        level = "mixed"
        headline = "Some methods did better than the random walk, but not by more than luck could explain"
        reasons.append(f"The best was {best_m.get('short') or best_m['label']} at {pct(best_s['skill'], 1)}, with a p-value of {best_s.get('p', float('nan')):.3f} ({best_s.get('p_adjusted', float('nan')):.3f} after allowing for the number of methods compared; {ALPHA:.0%} is asked for).")
    else:
        level = "bad"
        headline = f"No method forecast the spread of {hz}-day returns better than the random walk"
        reasons.append(f"The best, {best_m.get('short') or best_m['label']}, scored {pct(best_s['skill'], 1)}: no better than a constant spread read off the last year.")
    worse = [(m, s) for m, s in ranked if (s["skill"] or 0) < -0.02]
    if worse:
        reasons.append("Worse than the random walk: " + ", ".join(f"{m.get('short') or m['label']} ({pct(s['skill'], 0)})" for m, s in worse[-3:][::-1]) + ". A method can be more elaborate and less honest about what it does not know.")
    cal = best_m if not winners else winners[0][0]
    cs = cal["scores"][hz]
    if cs.get("coverage") and cs["coverage"].get("90") is not None:
        c90, p90 = cs["coverage"]["90"], cs.get("coverage_p", {}).get("90")
        off = p90 is not None and p90 < 0.05
        reasons.append(f"Calibration of {cal.get('short') or cal['label']}: the central 90% range of its forecast held the outcome {c90 * 100:.0f}% of the time ({'not consistent with 90%' if off else 'consistent with 90%'}), the central 50% range {cs['coverage']['50'] * 100:.0f}%, "
                       f"and the outcome fell below its 5% quantile {cs['var5'] * 100:.0f}% of the time (5% is the target).")
    mean_rows = [(m, s) for m, s in rows if "mean" in s and s["mean"].get("r2") is not None]
    if mean_rows:
        useful = [(m, s) for m, s in mean_rows if s["mean"]["r2"] > 0 and (s["mean"].get("p") or 1) < ALPHA / max(len(mean_rows), 1)]
        best_mean = max(mean_rows, key=lambda t: t[1]["mean"]["r2"])
        if useful:
            m, s = max(useful, key=lambda t: t[1]["mean"]["r2"])
            reasons.append(f"Average return: {m.get('short') or m['label']} forecast it better than the historical average of the last five years (out-of-sample R-squared {s['mean']['r2'] * 100:+.1f}%, Clark-West p = {s['mean']['p']:.3f}, Bonferroni-adjusted threshold {ALPHA / len(mean_rows):.3f}).")
        else:
            reasons.append(f"Average return: none of the {len(mean_rows)} methods that forecast one beat simply using the historical average of the last five years. The best was {best_mean[0].get('short') or best_mean[0]['label']} with an out-of-sample R-squared of {best_mean[1]['mean']['r2'] * 100:+.1f}%.")
        dirs = [(m, s) for m, s in rows if "direction" in s and s["direction"].get("edge") is not None]
        if dirs:
            top = max(dirs, key=lambda t: t[1]["direction"]["edge"])
            d = top[1]["direction"]
            reasons.append(f"Direction: always saying up was right {d['always_up'] * 100:.0f}% of the time over this stretch. The best caller was {top[0].get('short') or top[0]['label']} at {d['hit'] * 100:.0f}% ({d['edge'] * 100:+.0f} points, p = {d['p']:.2f}); "
                           + ("none beat always saying up by more than luck allows." if all((s['direction'].get('p') or 1) * len(dirs) >= ALPHA for _, s in dirs) else "at least one beat always saying up, which needs checking on other stocks and periods before it is believed."))
    for c in result.get("controls", []):
        if str(c["horizon"]) == hz and c["skill"] is not None:
            reasons.append(f"Does {_label(result, c['method'])} add anything to its simpler control, {_label(result, c['control'])}? Its forecasts scored {pct(c['skill'], 1)} against it (p = {c['p']:.2f}): "
                           + ("yes, by more than luck allows." if (c["p"] or 1) < ALPHA and c["skill"] > 0 else "no, what it shows is what the control already has."))
    failed = [m for m in result["methods"] if m["failures"] and m["id"] != base]
    if failed:
        reasons.append("Could not forecast at some origins: " + ", ".join(f"{m.get('short') or m['label']} ({m['failures']} of {o['count']})" for m in failed) + ". Those origins are left out of that method's scores.")
    reasons.append("A forecast is not a strategy: a method can describe the spread of returns well and still make no money, and the other way round. These results are for one stock over one history; another period would rank the methods differently, and the ranking among the best few is within what luck could change.")
    return {"level": level, "headline": headline, "reasons": reasons}
