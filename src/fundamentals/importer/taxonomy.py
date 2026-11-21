"""Financial taxonomy mapping: what a row label *means*, kept apart from how the document was read.

A mapping takes the label of a row, the statement it sits in, and returns the canonical line it names (or nothing). It never reads a file. There is one mapping per accounting standard, all built
on the shared multilingual label list and each adding the labels and exclusions that are specific to it:

``us-gaap`` (US GAAP), ``ifrs`` (IFRS: Hong Kong, Europe and the default), ``sfrs-i`` (SFRS(I), Singapore), ``k-ifrs`` (Korea), ``j-gaap`` (Japanese GAAP and J-IFRS reports), ``cas`` (Chinese Accounting
Standards), and any number of **custom** mappings registered with :func:`register`.

Matching is by the whole label (normalised), after removing a bracketed note and an outline number: ``Ⅰ. 매출액``, ``Net income (loss)``. It is never "the label contains the word revenue": a line named
"Revenue from related parties" or "営業外収益" is not the revenue line. Whether a matched line is *used* is decided by the matcher, from the statement it appears in.

Every mapping carries a version. A stored figure records the mapping version that produced it, so a later version can be run over the same IR or the same file and the two readings compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import canonical, manual

TAXONOMY_VERSION = "1.0.0"
EXTRACTOR_VERSION = "2.0.0"


ADJUSTED = re.compile(r"\b(?:adjusted|non-?ifrs|non-?gaap|underlying|core|excluding|before\s+(?:special|exceptional)\s+items|pro\s*forma)\b|调整后|調整後|调整|非国际财务报告准则|非gaap|조정|ajusté|bereinigt|ajustado", re.I)
MINORITY = re.compile(r"non-?controlling|minority|少数股东|少數股東|非支配|非控制|비지배|intérêts\s+minoritaires|nicht\s+beherrschende|intereses\s+minoritarios|participações\s+não", re.I)
ATTRIBUTABLE = re.compile(r"attributable\s+to|attributable|归属于|歸屬於|帰属する|귀속|지배기업|part\s+du\s+groupe|davon|entfallen|atribuible|atribuível|equity\s+holders\s+of|owners\s+of\s+the\s+(?:parent|company)", re.I)
PROFIT_WORD = re.compile(r"profit|income|earnings|loss|利润|利潤|溢利|利益|이익|résultat|ergebnis|resultado|beneficio|lucro|^attributable\s+to", re.I)


def _script_parts(label: str) -> list[str]:
    """A bilingual label ("Revenue 营业收入", "売上高 Net sales") in its Latin and CJK/Hangul parts, so that each is matched on its own and the two are never counted twice."""
    cjk = re.sub(r"[^\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]+", " ", label).strip()
    latin = re.sub(r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]+", " ", label).strip()
    return [p for p in (latin, cjk) if p and re.search(r"\w", p)] if cjk and latin and len(latin) > 2 and len(cjk) > 1 else []


_VERB = r"(?:(?:net|total|cash|flows?|generated|provided|used|inflow|outflow|inflows|outflows|from|by|in|for|and|or|of|continuing|operations|discontinued)\s+)*"
# Anchored patterns over the *whole* normalised label (lower case, punctuation folded to spaces), for the lines whose wording varies most between companies. A pattern names the line only if the label is
# nothing but that phrase: "cash used in investing activities" is the investing total; "proceeds from the sale of property" and "depreciation of a lease" are not matched by the capex and total patterns.
RULES: tuple = tuple((re.compile(p), code) for p, code in (
    (r"^" + _VERB + r"operating activities(?: continuing operations)?$", "operating_cash_flow"),
    (r"^" + _VERB + r"investing activities(?: continuing operations)?$", "investing_cash_flow"),
    (r"^" + _VERB + r"financing activities(?: continuing operations)?$", "financing_cash_flow"),
    (r"^(?:purchases?|purchase|payments?|additions?|acquisitions?|expenditures?|investments?)(?: made)?(?: of| for| on| to| in)?(?: acquisition of| purchase of)?(?: the)? (?:property|plant|equipment|fixed assets|capital assets|premises|pp e)(?: .*)?$", "capital_expenditure"),
    (r"^capital (?:expenditures?|spending|investments?)(?: .*)?$|^capex$", "capital_expenditure"),
    (r"^depreciation(?: depletion)?(?: ?(?:and|&) ?| )?(?:amorti[sz]ation)?(?: of (?:property|plant|equipment|fixed assets|tangible assets|intangible assets|assets)(?: .*)?)?(?: expense)?$|^amorti[sz]ation(?: expense)?$|^depreciation ?(?:and|&) ?amorti[sz]ation(?: expense)?$", "depreciation_amortization"),
    (r"^(?:share|stock) based (?:compensation|payments?)(?: expense)?$|^equity based compensation$", "stock_based_compensation"),
    (r"^(?:acquisitions?|acquisition of (?:businesses|subsidiaries|a business|business)|payments? for acquisitions?|business (?:acquisitions?|combinations?))(?: net of cash acquired)?$", "acquisitions"),
    (r"^(?:(?:repurchases?|purchases?|buybacks?|payments? for(?: the)? (?:repurchases?|purchase)) (?:of )?(?:common |ordinary |our own )?(?:stock|shares)(?: .*)?|(?:repurchases?|purchases?) of treasury (?:stock|shares)|share (?:repurchases?|buybacks?)|treasury (?:stock|shares) purchased)$", "share_repurchases"),
    (r"^(?:payments? (?:of|for) (?:cash )?dividends?|(?:cash )?dividends? (?:paid|payments?)(?: .*)?)$", "dividends_paid"),
    (r"^(?:long|non current|noncurrent) ?(?:term )?(?:debt|borrowings?|loans?|notes? payable|bonds? payable|financial liabilities|interest bearing (?:loans|borrowings))(?: net of (?:current|short term) .*| less (?:current|short term) .*| non current)?$|^term debt$|^bonds? payable$|^debentures?$", "long_term_debt"),
    (r"^(?:short term|current)(?: portion of (?:long term )?)? ?(?:debt|borrowings?|loans?)(?: .*)?$|^commercial paper$|^notes? payable$|^bank (?:overdrafts?|loans?)$", "short_term_debt"),
    (r"^(?:net )?(?:income|earnings|profit|loss)? ?(?:per (?:ordinary |common )?share)?[ :]*basic(?: .*)?$|^basic (?:net )?(?:income|earnings|profit|loss)? ?(?:per (?:ordinary |common )?share)?(?: .*)?$|^basic eps$|^earnings per share basic$", "eps_basic"),
    (r"^(?:net )?(?:income|earnings|profit|loss)? ?(?:per (?:ordinary |common )?share)?[ :]*diluted(?: .*)?$|^diluted (?:net )?(?:income|earnings|profit|loss)? ?(?:per (?:ordinary |common )?share)?(?: .*)?$|^diluted eps$|^earnings per share diluted$", "eps_diluted"),
    (r"^(?:weighted[ -]?average )?(?:number of )?(?:common |ordinary )?shares(?: outstanding| used)?[ :,-]* ?(?:in (?:computing|the computation of|calculating) )?diluted(?: .*)?$|^diluted (?:weighted[ -]?average )?(?:number of )?(?:common |ordinary )?shares(?: outstanding)?$|^shares used in (?:computing|computation of|calculating|the calculation of) diluted .*$|^weighted[ -]?average (?:number of )?(?:common |ordinary )?shares(?: outstanding)?[ :,-]* ?diluted$", "weighted_avg_shares_diluted"),
    (r"^(?:number of )?(?:common |ordinary )?shares (?:issued and )?outstanding$|^number of (?:common |ordinary )?shares(?: outstanding| in issue)?$|^shares in issue$", "shares_outstanding"),
    (r"^(?:interest|finance) (?:expenses?|costs?)(?:,? net)?$|^interest and finance (?:costs?|expenses?)$|^interest expenses? and similar charges$", "interest_expense"),
    (r"^r ?&? ?d(?: expenses?| costs?| spending)?$|^research and development(?: costs?| expenses?| expense)?$", "research_and_development"),
    (r"^inventor(?:y|ies)(?:,? net)?$|^stocks? and work in progress$", "inventory"),
    (r"^property,? plant,? (?:and|&) equipment(?:,? net(?: of .*)?)?$|^property (?:and|&) equipment(?:,? net)?$|^fixed assets(?:,? net)?$|^tangible fixed assets$", "ppe"),
    (r"^current assets,? total$|^total current assets$", "current_assets"),
    (r"^current liabilities,? total$|^total current liabilities$", "current_liabilities"),
    (r"^cash and (?:bank|cash) balances$|^cash at bank and in hand$", "cash_and_equivalents"),
    (r"^cash and cash equivalents,? (?:at )?(?:the )?end of (?:the )?(?:year|period|fiscal year)$|^cash and cash equivalents at (?:the )?end$", "cash_and_equivalents"),
    (r"^(?:retained earnings|retained profits?|reinvested earnings|accumulated (?:deficit|earnings|profits?))(?:.*)?$", "retained_earnings"),
    (r"^(?:preferred|preference) (?:stock|shares?|equity)(?:.*)?$", "preferred_equity"),
    (r"^net loss$|^net (?:income|profit|earnings)(?: loss| \(loss\))?$", "net_income"),
    (r"^earnings before (?:income )?taxes$|^earnings before interest,? taxes,? depreciation and amorti[sz]ation$", None),
))
RULES = tuple((rx, c) for rx, c in RULES if c)
_EBITDA = re.compile(r"^earnings before interest,? taxes,? depreciation and amorti[sz]ation$")
_EBT = re.compile(r"^earnings before (?:income )?taxes$")


@dataclass(frozen=True)
class Match:
    code: str
    tier: str            # exact | stripped | overlay
    rule: str            # the rule that matched, for provenance
    confidence: float


@dataclass
class Mapping:
    standard: str
    version: str
    label: str
    labels: dict[str, str] = field(default_factory=dict)         # normalised label -> canonical code (this standard's additions)
    never: set[str] = field(default_factory=set)                 # normalised labels that look like a line and are not (ordinary income, non-operating income)
    parent_words: tuple = ()                                     # words that mark a parent-only (separate) statement

    def match(self, label: str) -> Match | None:
        if not label or not label.strip():
            return None
        parts = _script_parts(label)
        if parts:                                                                                    # bilingual: both halves must agree (or one be silent); they are one line, never two
            found = [m for m in (self._match(p) for p in parts) if m]
            codes = {m.code for m in found}
            if len(codes) == 1:
                return found[0]
            if len(codes) > 1:
                return None
        return self._match(label)

    def _match(self, label: str) -> Match | None:
        raw = label.strip()
        if ADJUSTED.search(raw):                                                                     # an adjusted or non-IFRS figure is never the statutory line
            return None
        if ATTRIBUTABLE.search(raw) and PROFIT_WORD.search(raw):
            if MINORITY.search(raw):
                return None
            return Match("net_income_attributable", "overlay", "rule:attributable-to-parent", 0.9)
        cands = [("exact", raw)]
        stripped = re.sub(r"\s*[(（][^)）]*[)）]\s*$", "", raw).strip()
        cands.append(("stripped", stripped))
        cands.append(("stripped", re.sub(r"^\s*(?:[IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\d{1,2}|[가-힣]|[a-z])\s*[.)、]\s*", "", stripped, flags=re.I)))
        for tier, text in cands:
            if not text:
                continue
            n = manual._norm(text)
            if n in self.never:
                return None
            if n in self.labels:
                return Match(self.labels[n], "overlay", f"{self.standard}:{n}", 0.95 if tier == "exact" else 0.85)
            code = manual.ALIASES.get(n)
            if code:
                return Match(code, tier, f"alias:{n}", 1.0 if tier == "exact" else 0.9)
            if _EBITDA.match(n):
                return Match("ebitda", tier, "rule:ebitda", 0.85)
            if _EBT.match(n):
                return Match("income_before_tax", tier, "rule:ebt", 0.85)
            for rx, code in RULES:
                if rx.match(n):
                    return Match(code, "rule", f"rule:{code}", 0.88 if tier == "exact" else 0.8)
        return None

    def to_dict(self) -> dict:
        return {"standard": self.standard, "version": self.version, "label": self.label, "overlay_labels": len(self.labels), "exclusions": len(self.never), "taxonomy_version": TAXONOMY_VERSION}


def _n(*texts):
    return {manual._norm(t) for t in texts}


_REGISTRY: dict[str, Mapping] = {}


def register(mapping: Mapping) -> Mapping:
    """Add or replace a mapping (a custom taxonomy: a company's own chart of accounts, a regulator's list)."""
    _REGISTRY[mapping.standard] = mapping
    return mapping


def custom(standard: str, labels: dict[str, str], version: str = "1", never: tuple = ()) -> Mapping:
    """A custom taxonomy from a ``{label: canonical code}`` dictionary. The codes are checked: a typo is an error here, not a silent miss later."""
    bad = [c for c in labels.values() if c not in canonical.METRICS]
    if bad:
        raise ValueError(f"not canonical lines: {sorted(set(bad))}")
    return register(Mapping(standard, version, standard, {manual._norm(k): v for k, v in labels.items()}, _n(*never)))


register(Mapping("us-gaap", "1.0.0", "US GAAP", {manual._norm(k): v for k, v in {"Net sales": "revenue", "Total net sales": "revenue", "Net revenues": "revenue", "Net earnings": "net_income", "Total stockholders' equity": "shareholders_equity",
                                                                                      "Total shareholders' equity": "shareholders_equity", "Cost of sales": "cost_of_revenue"}.items()}, set(), ("parent company only", "condensed financial information of registrant")))
register(Mapping("ifrs", "1.0.0", "IFRS", {manual._norm(k): v for k, v in {"Revenue from contracts with customers": "revenue", "Profit for the year": "net_income", "Profit for the period": "net_income",
                                                                           "Total equity": "shareholders_equity", "Cash and cash equivalents at end of year": "cash_and_equivalents"}.items()}, set(), ("separate financial statements", "company only")))
register(Mapping("sfrs-i", "1.0.0", "SFRS(I) (Singapore)", {manual._norm(k): v for k, v in {"Revenue": "revenue", "Profit for the year": "net_income", "Profit net of tax": "net_income", "Total equity": "shareholders_equity",
                                                                                            "Profit before income tax": "income_before_tax", "Profit before tax": "income_before_tax"}.items()}, set(), ("the company", "company statement")))
register(Mapping("k-ifrs", "1.0.0", "K-IFRS (Korea)", {manual._norm(k): v for k, v in {"영업수익": "revenue", "수익(매출액)": "revenue", "당기순이익(손실)": "net_income", "연결당기순이익": "net_income", "자산총계": "total_assets",
                                                                                        "자본총계": "shareholders_equity", "부채총계": "total_liabilities", "법인세비용차감전순이익": "income_before_tax"}.items()},
                 _n("영업외수익", "금융수익", "기타수익"), ("별도", "별도재무제표", "재무제표(별도)")))
register(Mapping("j-gaap", "1.0.0", "J-GAAP / J-IFRS (Japan)", {manual._norm(k): v for k, v in {"売上高": "revenue", "営業収益": "revenue", "売上収益": "revenue", "親会社株主に帰属する当期純利益": "net_income", "当期純利益": "net_income",
                                                                                                "税金等調整前当期純利益": "income_before_tax", "純資産合計": "shareholders_equity", "資産合計": "total_assets", "負債合計": "total_liabilities"}.items()},
                 _n("経常利益", "営業外収益", "営業外費用", "特別利益", "特別損失", "ordinary income", "ordinary profit"), ("個別", "単体", "個別財務諸表")))
register(Mapping("cas", "1.0.0", "Chinese Accounting Standards", {manual._norm(k): v for k, v in {"营业收入": "revenue", "营业总收入": "revenue", "營業收入": "revenue", "净利润": "net_income", "归属于母公司所有者的净利润": "net_income",
                                                                                                   "资产总计": "total_assets", "负债合计": "total_liabilities", "所有者权益合计": "shareholders_equity", "股东权益合计": "shareholders_equity"}.items()},
                 _n("营业外收入", "营业外支出", "营业总成本"), ("母公司", "母公司资产负债表", "母公司利润表")))

COUNTRY_STANDARD = {"US": "us-gaap", "SG": "sfrs-i", "KR": "k-ifrs", "JP": "j-gaap", "CN": "cas", "HK": "ifrs", "GB": "ifrs", "FR": "ifrs", "DE": "ifrs", "ES": "ifrs", "IT": "ifrs", "NL": "ifrs", "BR": "ifrs", "CA": "ifrs"}
LANGUAGE_STANDARD = {"ko": "k-ifrs", "ja": "j-gaap", "zh": "cas"}


def standards() -> list[dict]:
    return [m.to_dict() for m in _REGISTRY.values()]


def standards_by_name() -> dict[str, Mapping]:
    return dict(_REGISTRY)


def get(standard: str | None) -> Mapping:
    if standard and standard in _REGISTRY:
        return _REGISTRY[standard]
    return _REGISTRY["ifrs"]


def detect_standard(hint: dict | None, text: str = "", language: str = "en") -> tuple[str, str]:
    """``(standard, how)``: the hint if one is given, else the country, else the script of the document, else the words of the report ("SFRS(I)", "K-IFRS", "US GAAP"), else IFRS."""
    hint = hint or {}
    if hint.get("standard") in _REGISTRY:
        return hint["standard"], "hint"
    c = str(hint.get("country") or "").upper()
    if c in COUNTRY_STANDARD:
        return COUNTRY_STANDARD[c], "country"
    low = (text or "")[:20000].lower()
    for words, std in (("sfrs(i)", "sfrs-i"), ("singapore financial reporting standards", "sfrs-i"), ("k-ifrs", "k-ifrs"), ("korean international financial reporting", "k-ifrs"), ("us gaap", "us-gaap"),
                       ("u.s. generally accepted", "us-gaap"), ("generally accepted accounting principles", "us-gaap"), ("企業会計基準", "j-gaap"), ("日本基準", "j-gaap"), ("企业会计准则", "cas")):
        if words in low:
            return std, "document"
    if language in LANGUAGE_STANDARD:
        return LANGUAGE_STANDARD[language], "language"
    if re.search(r"[가-힣]{3,}", text or ""):
        return "k-ifrs", "script"
    if re.search(r"[぀-ヿ]{2,}", text or ""):
        return "j-gaap", "script"
    if re.search(r"[一-鿿]{3,}", text or ""):
        return "cas", "script"
    return "ifrs", "default"


__all__ = ["Mapping", "Match", "register", "custom", "get", "standards", "detect_standard", "TAXONOMY_VERSION", "EXTRACTOR_VERSION"]
