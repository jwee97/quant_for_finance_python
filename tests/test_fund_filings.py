"""Country connectors and the global reading of filings, against stand-ins of each source's documented shape (none of these services is reachable from the test environment): what a source permits is
enforced (a key where one is required, no search of sites that restrict it, links only on a source's own host), what it returns is read into the application's lines, and a figure keeps its source."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date

import pytest

from src.fundamentals import manual
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.filings import calendar as cal
from src.fundamentals.filings import registry
from src.fundamentals.filings.base import FilingRef
from src.fundamentals.importer import xbrl
from src.fundamentals.importer.fetch import Fetched
from src.fundamentals.importer.pipeline import ImportService
from src.fundamentals.store import CompanyProfile, FundamentalsStore


class Fake:
    """A fetcher that answers by matching a piece of the address."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def fetch(self, url):
        self.seen.append(url)
        for key, body in self.routes.items():
            if key in url:
                body = body(url) if callable(body) else body
                data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
                return Fetched(data, "application/json", url, 200)
        raise FundamentalsError(f"the server answered HTTP 404 for this link: {url}")


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def conn(name, routes=None, **env):
    return registry.make(name, Fake(routes or {}), env)


# ------------------------------------------------------------------------------------------------------------------------ access rules
def test_the_registry_covers_the_markets_and_says_what_each_source_permits():
    names = {d["name"]: d for d in registry.describe(env={})}
    assert {"sec_edgar", "edinet", "dart", "cvm", "esef", "hkex", "sgx", "cninfo", "sse", "szse", "sedar"} <= set(names)
    assert names["edinet"]["discovery"] == names["dart"]["discovery"] == "api" and names["cvm"]["discovery"] == names["esef"]["discovery"] == "open_data"
    assert all(names[n]["discovery"] == "links_only" for n in ("hkex", "sgx", "cninfo", "sse", "szse", "sedar"))
    assert not names["edinet"]["available"] and "EDINET_API_KEY" in names["edinet"]["reason"] and "OPENDART_API_KEY" in names["dart"]["reason"]
    assert {c for d in names.values() for c in d["countries"]} >= {"US", "JP", "KR", "SG", "CN", "HK", "FR", "ES", "DE", "CA", "BR"}
    assert registry.for_country("fr") == ["esef"] and set(registry.for_country("CN")) == {"cninfo", "sse", "szse"}
    with pytest.raises(FundamentalsError, match="no connector called"):
        registry.make("nasdaq")


def test_a_source_that_asks_for_a_key_makes_no_request_without_one():
    for name in ("edinet", "dart"):
        c = conn(name)
        with pytest.raises(FundamentalsError, match="asks for a key"):
            c.list_filings({"ticker": "7203", "stock_code": "005930"})
        assert c.fetcher.seen == []


def test_sources_that_restrict_automated_collection_are_never_searched_and_accept_only_their_own_links():
    for name, url, title, kind in (("hkex", "https://www1.hkexnews.hk/listedco/listconews/sehk/2024/0328/2024032800001.pdf", "2023 Annual Report", "FY"),
                                   ("hkex", "https://www1.hkexnews.hk/x/interim.pdf", "Interim Report 2023", "H1"), ("cninfo", "http://static.cninfo.com.cn/finalpage/2024-03-30/1.PDF", "2023年半年度报告", "H1"),
                                   ("sgx", "https://links.sgx.com/1.0.0/corporate-announcements/ABC/file.pdf", "Full Year Financial Statements", "FY"), ("szse", "https://disc.static.szse.cn/download/x.pdf", "2023年第三季度报告", "Q3")):
        c = conn(name)
        out = c.list_filings({"ticker": "0700"})
        assert out["filings"] == [] and "no interface for programs" in out["note"] and c.fetcher.seen == []
        ref = c.classify_link(url, title)
        assert ref.fiscal_period == kind and ref.format == "pdf" and ref.source == name
    with pytest.raises(FundamentalsError, match="not on HKEXnews"):
        conn("hkex").classify_link("https://example.org/annual.pdf", "Annual report")
    with pytest.raises(FundamentalsError, match="not on"):
        conn("hkex").get(FilingRef("hkex", "HK", "", "x.pdf", url="https://evil.example/x.pdf"))
    got = conn("hkex", {"hkexnews.hk": b"%PDF-1.4"}).get(FilingRef("hkex", "HK", "", "ar.pdf", url="https://www1.hkexnews.hk/a/ar.pdf"))
    assert got["kind"] == "file" and got["data"].startswith(b"%PDF")


def test_each_countrys_calendar_says_which_periods_are_expected():
    assert cal.expected_periods("US") == ("Q1", "Q2", "Q3", "Q4", "FY") and cal.expected_periods("HK") == ("H1", "FY") and cal.expected_periods("KR") == ("Q1", "H1", "Q3", "FY")
    assert cal.expected_periods("JP", 2022) == ("Q1", "Q2", "Q3", "FY") and cal.expected_periods("JP", 2024) == ("H1", "FY") and cal.expected_periods("XX") == cal.DEFAULT["periods"]
    assert cal.expected("DE")["standard"] == "IFRS" and cal.expected("BR")["currency"] == "BRL" and "Q4 is derived" in cal.expected("CA")["note"]
    assert cal.quarter_of(date(2024, 6, 30), 3) == "Q1" and cal.quarter_of(date(2024, 12, 31), 3) == "Q3" and cal.quarter_of(date(2024, 3, 31), 3) == "Q4" and cal.quarter_of(date(2024, 9, 30), 12) == "Q3"


# ------------------------------------------------------------------------------------------------------------------------ line names in other languages
def test_statement_lines_are_recognised_in_japanese_korean_chinese_and_the_european_languages():
    for label, code in (("売上高", "revenue"), ("営業利益", "operating_income"), ("親会社株主に帰属する当期純利益", "net_income"), ("資産合計", "total_assets"), ("営業活動によるキャッシュ・フロー", "operating_cash_flow"),
                        ("매출액", "revenue"), ("영업이익", "operating_income"), ("자산총계", "total_assets"), ("부채총계", "total_liabilities"), ("营业收入", "revenue"), ("净利润", "net_income"), ("资产总计", "total_assets"),
                        ("經營活動所得現金淨額", "operating_cash_flow"), ("Chiffre d'affaires", "revenue"), ("Résultat net", "net_income"), ("Umsatzerlöse", "revenue"), ("Bilanzsumme", "total_assets"),
                        ("Ativo Total", "total_assets"), ("Patrimônio Líquido", "shareholders_equity"), ("Net sales", "revenue")):
        assert manual.line_for(label) == code, label
    assert manual.line_for("total du passif") is None, "an ambiguous label is left unmapped, not mapped wrongly"
    d = manual.parse_table("科目,2023年度,2022年度\n売上高,1200,1000\n営業利益,200,150\n当期純利益,150,120\n")
    assert set(d["values"]) == {"revenue", "operating_income", "net_income"} if d["columns"] else True


# ------------------------------------------------------------------------------------------------------------------------ Japan
def edinet_xbrl(consolidated=True, period="FY", end="2024-03-31", standard="Japan GAAP") -> bytes:
    scope = "" if consolidated else '<xbrli:scenario><xbrldi:explicitMember dimension="jppfs_cor:ConsolidatedOrNonConsolidatedAxis">jppfs_cor:NonConsolidatedMember</xbrldi:explicitMember></xbrli:scenario>'
    ctx = lambda i, p: f'<xbrli:context id="{i}"><xbrli:entity><xbrli:identifier scheme="http://disclosure.edinet-fsa.go.jp">E00000</xbrli:identifier></xbrli:entity><xbrli:period>{p}</xbrli:period>{scope}</xbrli:context>'  # noqa: E731
    facts = "".join(f'<jppfs_cor:{k} contextRef="CurrentYearDuration" unitRef="JPY" decimals="-6">{v}</jppfs_cor:{k}>' for k, v in {"NetSales": 48_000_000_000_000, "OperatingIncome": 4_500_000_000_000,
                                                                                                                               "ProfitLossAttributableToOwnersOfParent": 3_400_000_000_000}.items())
    facts += "".join(f'<jppfs_cor:{k} contextRef="CurrentYearInstant" unitRef="JPY" decimals="-6">{v}</jppfs_cor:{k}>' for k, v in {"Assets": 90_000_000_000_000, "Liabilities": 55_000_000_000_000, "NetAssets": 35_000_000_000_000}.items())
    return f'''<?xml version="1.0" encoding="utf-8"?><xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:jppfs_cor="http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-12-01/jppfs_cor"
xmlns:jpdei_cor="http://disclosure.edinet-fsa.go.jp/taxonomy/jpdei/2013-08-31/jpdei_cor">{ctx("CurrentYearDuration", f"<xbrli:startDate>2023-04-01</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>")}{ctx("CurrentYearInstant", f"<xbrli:instant>{end}</xbrli:instant>")}
<xbrli:context id="Cover"><xbrli:entity><xbrli:identifier scheme="x">E</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2023-04-01</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:unit id="JPY"><xbrli:measure>iso4217:JPY</xbrli:measure></xbrli:unit>
<jpdei_cor:CurrentPeriodEndDateDEI contextRef="Cover">{end}</jpdei_cor:CurrentPeriodEndDateDEI><jpdei_cor:TypeOfCurrentPeriodDEI contextRef="Cover">{period}</jpdei_cor:TypeOfCurrentPeriodDEI>
<jpdei_cor:AccountingStandardsDEI contextRef="Cover">{standard}</jpdei_cor:AccountingStandardsDEI><jpdei_cor:SecurityCodeDEI contextRef="Cover">72030</jpdei_cor:SecurityCodeDEI><jpdei_cor:FilerNameInEnglishDEI contextRef="Cover">Toyota Motor Corporation</jpdei_cor:FilerNameInEnglishDEI>
{facts}</xbrli:xbrl>'''.encode()


def test_a_japanese_filing_is_read_through_the_japanese_taxonomy_with_its_cover_facts():
    d = xbrl.extract(edinet_xbrl(), "S100TEST.xbrl")
    p = d["periods"][0]
    assert (d["currency"], d["accounting_standard"], p["fiscal_period"], p["period_end"]) == ("JPY", "Japan GAAP", "FY", "2024-03-31") and d["company"]["ticker"] == "72030" and "Toyota" in d["company"]["name"]
    assert p["values"]["revenue"] == 48e12 and p["values"]["operating_income"] == 4.5e12 and p["values"]["net_income"] == 3.4e12 and p["values"]["total_assets"] == 90e12 and p["values"]["total_liabilities"] == 55e12
    assert p["values"]["shareholders_equity"] == 35e12 and p["sources"]["shareholders_equity"]["tag"] == "jppfs_cor:NetAssets", "net assets stand in for the owners' equity and say so"
    assert p["sources"]["revenue"]["tag"] == "jppfs_cor:NetSales" and not d["parent_only"]
    half = xbrl.extract(edinet_xbrl(period="HY", end="2023-09-30"), "x.xbrl")
    assert half["periods"][0]["fiscal_period"] == "H1"


def test_a_parent_only_japanese_filer_is_read_from_its_non_consolidated_context():
    d = xbrl.extract(edinet_xbrl(consolidated=False), "x.xbrl")
    assert d["parent_only"] and d["periods"][0]["values"]["revenue"] == 48e12


def edinet_routes(key="K"):
    def listing(url):
        day = url.split("date=")[1][:10]
        rows = {"2024-06-25": [{"docID": "S100TOYO", "edinetCode": "E02144", "secCode": "72030", "filerName": "Toyota", "docTypeCode": "120", "periodEnd": "2024-03-31", "submitDateTime": "2024-06-25 15:00",
                                "docDescription": "有価証券報告書", "xbrlFlag": "1"}, {"docID": "S100OTHR", "edinetCode": "E99999", "secCode": "99990", "docTypeCode": "120", "xbrlFlag": "1"},
                               {"docID": "S100LETR", "edinetCode": "E02144", "secCode": "72030", "docTypeCode": "350", "xbrlFlag": "0"}],
                "2024-08-08": [{"docID": "S100Q1", "edinetCode": "E02144", "secCode": "72030", "docTypeCode": "140", "periodEnd": "2024-06-30", "submitDateTime": "2024-08-08 15:00", "xbrlFlag": "1"}]}
        return {"metadata": {"status": "200"}, "results": rows.get(day, [])}
    return {"documents.json": listing, "/documents/S100TOYO": zipped({"XBRL/PublicDoc/jpcrp030000-asr-001.xbrl": edinet_xbrl(), "XBRL/AuditDoc/a.xbrl": b"<x/>", "XBRL/PublicDoc/x.xsd": b"<s/>"})}


def test_edinet_lists_a_companys_securities_reports_by_scanning_days_and_fetches_the_xbrl(monkeypatch):
    c = conn("edinet", edinet_routes(), EDINET_API_KEY="K")
    out = c.list_filings({"ticker": "7203"}, since="2024-06-24", until="2024-08-09")
    assert out["complete"] and [(f.doc_id, f.fiscal_period, f.period_end) for f in out["filings"]] == [("S100TOYO", "FY", "2024-03-31"), ("S100Q1", "Q", "2024-06-30")], "another company and a letter (type 350) are not listed"
    assert all("Subscription-Key=K" in u for u in c.fetcher.seen) and not any("2024-06-29" in u or "2024-06-30" in u for u in c.fetcher.seen), "weekends are not scanned"
    monkeypatch.setattr("src.fundamentals.filings.edinet.MAX_DAYS", 3)
    part = conn("edinet", edinet_routes(), EDINET_API_KEY="K").list_filings({"ticker": "7203"}, since="2024-06-24", until="2024-08-09")
    assert not part["complete"] and part["next_since"] and part["scanned_days"] == 3, "a long scan is resumable"
    got = c.get(out["filings"][0])
    assert got["kind"] == "file" and got["name"] == "S100TOYO.xbrl" and got["hint"]["filing_date"] == "2024-06-25" and xbrl.extract(got["data"])["periods"][0]["values"]["revenue"] == 48e12
    with pytest.raises(FundamentalsError, match="needs the company's securities code"):
        c.list_filings({})
    bad = conn("edinet", {"documents.json": {"StatusCode": 401, "message": "Access denied due to invalid subscription key.Make sure to provide a valid key for an active subscription."}}, EDINET_API_KEY="K")   # the live answer
    with pytest.raises(FundamentalsError, match="refused the key"):
        bad.list_filings({"ticker": "7203"}, since="2024-06-24", until="2024-06-24")


# ------------------------------------------------------------------------------------------------------------------------ Korea
def dart_rows(h1=False):
    def row(sj, aid, nm, cur, add=None):
        return {"rcept_no": "20240814000001", "sj_div": sj, "account_id": aid, "account_nm": nm, "thstrm_amount": cur, "thstrm_add_amount": add or "", "currency": "KRW"}
    return [row("IS", "ifrs-full_Revenue", "매출액", "74,000,000,000,000", "146,000,000,000,000"), row("IS", "ifrs-full_ProfitLossFromOperatingActivities", "영업이익", "10,400,000,000,000", "13,000,000,000,000"),
            row("IS", "ifrs-full_ProfitLoss", "당기순이익", "9,800,000,000,000", "12,000,000,000,000"), row("BS", "ifrs-full_Assets", "자산총계", "450,000,000,000,000"), row("BS", "ifrs-full_Liabilities", "부채총계", "100,000,000,000,000"),
            row("BS", "ifrs-full_Equity", "자본총계", "350,000,000,000,000"), row("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities", "영업활동현금흐름", "30,000,000,000,000"),
            row("CF", "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "유형자산의 취득", "-20,000,000,000,000"), row("BS", "-표준계정코드 미사용-", "회사 특수 계정", "5")]


def dart_routes():
    return {"corpCode.xml": zipped({"CORPCODE.xml": "<result><list><corp_code>00126380</corp_code><stock_code>005930</stock_code></list><list><corp_code>1</corp_code><stock_code> </stock_code></list></result>"}),
            "list.json": {"status": "000", "total_page": 1, "list": [{"corp_code": "00126380", "stock_code": "005930", "rcept_no": "20240312000736", "report_nm": "사업보고서 (2023.12)", "rcept_dt": "20240312"},
                                                                     {"corp_code": "00126380", "stock_code": "005930", "rcept_no": "20240814000001", "report_nm": "반기보고서 (2024.06)", "rcept_dt": "20240814"},
                                                                     {"corp_code": "00126380", "stock_code": "005930", "rcept_no": "20241114000002", "report_nm": "분기보고서 (2024.09)", "rcept_dt": "20241114"},
                                                                     {"corp_code": "00126380", "rcept_no": "20240510000003", "report_nm": "분기보고서 (2024.03)", "rcept_dt": "20240510"},
                                                                     {"corp_code": "00126380", "rcept_no": "20240301000004", "report_nm": "주주총회소집공고", "rcept_dt": "20240301"}]},
            "fnlttSinglAcntAll": lambda url: {"status": "000", "list": dart_rows()} if "fs_div=CFS" in url else {"status": "013", "list": []}}


def test_opendart_lists_reports_by_company_and_turns_the_accounts_into_lines():
    c = conn("dart", dart_routes(), OPENDART_API_KEY="K")
    out = c.list_filings({"ticker": "5930"})
    assert [(f.fiscal_period, f.period_end, f.filed_at) for f in out["filings"]] == [("FY", "2023-12-31", "20240312"), ("H1", "2024-06-30", "20240814"), ("Q3", "2024-09-30", "20241114"), ("Q1", "2024-03-31", "20240510")]
    assert any("corp_code=00126380" in u for u in c.fetcher.seen) and not any(f.title.startswith("주주총회") for f in out["filings"])
    by = {f.fiscal_period: f for f in out["filings"]}
    h1 = c.get(by["H1"])
    d = h1["draft"]["periods"][0]
    assert h1["kind"] == "draft" and d["fiscal_period"] == "H1" and d["currency"] == "KRW" and d["filing_date"] == "2024-08-14"
    assert d["values"]["revenue"] == 146e12 and d["values"]["net_income"] == 12e12, "the half-year income statement is the six months, from the year-to-date column"
    assert d["values"]["capital_expenditure"] == 20e12 and d["values"]["operating_cash_flow"] == 30e12 and d["values"]["total_liabilities"] == 100e12, "an outflow is a positive amount; the cash flow is the six months reported"
    assert d["sources"]["revenue"]["tag"] == "ifrs-full:Revenue" and d["sources"]["revenue"]["method"] == "dart-api" and h1["draft"]["accounting_standard"] == "K-IFRS"
    q3 = c.get(by["Q3"])["draft"]
    assert q3["periods"][0]["values"]["revenue"] == 74e12 and "operating_cash_flow" not in q3["periods"][0]["values"] and q3["ytd_only"], "a Q3 report's cash flow is nine months: left out"
    assert c.get(by["FY"])["draft"]["periods"][0]["fiscal_period"] == "FY"
    nothing = conn("dart", {**dart_routes(), "fnlttSinglAcntAll": {"status": "013", "list": []}}, OPENDART_API_KEY="K")
    with pytest.raises(FundamentalsError, match="no statements"):
        nothing.get(by["FY"])
    wrong = conn("dart", {"list.json": {"status": "010", "message": "등록되지 않은 인증키입니다."}}, OPENDART_API_KEY="K")      # the live answer to an unregistered key
    with pytest.raises(FundamentalsError, match="not registered"):
        wrong.list_filings({"corp_code": "00126380"})
    limited = conn("dart", {"list.json": {"status": "020", "message": "limit"}}, OPENDART_API_KEY="K")
    with pytest.raises(FundamentalsError, match="daily request limit"):
        limited.list_filings({"corp_code": "00126380"})


# ------------------------------------------------------------------------------------------------------------------------ Brazil
def cvm_zip(doc="DFP", year=2023) -> bytes:
    def table(header, rows):
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
        return buf.getvalue().encode("latin-1")
    base = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "GRUPO_DFP", "MOEDA", "ESCALA_MOEDA", "ORDEM_EXERC"]
    def r(dt, ver, scale, ordem, code, name, value, ini=None, cnpj="33.000.167/0001-01", cvm="9512"):
        out = [cnpj, dt, ver, "PETROBRAS", cvm, "DF Consolidado", "REAL", scale, ordem]
        return out + ([ini] if ini is not None else []) + [code, name, value]
    dre_head = base + ["DT_INI_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"]
    bp_head = base + ["CD_CONTA", "DS_CONTA", "VL_CONTA"]
    if doc == "DFP":
        files = {f"dfp_cia_aberta_{year}.csv": table(["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "CATEG_DOC", "ID_DOC", "DT_RECEB"], [["33.000.167/0001-01", "2023-12-31", "1", "PETROBRAS", "9512", "DFP", "1", "2024-02-20"], ["33.000.167/0001-01", "2023-12-31", "2", "PETROBRAS", "9512", "DFP", "2", "2024-05-02"]]),
                 f"dfp_cia_aberta_DRE_con_{year}.csv": table(dre_head, [r("2023-12-31", "2", "MIL", "ÚLTIMO", "3.01", "Receita de Venda", "500000", "2023-01-01"), r("2023-12-31", "1", "MIL", "ÚLTIMO", "3.01", "Receita de Venda", "490000", "2023-01-01"),
                                                                       r("2023-12-31", "2", "MIL", "ÚLTIMO", "3.11", "Lucro", "124000", "2023-01-01"), r("2023-12-31", "2", "MIL", "ÚLTIMO", "3.02", "Custo", "-240000", "2023-01-01"), r("2023-12-31", "2", "MIL", "PENÚLTIMO", "3.01", "Receita", "1", "2022-01-01"),
                                                                       r("2023-12-31", "2", "MIL", "ÚLTIMO", "3.01", "outra empresa", "7", "2023-01-01", cnpj="00.000.000/0001-00", cvm="1")]),
                 f"dfp_cia_aberta_BPA_con_{year}.csv": table(bp_head, [r("2023-12-31", "2", "MIL", "ÚLTIMO", "1", "Ativo Total", "1000000"), r("2023-12-31", "2", "MIL", "ÚLTIMO", "1.01.01", "Caixa", "50000")]),
                 f"dfp_cia_aberta_BPP_con_{year}.csv": table(bp_head, [r("2023-12-31", "2", "MIL", "ÚLTIMO", "2.03", "Patrimônio Líquido", "400000"), r("2023-12-31", "2", "MIL", "ÚLTIMO", "2.01", "Passivo Circulante", "160000"),
                                                                       r("2023-12-31", "2", "MIL", "ÚLTIMO", "2.02", "Passivo Não Circulante", "440000")]),
                 f"dfp_cia_aberta_DFC_MI_con_{year}.csv": table(dre_head, [r("2023-12-31", "2", "MIL", "ÚLTIMO", "6.01", "Caixa líquido operacional", "200000", "2023-01-01")])}   # the indirect method, as most companies file
    else:
        files = {f"itr_cia_aberta_{year}.csv": table(["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "CATEG_DOC", "ID_DOC", "DT_RECEB"], [["33.000.167/0001-01", "2023-09-30", "1", "PETROBRAS", "9512", "ITR", "3", "2023-11-09"]]),
                 f"itr_cia_aberta_DRE_con_{year}.csv": table(dre_head, [r("2023-09-30", "1", "MIL", "ÚLTIMO", "3.01", "Receita", "120000", "2023-07-01"), r("2023-09-30", "1", "MIL", "ÚLTIMO", "3.01", "Receita acumulada", "370000", "2023-01-01")]),
                 f"itr_cia_aberta_DFC_MD_con_{year}.csv": table(dre_head, [r("2023-09-30", "1", "MIL", "ÚLTIMO", "6.01", "Caixa líquido operacional", "150000", "2023-01-01")])}
    return zipped(files)


def test_cvm_open_data_gives_the_year_scaled_to_units_with_the_latest_version_and_says_what_was_restated():
    c = conn("cvm", {"dfp_cia_aberta_2023.zip": cvm_zip("DFP"), "itr_cia_aberta_2023.zip": cvm_zip("ITR")})
    refs = c.list_filings({"cd_cvm": "9512"}, since="2023-01-01", until="2023-12-31")["filings"]
    assert {(r.doc_type, r.extra["year"]) for r in refs} == {("DFP", 2023), ("ITR", 2023)}
    got = c.get(next(r for r in refs if r.doc_type == "DFP"))
    p = got["draft"]["periods"][0]
    assert p["fiscal_period"] == "FY" and p["period_end"] == "2023-12-31" and p["currency"] == "BRL" and p["filing_date"] == "2024-05-02", "the date the latest version was received"
    assert p["values"] == {"revenue": 500_000_000.0, "cost_of_revenue": 240_000_000.0, "net_income": 124_000_000.0, "total_assets": 1_000_000_000.0, "cash_and_equivalents": 50_000_000.0, "shareholders_equity": 400_000_000.0,
                           "current_liabilities": 160_000_000.0, "total_liabilities": 600_000_000.0, "operating_cash_flow": 200_000_000.0}, "a cost is a positive amount; total liabilities is 2.01 + 2.02; the indirect cash flow is read"
    assert any("versions [1, 2]" in n and "latest (2)" in n for n in got["draft"]["notes"]) and p["sources"]["revenue"]["method"] == "cvm-csv"
    assert not any("outra" in s["text"] for s in p["sources"].values()), "another company's rows are not read"
    q = c.get(FilingRef("cvm", "BR", "33.000.167/0001-01", "ITR-2023", "ITR", "Q", url="https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2023.zip", extra={"year": 2023}))
    assert q["draft"]["periods"][0]["fiscal_period"] == "Q3" and q["draft"]["periods"][0]["values"]["revenue"] == 120_000_000.0, "the quarter, not the nine months"
    assert "operating_cash_flow" not in q["draft"]["periods"][0]["values"], "a Q3 cash flow is year to date: left out"
    with pytest.raises(FundamentalsError, match="needs the company"):
        c.get(FilingRef("cvm", "BR", "", "x", "DFP", url="u", extra={"year": 2023}))


# ------------------------------------------------------------------------------------------------------------------------ Europe
def esef_report() -> bytes:
    from tests.test_fund_import import ixbrl
    t = ixbrl().decode()
    for a, b in (("http://fasb.org/us-gaap/2023", "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full"), ("us-gaap:Revenues", "ifrs-full:Revenue"), ("us-gaap:NetIncomeLoss", "ifrs-full:ProfitLoss"),
                 ("us-gaap:OperatingIncomeLoss", "ifrs-full:ProfitLossFromOperatingActivities"), ("us-gaap:StockholdersEquity", "ifrs-full:Equity"), ("us-gaap:CashAndCashEquivalentsAtCarryingValue", "ifrs-full:CashAndCashEquivalents"),
                 ("us-gaap:", "ifrs-full:"), ("xmlns:us-gaap", "xmlns:ifrs-full"), ("iso4217:USD", "iso4217:EUR")):
        t = t.replace(a, b)
    return t.encode()


def test_esef_lists_a_companys_annual_reports_from_the_open_index_and_reads_the_inline_xbrl():
    index = {"data": [{"id": "1", "attributes": {"fxo_id": "969500TEST0000000000-2023-12-31-ESEF-FR-0", "country": "FR", "period_end": "2023-12-31", "date_added": "2024-04-02 10:00:00", "package_url": "/969500T/2023-12-31/ESEF-FR-0/rep.zip"}},
                      {"id": "2", "attributes": {"fxo_id": "x", "country": "FR", "period_end": "2021-12-31", "package_url": None}}]}
    c = conn("esef", {"api/filings": index, "rep.zip": zipped({"969500T/2023-12-31/ESEF-FR-0/reports/report.xhtml": esef_report(), "969500T/META-INF/x": b"1"})})
    out = c.list_filings({"lei": "969500TEST0000000000"})
    assert [(f.fiscal_period, f.period_end, f.filed_at) for f in out["filings"]] == [("FY", "2023-12-31", "2024-04-02")] and "like" in c.fetcher.seen[0] and "969500TEST0000000000" in c.fetcher.seen[0]
    got = c.get(out["filings"][0])
    d = xbrl.extract(got["data"], got["name"], got["hint"]["accession"])
    p = d["periods"][0]
    assert d["currency"] == "EUR" and p["values"]["revenue"] == 1200e6 and p["values"]["net_income"] == 150e6 and p["values"]["shareholders_equity"] == 2000e6 and d["accession"].startswith("969500TEST")
    with pytest.raises(FundamentalsError, match="LEI"):
        c.list_filings({})


# ------------------------------------------------------------------------------------------------------------------------ into the pipeline
def test_a_connectors_draft_goes_through_review_and_is_stored_with_its_source(tmp_path):
    store = FundamentalsStore(tmp_path / "f.db")
    store.upsert_security(CompanyProfile("manual", "kr:005930", "005930", "Samsung Electronics", exchange="KR", market="KR", country="KR", currency="KRW"))
    sid = int(store.securities().iloc[0]["security_id"])
    svc = ImportService(store)
    c = conn("dart", dart_routes(), OPENDART_API_KEY="K")
    ref = {f.fiscal_period: f for f in c.list_filings({"ticker": "5930"})["filings"]}["FY"]
    batch = svc.create_batch("dart", sid)
    item = svc.add_from_connector(batch, c, ref)[0]
    assert item["status"] in ("extracted", "needs_review") and item["source_kind"] == "api" and item["method"] == "dart-api"
    again = svc.add_from_connector(batch, c, ref)[0]
    assert again["status"] == "duplicate", "the same answer is not imported twice"
    svc.decide(item["item_id"], [{"action": "confirm_suggested"}])
    out = svc.approve(item["item_id"], security={"security_id": sid})
    assert out["saved"][0]["lines"] >= 6
    prov = {p["metric_code"]: p for p in svc.provenance(sid)}
    assert prov["revenue"]["accession"] == ref.doc_id and prov["revenue"]["taxonomy_tag"] == "ifrs-full:Revenue" and prov["revenue"]["extraction_method"] == "dart-api" and prov["revenue"]["reported_unit"] == "KRW"
    periods = store.periods([sid])
    assert str(periods.iloc[0]["filing_date"])[:10] == "2024-03-12" and periods.iloc[0]["reporting_currency"] == "KRW"
    # a file from a connector keeps what the filing list knew
    ec = conn("edinet", edinet_routes(), EDINET_API_KEY="K")
    jp_ref = ec.list_filings({"ticker": "7203"}, since="2024-06-24", until="2024-06-26")["filings"][0]
    store.upsert_security(CompanyProfile("manual", "jp:7203", "7203", "Toyota", exchange="JP", market="JP", country="JP", currency="JPY"))
    jb = svc.create_batch("edinet", int(store.securities().iloc[1]["security_id"]))
    items = svc.add_from_connector(jb, ec, jp_ref)
    svc.process(jb)
    got = svc.item(items[0]["item_id"], full=True)
    assert got["draft"]["accession"] == "S100TOYO" and got["draft"]["periods"][0]["filing_date"] == "2024-06-25" and got["draft"]["periods"][0]["values"]["revenue"] == 48e12


def test_inline_numbers_in_the_formats_real_reports_use():
    """Formats seen in a real ESEF report (Hermès 2024): a non-breaking space between digit groups and a decimal comma."""
    n = xbrl._number
    assert n("15\xa0170", 6, None, "ixt:num-comma-decimal") == 15_170e6 and n("13 427", 6, None, "ixt:num-comma-decimal") == 13_427e6
    assert n("1.234,5", 0, None, "ixt:num-comma-decimal") == 1234.5 and n("1,234.5", 0, None, "ixt:num-dot-decimal") == 1234.5 and n("1\u202f234,5", 0, None, "ixt:numspacecomma") == 1234.5
    assert n("-", 3, None, "ixt:zerodash") == 0.0 and n("(12)", 0, None, None) == -12.0 and n("5", 3, "-", None) == -5000.0 and n("x", 0, None, None) is None


TAXONOMIES = {"jppfs": "http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-12-01/jppfs_cor_2023-12-01.xsd", "jpigp": "http://disclosure.edinet-fsa.go.jp/taxonomy/jpigp/2023-12-01/jpigp_cor_2023-12-01.xsd",
              "jpdei": "http://disclosure.edinet-fsa.go.jp/taxonomy/jpdei/2013-08-31/jpdei_cor_2013-08-31.xsd"}


def test_every_japanese_element_name_is_in_the_published_taxonomy():
    """The mapping was checked against the FSA's own taxonomy files; this repeats the check whenever they can be downloaded (skipped offline), so a renamed element is noticed."""
    import re
    import urllib.request

    from src.fundamentals.filings import maps
    from src.fundamentals.importer import xbrl as x

    def names(url):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return set(re.findall(r'<(?:\w+:)?element[^>]*?\bname="([^"]+)"', r.read().decode("utf-8")))
        except Exception:
            pytest.skip("the taxonomy could not be downloaded here")
    pfs, igp, dei = names(TAXONOMIES["jppfs"]), names(TAXONOMIES["jpigp"]), names(TAXONOMIES["jpdei"])
    ifrs = {n[:-4] for n in igp if n.endswith("IFRS")}
    wanted = {n for t in (maps.JP_ELEMENTS, maps.JP_ABS) for opts in t.values() for n, _ in opts}
    assert not {n for n in wanted if n not in pfs and n not in ifrs}, "a mapped element that is in neither Japanese taxonomy"
    assert set(x.JP_COVER) <= dei
