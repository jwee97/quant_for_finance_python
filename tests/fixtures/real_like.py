"""Synthetic PDFs that reproduce the *structures* of seven real FY2020 annual reports (not their text, and not all of their numbers).

Values the task gives as reference values are used where a test asserts them; every other figure is invented and says nothing about the real company. What is reproduced: which columns a statement has
(current, prior, change, percent change, forecast, alternative currency, Group and Company side by side, note references), where the fiscal year ends, how a label wraps, how a statement continues on the
next page, bilingual labels, five-year summaries, quarterly tables inside an annual report, adjusted profit next to statutory profit, and look-alike tables (segment, parent-only, non-consolidated).

None of this proves anything about the original PDFs: see ``benchmarks/fundamentals_importer/real_filings``.
"""

from __future__ import annotations

import pymupdf

LATIN, CJK = "helv", "china-s"


class Doc:
    def __init__(self, font: str = LATIN):
        self.doc = pymupdf.open()
        self.font = font
        self.page = None
        self.y = 0.0

    def new_page(self):
        self.page = self.doc.new_page()
        self.y = 56.0
        return self

    def at(self, y: float):
        self.y = y
        return self

    def _font(self, s: str, font: str | None) -> str:
        """A string with CJK characters uses the CJK font; digits and Latin text use the Latin font (the CJK font sets Latin characters one full width wide)."""
        return font or (self.font if any(ord(ch) > 0x2E80 for ch in s) else LATIN)

    def left(self, x: float, s: str, size: float = 10, font: str | None = None):
        self.page.insert_text((x, self.y), s, fontsize=size, fontname=self._font(s, font))

    def right(self, x: float, s: str, size: float = 10, font: str | None = None):
        f = self._font(s, font)
        w = pymupdf.get_text_length(s, fontname=f, fontsize=size)
        self.page.insert_text((x - w, self.y), s, fontsize=size, fontname=f)

    def centre(self, x: float, s: str, size: float = 10, font: str | None = None):
        f = self._font(s, font)
        w = pymupdf.get_text_length(s, fontname=f, fontsize=size)
        self.page.insert_text((x - w / 2, self.y), s, fontsize=size, fontname=f)

    def nl(self, gap: float = 14.0):
        self.y += gap

    def table(self, title: str, unit: str | None, headers: list[list[tuple[float, str]]], rows: list[tuple], edges: list[float], note_edge: float | None = None, label_x: float = 40):
        """``rows``: ``(label, [values...])`` or ``(label, [values...], note)``; a label of ``None`` values is a section line."""
        if title:
            self.left(label_x, title, 12)
            self.nl(18)
        if unit:
            self.left(label_x, unit)
            self.nl(14)
        for h in headers:
            for x, s in h:
                (self.centre if s in ("Group", "Company") else self.right)(x, s)
            self.nl(14)
        for row in rows:
            label, vals = row[0], row[1]
            note = row[2] if len(row) > 2 else None
            self.left(label_x + (14 if label.startswith("  ") else 0), label.strip())
            if note and note_edge:
                self.right(note_edge, str(note))
            for e, v in zip(edges, vals):
                if v is not None:
                    self.right(e, v)
            self.nl(14)

    def bytes(self) -> bytes:
        return self.doc.tobytes()


def n(v, dec=0) -> str:
    s = f"{abs(v):,.{dec}f}"
    return f"({s})" if v < 0 else s


# ------------------------------------------------------------------------------------------------------------------------ Toyota: FY ends 31 March, change and percent columns, mezzanine equity, forecast, non-consolidated
def toyota() -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    d.left(40, "Consolidated Financial Results for the Year Ended March 31, 2020 (US GAAP)", 12)
    d.nl(20)
    d.left(40, "Financial Results Summary (Forecast for FY2021)")
    d.nl(14)
    for x, s in ((330, "FY2020"), (420, "FY2021 Forecast"), (500, "Change %")):
        d.right(x, s)
    d.nl(14)
    d.left(40, "Sales revenues")
    d.right(330, n(29929992)); d.right(420, n(26000000)); d.right(500, "(13.1)%")
    d.nl(14)
    d.left(40, "Operating income")
    d.right(330, n(2442869)); d.right(420, n(1500000)); d.right(500, "(38.6)%")
    d.new_page()
    edges = [330, 420, 500, 560]
    hdr = [[(330, "March 31, 2019"), (420, "March 31, 2020"), (500, "Increase/"), (560, "%")], [(500, "(Decrease)"), (560, "Change")]]
    d.table("Consolidated Balance Sheets", "(Yen in millions)", hdr, [
        ("Total current assets", [n(21250000), n(21000000), n(-250000), "(1.2)%"]),
        ("Total assets", [n(51936949), n(52680436), n(743487), "1.4%"]),
        ("Total current liabilities", [n(17000000), n(17500000), n(500000), "2.9%"]),
        ("Total liabilities", [n(30900000), n(31438585), n(538585), "1.7%"]),
        ("Mezzanine equity", [n(520000), n(504169), n(-15831), "(3.0)%"]),
        ("Total shareholders' equity", [n(20516949), n(20737682), n(220733), "1.1%"]),
    ], edges)
    d.new_page()
    hdr = [[(330, "Year ended March 31, 2019"), (420, "Year ended March 31, 2020")]]
    hdr = [[(330, "FY2019"), (420, "FY2020"), (500, "Change")], [(330, "March 31, 2019"), (420, "March 31, 2020")]]
    d.table("Consolidated Statements of Income", "(Yen in millions)", [[(330, "2019"), (420, "2020"), (500, "Change")]], [
        ("Sales revenues", [n(30225681), n(29929992), n(-295689)]),
        ("Cost of products sold", [n(24500000), n(24200000), n(-300000)]),
        ("Operating income", [n(2467545), n(2442869), n(-24676)]),
        ("Net income", [n(2036122), n(2142329), n(106207)]),
        ("Net income attributable to Toyota Motor Corporation", [n(1882873), n(2076183), n(193310)]),
    ], [330, 420, 500])
    d.new_page()
    d.table("Consolidated Statements of Cash Flows", "(Yen in millions)", [[(330, "2019"), (420, "2020")]], [
        ("Net cash provided by operating activities", [n(3797978), n(3590643)]),
        ("Net cash used in investing activities", [n(-2400000), n(-3100000)]),
    ], [330, 420])
    d.new_page()
    d.table("Non-consolidated Financial Results (J-GAAP)", "(Yen in millions)", [[(330, "FY2019"), (420, "FY2020")]], [
        ("Net sales", [n(8300000), n(8500000)]), ("Total assets", [n(17000000), n(17400000)]), ("Net income", [n(1300000), n(1400000)])], [330, 420])
    truth = {"currency": "JPY", "unit": 1_000_000, "period_end": "2020-03-31", "values": {"revenue": 29929992, "operating_income": 2442869, "net_income": 2142329, "net_income_attributable": 2076183, "total_assets": 52680436,
                                                                                        "total_liabilities": 31438585, "temporary_equity": 504169, "shareholders_equity": 20737682, "operating_cash_flow": 3590643},
             "prior": {"total_assets": 51936949}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ Samsung: KRW and USD columns for the same two years, different units
def samsung() -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    d.left(40, "Quarterly Segment Information (Q4 2020)", 12); d.nl(18)
    d.left(40, "(In millions of Korean won)"); d.nl(14)
    d.right(330, "Q4 2020"); d.right(420, "Q4 2019"); d.nl(14)
    d.left(40, "Revenue"); d.right(330, n(30474830)); d.right(420, n(27155000)); d.nl(14)
    d.new_page()
    edges = [320, 390, 470, 550]
    hdr = [[(390, "In millions of Korean won"), (550, "In thousands of US dollars")], [(320, "2020"), (390, "2019"), (470, "2020"), (550, "2019")]]
    d.table("Consolidated Statements of Profit or Loss", "(KRW in millions, USD in thousands, except per share)", hdr, [
        ("Revenue", [n(236806988), n(230400881), n(200606179), n(195179376)], 21),
        ("Cost of sales", [n(144488296), n(147239549), n(122402905), n(124729898)]),
        ("Gross profit", [n(92318692), n(83161332), n(78203274), n(70449478)]),
        ("Operating profit", [n(35993876), n(27768509), n(30491473), n(23523522)]),
        ("Profit for the year", [n(26407832), n(21738865), n(22370116), n(18416000)]),
    ], edges, note_edge=250)
    d.new_page()
    d.table("Consolidated Statements of Financial Position", "(KRW in millions, USD in thousands)", [[(390, "In millions of Korean won"), (550, "In thousands of US dollars")], [(320, "2020"), (390, "2019"), (470, "2020"), (550, "2019")]], [
        ("Total assets", [n(378235718), n(352564497), n(320404000), n(298650000)]),
        ("Total liabilities", [n(102287702), n(89684000), n(86650000), n(75970000)]),
        ("Total equity", [n(275948016), n(262880000), n(233754000), n(222680000)]),
    ], edges)
    d.new_page()
    d.table("Consolidated Statements of Cash Flows", "(KRW in millions, USD in thousands)", [[(390, "In millions of Korean won"), (550, "In thousands of US dollars")], [(320, "2020"), (390, "2019"), (470, "2020"), (550, "2019")]], [
        ("Net cash provided by operating activities", [n(65287009), n(45382000), n(55303000), n(38442000)])], edges)
    truth = {"currency": "KRW", "unit": 1_000_000, "period_end": "2020-12-31", "values": {"revenue": 236806988, "operating_income": 35993876, "net_income": 26407832, "total_assets": 378235718, "total_liabilities": 102287702,
                                                                                        "shareholders_equity": 275948016, "operating_cash_flow": 65287009},
             "alt": {"currency": "USD", "unit": 1000, "revenue": 200606179, "operating_income": 30491473}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ Apple: fiscal year ends in September, three years, calendar-year look-alikes
def apple() -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    d.left(40, "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", 11); d.nl(16)
    d.left(40, "For the fiscal year ended September 26, 2020"); d.nl(14)
    d.left(40, "Fiscal 2021 will include 53 weeks; calendar 2021 outlook and 2022 plans"); d.nl(14)
    d.new_page()
    d.table("Net sales by category", "(Dollars in millions)", [[(330, "2020"), (410, "2019"), (490, "Change")]], [
        ("iPhone", [n(137781), n(142381), "(3)%"]), ("Total net sales", [n(274515), n(260174), "6%"])], [330, 410, 490])
    d.new_page()
    d.table("CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions, except number of shares which are reflected in thousands and per share amounts)", [[(300, "Years ended")], [(300, "September 26, 2020"), (400, "September 28, 2019"), (500, "September 29, 2018")]], [
        ("Total net sales", ["$ " + n(274515), "$ " + n(260174), "$ " + n(265595)]), ("Total cost of sales", [n(169559), n(161782), n(163756)]), ("Gross margin", [n(104956), n(98392), n(101839)]),
        ("Operating income", [n(66288), n(63930), n(70898)]), ("Net income", [n(57411), n(55256), n(59531)])], [300, 400, 500])
    d.new_page()
    d.table("CONSOLIDATED BALANCE SHEETS", "(In millions, except number of shares which are reflected in thousands and par value)", [[(300, "September 26, 2020"), (400, "September 28, 2019")]], [
        ("Total current assets", [n(143713), n(162819)]), ("Total assets", [n(323888), n(338516)]), ("Total liabilities", [n(258549), n(248028)]), ("Total shareholders' equity", [n(65339), n(90488)])], [300, 400])
    d.new_page()
    d.table("CONSOLIDATED STATEMENTS OF CASH FLOWS", "(In millions)", [[(300, "Years ended")], [(300, "September 26, 2020"), (400, "September 28, 2019"), (500, "September 29, 2018")]], [
        ("Cash generated by operating activities", [n(80674), n(69391), n(77434)])], [300, 400, 500])
    truth = {"currency": "USD", "unit": 1_000_000, "period_end": "2020-09-26", "values": {"revenue": 274515, "net_income": 57411, "total_assets": 323888, "total_liabilities": 258549, "shareholders_equity": 65339,
                                                                                        "operating_cash_flow": 80674}, "period_ends": {"2020-09-26", "2019-09-28", "2018-09-29"}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ Singtel: Group and Company side by side; statement of financial position continues on the next page
def singtel(continued_header: bool = True) -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    edges = [330, 400]
    d.table("Consolidated Income Statement", "(S$ million)", [[(330, "2020"), (400, "2019")]], [
        ("Operating revenue", [n(17371), n(17224)], 4), ("Operating expenses", [n(-12800), n(-12700)]), ("Profit attributable to shareholders of the Company", [n(3141), n(3359)])], edges, note_edge=270)
    d.new_page()
    hdr = [[(350, "Group"), (505, "Company")], [(330, "2020"), (400, "2019"), (470, "2020"), (540, "2019")]]
    e4 = [330, 400, 470, 540]
    d.table("Statements of Financial Position", "As at 31 March 2020 (S$ million)", hdr, [
        ("Current assets", [None, None, None, None]),
        ("Cash and cash equivalents", [n(1800), n(1900), n(300), n(200)], 18),
        ("Trade and other", [None, None, None, None]),
        ("  receivables", [n(4200), n(4100), n(900), n(800)], 19),
        ("Total current assets", [n(6000), n(6000), n(1200), n(1000)]),
        ("Total assets", [n(35000), n(34000), n(20000), n(19000)]),
    ], e4, note_edge=270)
    d.new_page()
    if continued_header:
        d.left(40, "Statements of Financial Position (continued)", 12); d.nl(18)
        for x, s in ((330, "2020"), (400, "2019"), (470, "2020"), (540, "2019")):
            d.right(x, s)
        d.nl(14)
    for label, vals in (("Current liabilities", [None] * 4), ("Trade and other payables", [n(5000), n(4800), n(700), n(650)]), ("Total current liabilities", [n(8000), n(7500), n(1500), n(1400)]),
                        ("Total liabilities", [n(15000), n(14500), n(6000), n(5900)]), ("Total equity", [n(20000), n(19500), n(14000), n(13100)])):
        d.left(40, label)
        for e, v in zip(e4, vals):
            if v:
                d.right(e, v)
        d.nl(14)
    d.new_page()
    d.table("Consolidated Cash Flow Statement", "(S$ million)", [[(330, "2020"), (400, "2019")]], [("Cash flows from operating activities", [None, None]), ("Net cash inflow from operating activities", [n(5300), n(5100)])], edges)
    truth = {"currency": "SGD", "unit": 1_000_000, "period_end": "2020-03-31", "values": {"revenue": 17371, "total_assets": 35000, "total_liabilities": 15000, "shareholders_equity": 20000, "operating_cash_flow": 5300},
             "company": {"total_assets": 20000, "total_liabilities": 6000}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ BYD: bilingual labels, five-year summary, forecast
def byd() -> tuple[bytes, dict]:
    d = Doc(font=CJK)
    d.new_page()
    E5 = [330, 400, 470, 540, 610]
    d.table("五年财务摘要 Five-Year Summary", "(RMB'000 人民币千元)", [[(330, "2020"), (400, "2019"), (470, "2018"), (540, "2017"), (610, "2016")]], [
        ("Revenue 营业收入", [n(153469184), n(127739000), n(130047000), n(105914000), n(103470000)]), ("Total assets 资产总额", [n(201017321), n(195000000), n(180000000), n(170000000), n(160000000)])], E5, label_x=30)
    d.new_page()
    d.left(30, "Outlook 展望: revenue forecast", 11); d.nl(14)
    for x, s_ in ((470, "2020"), (560, "2021 (Forecast)")):
        d.right(x, s_)
    d.nl(14)
    d.left(30, "Revenue 营业收入"); d.right(470, n(153469184)); d.right(560, n(190000000)); d.nl(14)
    d.new_page()
    E2 = [500, 580]
    d.table("合并利润表 Consolidated Income Statement", "(RMB'000 人民币千元)", [[(500, "2020"), (580, "2019")], [(500, "2020年"), (580, "2019年")]], [
        ("Revenue 营业收入", [n(153469184), n(127739000)]), ("Cost of sales 营业成本", [n(126225593), n(105000000)]), ("Gross profit 毛利", [n(27243591), n(22739000)]),
        ("Net profit attributable 归属于母公司股东的净利润", [n(4234267), n(1614000)])], E2, label_x=30)
    d.new_page()
    d.table("合并资产负债表 Consolidated Balance Sheet", "(RMB'000 人民币千元)", [[(500, "31 December 2020"), (580, "31 December 2019")]], [("Total assets 资产总计", [n(201017321), n(195000000)])], E2, label_x=30)
    truth = {"currency": "CNY", "unit": 1000, "period_end": "2020-12-31", "values": {"revenue": 153469184, "gross_profit": 27243591, "net_income_attributable": 4234267, "total_assets": 201017321}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ Tencent: annual statements, a Q4 table, statutory and non-IFRS profit
def tencent() -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    d.table("Fourth Quarter 2020 Results", "(RMB in millions)", [[(330, "Three months ended 31 December 2020"), (440, "31 December 2019"), (520, "YoY change")]], [
        ("Revenues", [n(133669), n(105767), "26%"]), ("Operating profit", [n(63713), n(33750), "89%"]), ("Non-IFRS operating profit", [n(53400), n(40000), "33%"]), ("Profit for the period", [n(51000), n(27000), "89%"])], [330, 440, 520])
    d.new_page()
    d.table("Consolidated Statement of Comprehensive Income", "(RMB in millions)", [[(330, "Year ended 31 December"), ], [(330, "2020"), (410, "2019")]], [
        ("Revenues", [n(482064), n(377289)]), ("Cost of revenues", [n(-260532), n(-209756)]), ("Gross profit", [n(221532), n(167533)]), ("Operating profit", [n(184237), n(121898)]), ("Profit before income tax", [n(180022), n(130000)]),
        ("Profit for the year", [n(160125), n(95000)]), ("Attributable to: Equity holders of the Company", [n(159847), n(94000)])], [330, 410])
    d.new_page()
    d.table("Consolidated Statement of Financial Position", "(RMB in millions)", [[(330, "As at 31 December 2020"), (410, "2019")]], [("Total assets", [n(1333425), n(969000)]), ("Total liabilities", [n(555382), n(430000)]), ("Total equity", [n(778043), n(539000)])], [330, 410])
    d.new_page()
    d.table("Non-IFRS Reconciliation", "(RMB in millions)", [[(330, "2020"), (410, "2019")]], [("Non-IFRS profit attributable to equity holders of the Company", [n(122000), n(98000)]), ("Profit attributable to equity holders of the Company", [n(159847), n(94000)])], [330, 410])
    truth = {"currency": "CNY", "unit": 1_000_000, "period_end": "2020-12-31", "values": {"revenue": 482064, "gross_profit": 221532, "operating_income": 184237, "income_before_tax": 180022, "net_income": 160125,
                                                                                        "net_income_attributable": 159847, "total_assets": 1333425, "total_liabilities": 555382, "shareholders_equity": 778043},
             "q4": {"revenue": 133669, "operating_income": 63713}}
    return d.bytes(), truth


# ------------------------------------------------------------------------------------------------------------------------ ASML: IFRS (EU) with a US GAAP section, totals and their components
def asml() -> tuple[bytes, dict]:
    d = Doc()
    d.new_page()
    d.left(40, "Basis of preparation: the Consolidated Financial Statements are prepared in accordance with IFRS as adopted by the European Union."); d.nl(14)
    d.left(40, "A US GAAP presentation is provided in a separate section for reference."); d.nl(14)
    d.new_page()
    d.table("Consolidated Statement of Profit or Loss", "(EUR millions, unless otherwise indicated)", [[(330, "2020"), (410, "2019")]], [
        ("Net system sales", [n(11298.4, 1), n(9573.5, 1)]), ("Net service and field option sales", [n(2680.1, 1), n(2840.4, 1)]), ("Total net sales", [n(13978.5, 1), n(11820.5, 1)]),
        ("Cost of system sales", [n(-5972.7, 1), n(-5107.7, 1)]), ("Cost of service and field option sales", [n(-1521.2, 1), n(-1500.0, 1)]), ("Total cost of sales", [n(-7493.9, 1), n(-6607.7, 1)]),
        ("Gross profit", [n(6484.6, 1), n(5212.8, 1)]), ("Research and development costs", [n(-2200.0, 1), n(-1500.0, 1)]), ("Selling, general and administrative costs", [n(-600.0, 1), n(-550.0, 1)]),
        ("Income from operations", [n(3684.6, 1), n(3162.8, 1)])], [330, 410])
    d.new_page()
    d.left(40, "US GAAP Consolidated Statement of Operations (reference)", 12); d.nl(18)
    d.left(40, "(EUR millions)"); d.nl(14)
    d.right(330, "2020"); d.nl(14)
    d.left(40, "Total net sales"); d.right(330, n(13978.5, 1)); d.nl(14)
    d.left(40, "Cost of sales"); d.right(330, n(-7500.3, 1)); d.nl(14)
    truth = {"currency": "EUR", "unit": 1_000_000, "period_end": "2020-12-31", "values": {"revenue": 13978.5, "cost_of_revenue": 7493.9, "gross_profit": 6484.6, "operating_income": 3684.6}, "basis": "ifrs"}
    return d.bytes(), truth


BUILDERS = {"toyota": toyota, "samsung": samsung, "apple": apple, "singtel": singtel, "byd": byd, "tencent": tencent, "asml": asml}
