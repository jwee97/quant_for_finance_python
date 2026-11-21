"""The names of the statement lines in the languages of the filings this application reads: Japanese, Korean, Chinese (simplified and traditional), French, Spanish, German and Portuguese.

A report on paper or in a PDF names its lines in words, not in a taxonomy's element names, so the reader of a PDF, an HTML table or a spreadsheet needs the words. Each canonical line has the names in
common use in a statement; they are matched after the same normalisation as the English ones (lower case, punctuation and spaces folded), so the table and the lookup agree. Where a word is ambiguous
(the French "total du passif" is liabilities *and* equity; the Chinese 销售费用 is selling expense, not selling and administrative) it is left out rather than mapped wrongly: a line that is not
recognised is reported as such and a person types it. The names were written from the usual statement formats of each market, not checked against every company's wording.
"""

from __future__ import annotations

import re

NAMES: dict[str, list[str]] = {
    "revenue": ["売上高", "営業収益", "売上収益", "순매출액", "매출액", "수익(매출액)", "营业收入", "营业总收入", "營業額", "收益", "營業收入", "chiffre d'affaires", "produits des activités ordinaires", "importe neto de la cifra de negocios",
                "ingresos de actividades ordinarias", "umsatzerlöse", "umsatz", "receita de venda de bens e/ou serviços", "receita líquida", "receita operacional líquida"],
    "cost_of_revenue": ["売上原価", "매출원가", "营业成本", "銷售成本", "銷貨成本", "coût des ventes", "coste de ventas", "herstellungskosten", "custo dos bens e/ou serviços vendidos"],
    "gross_profit": ["売上総利益", "매출총이익", "毛利", "毛利潤", "marge brute", "beneficio bruto", "bruttoergebnis", "resultado bruto"],
    "selling_general_admin": ["販売費及び一般管理費", "판매비와관리비", "판매비와 관리비", "销售及管理费用", "行政开支", "frais généraux", "gastos generales", "vertriebs- und verwaltungskosten", "despesas gerais e administrativas"],
    "research_and_development": ["研究開発費", "연구개발비", "研发费用", "研發費用", "frais de recherche et développement", "gastos de investigación y desarrollo", "forschungs- und entwicklungskosten", "pesquisa e desenvolvimento"],
    "operating_income": ["営業利益", "영업이익", "营业利润", "經營溢利", "營運溢利", "résultat opérationnel", "résultat d'exploitation", "resultado de explotación", "betriebsergebnis", "ebit",
                         "resultado antes do resultado financeiro e dos tributos"],
    "interest_expense": ["支払利息", "이자비용", "利息支出", "财务费用", "融資成本", "charges d'intérêts", "gastos financieros", "zinsaufwendungen", "despesas financeiras"],
    "income_before_tax": ["税引前利益", "税引前当期純利益", "税金等調整前当期純利益", "법인세비용차감전순이익", "법인세차감전순이익", "利润总额", "除稅前溢利", "résultat avant impôt", "resultado antes de impuestos", "ergebnis vor steuern",
                          "resultado antes dos tributos sobre o lucro"],
    "income_tax": ["法人税等", "法人税、住民税及び事業税", "법인세비용", "所得税费用", "所得稅", "所得稅開支", "impôt sur les résultats", "impuesto sobre beneficios", "ertragsteuern", "imposto de renda e contribuição social sobre o lucro"],
    "net_income": ["当期純利益", "親会社株主に帰属する当期純利益", "当期利益", "親会社の所有者に帰属する当期利益", "당기순이익", "지배기업 소유주지분 당기순이익", "净利润", "归属于母公司股东的净利润", "归属于母公司所有者的净利润",
                   "本年度溢利", "年內溢利", "本公司擁有人應佔溢利", "résultat net", "resultado del ejercicio", "jahresüberschuss", "konzernergebnis", "lucro/prejuízo consolidado do período", "lucro líquido"],
    "eps_basic": ["1株当たり当期純利益", "基本的1株当たり当期利益", "기본주당이익", "기본주당순이익", "基本每股收益", "每股基本盈利", "résultat de base par action", "beneficio básico por acción", "ergebnis je aktie", "lucro básico por ação"],
    "eps_diluted": ["潜在株式調整後1株当たり当期純利益", "希薄化後1株当たり当期利益", "희석주당이익", "희석주당순이익", "稀释每股收益", "每股攤薄盈利", "résultat dilué par action", "beneficio diluido por acción", "verwässertes ergebnis je aktie",
                    "lucro diluído por ação"],
    "cash_and_equivalents": ["現金及び現金同等物", "현금및현금성자산", "现金及现金等价物", "货币资金", "現金及現金等價物", "trésorerie et équivalents de trésorerie", "efectivo y equivalentes al efectivo", "zahlungsmittel und zahlungsmitteläquivalente",
                             "caixa e equivalentes de caixa"],
    "accounts_receivable": ["受取手形及び売掛金", "売掛金", "매출채권", "应收账款", "應收賬款", "créances clients", "deudores comerciales", "forderungen aus lieferungen und leistungen", "contas a receber"],
    "inventory": ["棚卸資産", "재고자산", "存货", "存貨", "stocks", "existencias", "vorräte", "estoques"],
    "current_assets": ["流動資産合計", "流動資産", "유동자산", "流动资产合计", "流動資產", "流動資產總值", "actifs courants", "activo corriente", "kurzfristige vermögenswerte", "ativo circulante"],
    "total_assets": ["資産合計", "総資産", "자산총계", "资产总计", "总资产", "資產總值", "總資產", "total de l'actif", "total activo", "bilanzsumme", "ativo total"],
    "ppe": ["有形固定資産", "유형자산", "固定资产", "物業、廠房及設備", "immobilisations corporelles", "inmovilizado material", "sachanlagen", "imobilizado"],
    "goodwill": ["のれん", "영업권", "商誉", "商譽", "écarts d'acquisition", "fondo de comercio", "geschäfts- oder firmenwert", "ágio"],
    "accounts_payable": ["支払手形及び買掛金", "買掛金", "매입채무", "应付账款", "應付賬款", "dettes fournisseurs", "acreedores comerciales", "verbindlichkeiten aus lieferungen und leistungen", "fornecedores"],
    "current_liabilities": ["流動負債合計", "流動負債", "유동부채", "流动负债合计", "流動負債", "流動負債總額", "passifs courants", "pasivo corriente", "kurzfristige schulden", "passivo circulante"],
    "short_term_debt": ["短期借入金", "단기차입금", "短期借款", "短期借貸", "emprunts à court terme", "deuda a corto plazo", "kurzfristige finanzverbindlichkeiten", "empréstimos e financiamentos de curto prazo"],
    "long_term_debt": ["長期借入金", "장기차입금", "长期借款", "長期借貸", "emprunts à long terme", "deuda a largo plazo", "langfristige finanzverbindlichkeiten"],
    "total_liabilities": ["負債合計", "부채총계", "负债合计", "總負債", "負債總額", "total des passifs", "total pasivo", "gesamte schulden", "passivo total"],
    "shareholders_equity": ["純資産合計", "株主資本合計", "親会社の所有者に帰属する持分合計", "자본총계", "지배기업 소유주지분", "所有者权益合计", "股东权益合计", "归属于母公司所有者权益合计", "權益總額", "股東權益", "本公司擁有人應佔權益",
                            "capitaux propres", "patrimonio neto", "eigenkapital", "patrimônio líquido"],
    "minority_interest": ["非支配株主持分", "비지배지분", "少数股东权益", "非控股權益", "participations ne donnant pas le contrôle", "participaciones no dominantes", "anteile ohne beherrschenden einfluss", "participação dos não controladores"],
    "retained_earnings": ["利益剰余金", "이익잉여금", "未分配利润", "保留溢利", "report à nouveau", "reservas", "gewinnrücklagen", "lucros acumulados"],
    "operating_cash_flow": ["営業活動によるキャッシュ・フロー", "영업활동현금흐름", "영업활동으로 인한 현금흐름", "经营活动产生的现金流量净额", "經營活動所得現金淨額", "flux de trésorerie liés aux activités opérationnelles",
                            "flujos de efectivo de las actividades de explotación", "cashflow aus laufender geschäftstätigkeit", "caixa líquido atividades operacionais"],
    "investing_cash_flow": ["投資活動によるキャッシュ・フロー", "투자활동현금흐름", "투자활동으로 인한 현금흐름", "投资活动产生的现金流量净额", "投資活動所用現金淨額", "flux de trésorerie liés aux activités d'investissement",
                            "flujos de efectivo de las actividades de inversión", "cashflow aus investitionstätigkeit", "caixa líquido atividades de investimento"],
    "financing_cash_flow": ["財務活動によるキャッシュ・フロー", "재무활동현금흐름", "재무활동으로 인한 현금흐름", "筹资活动产生的现金流量净额", "融資活動所得現金淨額", "flux de trésorerie liés aux activités de financement",
                            "flujos de efectivo de las actividades de financiación", "cashflow aus finanzierungstätigkeit", "caixa líquido atividades de financiamento"],
    "capital_expenditure": ["有形固定資産の取得による支出", "유형자산의 취득", "购建固定资产、无形资产和其他长期资产支付的现金", "購買物業、廠房及設備", "acquisitions d'immobilisations corporelles", "adquisición de inmovilizado material",
                            "investitionen in sachanlagen", "aquisição de imobilizado"],
    "dividends_paid": ["配当金の支払額", "배당금지급", "배당금의 지급", "支付股利", "已付股息", "dividendes versés", "dividendos pagados", "gezahlte dividenden", "dividendos pagos"],
    "depreciation_amortization": ["減価償却費", "감가상각비", "折旧与摊销", "折舊及攤銷", "dotations aux amortissements", "amortización", "abschreibungen", "depreciação e amortização"],
    "shares_outstanding": ["発行済株式総数", "발행주식총수", "总股本", "已發行股份", "nombre d'actions", "número de acciones", "anzahl der aktien", "número de ações"],
}


def normalise(text) -> str:
    """Lower case, accents kept, every run of punctuation and spaces folded to one space (the same rule as :func:`manual._norm`, so a name and a label meet)."""
    return re.sub(r"[^\w&]+", " ", str(text).lower().replace("_", " ").replace("&", " & ")).replace(" & ", "&").strip()


def table() -> dict[str, str]:
    """``{normalised name: canonical line}``; a name that two lines claim goes to the first."""
    out: dict[str, str] = {}
    for code, names in NAMES.items():
        for n in names:
            out.setdefault(normalise(n), code)
    return out


ALIASES_I18N = table()
