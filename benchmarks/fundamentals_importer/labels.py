"""Row labels per language, in two wordings. ``dev`` is the wording the extraction rules were developed against; ``alt`` is wording written *separately* (common alternatives in real reports) and used by the
validation and held-out sets, so the held-out result measures how the rules generalise to wording they were not written for."""

STATEMENT_TITLES = {
    "en": {"IS": ("Consolidated Income Statement", "Consolidated Statement of Profit or Loss"), "BS": ("Consolidated Balance Sheet", "Consolidated Statement of Financial Position"),
           "CF": ("Consolidated Statement of Cash Flows", "Consolidated Cash Flow Statement")},
    "ja": {"IS": ("連結損益計算書", "連結損益及び包括利益計算書"), "BS": ("連結貸借対照表", "連結貸借対照表"), "CF": ("連結キャッシュ・フロー計算書", "連結キャッシュ・フロー計算書")},
    "ko": {"IS": ("연결손익계산서", "연결포괄손익계산서"), "BS": ("연결재무상태표", "연결재무상태표"), "CF": ("연결현금흐름표", "연결현금흐름표")},
    "zh": {"IS": ("合并利润表", "合并利润表"), "BS": ("合并资产负债表", "合并资产负债表"), "CF": ("合并现金流量表", "合并现金流量表")},
    "zh-hk": {"IS": ("綜合損益表", "綜合收益表"), "BS": ("綜合財務狀況表", "綜合財務狀況表"), "CF": ("綜合現金流量表", "綜合現金流量表")},
    "fr": {"IS": ("Compte de résultat consolidé", "Compte de résultat consolidé"), "BS": ("Bilan consolidé", "État consolidé de la situation financière"), "CF": ("Tableau consolidé des flux de trésorerie", "Tableau des flux de trésorerie consolidés")},
    "de": {"IS": ("Konzern-Gewinn- und Verlustrechnung", "Konzern-Gewinn- und Verlustrechnung"), "BS": ("Konzernbilanz", "Konzernbilanz"), "CF": ("Konzern-Kapitalflussrechnung", "Konzern-Kapitalflussrechnung")},
    "es": {"IS": ("Cuenta de resultados consolidada", "Cuenta de pérdidas y ganancias consolidada"), "BS": ("Balance consolidado", "Estado de situación financiera consolidado"), "CF": ("Estado de flujos de efectivo consolidado", "Estado de flujos de efectivo consolidado")},
}
# code -> (dev wording, alt wording)
LABELS = {
    "en": {"revenue": ("Revenue", "Total revenues"), "cost_of_revenue": ("Cost of revenue", "Cost of sales"), "gross_profit": ("Gross profit", "Gross profit"), "operating_income": ("Operating income", "Profit from operations"),
           "income_before_tax": ("Income before income taxes", "Profit before tax"), "income_tax": ("Income tax expense", "Taxation"), "net_income": ("Net income", "Profit for the year"),
           "eps_basic": ("Basic earnings per share", "Earnings per share - basic"),
           "cash_and_equivalents": ("Cash and cash equivalents", "Cash and cash equivalents"), "accounts_receivable": ("Accounts receivable", "Trade and other receivables"), "inventory": ("Inventories", "Inventories"),
           "current_assets": ("Total current assets", "Total current assets"), "ppe": ("Property, plant and equipment", "Property, plant and equipment"), "total_assets": ("Total assets", "Total assets"),
           "accounts_payable": ("Accounts payable", "Trade and other payables"), "current_liabilities": ("Total current liabilities", "Total current liabilities"), "total_liabilities": ("Total liabilities", "Total liabilities"),
           "shareholders_equity": ("Total shareholders' equity", "Total equity"),
           "operating_cash_flow": ("Net cash provided by operating activities", "Net cash from operating activities"), "investing_cash_flow": ("Net cash used in investing activities", "Net cash from investing activities"),
           "financing_cash_flow": ("Net cash used in financing activities", "Net cash from financing activities")},
    "ja": {"revenue": ("売上高", "売上収益"), "cost_of_revenue": ("売上原価", "売上原価"), "gross_profit": ("売上総利益", "売上総利益"), "operating_income": ("営業利益", "営業利益"),
           "income_before_tax": ("税金等調整前当期純利益", "税引前利益"), "income_tax": ("法人税等", "法人所得税費用"), "net_income": ("親会社株主に帰属する当期純利益", "親会社の所有者に帰属する当期利益"),
           "cash_and_equivalents": ("現金及び預金", "現金及び現金同等物"), "accounts_receivable": ("受取手形及び売掛金", "営業債権及びその他の債権"), "inventory": ("棚卸資産", "棚卸資産"),
           "current_assets": ("流動資産合計", "流動資産合計"), "ppe": ("有形固定資産", "有形固定資産"), "total_assets": ("資産合計", "資産合計"), "accounts_payable": ("支払手形及び買掛金", "営業債務及びその他の債務"),
           "current_liabilities": ("流動負債合計", "流動負債合計"), "total_liabilities": ("負債合計", "負債合計"), "shareholders_equity": ("純資産合計", "資本合計"),
           "operating_cash_flow": ("営業活動によるキャッシュ・フロー", "営業活動によるキャッシュ・フロー"), "investing_cash_flow": ("投資活動によるキャッシュ・フロー", "投資活動によるキャッシュ・フロー"),
           "financing_cash_flow": ("財務活動によるキャッシュ・フロー", "財務活動によるキャッシュ・フロー")},
    "ko": {"revenue": ("매출액", "수익(매출액)"), "cost_of_revenue": ("매출원가", "매출원가"), "gross_profit": ("매출총이익", "매출총이익"), "operating_income": ("영업이익", "영업이익"),
           "income_before_tax": ("법인세비용차감전순이익", "법인세비용차감전계속사업이익"), "income_tax": ("법인세비용", "법인세비용"), "net_income": ("당기순이익", "연결당기순이익"),
           "cash_and_equivalents": ("현금및현금성자산", "현금및현금성자산"), "accounts_receivable": ("매출채권", "매출채권 및 기타채권"), "inventory": ("재고자산", "재고자산"), "current_assets": ("유동자산", "유동자산"),
           "ppe": ("유형자산", "유형자산"), "total_assets": ("자산총계", "자산총계"), "accounts_payable": ("매입채무", "매입채무 및 기타채무"), "current_liabilities": ("유동부채", "유동부채"),
           "total_liabilities": ("부채총계", "부채총계"), "shareholders_equity": ("자본총계", "자본총계"),
           "operating_cash_flow": ("영업활동현금흐름", "영업활동으로 인한 현금흐름"), "investing_cash_flow": ("투자활동현금흐름", "투자활동으로 인한 현금흐름"), "financing_cash_flow": ("재무활동현금흐름", "재무활동으로 인한 현금흐름")},
    "zh": {"revenue": ("营业收入", "营业总收入"), "cost_of_revenue": ("营业成本", "营业成本"), "operating_income": ("营业利润", "营业利润"), "income_before_tax": ("利润总额", "利润总额"), "income_tax": ("所得税费用", "所得税费用"),
           "net_income": ("净利润", "归属于母公司股东的净利润"), "cash_and_equivalents": ("货币资金", "货币资金"), "accounts_receivable": ("应收账款", "应收账款"), "inventory": ("存货", "存货"),
           "current_assets": ("流动资产合计", "流动资产合计"), "ppe": ("固定资产", "固定资产"), "total_assets": ("资产总计", "资产总计"), "accounts_payable": ("应付账款", "应付账款"),
           "current_liabilities": ("流动负债合计", "流动负债合计"), "total_liabilities": ("负债合计", "负债合计"), "shareholders_equity": ("所有者权益合计", "股东权益合计"),
           "operating_cash_flow": ("经营活动产生的现金流量净额", "经营活动产生的现金流量净额"), "investing_cash_flow": ("投资活动产生的现金流量净额", "投资活动产生的现金流量净额"),
           "financing_cash_flow": ("筹资活动产生的现金流量净额", "筹资活动产生的现金流量净额")},
    "zh-hk": {"revenue": ("收益", "營業額"), "cost_of_revenue": ("銷售成本", "銷售成本"), "gross_profit": ("毛利", "毛利"), "operating_income": ("經營溢利", "營運溢利"), "income_before_tax": ("除稅前溢利", "除稅前溢利"),
              "income_tax": ("所得稅開支", "稅項"), "net_income": ("年內溢利", "本公司擁有人應佔溢利"), "cash_and_equivalents": ("現金及現金等價物", "現金及現金等價物"), "inventory": ("存貨", "存貨"),
              "current_assets": ("流動資產總值", "流動資產總額"), "ppe": ("物業、廠房及設備", "物業、廠房及設備"), "total_assets": ("資產總值", "總資產"), "current_liabilities": ("流動負債總額", "流動負債總額"),
              "total_liabilities": ("負債總額", "總負債"), "shareholders_equity": ("權益總額", "總權益"), "operating_cash_flow": ("經營活動所得現金淨額", "經營活動產生之現金淨額"),
              "investing_cash_flow": ("投資活動所用現金淨額", "投資活動所用現金淨額"), "financing_cash_flow": ("融資活動所用現金淨額", "融資活動所用現金淨額")},
    "fr": {"revenue": ("Chiffre d'affaires", "Produits des activités ordinaires"), "cost_of_revenue": ("Coût des ventes", "Coût des ventes"), "gross_profit": ("Marge brute", "Marge brute"),
           "operating_income": ("Résultat opérationnel", "Résultat d'exploitation"), "income_before_tax": ("Résultat avant impôt", "Résultat avant impôts"), "income_tax": ("Impôt sur le résultat", "Impôts sur les bénéfices"),
           "net_income": ("Résultat net", "Résultat net de l'exercice"), "cash_and_equivalents": ("Trésorerie et équivalents de trésorerie", "Trésorerie et équivalents de trésorerie"),
           "accounts_receivable": ("Créances clients", "Clients et comptes rattachés"), "inventory": ("Stocks", "Stocks et en-cours"), "current_assets": ("Total des actifs courants", "Total actifs courants"),
           "ppe": ("Immobilisations corporelles", "Immobilisations corporelles"), "total_assets": ("Total de l'actif", "Total actif"), "accounts_payable": ("Dettes fournisseurs", "Fournisseurs et comptes rattachés"),
           "current_liabilities": ("Total des passifs courants", "Total passifs courants"), "total_liabilities": ("Total des passifs", "Total passif hors capitaux propres"), "shareholders_equity": ("Total des capitaux propres", "Capitaux propres"),
           "operating_cash_flow": ("Flux de trésorerie liés aux activités opérationnelles", "Flux net de trésorerie généré par l'activité"), "investing_cash_flow": ("Flux de trésorerie liés aux activités d'investissement", "Flux net de trésorerie lié aux opérations d'investissement"),
           "financing_cash_flow": ("Flux de trésorerie liés aux activités de financement", "Flux net de trésorerie lié aux opérations de financement")},
    "de": {"revenue": ("Umsatzerlöse", "Umsatz"), "cost_of_revenue": ("Herstellungskosten", "Umsatzkosten"), "gross_profit": ("Bruttoergebnis", "Bruttoergebnis vom Umsatz"), "operating_income": ("Betriebsergebnis", "Ergebnis der betrieblichen Tätigkeit"),
           "income_before_tax": ("Ergebnis vor Steuern", "Ergebnis vor Ertragsteuern"), "income_tax": ("Ertragsteuern", "Steuern vom Einkommen und Ertrag"), "net_income": ("Konzernergebnis", "Jahresüberschuss"),
           "cash_and_equivalents": ("Zahlungsmittel und Zahlungsmitteläquivalente", "Liquide Mittel"), "inventory": ("Vorräte", "Vorräte"), "current_assets": ("Kurzfristige Vermögenswerte", "Summe kurzfristige Vermögenswerte"),
           "ppe": ("Sachanlagen", "Sachanlagen"), "total_assets": ("Bilanzsumme", "Summe Aktiva"), "current_liabilities": ("Kurzfristige Schulden", "Summe kurzfristige Schulden"), "total_liabilities": ("Summe Schulden", "Summe Verbindlichkeiten"),
           "shareholders_equity": ("Eigenkapital", "Summe Eigenkapital"), "operating_cash_flow": ("Cashflow aus betrieblicher Tätigkeit", "Mittelzufluss aus laufender Geschäftstätigkeit"),
           "investing_cash_flow": ("Cashflow aus Investitionstätigkeit", "Mittelabfluss aus Investitionstätigkeit"), "financing_cash_flow": ("Cashflow aus Finanzierungstätigkeit", "Mittelabfluss aus Finanzierungstätigkeit")},
    "es": {"revenue": ("Importe neto de la cifra de negocios", "Ingresos de actividades ordinarias"), "operating_income": ("Resultado de explotación", "Resultado de explotación"), "income_before_tax": ("Resultado antes de impuestos", "Resultado antes de impuestos"),
           "income_tax": ("Impuesto sobre beneficios", "Impuesto sobre las ganancias"), "net_income": ("Resultado del ejercicio", "Beneficio del ejercicio"), "cash_and_equivalents": ("Efectivo y equivalentes de efectivo", "Efectivo y otros activos líquidos equivalentes"),
           "inventory": ("Existencias", "Inventarios"), "current_assets": ("Total activo corriente", "Activos corrientes totales"), "total_assets": ("Total activo", "Total activos"), "current_liabilities": ("Total pasivo corriente", "Pasivos corrientes totales"),
           "total_liabilities": ("Total pasivo", "Total pasivos"), "shareholders_equity": ("Total patrimonio neto", "Patrimonio neto total"), "operating_cash_flow": ("Flujos de efectivo de las actividades de explotación", "Flujos de efectivo de actividades de operación"),
           "investing_cash_flow": ("Flujos de efectivo de las actividades de inversión", "Flujos de efectivo de actividades de inversión"), "financing_cash_flow": ("Flujos de efectivo de las actividades de financiación", "Flujos de efectivo de actividades de financiación")},
}
STATEMENT_LINES = {"IS": ["revenue", "cost_of_revenue", "gross_profit", "operating_income", "income_before_tax", "income_tax", "net_income", "eps_basic"],
                   "BS": ["cash_and_equivalents", "accounts_receivable", "inventory", "current_assets", "ppe", "total_assets", "accounts_payable", "current_liabilities", "total_liabilities", "shareholders_equity"],
                   "CF": ["operating_cash_flow", "investing_cash_flow", "financing_cash_flow"]}
UNIT_WORDS = {"en": {"units": "", "thousands": "(in thousands)", "millions": "(in millions)"}, "ja": {"millions": "(単位：百万円)", "thousands": "(単位：千円)"}, "ko": {"millions": "(단위 : 백만원)", "thousands": "(단위 : 천원)"},
              "zh": {"thousands": "(单位：千元)", "units": "(单位：元)"}, "zh-hk": {"thousands": "(以千港元列示)"}, "fr": {"thousands": "(en milliers d'euros)", "millions": "(en millions d'euros)"},
              "de": {"thousands": "(in Tsd. EUR)", "millions": "(in Mio. EUR)"}, "es": {"thousands": "(en miles de euros)", "millions": "(en millones de euros)"}}


# ---- a third wording, for the held-out split only. Written separately from the other two (other common choices of words in each language) and never used to add a name to the label lists.
HELD = {
    "en": {"revenue": "Turnover", "cost_of_revenue": "Cost of goods sold", "gross_profit": "Gross margin", "operating_income": "Operating profit", "income_before_tax": "Profit before income tax", "income_tax": "Income taxes",
           "net_income": "Net profit", "eps_basic": "Basic EPS", "cash_and_equivalents": "Cash and bank balances", "accounts_receivable": "Receivables", "ppe": "Property and equipment, net", "accounts_payable": "Payables",
           "shareholders_equity": "Shareholders' equity", "operating_cash_flow": "Net cash flows from operating activities", "investing_cash_flow": "Net cash flows used in investing activities",
           "financing_cash_flow": "Net cash flows from financing activities"},
    "ja": {"revenue": "営業収益", "income_before_tax": "税引前当期純利益", "income_tax": "法人税、住民税及び事業税", "net_income": "当期純利益", "cash_and_equivalents": "現金及び現金同等物", "accounts_receivable": "売掛金",
           "inventory": "たな卸資産", "ppe": "有形固定資産合計", "total_assets": "総資産", "accounts_payable": "買掛金", "total_liabilities": "負債の部合計", "shareholders_equity": "純資産の部合計"},
    "ko": {"revenue": "영업수익", "operating_income": "영업이익(손실)", "income_before_tax": "법인세비용차감전계속영업이익", "net_income": "당기순이익(손실)", "accounts_receivable": "매출채권및기타채권", "total_assets": "자산 총계",
           "accounts_payable": "매입채무및기타채무", "total_liabilities": "부채 총계", "shareholders_equity": "자본 총계", "operating_cash_flow": "영업활동으로 인한 현금흐름"},
    "zh": {"revenue": "营业总收入", "accounts_receivable": "应收票据及应收账款", "accounts_payable": "应付票据及应付账款", "shareholders_equity": "归属于母公司所有者权益合计", "income_tax": "所得税"},
    "zh-hk": {"revenue": "收入", "income_before_tax": "除所得稅前溢利", "income_tax": "所得稅開支", "net_income": "年度溢利", "cash_and_equivalents": "現金及銀行結餘", "operating_cash_flow": "經營活動所得現金流量淨額",
              "investing_cash_flow": "投資活動所用現金流量淨額", "financing_cash_flow": "融資活動所得現金流量淨額"},
    "fr": {"revenue": "Ventes nettes", "cost_of_revenue": "Coût des produits vendus", "operating_income": "Résultat opérationnel courant", "income_tax": "Charge d'impôt", "net_income": "Résultat net - part du groupe",
           "cash_and_equivalents": "Disponibilités", "accounts_receivable": "Créances clients et comptes rattachés", "current_assets": "Total de l'actif courant", "ppe": "Immobilisations corporelles nettes",
           "accounts_payable": "Dettes fournisseurs et comptes rattachés", "current_liabilities": "Total du passif courant", "total_liabilities": "Total dettes"},
    "de": {"revenue": "Erlöse", "cost_of_revenue": "Herstellungskosten des Umsatzes", "gross_profit": "Rohergebnis", "operating_income": "EBIT", "income_before_tax": "Ergebnis vor Steuern (EBT)", "income_tax": "Steuern vom Einkommen",
           "net_income": "Periodenergebnis", "cash_and_equivalents": "Flüssige Mittel", "inventory": "Vorratsvermögen", "current_assets": "Kurzfristiges Vermögen", "ppe": "Sachanlagevermögen", "total_assets": "Summe der Aktiva",
           "current_liabilities": "Kurzfristige Verbindlichkeiten", "total_liabilities": "Verbindlichkeiten gesamt", "shareholders_equity": "Eigenkapital gesamt", "operating_cash_flow": "Mittelzufluss aus betrieblicher Tätigkeit"},
    "es": {"revenue": "Ventas netas", "income_before_tax": "Resultado antes de impuestos sobre ganancias", "income_tax": "Gasto por impuesto", "net_income": "Beneficio neto", "cash_and_equivalents": "Efectivo y equivalentes al efectivo",
           "inventory": "Inventarios", "current_assets": "Activo corriente", "total_assets": "Activo total", "current_liabilities": "Pasivo corriente", "total_liabilities": "Pasivo total", "shareholders_equity": "Patrimonio neto",
           "operating_cash_flow": "Flujos netos de efectivo de actividades de explotación"},
}
HELD_TITLES = {"en": {"IS": "Consolidated Statements of Operations", "BS": "Consolidated Statements of Financial Position", "CF": "Consolidated Cash Flows"},
               "ja": {"IS": "連結損益計算書", "BS": "連結貸借対照表", "CF": "連結キャッシュ・フロー計算書"}, "ko": {"IS": "연결 손익계산서", "BS": "연결 재무상태표", "CF": "연결 현금흐름표"},
               "zh": {"IS": "合并利润表", "BS": "合并资产负债表", "CF": "合并现金流量表"}, "zh-hk": {"IS": "綜合損益及其他全面收益表", "BS": "綜合財務狀況表", "CF": "綜合現金流量表"},
               "fr": {"IS": "État du résultat global consolidé", "BS": "Bilan consolidé", "CF": "Tableau des flux de trésorerie consolidés"},
               "de": {"IS": "Konzern-Gesamtergebnisrechnung", "BS": "Konzernbilanz", "CF": "Konzernkapitalflussrechnung"}, "es": {"IS": "Estado de resultados consolidado", "BS": "Balance de situación consolidado", "CF": "Estado de flujos de efectivo consolidado"}}
for _lang, _d in LABELS.items():
    for _code, _pair in list(_d.items()):
        _d[_code] = (_pair[0], _pair[1], HELD.get(_lang, {}).get(_code, _pair[1]))
for _lang, _t in STATEMENT_TITLES.items():
    for _s, _pair in list(_t.items()):
        _t[_s] = (_pair[0], _pair[1], HELD_TITLES.get(_lang, {}).get(_s, _pair[1]))
