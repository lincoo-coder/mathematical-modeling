import pandas as pd
import os


# 日志函数
def log(message):
    print(f"[INFO] {message}")


# 替换错误标签
def replace_invalid_labels(df, column_name, label_replacement_map):
    """
    替换NOC列中的错误标签为目标内容
    :param df: DataFrame
    :param column_name: 需要替换标签的列名
    :param label_replacement_map: 错误标签到目标标签的映射字典
    """
    # 先去除每个值两端的空白字符
    df[column_name] = df[column_name].str.strip()

    # 记录原始NOC列的值
    original_noc_values = df[column_name].copy()

    # 强制替换
    df[column_name] = df[column_name].replace(label_replacement_map)

    # 记录修改的行数
    modified_rows = (df[column_name] != original_noc_values).sum()
    log(f"替换了 {modified_rows} 行数据")
    return df


# ---------------------------------------------------------------main部分----------------------------------------------------------------

def preprocess_data():
    # 错误标签与目标标签的映射（即三字母的简称）
    label_replacement_map = {
        "United States": "USA", "Greece": "GRE", "Germany": "GER", "France": "FRA",
        "Great Britain": "GBR", "Hungary": "HUN", "Austria": "AUT", "Australia": "AUS",
        "Denmark": "DEN", "Switzerland": "SUI", "Mixed team": "MIX", "Belgium": "BEL",
        "Italy": "ITA", "Cuba": "CUB", "Canada": "CAN", "Spain": "ESP", "Luxembourg": "LUX",
        "Norway": "NOR", "Netherlands": "NED", "India": "IND", "Bohemia": "BOH", "Sweden": "SWE",
        "Australasia": "OCE", "Russian Empire": "RUS", "Finland": "FIN", "South Africa": "RSA",
        "Estonia": "EST", "Brazil": "BRA", "Japan": "JPN", "Czechoslovakia": "TCH",
        "New Zealand": "NZL", "Yugoslavia": "YUG", "Argentina": "ARG", "Uruguay": "URY",
        "Poland": "POL", "Haiti": "HTI", "Portugal": "PRT", "Romania": "ROU", "Egypt": "EGY",
        "Ireland": "IRL", "Chile": "CHI", "Philippines": "PHI", "Latvia": "LVA", "Mexico": "MEX",
        "Jamaica": "JAM", "Peru": "PER", "Ceylon": "CEY", "Trinidad and Tobago": "TTO",
        "Panama": "PAN", "South Korea": "KOR", "Iran": "IRI", "Puerto Rico": "PRI",
        "Soviet Union": "URS", "Lebanon": "LBN", "Bulgaria": "BUL", "Venezuela": "VEN",
        "United Team of Germany": "GDR", "Iceland": "ISL", "Pakistan": "PAK", "Bahamas": "BHS",
        "Turkey": "TUR", "Romania": "ROU", "Yugoslavia": "YUG", "Ethiopia": "ETH",
        "Greece": "GRE", "Norway": "NOR", "Iran": "IRI", "Egypt": "EGY", "Formosa": "TPE",
        "Ghana": "GHA", "Morocco": "MAR", "Portugal": "PRT", "Singapore": "SGP", "Brazil": "BRA",
        "British West Indies": "BWI", "Iraq": "IRQ", "Venezuela": "VEN", "Ethiopia": "ETH",
        "Tunisia": "TUN", "Kenya": "KEN", "Nigeria": "NGA", "East Germany": "GDR",
        "West Germany": "FRG", "Mongolia": "MNG", "Uganda": "UGA", "Cameroon": "CMR",
        "Taiwan": "TWN", "North Korea": "PRK", "Colombia": "COL", "Niger": "NER",
        "Bermuda": "BMU", "Thailand": "THA", "Zimbabwe": "ZWE", "Tanzania": "TAN",
        "Guyana": "GUY", "China": "CHN", "Ivory Coast": "CIV", "Syria": "SYR", "Algeria": "DZA",
        "Chinese Taipei": "TPE", "Dominican Republic": "DOM", "Zambia": "ZMB", "Suriname": "SUR",
        "Costa Rica": "CRI", "Indonesia": "IDN", "Netherlands Antilles": "NCA", "Senegal": "SEN",
        "Virgin Islands": "VGB", "Djibouti": "DJI", "Lithuania": "LTU", "Namibia": "NAM",
        "Croatia": "HRV", "Israel": "ISR", "Slovenia": "SVN", "Malaysia": "MAS",
        "Qatar": "QAT", "Russia": "RUS", "Ukraine": "UKR", "Czech Republic": "CZE",
        "Kazakhstan": "KAZ", "Belarus": "BLR", "FR Yugoslavia": "YUG", "Slovakia": "SVK",
        "Armenia": "ARM", "Burundi": "BDI", "Ecuador": "ECU", "Hong Kong": "HKG",
        "Moldova": "MDA", "Uzbekistan": "UZB", "Azerbaijan": "AZE", "Tonga": "TON",
        "Georgia": "GEO", "Mozambique": "MOZ", "Saudi Arabia": "KSA", "Sri Lanka": "SRI",
        "Vietnam": "VNM", "Barbados": "BRB", "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
        "Macedonia": "MKD", "United Arab Emirates": "UAE", "Serbia and Montenegro": "SCG",
        "Paraguay": "PRY", "Eritrea": "ERI", "Serbia": "SRB", "Tajikistan": "TJK",
        "Samoa": "SAM", "Singapore": "SGP", "Sudan": "SDN", "Afghanistan": "AFG",
        "Mauritius": "MUS", "Togo": "TGO", "Bahrain": "BHR", "Grenada": "GRD",
        "Botswana": "BWA", "Cyprus": "CYP", "Gabon": "GAB", "Guatemala": "GTM",
        "Montenegro": "MNE", "Independent Olympic Athletes": "IOA", "Fiji": "FJI",
        "Jordan": "JOR", "Kosovo": "KOS", "San Marino": "SMR", "North Macedonia": "MKD",
        "Turkmenistan": "TKM", "Burkina Faso": "BFA", "Saint Lucia": "LCA", "Dominica": "DMA",
        "Albania": "ALB", "Cabo Verde": "CPV", "Refugee Olympic Team": "ROT"
    }

    log("开始处理夏季奥运奖牌数据（summerOly_medal_counts.csv）")

    # 读取数据
    medals_df = pd.read_csv("../../data/第一问处理的数据/第一阶段/summerOly_medal_counts.csv")

    # 替换NOC列中的错误标签
    medals_df = replace_invalid_labels(medals_df, 'NOC', label_replacement_map)

    # 保存修改后的数据框
    medals_df.to_csv("./第二阶段/处理过NOC的原版本medal数据.csv", index=False)

    log("处理后的数据保存完毕")


# 调用数据处理函数
preprocess_data()
