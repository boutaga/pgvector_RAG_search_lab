#!/usr/bin/env python3
"""
00_generate_parquet.py — Generate Parquet data lake on MinIO (S3)

Creates realistic Swiss private bank trading data as Parquet files
and uploads them to a MinIO bucket (S3-compatible).

Usage:
    python python/00_generate_parquet.py

Data generated (seed=42, reproducible):
    exchanges, instruments, counterparties, clients, accounts,
    orders, executions, positions, cash_balances, market_prices,
    risk_limits, risk_metrics, compliance_checks, aml_alerts
"""

import os
import random
import io
from datetime import date, datetime, timedelta
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.client import Config

import numpy as np
import config

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# S3 / MINIO CLIENT
# ---------------------------------------------------------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

def upload_parquet(s3, bucket: str, key: str, df: pd.DataFrame):
    """Upload DataFrame as Parquet to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    size_kb = buf.getbuffer().nbytes / 1024
    print(f"   ✓ {key}: {len(df):,} rows ({size_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# REFERENCE DATA
# ---------------------------------------------------------------------------

EXCHANGES = [
    {"exchange_id":1,"mic_code":"XSWX","exchange_name":"SIX Swiss Exchange","country_code":"CH","timezone":"Europe/Zurich"},
    {"exchange_id":2,"mic_code":"XETR","exchange_name":"Xetra (Deutsche Börse)","country_code":"DE","timezone":"Europe/Berlin"},
    {"exchange_id":3,"mic_code":"XPAR","exchange_name":"Euronext Paris","country_code":"FR","timezone":"Europe/Paris"},
    {"exchange_id":4,"mic_code":"XLON","exchange_name":"London Stock Exchange","country_code":"GB","timezone":"Europe/London"},
    {"exchange_id":5,"mic_code":"XNYS","exchange_name":"New York Stock Exchange","country_code":"US","timezone":"America/New_York"},
    {"exchange_id":6,"mic_code":"XNAS","exchange_name":"NASDAQ","country_code":"US","timezone":"America/New_York"},
    {"exchange_id":7,"mic_code":"XTKS","exchange_name":"Tokyo Stock Exchange","country_code":"JP","timezone":"Asia/Tokyo"},
    {"exchange_id":8,"mic_code":"XHKG","exchange_name":"Hong Kong Stock Exchange","country_code":"HK","timezone":"Asia/Hong_Kong"},
]

INSTRUMENTS_RAW = [
    # (isin, sedol, ticker, name, asset_class, sub_class, sector, industry_group, issuer, ccy, exchange_mic, country_of_risk, maturity, coupon)
    ("CH0012005267","0QRE","NESN SW","Nestlé SA","equity","large_cap","Consumer Staples","Food Products","Nestlé","CHF","XSWX","CH",None,None),
    ("CH0012032048","0QMH","ROG SW","Roche Holding AG","equity","large_cap","Health Care","Pharmaceuticals","Roche","CHF","XSWX","CH",None,None),
    ("CH0038863350","0QN6","NOVN SW","Novartis AG","equity","large_cap","Health Care","Pharmaceuticals","Novartis","CHF","XSWX","CH",None,None),
    ("CH0244767585","BYZ45K","UBSG SW","UBS Group AG","equity","large_cap","Financials","Banks","UBS","CHF","XSWX","CH",None,None),
    ("CH0210483332","BFNR3T","ZURN SW","Zurich Insurance Group","equity","large_cap","Financials","Insurance","Zurich Insurance","CHF","XSWX","CH",None,None),
    ("CH0012221716","BN0MJ5","ABBN SW","ABB Ltd","equity","large_cap","Industrials","Electrical Equip","ABB","CHF","XSWX","CH",None,None),
    ("CH1175448666","BNKGYB","SREN SW","Swiss Re AG","equity","large_cap","Financials","Insurance","Swiss Re","CHF","XSWX","CH",None,None),
    ("CH0418792922","BFZXQR","SIKA SW","Sika AG","equity","mid_cap","Materials","Chemicals","Sika","CHF","XSWX","CH",None,None),
    ("CH0013841017","0QOT","LONN SW","Lonza Group AG","equity","mid_cap","Health Care","Life Sciences","Lonza","CHF","XSWX","CH",None,None),
    ("CH0030170408","0QQ4","GEBN SW","Geberit AG","equity","mid_cap","Industrials","Building Products","Geberit","CHF","XSWX","CH",None,None),
    ("DE0007164600","4927","SAP GY","SAP SE","equity","large_cap","Information Technology","Software","SAP","EUR","XETR","DE",None,None),
    ("FR0000121014","B0KGR9","MC FP","LVMH Moët Hennessy","equity","large_cap","Consumer Discretionary","Luxury Goods","LVMH","EUR","XPAR","FR",None,None),
    ("NL0010273215","B7KR2L","ASML NA","ASML Holding NV","equity","large_cap","Information Technology","Semiconductors","ASML","EUR","XETR","NL",None,None),
    ("DE0007100000","5765MH","DAI GY","Mercedes-Benz Group","equity","large_cap","Consumer Discretionary","Automobiles","Mercedes-Benz","EUR","XETR","DE",None,None),
    ("FR0000120271","5765KR","TTE FP","TotalEnergies SE","equity","large_cap","Energy","Oil & Gas","TotalEnergies","EUR","XPAR","FR",None,None),
    ("US0378331005","2046251","AAPL US","Apple Inc","equity","large_cap","Information Technology","Consumer Elec","Apple","USD","XNAS","US",None,None),
    ("US5949181045","2588173","MSFT US","Microsoft Corp","equity","large_cap","Information Technology","Software","Microsoft","USD","XNAS","US",None,None),
    ("US02079K3059","BN0X8D","GOOGL US","Alphabet Inc","equity","large_cap","Communication Services","Internet","Alphabet","USD","XNAS","US",None,None),
    ("US0231351067","2026082","AMZN US","Amazon.com Inc","equity","large_cap","Consumer Discretionary","Internet Retail","Amazon","USD","XNAS","US",None,None),
    ("US67066G1040","B7TL820","NVDA US","NVIDIA Corp","equity","large_cap","Information Technology","Semiconductors","NVIDIA","USD","XNAS","US",None,None),
    ("JP3633400001","6900643","7203 JT","Toyota Motor Corp","equity","large_cap","Consumer Discretionary","Automobiles","Toyota","JPY","XTKS","JP",None,None),
    ("HK0941009539","BP3R340","941 HK","China Mobile Ltd","equity","large_cap","Communication Services","Telecom","China Mobile","HKD","XHKG","CN",None,None),
    ("CH0224397007",None,"SWISS10Y","Swiss Govt Bond 1.5% 2034","fixed_income","govt_bond","Government",None,"Swiss Confed","CHF","XSWX","CH","2034-06-24",1.5),
    ("DE0001102580",None,"DBR 0 02/35","German Bund 0% 2035","fixed_income","govt_bond","Government",None,"Germany","EUR","XETR","DE","2035-02-15",0.0),
    ("US912810TA88",None,"T 3⅞ 08/34","US Treasury 3.875% 2034","fixed_income","govt_bond","Government",None,"US Treasury","USD","XNYS","US","2034-08-15",3.875),
    ("CH0537261858",None,"NESTLE 25","Nestlé 0.75% 2025","fixed_income","corp_bond","Consumer Staples","Food Products","Nestlé","CHF","XSWX","CH","2025-09-15",0.75),
    ("XS2310511717",None,"UBS 28","UBS 1.5% 2028","fixed_income","corp_bond","Financials","Banks","UBS","EUR","XSWX","CH","2028-03-20",1.5),
    ("US594918CC90",None,"MSFT 33","Microsoft 2.525% 2033","fixed_income","corp_bond","Information Technology","Software","Microsoft","USD","XNYS","US","2033-06-01",2.525),
    ("IE00B4L5Y983","B4L5Y98","IWDA LN","iShares MSCI World ETF","etf","global_equity","Diversified",None,"iShares","USD","XLON","IE",None,None),
    ("IE00B5BMR087","B5BMR08","CSPX LN","iShares S&P 500 ETF","etf","us_equity","Diversified",None,"iShares","USD","XLON","IE",None,None),
    ("LU0274208692","B1XFGM","XSMI SW","Xtrackers SMI ETF","etf","swiss_equity","Diversified",None,"Xtrackers","CHF","XSWX","CH",None,None),
    ("IE00B4L5YC18","B4L5YC1","EIMI LN","iShares EM ETF","etf","em_equity","Diversified",None,"iShares","USD","XLON","IE",None,None),
    ("IE00BZ163K21","BZ163K2","VDTA LN","Vanguard USD Treasury","etf","govt_bond","Fixed Income",None,"Vanguard","USD","XLON","IE",None,None),
    ("CH0599123456",None,"STRUC01","Capital Protected Note on SMI","structured_product","capital_protected","Diversified",None,"UBS","CHF","XSWX","CH","2026-12-15",None),
]

COUNTERPARTIES = [
    {"counterparty_id":1,"cp_code":"BRK-GS","cp_name":"Goldman Sachs International","cp_type":"broker","lei":"5493002R23N8JL3WGR95","country_code":"GB"},
    {"counterparty_id":2,"cp_code":"BRK-JPM","cp_name":"J.P. Morgan Securities","cp_type":"broker","lei":"ZBUT11V806EZRVTWT807","country_code":"US"},
    {"counterparty_id":3,"cp_code":"BRK-UBS","cp_name":"UBS Securities","cp_type":"broker","lei":"BFM8T61CT2L1QCEMIK50","country_code":"CH"},
    {"counterparty_id":4,"cp_code":"CST-SIX","cp_name":"SIX SIS AG","cp_type":"custodian","lei":"529900L25DE7A3P4P865","country_code":"CH"},
    {"counterparty_id":5,"cp_code":"CST-CBK","cp_name":"Clearstream Banking","cp_type":"custodian","lei":"549300OL514RA0SBER12","country_code":"LU"},
    {"counterparty_id":6,"cp_code":"CLR-EURX","cp_name":"Eurex Clearing AG","cp_type":"clearing_house","lei":"529900LN3S50JPU47S06","country_code":"DE"},
]

CLIENTS_RAW = [
    # (code, legal_name, short_name, type, segment, risk_profile, risk_rating, dom, tax, mifid, rm, onboard, pep)
    ("CLI-CH-001","Stiftung Müller","Müller Found.","legal_entity","institutional","balanced","low","CH","CH","professional_client","Thomas Weber","2019-03-15",False),
    ("CLI-CH-002","Dr. Elena Brunner","E. Brunner","natural_person","uhnwi","growth","medium","CH","CH","professional_client","Sophie Martin","2018-06-01",False),
    ("CLI-CH-003","Horizon Capital Partners AG","Horizon Cap","legal_entity","institutional","aggressive","medium","CH","CH","professional_client","Thomas Weber","2017-01-20",False),
    ("CLI-CH-004","Famille Dumont Trust","Dumont Trust","trust","uhnwi","conservative","low","CH","CH","professional_client","Sophie Martin","2016-08-10",False),
    ("CLI-CH-005","Marco Bianchi","M. Bianchi","natural_person","hnwi","balanced","low","CH","IT","retail_client","Maria Chen","2020-02-28",False),
    ("CLI-CH-006","TechVentures Zürich GmbH","TechVentures","legal_entity","corporate","growth","medium","CH","CH","professional_client","Thomas Weber","2021-05-12",False),
    ("CLI-CH-007","Anna Schneider","A. Schneider","natural_person","affluent","conservative","low","CH","CH","retail_client","Maria Chen","2022-01-15",False),
    ("CLI-CH-008","Pension Fund Bern","PK Bern","legal_entity","institutional","conservative","low","CH","CH","eligible_counterparty","Thomas Weber","2015-03-01",False),
    ("CLI-DE-009","Katarina Hoffmann","K. Hoffmann","natural_person","hnwi","growth","medium","DE","DE","professional_client","Sophie Martin","2019-11-20",False),
    ("CLI-DE-010","Rhein Industrie Holding","Rhein Holding","legal_entity","corporate","balanced","low","DE","DE","professional_client","Thomas Weber","2018-04-18",False),
    ("CLI-FR-011","Pierre & Marie Lefèvre","Lefèvre","natural_person","uhnwi","aggressive","medium","FR","FR","professional_client","Sophie Martin","2017-09-05",False),
    ("CLI-FR-012","Fondation Arts et Culture","FAC Paris","legal_entity","institutional","balanced","low","FR","FR","professional_client","Sophie Martin","2020-07-22",False),
    ("CLI-UK-013","Sir James Whitfield","J. Whitfield","natural_person","uhnwi","growth","low","GB","GB","professional_client","Thomas Weber","2016-12-01",True),
    ("CLI-UK-014","Albion Partners LLP","Albion","legal_entity","institutional","aggressive","medium","GB","GB","eligible_counterparty","Thomas Weber","2019-02-14",False),
    ("CLI-US-015","Jennifer Chen","J. Chen","natural_person","hnwi","aggressive","medium","US","US","professional_client","Maria Chen","2020-10-05",False),
    ("CLI-US-016","Pacific Endowment Fund","Pacific EF","legal_entity","institutional","balanced","low","US","US","eligible_counterparty","Thomas Weber","2018-08-30",False),
    ("CLI-JP-017","Yamamoto Kenji","K. Yamamoto","natural_person","hnwi","balanced","low","JP","JP","retail_client","Maria Chen","2021-04-10",False),
    ("CLI-SG-018","Temasek Family Office","Temasek FO","legal_entity","uhnwi","growth","medium","SG","SG","professional_client","Sophie Martin","2019-06-20",False),
    ("CLI-RU-019","Dmitri Volkov","D. Volkov","natural_person","hnwi","aggressive","critical","RU","RU","professional_client","Sophie Martin","2017-03-15",True),
    ("CLI-AE-020","Al Rashid Investment Co","Al Rashid","legal_entity","uhnwi","growth","high","AE","AE","professional_client","Thomas Weber","2020-01-08",False),
    ("CLI-CH-021","Reto Gerber","R. Gerber","natural_person","retail","conservative","low","CH","CH","retail_client","Maria Chen","2023-06-01",False),
    ("CLI-CH-022","Zürcher Familienstiftung","ZFS","legal_entity","institutional","balanced","low","CH","CH","professional_client","Thomas Weber","2014-09-12",False),
    ("CLI-IT-023","Conte Alessandro Moretti","A. Moretti","natural_person","uhnwi","balanced","medium","IT","IT","professional_client","Sophie Martin","2018-11-30",False),
    ("CLI-LI-024","Liechtenstein Royal Trust","LRT","trust","uhnwi","conservative","low","LI","LI","professional_client","Sophie Martin","2016-02-20",False),
    ("CLI-BR-025","Fernanda Oliveira","F. Oliveira","natural_person","hnwi","growth","high","BR","BR","retail_client","Maria Chen","2021-08-25",False),
]

# ---------------------------------------------------------------------------
# GENERATE DATAFRAMES
# ---------------------------------------------------------------------------

def gen_instruments():
    exchange_map = {e["mic_code"]: e["exchange_id"] for e in EXCHANGES}
    rows = []
    for i, inst in enumerate(INSTRUMENTS_RAW, 1):
        isin, sedol, ticker, name, ac, sc, sector, ig, issuer, ccy, mic, cor, mat, coupon = inst
        rows.append({
            "instrument_id": i, "isin": isin, "sedol": sedol,
            "bloomberg_ticker": ticker, "instrument_name": name,
            "asset_class": ac, "sub_class": sc, "sector": sector,
            "industry_group": ig, "issuer": issuer, "currency": ccy,
            "exchange_id": exchange_map.get(mic), "country_of_risk": cor,
            "maturity_date": mat, "coupon_rate": coupon,
        })
    return pd.DataFrame(rows)

def gen_clients():
    rows = []
    for i, c in enumerate(CLIENTS_RAW, 1):
        code, legal, short, ctype, seg, rp, rr, dom, tax, mifid, rm, onb, pep = c
        rows.append({
            "client_id": i, "client_code": code, "legal_name": legal,
            "short_name": short, "client_type": ctype, "segment": seg,
            "risk_profile": rp, "risk_rating": rr, "country_domicile": dom,
            "country_tax": tax, "mifid_category": mifid,
            "relationship_manager": rm, "onboarding_date": onb, "is_pep": pep,
        })
    return pd.DataFrame(rows)

def gen_accounts(clients_df):
    account_types = ["trading", "custody", "cash"]
    subtypes = {"trading": ["discretionary", "advisory", "execution_only"],
                "custody": ["discretionary", "advisory"], "cash": [None]}
    rows = []
    acc_id = 0
    for _, cl in clients_df.iterrows():
        n = 3 if cl.segment in ("uhnwi", "institutional") else 2 if cl.segment in ("hnwi", "corporate") else 1
        for a in range(n):
            acc_id += 1
            atype = account_types[a % 3]
            ccy = "CHF" if cl.country_domicile in ("CH", "LI") else "EUR" if cl.country_domicile in ("DE", "FR", "IT") else "USD"
            rows.append({
                "account_id": acc_id,
                "account_number": f"{cl.country_domicile}{93+acc_id:02d}{acc_id:04d}{random.randint(1000,9999):04d}",
                "client_id": cl.client_id, "account_type": atype,
                "account_subtype": random.choice(subtypes[atype]),
                "base_currency": ccy, "custodian_id": random.choice([4, 5]),
                "opened_date": cl.onboarding_date, "status": "active",
            })
    return pd.DataFrame(rows)

def gen_market_prices(instruments_df):
    today = date(2026, 2, 20)
    trading_days = [today - timedelta(days=d) for d in range(45) if (today - timedelta(days=d)).weekday() < 5][:30]
    trading_days.reverse()
    base = {}
    for _, inst in instruments_df.iterrows():
        bp = {"equity": random.uniform(20, 800), "fixed_income": random.uniform(95, 105),
              "etf": random.uniform(30, 500)}.get(inst.asset_class, random.uniform(90, 110))
        base[inst.instrument_id] = bp
    rows = []
    pid = 0
    for inst_id, bp in base.items():
        p = bp
        for td in trading_days:
            pid += 1
            p *= (1 + random.gauss(0.0002, 0.015))
            rows.append({
                "price_id": pid, "instrument_id": inst_id, "price_date": str(td),
                "open_price": round(p*(1+random.gauss(0,0.005)), 4),
                "high_price": round(p*(1+abs(random.gauss(0,0.008))), 4),
                "low_price": round(p*(1-abs(random.gauss(0,0.008))), 4),
                "close_price": round(p, 4), "adj_close": round(p, 4),
                "volume": random.randint(10000, 5000000),
                "source": random.choice(["bloomberg", "reuters", "exchange"]),
            })
    return pd.DataFrame(rows), base

def gen_orders_and_executions(accounts_df, instruments_df, base_prices):
    today = date(2026, 2, 20)
    trading_days = [today - timedelta(days=d) for d in range(30) if (today - timedelta(days=d)).weekday() < 5][:20]
    trading_days.reverse()
    trading_accts = accounts_df[accounts_df.account_type.isin(["trading", "custody"])]
    orders, execs = [], []
    oid, eid = 0, 0
    for td in trading_days:
        for _ in range(random.randint(15, 35)):
            oid += 1
            acct = trading_accts.sample(1, random_state=random.randint(0, 2**31)).iloc[0]
            inst = instruments_df.sample(1, random_state=random.randint(0, 2**31)).iloc[0]
            bp = base_prices.get(inst.instrument_id, 100)
            side = random.choice(["BUY","BUY","BUY","SELL"])
            otype = random.choices(["MARKET","LIMIT","LIMIT","STOP_LIMIT","VWAP"],[3,4,3,1,1])[0]
            qty = random.choice([10,25,50,100,200,500,1000,2500])
            comp = "passed"
            comp_note = None
            if acct.client_id == 19:
                comp = random.choice(["passed","flagged","flagged","rejected"])
                if comp != "passed": comp_note = "Sanctions screening — RU domicile"
            status = "REJECTED" if comp == "rejected" else random.choices(
                ["FILLED","FILLED","FILLED","PARTIALLY_FILLED","CANCELLED","REJECTED"],[5,4,3,1,1,1])[0]
            fq = qty if status == "FILLED" else (int(qty*random.uniform(0.3,0.8)) if status == "PARTIALLY_FILLED" else 0)
            avg_p = round(bp*random.uniform(0.995,1.005),4) if fq > 0 else None
            orders.append({
                "order_id": oid, "cl_ord_id": f"ORD-{oid:06d}",
                "account_id": acct.account_id, "instrument_id": inst.instrument_id,
                "side": side, "order_type": otype, "quantity": qty,
                "limit_price": round(bp*random.uniform(0.97,1.03),4) if otype != "MARKET" else None,
                "currency": inst.currency, "status": status,
                "filled_qty": fq, "avg_fill_price": avg_p,
                "remaining_qty": qty - fq, "broker_id": random.choice([1,2,3]),
                "trading_desk": {"equity":"equities","fixed_income":"fixed_income","etf":"equities"}.get(inst.asset_class,"equities"),
                "order_source": random.choices(["manual","algo","dma","api"],[4,3,2,1])[0],
                "order_date": str(td), "compliance_check": comp, "compliance_note": comp_note,
            })
            if fq > 0:
                n_fills = 1 if fq < 200 else random.randint(1,3)
                rem = fq
                for f in range(n_fills):
                    eid += 1
                    fill_q = rem if f == n_fills-1 else max(1, int(rem*random.uniform(0.3,0.7)))
                    rem -= fill_q
                    fp = round(bp*random.uniform(0.995,1.005), 4)
                    execs.append({
                        "execution_id": eid, "exec_id": f"EXE-{eid:06d}",
                        "order_id": oid, "fill_qty": fill_q, "fill_price": fp,
                        "commission": round(fill_q*fp*random.uniform(0.0001,0.0008),2),
                        "commission_ccy": inst.currency,
                        "fees": round(fill_q*fp*random.uniform(0.00005,0.0002),2),
                        "venue": random.choice(["XSWX","XETR","XLON","XNAS","OTC"]),
                        "settlement_date": str(td + timedelta(days=2)),
                        "executed_at": f"{td} {random.randint(9,16)}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
                    })
    return pd.DataFrame(orders), pd.DataFrame(execs)

def gen_positions(accounts_df, instruments_df, base_prices):
    rows = []
    pid = 0
    for _, acct in accounts_df[accounts_df.account_type != "cash"].iterrows():
        for _, inst in instruments_df.sample(min(random.randint(3,12), len(instruments_df)), random_state=random.randint(0, 2**31)).iterrows():
            pid += 1
            bp = base_prices.get(inst.instrument_id, 100)
            qty = random.choice([10,25,50,100,200,500,1000,2500,5000])
            avg_cost = round(bp*random.uniform(0.85,1.05),4)
            mkt = round(bp*random.uniform(0.98,1.02),4)
            rows.append({
                "position_id": pid, "account_id": acct.account_id,
                "instrument_id": inst.instrument_id, "quantity": qty,
                "avg_cost": avg_cost, "cost_basis": round(qty*avg_cost,2),
                "market_price": mkt, "market_value": round(qty*mkt,2),
                "unrealized_pnl": round(qty*(mkt-avg_cost),2),
                "realized_pnl": round(random.uniform(-5000,15000),2),
                "currency": inst.currency, "valuation_date": "2026-02-20",
                "weight_pct": round(random.uniform(1,25),4),
            })
    return pd.DataFrame(rows)

def gen_cash_balances(accounts_df):
    rows = []
    for i, (_, acct) in enumerate(accounts_df.iterrows(), 1):
        bal = round(random.uniform(50000, 5000000), 2)
        avail = round(bal * random.uniform(0.7, 0.95), 2)
        rows.append({
            "balance_id": i, "account_id": acct.account_id,
            "currency": acct.base_currency, "balance": bal,
            "available": avail, "reserved": round(bal - avail, 2),
            "balance_date": "2026-02-20",
        })
    return pd.DataFrame(rows)

def gen_risk_limits(clients_df):
    rows = []
    lid = 0
    for _, cl in clients_df.iterrows():
        for lt in random.sample(["gross_exposure","net_exposure","single_name","sector","leverage"], random.randint(2,4)):
            lid += 1
            lv = round(random.uniform(500000, 50000000), 2)
            usage = round(lv * random.uniform(0.1, 0.95), 2)
            util = round(usage / lv * 100, 2)
            rows.append({
                "limit_id": lid, "client_id": cl.client_id, "limit_type": lt,
                "limit_value": lv, "current_usage": usage, "utilization_pct": util,
                "breach_action": "alert" if util < 80 else ("block_new_orders" if util < 95 else "force_reduce"),
                "approved_by": "Risk Committee",
            })
    return pd.DataFrame(rows)

def gen_risk_metrics(clients_df):
    today = date(2026, 2, 20)
    days = [today - timedelta(days=d) for d in range(7) if (today - timedelta(days=d)).weekday() < 5][:5]
    rows = []
    mid = 0
    for _, cl in clients_df.iterrows():
        for d in days:
            mid += 1
            v95 = round(random.uniform(5000,500000),2)
            rows.append({
                "metric_id": mid, "client_id": cl.client_id, "metric_date": str(d),
                "var_1d_95": v95, "var_1d_99": round(v95*random.uniform(1.2,1.8),2),
                "cvar_1d_95": round(v95*random.uniform(1.3,2.0),2),
                "gross_exposure": round(random.uniform(500000,50000000),2),
                "net_exposure": round(random.uniform(200000,30000000),2),
                "leverage_ratio": round(random.uniform(0.5,3.0),4),
                "sharpe_30d": round(random.uniform(-0.5,2.5),4),
                "volatility_30d": round(random.uniform(0.05,0.40),4),
                "max_drawdown": round(random.uniform(-0.30,-0.01),4),
                "beta_to_market": round(random.uniform(0.5,1.8),4),
            })
    return pd.DataFrame(rows)

def gen_compliance_checks(orders_df):
    rows = []
    cid = 0
    for _, o in orders_df.iterrows():
        for _ in range(random.randint(1, 3)):
            cid += 1
            ct = random.choice(["pre_trade_limit","suitability","concentration","sanctions","insider_list"])
            result, breach = "passed", None
            if o["compliance_check"] == "rejected" and ct == "sanctions":
                result, breach = random.choice(["warning","breach"]), "Sanctions screening flag"
            elif random.random() < 0.05:
                result, breach = "warning", f"Soft breach on {ct}"
            rows.append({
                "check_id": cid, "order_id": o.order_id, "check_type": ct,
                "check_result": result, "rule_name": f"{ct}_rule", "breach_detail": breach,
            })
    return pd.DataFrame(rows)

def gen_aml_alerts():
    rows = []
    aid = 0
    descs = {"unusual_volume":"Abnormal trading volume — 3x 30-day average",
             "sanctions_hit":"Name match against OFAC/EU sanctions list",
             "pep_activity":"Large transfer by PEP-flagged client",
             "structuring":"Multiple sub-threshold transfers detected",
             "geographic_risk":"Transfer to/from high-risk jurisdiction"}
    for ci in [19, 20, 13, 25, 11]:
        for _ in range(random.randint(1, 4)):
            aid += 1
            at = random.choice(["sanctions_hit","pep_activity","geographic_risk"]) if ci == 19 else random.choice(list(descs.keys()))
            sev = random.choice(["high","critical"]) if ci == 19 else random.choice(["low","medium","high","critical"])
            rows.append({
                "alert_id": aid, "client_id": ci, "alert_type": at,
                "severity": sev, "description": descs[at],
                "status": random.choice(["open","investigating","escalated","closed_false_positive"]),
                "assigned_to": "Sophie Martin" if sev in ("high","critical") else "Maria Chen",
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("  Generating Parquet data lake → MinIO (S3)")
    print("="*60)

    s3 = get_s3_client()
    bucket = config.S3_BUCKET
    ensure_bucket(s3, bucket)
    print(f"\n📦 Bucket: {bucket}")

    # Generate all dataframes
    exchanges_df = pd.DataFrame(EXCHANGES)
    instruments_df = gen_instruments()
    counterparties_df = pd.DataFrame(COUNTERPARTIES)
    clients_df = gen_clients()
    accounts_df = gen_accounts(clients_df)
    prices_df, base_prices = gen_market_prices(instruments_df)
    orders_df, executions_df = gen_orders_and_executions(accounts_df, instruments_df, base_prices)
    positions_df = gen_positions(accounts_df, instruments_df, base_prices)
    cash_df = gen_cash_balances(accounts_df)
    risk_limits_df = gen_risk_limits(clients_df)
    risk_metrics_df = gen_risk_metrics(clients_df)
    compliance_df = gen_compliance_checks(orders_df)
    aml_df = gen_aml_alerts()

    print(f"\n📤 Uploading Parquet files...")
    tables = {
        "exchanges": exchanges_df,
        "instruments": instruments_df,
        "counterparties": counterparties_df,
        "clients": clients_df,
        "accounts": accounts_df,
        "market_prices": prices_df,
        "orders": orders_df,
        "executions": executions_df,
        "positions": positions_df,
        "cash_balances": cash_df,
        "risk_limits": risk_limits_df,
        "risk_metrics": risk_metrics_df,
        "compliance_checks": compliance_df,
        "aml_alerts": aml_df,
    }

    for name, df in tables.items():
        upload_parquet(s3, bucket, f"{name}.parquet", df)

    total_rows = sum(len(df) for df in tables.values())
    print(f"\n✅ Data lake ready: {len(tables)} tables, {total_rows:,} total rows")
    print(f"   MinIO console: http://localhost:9001 (minio_admin / minio_2026!)")
    print("="*60)


if __name__ == "__main__":
    main()
