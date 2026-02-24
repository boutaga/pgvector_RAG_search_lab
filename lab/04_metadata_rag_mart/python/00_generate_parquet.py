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
import json
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
                "account_number": f"{cl.country_domicile}{(93+acc_id) % 100:02d}{acc_id:04d}{random.randint(1000,9999):04d}",
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
            acct = trading_accts.sample(1, random_state=random.randint(0, 2**31 - 1)).iloc[0]
            inst = instruments_df.sample(1, random_state=random.randint(0, 2**31 - 1)).iloc[0]
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
                    if f == n_fills - 1 or rem <= 1:
                        fill_q = max(1, rem)
                    else:
                        fill_q = max(1, int(rem * random.uniform(0.3, 0.7)))
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
        for _, inst in instruments_df.sample(min(random.randint(3,12), len(instruments_df)), random_state=random.randint(0, 2**31 - 1)).iterrows():
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

def gen_financial_news(instruments_df):
    """Generate ~200 synthetic financial news articles with Swiss/EU focus."""
    today = date(2026, 2, 20)
    sources = ["Bloomberg", "Reuters", "AWP", "FT", "NZZ"]
    categories = ["earnings", "regulatory", "market_move", "m_and_a", "macro", "downgrade", "upgrade"]
    sentiments = ["positive", "negative", "neutral"]
    regions = ["CH", "EU", "US", "APAC", "Global"]
    impact_levels = ["low", "medium", "high", "critical"]
    sectors = sorted(set(i[6] for i in INSTRUMENTS_RAW if i[6] and i[6] != "Government"))
    isin_list = [i[0] for i in INSTRUMENTS_RAW]

    # Pre-built high-impact news that drive broker signals
    signal_news = [
        {"headline": "Nestlé Q4 revenue misses estimates; organic growth at 2.1% vs 3.4% expected",
         "summary": "Nestlé SA reported Q4 2025 revenue of CHF 23.1bn, missing consensus by 2.3%. Organic growth of 2.1% fell short of the 3.4% expected. Management cited consumer weakness in Europe and North America. Stock dropped 4.2% in early Zurich trading.",
         "source": "Bloomberg", "category": "earnings", "sentiment": "negative", "sentiment_score": -0.78,
         "related_isins": ["CH0012005267", "CH0537261858"], "related_sectors": ["Consumer Staples"],
         "region": "CH", "impact_level": "critical", "days_ago": 2},
        {"headline": "FINMA orders UBS to increase capital buffer by CHF 25bn",
         "summary": "Swiss regulator FINMA has directed UBS Group to raise additional capital buffers following its systemic risk review. The CHF 25bn requirement must be met within 18 months. UBS shares fell 3.1% on the announcement.",
         "source": "AWP", "category": "regulatory", "sentiment": "negative", "sentiment_score": -0.85,
         "related_isins": ["CH0244767585", "XS2310511717"], "related_sectors": ["Financials"],
         "region": "CH", "impact_level": "critical", "days_ago": 3},
        {"headline": "ASML warns of export restrictions impact on 2027 guidance",
         "summary": "ASML Holding NV warned that new US-led export restrictions to China could reduce 2027 revenue by EUR 2-3bn. The Dutch chipmaker maintained its 2026 outlook but flagged growing geopolitical headwinds for its EUV lithography business.",
         "source": "Reuters", "category": "downgrade", "sentiment": "negative", "sentiment_score": -0.62,
         "related_isins": ["NL0010273215"], "related_sectors": ["Information Technology"],
         "region": "EU", "impact_level": "high", "days_ago": 5},
        {"headline": "Roche receives FDA breakthrough designation for new Alzheimer's drug",
         "summary": "Roche's experimental Alzheimer's treatment gantenerumab-next received FDA Breakthrough Therapy designation. Phase 3 data showed 35% slowing of cognitive decline. Shares rose 2.8% in pre-market. Analysts see CHF 5bn+ peak sales potential.",
         "source": "Bloomberg", "category": "upgrade", "sentiment": "positive", "sentiment_score": 0.82,
         "related_isins": ["CH0012032048"], "related_sectors": ["Health Care"],
         "region": "CH", "impact_level": "high", "days_ago": 1},
        {"headline": "SNB holds policy rate at 1.50%, signals no near-term cuts",
         "summary": "The Swiss National Bank maintained its policy rate at 1.50% as expected. President Schlegel indicated no urgency to ease, citing resilient domestic inflation. CHF strengthened 0.3% against EUR on the hawkish hold.",
         "source": "AWP", "category": "macro", "sentiment": "neutral", "sentiment_score": 0.05,
         "related_isins": [], "related_sectors": ["Financials"],
         "region": "CH", "impact_level": "medium", "days_ago": 4},
        {"headline": "Goldman Sachs downgrades NVIDIA to Sell, cuts target to $95",
         "summary": "Goldman analyst Toshiya Hari downgraded NVIDIA from Buy to Sell with a $95 price target, citing peak AI capex cycle and rising competition from custom silicon. The downgrade follows a 180% YTD rally. Hari sees 25% downside risk.",
         "source": "Bloomberg", "category": "downgrade", "sentiment": "negative", "sentiment_score": -0.71,
         "related_isins": ["US67066G1040"], "related_sectors": ["Information Technology"],
         "region": "US", "impact_level": "high", "days_ago": 6},
        {"headline": "Swiss Re raises 2026 dividend by 15% after record combined ratio",
         "summary": "Swiss Re AG announced a 15% dividend increase to CHF 7.35 per share after achieving a 92.1% combined ratio in FY 2025. CEO Mumenthaler highlighted strong nat cat reserve releases and improved pricing discipline.",
         "source": "NZZ", "category": "earnings", "sentiment": "positive", "sentiment_score": 0.65,
         "related_isins": ["CH1175448666"], "related_sectors": ["Financials"],
         "region": "CH", "impact_level": "medium", "days_ago": 7},
        {"headline": "Zurich Insurance completes EUR 2.1bn acquisition of Italian insurer",
         "summary": "Zurich Insurance Group closed its acquisition of Assicurazioni Generali's personal lines portfolio for EUR 2.1bn. The deal expands Zurich's European retail presence by 3 million policyholders.",
         "source": "FT", "category": "m_and_a", "sentiment": "positive", "sentiment_score": 0.45,
         "related_isins": ["CH0210483332"], "related_sectors": ["Financials"],
         "region": "EU", "impact_level": "medium", "days_ago": 8},
        {"headline": "SAP Q4 cloud revenue beats estimates by 8%, raises FY 2026 guidance",
         "summary": "SAP SE reported Q4 cloud revenue of EUR 4.7bn, 8% above consensus. The German software giant raised its FY 2026 cloud revenue guidance to EUR 21-22bn. Operating profit margin expanded 150bp to 34.5%.",
         "source": "Reuters", "category": "earnings", "sentiment": "positive", "sentiment_score": 0.73,
         "related_isins": ["DE0007164600"], "related_sectors": ["Information Technology"],
         "region": "EU", "impact_level": "medium", "days_ago": 10},
        {"headline": "JP Morgan cuts ASML target price by 15%, maintains Hold",
         "summary": "JP Morgan analyst Sandeep Deshpande reduced ASML's target to EUR 580 from EUR 680, citing slower-than-expected EUV adoption in China and inventory build at key customers. Maintained Hold rating.",
         "source": "Bloomberg", "category": "downgrade", "sentiment": "negative", "sentiment_score": -0.48,
         "related_isins": ["NL0010273215"], "related_sectors": ["Information Technology"],
         "region": "EU", "impact_level": "medium", "days_ago": 4},
    ]

    rows = []
    nid = 0

    # Add signal news first
    for sn in signal_news:
        nid += 1
        pub_dt = datetime(today.year, today.month, today.day, random.randint(6, 20), random.randint(0, 59)) - timedelta(days=sn["days_ago"])
        rows.append({
            "news_id": nid, "published_at": pub_dt.isoformat(),
            "source": sn["source"], "headline": sn["headline"], "summary": sn["summary"],
            "category": sn["category"], "sentiment": sn["sentiment"],
            "sentiment_score": sn["sentiment_score"],
            "related_isins": sn["related_isins"], "related_sectors": sn["related_sectors"],
            "region": sn["region"], "impact_level": sn["impact_level"],
        })

    # Generate ~190 more random news
    headlines_templates = [
        ("{company} reports {direction} quarterly results, {metric} {beats_misses} consensus",
         lambda: {"company": random.choice([i[3] for i in INSTRUMENTS_RAW[:22]]),
                  "direction": random.choice(["strong", "mixed", "weak"]),
                  "metric": random.choice(["revenue", "EPS", "EBITDA"]),
                  "beats_misses": random.choice(["beats", "misses", "in line with"])}),
        ("Analyst {action} {company} citing {reason}",
         lambda: {"action": random.choice(["upgrades", "downgrades", "initiates coverage on", "maintains Outperform on"]),
                  "company": random.choice([i[3] for i in INSTRUMENTS_RAW[:22]]),
                  "reason": random.choice(["valuation concerns", "strong fundamentals", "sector rotation", "margin expansion", "regulatory headwinds"])}),
        ("{sector} sector {direction} amid {catalyst}",
         lambda: {"sector": random.choice(sectors),
                  "direction": random.choice(["rallies", "sells off", "underperforms", "outperforms"]),
                  "catalyst": random.choice(["rate expectations", "earnings season", "geopolitical tensions", "trade deal progress", "central bank signals"])}),
        ("{region} markets {direction} on {catalyst}",
         lambda: {"region": random.choice(["European", "Swiss", "US", "Asian", "Global"]),
                  "direction": random.choice(["rise", "fall", "mixed"]),
                  "catalyst": random.choice(["ECB signals", "Fed minutes", "China data", "oil price moves", "tech earnings"])}),
    ]

    for _ in range(190):
        nid += 1
        template, gen_params = random.choice(headlines_templates)
        params = gen_params()
        headline = template.format(**params)
        pub_dt = datetime(today.year, today.month, today.day, random.randint(6, 20), random.randint(0, 59)) - timedelta(days=random.randint(0, 29))
        cat = random.choice(categories)
        sent = random.choices(sentiments, weights=[25, 40, 35])[0]
        score = {"positive": random.uniform(0.1, 0.9), "negative": random.uniform(-0.9, -0.1),
                 "neutral": random.uniform(-0.15, 0.15)}[sent]
        n_isins = random.randint(0, 3)
        rel_isins = random.sample(isin_list[:22], min(n_isins, 22)) if n_isins > 0 else []
        n_sectors = random.randint(1, 2)
        rel_sectors = random.sample(sectors, min(n_sectors, len(sectors)))
        imp = random.choices(impact_levels, weights=[30, 40, 20, 10])[0]

        rows.append({
            "news_id": nid, "published_at": pub_dt.isoformat(),
            "source": random.choice(sources), "headline": headline,
            "summary": f"Summary for: {headline}. Market impact assessed as {imp}.",
            "category": cat, "sentiment": sent,
            "sentiment_score": round(score, 2),
            "related_isins": rel_isins, "related_sectors": rel_sectors,
            "region": random.choice(regions), "impact_level": imp,
        })

    df = pd.DataFrame(rows)
    # Convert list columns to string representation for Parquet
    df["related_isins"] = df["related_isins"].apply(json.dumps)
    df["related_sectors"] = df["related_sectors"].apply(json.dumps)
    return df


def gen_public_filings():
    """Generate ~80 synthetic SIX/FINMA-style filings."""
    today = date(2026, 2, 20)
    filing_types = ["ad_hoc_disclosure", "annual_report", "interim_report", "prospectus", "share_buyback", "mgmt_transaction"]
    regulators = ["FINMA", "SIX", "SEC", "BaFin"]
    financial_impacts = ["positive", "negative", "neutral", "material"]

    # Pre-built signal filings
    signal_filings = [
        {"filing_type": "ad_hoc_disclosure", "issuer": "UBS Group AG", "issuer_isin": "CH0244767585",
         "regulator": "FINMA", "title": "FINMA Capital Buffer Requirement — Ad Hoc Disclosure",
         "summary": "UBS Group AG is required to increase its regulatory capital buffer by CHF 25bn within 18 months per FINMA directive. The bank plans to meet the requirement through retained earnings and conditional capital instruments. No dividend impact expected for FY 2025.",
         "financial_impact": "material", "key_figures": '{"capital_requirement": "CHF 25bn", "timeline": "18 months"}',
         "sector": "Financials", "days_ago": 3},
        {"filing_type": "annual_report", "issuer": "Nestlé SA", "issuer_isin": "CH0012005267",
         "regulator": "SIX", "title": "Nestlé SA — FY 2025 Annual Report",
         "summary": "Nestlé reports FY 2025 revenue of CHF 91.2bn (-1.8% organic). Net income CHF 10.5bn. Board proposes dividend of CHF 3.00 (unchanged). Guidance for 2026: 3-4% organic growth target.",
         "financial_impact": "negative", "key_figures": '{"revenue": "CHF 91.2bn", "net_income": "CHF 10.5bn", "dividend": "CHF 3.00"}',
         "sector": "Consumer Staples", "days_ago": 5},
        {"filing_type": "share_buyback", "issuer": "Roche Holding AG", "issuer_isin": "CH0012032048",
         "regulator": "SIX", "title": "Roche announces CHF 2bn share buyback program",
         "summary": "Roche launches a CHF 2bn share buyback program over 12 months, reflecting confidence in cash flow generation. The buyback represents approximately 1% of market cap.",
         "financial_impact": "positive", "key_figures": '{"buyback_amount": "CHF 2bn", "duration": "12 months"}',
         "sector": "Health Care", "days_ago": 10},
    ]

    rows = []
    fid = 0

    for sf in signal_filings:
        fid += 1
        rows.append({
            "filing_id": fid, "filing_date": str(today - timedelta(days=sf["days_ago"])),
            "filing_type": sf["filing_type"], "issuer": sf["issuer"],
            "issuer_isin": sf["issuer_isin"], "regulator": sf["regulator"],
            "title": sf["title"], "summary": sf["summary"],
            "financial_impact": sf["financial_impact"], "key_figures": sf["key_figures"],
            "sector": sf["sector"],
            "filing_url": f"https://www.six-exchange-regulation.com/en/home/publications/significant-shareholders.html#{fid}",
        })

    # Generate ~77 more
    issuers = [(i[3], i[0], i[6]) for i in INSTRUMENTS_RAW if i[4] == "equity"]
    for _ in range(77):
        fid += 1
        issuer_name, issuer_isin, sector = random.choice(issuers)
        ft = random.choice(filing_types)
        fi = random.choice(financial_impacts)
        reg = "FINMA" if random.random() < 0.3 else random.choice(regulators)
        title_templates = {
            "ad_hoc_disclosure": f"{issuer_name} — Ad Hoc Disclosure: {random.choice(['Management Change', 'Contract Award', 'Legal Proceeding', 'Strategic Review'])}",
            "annual_report": f"{issuer_name} — FY 2025 Annual Report",
            "interim_report": f"{issuer_name} — {random.choice(['H1', 'Q3'])} 2025 Interim Report",
            "prospectus": f"{issuer_name} — {random.choice(['Bond', 'Equity', 'Convertible'])} Prospectus",
            "share_buyback": f"{issuer_name} — Share Buyback {random.choice(['Announcement', 'Update', 'Completion'])}",
            "mgmt_transaction": f"{issuer_name} — {random.choice(['CEO', 'CFO', 'Board Member'])} {random.choice(['Purchase', 'Sale'])} of Shares",
        }
        rows.append({
            "filing_id": fid, "filing_date": str(today - timedelta(days=random.randint(1, 90))),
            "filing_type": ft, "issuer": issuer_name, "issuer_isin": issuer_isin,
            "regulator": reg, "title": title_templates[ft],
            "summary": f"Regulatory filing by {issuer_name}. Type: {ft}. Impact assessed as {fi}.",
            "financial_impact": fi, "key_figures": "{}",
            "sector": sector or "Diversified",
            "filing_url": f"https://www.six-exchange-regulation.com/en/home/publications/{fid}",
        })

    return pd.DataFrame(rows)


def gen_analyst_recommendations(instruments_df):
    """Generate ~150 synthetic sell-side analyst recommendations."""
    today = date(2026, 2, 20)
    firms = ["Goldman Sachs", "UBS", "JP Morgan", "Vontobel", "ZKB", "Barclays", "Morgan Stanley", "BofA Securities"]
    analyst_names = ["Toshiya Hari", "Sandeep Deshpande", "Michael Levin", "Anna Schmidt", "Lukas Meier",
                     "Sarah Chen", "Pierre Dubois", "James Whitworth", "Elena Rossi", "Martin Huber",
                     "Lisa Wang", "Thomas Keller", "Rajesh Patel", "Maria Fischer", "David Brown"]
    recommendations = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
    rec_weights = [10, 30, 35, 18, 7]

    # Signal recommendations (drive broker alerts)
    signal_recs = [
        {"firm": "Goldman Sachs", "analyst": "Toshiya Hari", "isin": "US67066G1040", "name": "NVIDIA Corp",
         "recommendation": "sell", "prev_recommendation": "buy", "target_price": 95.0, "current_price": 127.50,
         "rationale": "Downgrade from Buy to Sell. Peak AI capex cycle, rising competition from custom silicon (Google TPU, Amazon Trainium). See 25% downside. Valuation at 45x forward P/E unsustainable.",
         "sector": "Information Technology", "risk_factors": '["AI capex peak", "Competition", "Valuation"]', "days_ago": 6},
        {"firm": "JP Morgan", "analyst": "Sandeep Deshpande", "isin": "NL0010273215", "name": "ASML Holding NV",
         "recommendation": "hold", "prev_recommendation": "hold", "target_price": 580.0, "current_price": 665.0,
         "rationale": "Maintain Hold, cut target to EUR 580 from EUR 680. China export restrictions reduce 2027 revenue visibility by EUR 2-3bn. EUV adoption slower than modeled. Wait for better entry point.",
         "sector": "Information Technology", "risk_factors": '["Export controls", "China exposure", "Order delays"]', "days_ago": 4},
        {"firm": "Vontobel", "analyst": "Lukas Meier", "isin": "CH0012005267", "name": "Nestlé SA",
         "recommendation": "hold", "prev_recommendation": "buy", "target_price": 82.0, "current_price": 78.50,
         "rationale": "Downgrade from Buy to Hold after Q4 miss. Organic growth disappointing at 2.1%. European consumer weakness persists. Need to see evidence of portfolio reshaping before turning constructive.",
         "sector": "Consumer Staples", "risk_factors": '["Consumer weakness", "Pricing power erosion", "Portfolio restructuring"]', "days_ago": 2},
        {"firm": "UBS", "analyst": "Anna Schmidt", "isin": "CH0012032048", "name": "Roche Holding AG",
         "recommendation": "buy", "prev_recommendation": "hold", "target_price": 310.0, "current_price": 262.0,
         "rationale": "Upgrade to Buy on FDA breakthrough designation for gantenerumab-next. CHF 5bn+ peak sales potential. CHF 2bn buyback provides floor. Risk/reward compelling at 12x forward P/E.",
         "sector": "Health Care", "risk_factors": '["Phase 3 execution", "Pricing pressure", "Biosimilar competition"]', "days_ago": 1},
        {"firm": "ZKB", "analyst": "Martin Huber", "isin": "CH0244767585", "name": "UBS Group AG",
         "recommendation": "hold", "prev_recommendation": "buy", "target_price": 26.0, "current_price": 24.80,
         "rationale": "Downgrade to Hold following FINMA capital buffer requirement. CHF 25bn additional capital reduces buyback capacity. Regulatory overhang likely to persist for 12-18 months.",
         "sector": "Financials", "risk_factors": '["Regulatory capital", "Buyback reduction", "Integration costs"]', "days_ago": 3},
    ]

    rows = []
    rid = 0

    for sr in signal_recs:
        rid += 1
        upside = round((sr["target_price"] / sr["current_price"] - 1) * 100, 1)
        rows.append({
            "recommendation_id": rid,
            "analyst_firm": sr["firm"], "analyst_name": sr["analyst"],
            "published_date": str(today - timedelta(days=sr["days_ago"])),
            "instrument_isin": sr["isin"], "instrument_name": sr["name"],
            "recommendation": sr["recommendation"], "prev_recommendation": sr["prev_recommendation"],
            "target_price": sr["target_price"], "current_price": sr["current_price"],
            "upside_pct": upside, "rationale": sr["rationale"],
            "sector": sr["sector"], "risk_factors": sr["risk_factors"],
        })

    # Generate ~145 more
    equity_instruments = [(i[0], i[3], i[6]) for i in INSTRUMENTS_RAW if i[4] == "equity"]
    etf_instruments = [(i[0], i[3], "Diversified") for i in INSTRUMENTS_RAW if i[4] == "etf"]
    all_rec_instruments = equity_instruments + etf_instruments[:3]

    for _ in range(145):
        rid += 1
        isin, name, sector = random.choice(all_rec_instruments)
        rec = random.choices(recommendations, weights=rec_weights)[0]
        prev_rec = random.choices(recommendations, weights=rec_weights)[0]
        curr_price = round(random.uniform(15, 900), 2)
        direction = 1 if rec in ("strong_buy", "buy") else (-1 if rec in ("sell", "strong_sell") else 0)
        target_price = round(curr_price * (1 + direction * random.uniform(0.05, 0.35)), 2)
        upside = round((target_price / curr_price - 1) * 100, 1) if curr_price > 0 else 0.0
        risk_list = random.sample(["Valuation", "Competition", "Macro", "Regulation", "Execution",
                                   "Currency", "Margin pressure", "Demand slowdown"], random.randint(1, 3))
        rows.append({
            "recommendation_id": rid,
            "analyst_firm": random.choice(firms), "analyst_name": random.choice(analyst_names),
            "published_date": str(today - timedelta(days=random.randint(0, 60))),
            "instrument_isin": isin, "instrument_name": name,
            "recommendation": rec, "prev_recommendation": prev_rec,
            "target_price": target_price, "current_price": curr_price,
            "upside_pct": upside,
            "rationale": f"{rec.replace('_', ' ').title()} on {name}. Key driver: {random.choice(['margin expansion', 'revenue growth', 'valuation', 'market share', 'cost cutting'])}.",
            "sector": sector or "Diversified",
            "risk_factors": json.dumps(risk_list),
        })

    return pd.DataFrame(rows)


def gen_financial_reports():
    """Generate ~60 synthetic earnings/financial reports."""
    today = date(2026, 2, 20)
    report_types = ["quarterly_earnings", "annual_results", "profit_warning", "guidance_update", "dividend_announcement"]
    guidances = ["raised", "maintained", "lowered", "withdrawn"]
    periods = ["Q3 2025", "Q4 2025", "FY 2025", "H1 2025"]

    # Signal reports
    signal_reports = [
        {"company": "Nestlé SA", "isin": "CH0012005267", "report_type": "quarterly_earnings",
         "period": "Q4 2025", "revenue": 23100.0, "revenue_currency": "CHF",
         "revenue_surprise_pct": -2.3, "eps": 4.15, "eps_surprise_pct": -1.8,
         "dividend_per_share": 3.00, "guidance": "maintained",
         "key_takeaway": "Q4 revenue missed consensus by 2.3%. Organic growth slowed to 2.1% vs 3.4% expected. European consumer weakness cited. Guidance maintained but market skeptical.",
         "sector": "Consumer Staples", "days_ago": 2},
        {"company": "Roche Holding AG", "isin": "CH0012032048", "report_type": "quarterly_earnings",
         "period": "Q4 2025", "revenue": 16400.0, "revenue_currency": "CHF",
         "revenue_surprise_pct": 1.5, "eps": 18.20, "eps_surprise_pct": 2.1,
         "dividend_per_share": 9.60, "guidance": "raised",
         "key_takeaway": "Q4 beat on Pharma strength. FDA breakthrough for Alzheimer's drug a catalyst. CHF 2bn buyback announced. Guidance raised for FY 2026 to mid-single digit growth.",
         "sector": "Health Care", "days_ago": 8},
        {"company": "UBS Group AG", "isin": "CH0244767585", "report_type": "annual_results",
         "period": "FY 2025", "revenue": 39800.0, "revenue_currency": "USD",
         "revenue_surprise_pct": 0.8, "eps": 1.92, "eps_surprise_pct": -0.5,
         "dividend_per_share": 0.70, "guidance": "maintained",
         "key_takeaway": "FY 2025 results broadly in line. Integration of Credit Suisse on track with CHF 13bn cost savings achieved. FINMA capital buffer requirement creates near-term uncertainty.",
         "sector": "Financials", "days_ago": 5},
        {"company": "NVIDIA Corp", "isin": "US67066G1040", "report_type": "quarterly_earnings",
         "period": "Q4 2025", "revenue": 38500.0, "revenue_currency": "USD",
         "revenue_surprise_pct": 3.2, "eps": 0.89, "eps_surprise_pct": 4.1,
         "dividend_per_share": 0.04, "guidance": "raised",
         "key_takeaway": "Q4 beat driven by data center. Revenue up 55% YoY. However, gross margin dipped 100bp. Guidance raised but Goldman downgrade raised peak-cycle concerns.",
         "sector": "Information Technology", "days_ago": 10},
    ]

    rows = []
    rid = 0

    for sr in signal_reports:
        rid += 1
        rows.append({
            "report_id": rid, "report_date": str(today - timedelta(days=sr["days_ago"])),
            "report_type": sr["report_type"], "company": sr["company"],
            "company_isin": sr["isin"], "period": sr["period"],
            "revenue": sr["revenue"], "revenue_currency": sr["revenue_currency"],
            "revenue_surprise_pct": sr["revenue_surprise_pct"],
            "eps": sr["eps"], "eps_surprise_pct": sr["eps_surprise_pct"],
            "dividend_per_share": sr["dividend_per_share"],
            "guidance": sr["guidance"], "key_takeaway": sr["key_takeaway"],
            "sector": sr["sector"],
        })

    # Generate ~56 more
    equity_companies = [(i[3], i[0], i[6], i[9]) for i in INSTRUMENTS_RAW if i[4] == "equity"]
    for _ in range(56):
        rid += 1
        company, isin, sector, ccy = random.choice(equity_companies)
        rt = random.choices(report_types, weights=[35, 25, 5, 15, 20])[0]
        period = random.choice(periods)
        revenue = round(random.uniform(500, 50000), 1)
        rev_surprise = round(random.uniform(-5, 5), 1)
        eps = round(random.uniform(0.5, 25), 2)
        eps_surprise = round(random.uniform(-5, 5), 1)
        div = round(random.uniform(0.5, 15), 2) if random.random() < 0.6 else None
        guidance = random.choice(guidances)

        rows.append({
            "report_id": rid, "report_date": str(today - timedelta(days=random.randint(1, 60))),
            "report_type": rt, "company": company,
            "company_isin": isin, "period": period,
            "revenue": revenue, "revenue_currency": ccy,
            "revenue_surprise_pct": rev_surprise,
            "eps": eps, "eps_surprise_pct": eps_surprise,
            "dividend_per_share": div,
            "guidance": guidance,
            "key_takeaway": f"{company} {period} report. Revenue surprise: {rev_surprise:+.1f}%. Guidance {guidance}.",
            "sector": sector or "Diversified",
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
    financial_news_df = gen_financial_news(instruments_df)
    public_filings_df = gen_public_filings()
    analyst_recommendations_df = gen_analyst_recommendations(instruments_df)
    financial_reports_df = gen_financial_reports()

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
        "financial_news": financial_news_df,
        "public_filings": public_filings_df,
        "analyst_recommendations": analyst_recommendations_df,
        "financial_reports": financial_reports_df,
    }

    for name, df in tables.items():
        upload_parquet(s3, bucket, f"{name}.parquet", df)

    total_rows = sum(len(df) for df in tables.values())
    print(f"\n✅ Data lake ready: {len(tables)} tables, {total_rows:,} total rows")
    print(f"   MinIO console: http://localhost:9001 (minio_admin / minio_2026!)")
    print("="*60)


if __name__ == "__main__":
    main()
