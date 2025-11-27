#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, jsonify, send_file
import io
import csv
import random
import math
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------- 数据生成器工具（不依赖 numpy/pandas） ----------

def sample_normal(mu=0.0, sigma=1.0, size=1):
    """使用 Box-Muller 生成正态分布样本"""
    out = []
    for _ in range(size):
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2 * math.pi * u2)
        out.append(mu + sigma * z0)
    return out

def sample_uniform(low=0.0, high=1.0, size=1):
    return [random.uniform(low, high) for _ in range(size)]

def sample_exponential(scale=1.0, size=1):
    # random.expovariate expects lambda = 1/scale
    lam = 1.0 / scale if scale != 0 else 1.0
    return [random.expovariate(lam) for _ in range(size)]

def sample_int(low=0, high=100, size=1):
    return [random.randint(low, high) for _ in range(size)]

def sample_categorical(categories, probs=None, size=1):
    if probs:
        # ensure sum ~=1 and lengths match
        return random.choices(population=categories, weights=probs, k=size)
    else:
        return random.choices(population=categories, k=size)

def sample_datetime(start_str="2020-01-01", end_str="2020-12-31", size=1):
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    out = []
    for _ in range(size):
        ts = random.randint(start_ts, end_ts)
        out.append(datetime.fromtimestamp(ts))
    return out

def sample_text(choices=None, size=1):
    if choices:
        return random.choices(choices, k=size)
    # fallback: random alphanumeric short strings
    import string
    out = []
    for _ in range(size):
        out.append(''.join(random.choices(string.ascii_letters + string.digits, k=8)))
    return out

# synthesize_table 根据 schema 生成行
def synthesize_table(n_rows, schema):
    """
    schema: list of column specs
    each column spec is a dict:
        {
            "name": "col1",
            "type": "numeric" | "int" | "categorical" | "datetime" | "text",
            "dist": "normal"/"uniform"/"exponential" (for numeric),
            "params": {...}  # params for distributions
            "categories": ["A","B"]
            "probs": [0.5,0.5]
        }
    """
    # Prepare empty columns
    cols = {col["name"]: [] for col in schema}

    for col in schema:
        name = col["name"]
        typ = col.get("type", "numeric")
        params = col.get("params", {})
        if typ == "numeric":
            dist = col.get("dist", "normal")
            if dist == "normal":
                mu = float(params.get("mu", 0))
                sigma = float(params.get("sigma", 1))
                vals = sample_normal(mu, sigma, n_rows)
            elif dist == "uniform":
                low = float(params.get("low", 0))
                high = float(params.get("high", 1))
                vals = sample_uniform(low, high, n_rows)
            elif dist == "exponential":
                scale = float(params.get("scale", 1.0))
                vals = sample_exponential(scale, n_rows)
            else:
                # fallback uniform
                vals = sample_uniform(0, 1, n_rows)
            # convert to Python floats
            cols[name] = [float(v) for v in vals]

        elif typ == "int":
            low = int(params.get("low", 0))
            high = int(params.get("high", 100))
            cols[name] = sample_int(low, high, n_rows)

        elif typ == "categorical":
            categories = col.get("categories", ["A", "B"])
            probs = col.get("probs", None)
            cols[name] = sample_categorical(categories, probs, n_rows)

        elif typ == "datetime":
            start = params.get("start", "2020-01-01")
            end = params.get("end", "2020-12-31")
            dtlist = sample_datetime(start, end, n_rows)
            # store ISO strings for JSON/CSV compatibility
            cols[name] = [dt.isoformat() for dt in dtlist]

        elif typ == "text":
            choices = col.get("choices", None)
            cols[name] = sample_text(choices, n_rows)

        else:
            # fallback: random short text
            cols[name] = sample_text(None, n_rows)

    # assemble rows (list of dicts)
    rows = []
    for i in range(n_rows):
        row = {}
        for col in schema:
            row[col["name"]] = cols[col["name"]][i]
        rows.append(row)
    return rows

# ---------- Flask 路由 ----------

@app.route("/")
def index():
    # 前端提供一个示例 schema，方便用户修改
    example_schema = [
        {"name":"id","type":"int","params":{"low":1,"high":10000}},
        {"name":"value","type":"numeric","dist":"normal","params":{"mu":0,"sigma":1}},
        {"name":"category","type":"categorical","categories":["A","B","C"],"probs":[0.5,0.3,0.2]},
        {"name":"event_time","type":"datetime","params":{"start":"2023-01-01","end":"2023-12-31"}},
        {"name":"note","type":"text"}
    ]
    import json
    return render_template("index.html", example_schema=json.dumps(example_schema, ensure_ascii=False, indent=2))

@app.route("/generate_preview", methods=["POST"])
def generate_preview():
    payload = request.json or {}
    n = int(payload.get("n_rows", 10))
    schema = payload.get("schema", [])
    # basic validation
    if not isinstance(schema, list):
        return jsonify({"error":"schema must be a list"}), 400
    rows = synthesize_table(n, schema)
    # return columns + rows (rows as list of lists)
    columns = [c["name"] for c in schema]
    rows_list = [[r[col] for col in columns] for r in rows]
    return jsonify({"columns": columns, "rows": rows_list})

@app.route("/download_csv", methods=["POST"])
def download_csv():
    payload = request.json or {}
    n = int(payload.get("n_rows", 1000))
    schema = payload.get("schema", [])
    rows = synthesize_table(n, schema)
    columns = [c["name"] for c in schema]

    # write CSV to memory
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for r in rows:
        # ensure all values are str/primitive
        writer.writerow({k: (v if v is not None else "") for k,v in r.items()})
    buf.seek(0)
    # send as file
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
