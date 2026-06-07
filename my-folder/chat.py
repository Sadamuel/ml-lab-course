'''
Local Natural-Language SQL Assistant using Ollama and SQLite


To setup, please install ollama before running:
pull qwen2.5:latest
or
ollama pull gemma3:4b
or do the command with any other model you want to test, but keep in mind only qwen2.5 and gemma3.4b have been tested.
Then, start a local API server with the command:
ollama serve


After doing setup, to run please use:
    python chat.py or py chat.py
    To run with a specific model or specific database location, add after chat.py:
    --model qwen2.5:1.5b
    --db path/to/db.db


    Such as: py chat.py --model qwen2.5:1.5b
'''

import argparse
import re
import sqlite3
import sys
import textwrap
from textwrap import dedent

import requests

DEFAULT_MODEL   = "qwen2.5:latest"
DEFAULT_DB      = "musicCC.db"
OLLAMA_URL      = "http://localhost:11434/api/chat"
STATELESS       = True
AI_SUMMARY      = True

SCHEMA = dedent("""
    Table: profiles
    Columns:
      id                   INTEGER  PRIMARY KEY
      profile_url          TEXT - Profile URL. Always include in results.
      display_name         TEXT - artist's display name, always include in results
      full_name            TEXT
      country_code         TEXT - ISO 3166-1 alpha-2 country code
      location             TEXT
      join_date            TEXT - ISO 8601 date string
      is_verified          INTEGER - ignore unless user specifically asks about verified status
      warnings             TEXT - NULL = confirmed Creative Commons artist; non-NULL = NOT a CC artist
      user_id              TEXT
      description          TEXT
      followers_count      INTEGER - popularity metric; higher = more popular
      followings_count     INTEGER
      track_count          INTEGER
      likes_count          INTEGER
      playlist_count       INTEGER
      playlist_likes_count INTEGER
      needs_review         INTEGER - 0 or 1

    Key facts:
      - Creative Commons artists = WHERE warnings IS NULL. Do NOT check any other column for CC.
      - Non-NULL warnings = NOT a CC artist.
      - Most popular = ORDER BY followers_count DESC
      - Total rows: 10,067
      - If no country is mentioned in the latest prompt, do NOT filter by country_code.

    Country codes:
      Sweden=SE UnitedStates=US Germany=DE UnitedKingdom=GB
      France=FR Netherlands=NL Norway=NO Finland=FI
      Denmark=DK Canada=CA Australia=AU Japan=JP
""").strip()

SQL_SYSTEM = dedent(f"""
    You are a SQL assistant for a SQLite database of SoundCloud profiles.

    {SCHEMA}

    RULES:
    1. Output ONLY a single SQL SELECT inside ```sql ... ```. Nothing else.
    2. Default columns: display_name, followers_count, profile_url, nothing else.
    3. Default LIMIT 20. No LIMIT for COUNT/aggregate queries.
    4. Creative Commons ALWAYS means WHERE warnings IS NULL, nothing else.
    5. Most popular ALWAYS means ORDER BY followers_count DESC.
    6. Ignore is_verified by default. Only touch this column if the user's question is specifically about verified or unverified artists.
    7. Never use DROP, DELETE, INSERT, UPDATE, or any write operation.
    8. For percentage/likelihood queries, always GROUP BY the dimension being compared.
       Use COUNT(*) as the denominator, it counts rows in that group, not the whole table.
       ALWAYS use 100.0 (not 100) to force float division, e.g. ROUND(100.0 * SUM(...) / COUNT(*), 1).
       Using plain 100 causes SQLite integer division and produces only 0 or 1, never do this.
       Never use a single-row CASE-per-column approach for comparing groups.
       Never mix display_name, followers_count, or profile_url into aggregate queries.
    9. When comparing two groups (e.g. verified vs unverified, country A vs country B),
       always use GROUP BY, never inline CASE columns.
       Example, "what % of verified vs non-verified are CC?":
         SELECT is_verified,
                COUNT(*) AS total,
                ROUND(100.0 * SUM(CASE WHEN warnings IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS cc_percentage
         FROM profiles
         GROUP BY is_verified
         ORDER BY is_verified DESC
       Do NOT filter by warnings in the WHERE clause for group comparison queries.
    10. If user ask for artist's descriptions, then include the row's description. 
""").strip()

CHAT_SYSTEM = dedent(f"""
    You are a knowledgeable assistant for a SoundCloud profile database.

    {SCHEMA}

    Answer the user's question conversationally in plain English.
    You can reason about the data, explain concepts, or discuss what the database contains.
    Do not generate SQL. Do not output code blocks.
    Keep answers concise, 2 to 4 sentences unless more detail is clearly needed.
""").strip()

SQL_SIGNALS = [
    r"\blist\b", r"\bshow\b", r"\bgive me\b", r"\bfind\b", r"\bwho\b",
    r"\bhow many\b", r"\bcount\b", r"\btop\b", r"\bbest\b", r"\bmost\b",
    r"\blast\b", r"\blatest\b", r"\boldest\b", r"\brank\b", r"\bwhich country\b",
    r"\baverage\b", r"\btotal\b", r"\bpercent\b", r"\bratio\b", r"\bcompare\b",
    r"\bartists? from\b", r"\bartists? with\b", r"\bartists? who\b",
    r"\bfollowers\b", r"\btracks?\b", r"\bverified\b",
    r"\blikely\b", r"\blikelihood\b", r"\bchance\b", r"\bprobab\w+\b",
]


def chat(messages, model):
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.1}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit("\n[ERROR] Cannot reach Ollama. Run: ollama serve\n")
    return r.json()["message"]["content"]


def needs_sql(question):
    return any(re.search(p, question.lower()) for p in SQL_SIGNALS)


def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = m.group(1).strip() if m else None
    if not sql:
        m = re.search(r"(SELECT\b.+)", text, re.DOTALL | re.IGNORECASE)
        sql = m.group(1).strip() if m else None
    if sql:
        sql = re.sub(r";\s*(ORDER\s+BY|LIMIT|GROUP\s+BY|HAVING)", r" \1", sql, flags=re.IGNORECASE)
        sql = sql.rstrip(";").strip()
    return sql


def is_safe(sql):
    return not re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE|ATTACH)\b",
        sql, re.IGNORECASE
    )


def run_query(sql, db_path):
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [tuple(r) for r in cur.fetchall()]
        con.close()
        return cols, rows, None
    except sqlite3.Error as e:
        return [], [], str(e)


def col_idx(cols, name):
    return next((i for i, c in enumerate(cols) if c.lower() == name), None)


def print_results(cols, rows, model, question):
    if not rows:
        print("Assistant: No results found.\n")
        return

    name_i = col_idx(cols, 'display_name')
    fol_i  = col_idx(cols, 'followers_count')
    url_i  = col_idx(cols, 'profile_url')
    desc_i = col_idx(cols, 'description')
    is_stats = name_i is None

    print("Assistant:")

    if is_stats:
        widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
        print("  " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
        print("  " + "-+-".join("-" * w for w in widths))
        for row in rows:
            print("  " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
        print()

    else:
        for row in rows:
            parts = [str(row[name_i])]
            if fol_i is not None: parts.append(f"{row[fol_i]} followers")
            if url_i is not None: parts.append(str(row[url_i]))
            print("  " + " — ".join(parts))
            if desc_i is not None:
                desc = row[desc_i]
                if desc and str(desc).strip():
                    print(textwrap.fill(" ".join(str(desc).split()), width=100, initial_indent="    ", subsequent_indent="    "))
                else:
                    print("    (no description)")
            print()

    if AI_SUMMARY:
        if is_stats:
            compact = " | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in row) for row in rows)
        else:
            compact = "\n".join(
                str(row[name_i]) + (f" ({row[fol_i]} followers)" if fol_i is not None else "")
                for row in rows
            )
        summary = chat([
            {"role": "system", "content": "You comment briefly on database query results. Be concise and natural."},
            {"role": "user", "content": f'The user asked: "{question}"\n\nResults:\n{compact}\n\nGive a 1-4 sentence observation about these results. Be natural and specific. No lists.'}
        ], model)
        print(f"  Summary: {summary}\n")


def main():
    parser = argparse.ArgumentParser(description="NL->SQL assistant (local Ollama)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db",    default=DEFAULT_DB)
    args = parser.parse_args()

    print(f"\nMusic DB Assistant | model={args.model} | db={args.db}")
    print("Good day, ben, ask any question about the database and I will answer. Alternatively Type 'exit' or 'quit' to quit.\n")

    history     = [{"role": "system", "content": CHAT_SYSTEM}]
    sql_history = [{"role": "system", "content": SQL_SYSTEM}]

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Good, bye, ben.")
            break

        if not needs_sql(question):
            if STATELESS:
                history = [{"role": "system", "content": CHAT_SYSTEM}]
            history.append({"role": "user", "content": question})
            reply = chat(history, args.model)
            history.append({"role": "assistant", "content": reply})
            print(f"\nDB Assistant: {reply}\n")
            continue

        if STATELESS:
            sql_history = [{"role": "system", "content": SQL_SYSTEM}]
        sql_history.append({"role": "user", "content": question})
        sql_reply = chat(sql_history, args.model)
        sql_history.append({"role": "assistant", "content": sql_reply})

        sql = extract_sql(sql_reply)
        if not sql:
            print(f"\nAssistant: {sql_reply}\n")
            continue
        if not is_safe(sql):
            print("\nATTEMPTED ILLEGAL WRITE OPERATION!!!\n")
            continue

        print(f"\n[SQL] {sql}\n")
        cols, rows, error = run_query(sql, args.db)

        if error:
            print(f"[ERROR] SQL error: {error}\n")
            sql_history.append({"role": "user", "content": f"SQL error: {error}"})
            correction = chat(sql_history, args.model)
            sql_history.append({"role": "assistant", "content": correction})
            sql2 = extract_sql(correction)
            if sql2 and is_safe(sql2):
                print(f"[RETRY SQL] {sql2}\n")
                cols, rows, error = run_query(sql2, args.db)
                if error:
                    print(f"[FAILED] {error}\n")
                    continue
            else:
                continue

        print_results(cols, rows, args.model, question)


if __name__ == "__main__":
    main()
