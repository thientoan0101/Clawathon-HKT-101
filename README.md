# Money Insight
AI Analytics Assistant for Money Transfer

# Team: Money Radar
Money Radar lets anyone ask their transfer data a question in plain language and get a grounded answer with a chart in seconds — no SQL, no waiting on the data team.

# Problem
At ZaloPay’s scale, the people who need transfer insights most — PMs, ops, growth, risk — are furthest from the data. Today a single number means writing Presto/Trino SQL against tables like zpa_eventlog and zalopay_translog, or queuing the data team for days. Static dashboards can’t be interrogated: slice differently and you’re back to queries.
# Solution
Money Insight is a transfer-analytics dashboard with a built-in AI assistant. Ask “What’s the TPV this month?”, “Top 20 senders by volume”, or “Peak transfer hour”, and it returns the exact number plus a visualization. Each question is routed through a schema-grounded LLM function registry: the model maps language to a defined function, runs it against the real tables, and answers only from the results — correct numbers, not hallucinations. The same engine powers a Zalo OA chatbot for answers on the go.
# Use cases
Self-serve metrics for PMs; retention deep dives by cohort; top-sender and anomaly analysis; DAU/MAU health checks; peak-hour operational timing; next-month forecasting; and on-the-go queries via Zalo.
# Why it matters
It collapses “question” to “trustworthy answer” from days to seconds and removes SQL as a gatekeeper — natural-language analytics made safe for fintech, where a wrong number is worse than none.
