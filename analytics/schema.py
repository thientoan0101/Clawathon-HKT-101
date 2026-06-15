"""Transfer transaction CSV schema and column constants."""

CSV_COLUMNS = [
    "order_no",
    "app_id",
    "sender_id",
    "peer_id",
    "product_code",
    "amount",
    "order_time",
]

# order_time is epoch milliseconds (e.g. 1772713878061)
ORDER_TIME_UNIT = "ms"

DEFAULT_DATA_PATH = "data/transfers.csv"
