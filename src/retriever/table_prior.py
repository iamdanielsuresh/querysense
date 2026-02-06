# Simple table-level priors based on business domain knowledge

TABLE_PRIORS = {
    "revenue": ["orders", "order_product"],
    "sales": ["orders", "order_product"],
    "order": ["orders", "order_product"],
    "customer": ["customer"],
    "product": ["product", "order_product"],
}

# Keywords that signal the query needs a date/time column
TIME_KEYWORDS = [
    "last month", "this month", "yesterday", "today", "last week",
    "this week", "last year", "this year", "month", "year", "week",
    "day", "date", "recent", "ago", "since", "between", "before",
    "after", "period", "quarter",
]

# Columns considered date/time columns (matched by substring)
DATE_COLUMN_PATTERNS = ["date", "time", "created", "modified", "shipped"]