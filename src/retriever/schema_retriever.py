try:
    from src.retriever.table_prior import TABLE_PRIORS, TIME_KEYWORDS, DATE_COLUMN_PATTERNS
except ModuleNotFoundError:
    from table_prior import TABLE_PRIORS, TIME_KEYWORDS, DATE_COLUMN_PATTERNS
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Regex to strip temporal phrases so the embedding focuses on the
# "what" aspect of the question (value columns) rather than "when".
_TIME_STRIP_RE = re.compile(
    r"\b("
    + "|".join(re.escape(kw) for kw in sorted(TIME_KEYWORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


class SchemaRetriever:

    def __init__(self, schema_items):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        #store schema
        self.schema_items = schema_items

        #Convert each schema column text into an embedding vector
        texts = [item["text"] for item in schema_items]
        self.schema_embeddings = self.model.encode(
            texts,
            normalize_embeddings = True
        )

    def retrieve(self, question, top_k=10):

        """
       1. Given a natural language question, return top_k relevent schema columns
          as a list of (score, item) tuples.
       2. Table - level buisness priors
         """

        question_lower = question.lower()

        #detect intent keywords
        preferred_tables = set()
        for keyword, tables in TABLE_PRIORS.items():
            if keyword in question_lower:
                preferred_tables.update(tables)

        # Strip temporal phrases before embedding so that the semantic
        # search focuses on value columns (e.g. revenue, total_inc_tax)
        # rather than being pulled toward date/timestamp columns.
        needs_time = any(kw in question_lower for kw in TIME_KEYWORDS)
        embedding_question = _TIME_STRIP_RE.sub("", question).strip() if needs_time else question
        # Fallback: if stripping removed everything, use original
        if not embedding_question:
            embedding_question = question

        #encode question (stripped of temporal phrases)
        self._last_question_embedding = self.model.encode(
            [embedding_question],
            normalize_embeddings=True
        )
        q_emd = self._last_question_embedding
        self._embedding_question = embedding_question


        #compute cosine similarity between question and all schema columns
        sims = cosine_similarity(
            q_emd,
            self.schema_embeddings
        )[0]

        scored_items = []

        for idx, item in enumerate(self.schema_items):
            score = sims[idx]

            #Apply a small boost if column belongs to a preferred table
            if item['table'] in preferred_tables:
                score += 0.15
            scored_items.append((score, item)) 
        #sort by final score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        top_results = [(score, item) for score, item in scored_items[:top_k]]

        # If the question mentions time, ensure date columns from
        # preferred tables are included even if they didn't rank in top_k
        needs_time = any(kw in question_lower for kw in TIME_KEYWORDS)
        if needs_time and preferred_tables:
            already_included = {
                (item["table"], item["column"]) for _, item in top_results
            }
            for score, item in scored_items:
                col_lower = item["column"].lower()
                if (
                    item["table"] in preferred_tables
                    and any(p in col_lower for p in DATE_COLUMN_PATTERNS)
                    and (item["table"], item["column"]) not in already_included
                ):
                    top_results.append((score, item))
                    already_included.add((item["table"], item["column"]))

        return top_results    


