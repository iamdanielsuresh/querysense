"""Test the query decomposition module — complexity detection (no LLM needed)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decomposition.query_decomposer import classify_complexity

# Test cases: (question, expected_complexity)
test_cases = [
    # Simple queries
    ("How many singers are there?", "simple"),
    ("What is the name of the oldest singer?", "simple"),
    ("Show all concert names.", "simple"),
    ("Find the total revenue.", "simple"),
    ("List all customers from New York.", "simple"),

    # Complex queries
    ("Find the customers who placed more than 5 orders and also bought products from the electronics category.", "complex"),
    ("What is the average order value for each customer, and show only those who have an average higher than the overall average?", "complex"),
    ("Compare the total revenue from last month versus this month for each product category.", "complex"),
    ("First find the top 10 selling products, then calculate the average rating of those products.", "complex"),
    ("Show the customers who purchased product A but not product B, and list their total spending.", "complex"),
    ("For each store, count the number of orders that have at least 3 items and the total revenue of those orders, among those placed in 2025.", "complex"),
    ("What is the difference between the number of completed orders and cancelled orders per month?", "complex"),
]

print(f"{'Question':<90} {'Expected':<10} {'Got':<10} {'Score':<7} {'Signals'}")
print("-" * 150)

correct = 0
for question, expected in test_cases:
    result = classify_complexity(question)
    got = result["complexity"]
    match = "✓" if got == expected else "✗"
    if got == expected:
        correct += 1
    q_display = question[:87] + "..." if len(question) > 87 else question
    signals = ", ".join(result["signals"][:3]) if result["signals"] else "-"
    print(f"{match} {q_display:<88} {expected:<10} {got:<10} {result['score']:<7} {signals}")

print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
