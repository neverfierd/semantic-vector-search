import time
import numpy as np
from main import SemanticBTree, Record, linear_search


def run_benchmark(dimensions: int) -> None:
    k = 10
    n_values = [100, 500, 1000, 5000, 10000]

    print(f"\nТест для размерности D = {dimensions}")
    print("| n | линейный ms | дерево ms | ускорение |")
    print("|---|-------------|-----------|-----------|")

    for n in n_values:
        vectors = np.random.randn(n, dimensions).astype(np.float32)
        query = np.random.randn(dimensions).astype(np.float32)

        payloads = [Record(id=i, text=f"text {i}") for i in range(n)]

        tree = SemanticBTree(t=8)
        for i in range(n):
            tree.insert(vector=vectors[i], item_id=i, payload=payloads[i])

        start_tree = time.perf_counter()
        _ = tree.search(query, k=k)
        tree_time = (time.perf_counter() - start_tree) * 1000

        start_linear = time.perf_counter()
        _ = linear_search(vectors, payloads, query, k=k)
        linear_time = (time.perf_counter() - start_linear) * 1000

        speedup = linear_time / tree_time if tree_time > 0 else 0

        print(f"| {n} | {linear_time:.2f} | {tree_time:.2f} | {speedup:.1f}x |")


if __name__ == "__main__":
    run_benchmark(dimensions=2)
    run_benchmark(dimensions=384)
