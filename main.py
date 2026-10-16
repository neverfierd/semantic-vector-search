from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any
import heapq
import math


@dataclass(frozen=True)
class Record:
    """хранит оригинальный текст и его id"""

    id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """то что мы возвращаем при поиске"""

    id: str
    distance: float
    score: float
    path: tuple[str, ...]  # путь по узлам дерева от корня до листа
    payload: Record | None


@dataclass
class Item:
    """элемент который физически хранится в листе дерева"""

    item_id: int
    vector: np.ndarray
    payload: Record | None


class Node:
    """узел нашего ball tree"""

    def __init__(self, node_id: str, is_leaf: bool = True):
        self.node_id = node_id
        self.is_leaf = is_leaf

        # геометрия гиперсферы
        self.centroid: np.ndarray | None = None
        self.radius: float = 0.0

        # наполнение
        self.size: int = 0
        self.items: list[Item] = []
        self.children: list[Node] = []

        # метаданные для ui
        self.max_item_id: int | None = None


class SemanticBTree:
    """главный класс-контроллер индекса"""

    def __init__(self, t: int = 8):
        self.t = t
        self.root = Node(node_id="root", is_leaf=True)
        self.root.max_item_id = 0

    def insert(
        self, vector: np.ndarray, item_id: str | int, payload: Record | None = None
    ) -> None:
        """добавить вектор в индекс"""
        item = Item(item_id=int(item_id), vector=vector, payload=payload)
        self._insert_recursive(self.root, item)

        # обновляем счетчик максимального id для ui
        if self.root.max_item_id is None or item.item_id > self.root.max_item_id:
            self.root.max_item_id = item.item_id

    def _insert_recursive(self, node: Node, item: Item) -> None:
        """рекурсивный спуск и вставка"""
        if node.is_leaf:
            # если мы в листе просто кладем вектор
            node.items.append(item)
            self._update_node_geometry(node)

            # если лист переолнился элементов > t делаем сплит
            if len(node.items) > self.t:
                self._split_leaf(node)
            return

        # если это ветка ищем дочернюю сферу чей центроид геометрически ближе всего
        best_child = min(
            node.children,
            key=lambda child: (
                float(np.linalg.norm(child.centroid - item.vector))
                if child.centroid is not None
                else float("inf")
            ),
        )
        self._insert_recursive(best_child, item)

        # обновляем радиус текущей ветки из рекурсии
        self._update_node_geometry(node)

    def _update_node_geometry(self, node: Node) -> None:
        """пересчитывает центроид и радиус гиперсферы"""
        if node.is_leaf:
            if not node.items:
                return
            # центроид листа - среднее арифметическое всех его векторов
            vectors = np.stack([item.vector for item in node.items])
            node.centroid = np.mean(vectors, axis=0)
            node.size = len(node.items)

            # радиус - максимальное расстояние от центроида до векторов листа
            distances = np.linalg.norm(vectors - node.centroid, axis=1)
            node.radius = float(np.max(distances))
        else:
            if not node.children:
                return
            # центроид ветки - среднее центроидов детей
            child_centroids = np.stack(
                [c.centroid for c in node.children if c.centroid is not None]
            )
            node.centroid = np.mean(child_centroids, axis=0)
            node.size = sum(c.size for c in node.children)

            # радиус ветки должен покрывать дочерние сферы полностью
            max_r = 0.0
            for child in node.children:
                if child.centroid is not None:
                    dist = float(np.linalg.norm(child.centroid - node.centroid))
                    max_r = max(max_r, dist + child.radius)
            node.radius = max_r

    def _split_leaf(self, node: Node) -> None:
        """сплит переполненного листа на два новых"""
        items = node.items

        # ищем два самых далеких вектора это будут наши новые центры
        vectors = np.stack([item.vector for item in items])

        # считаем расстояния всех со всеми матрицей чтобы без циклов
        diffs = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)

        # достаем индексы двух самых отдаленных точек
        p1_idx, p2_idx = np.unravel_index(np.argmax(distances), distances.shape)

        pole1_vector = items[p1_idx].vector
        pole2_vector = items[p2_idx].vector

        # делаем две новые ноды
        left_child = Node(node_id=f"{node.node_id}_L", is_leaf=True)
        right_child = Node(node_id=f"{node.node_id}_R", is_leaf=True)

        # раскидываем векторы к какому центру они ближе
        for item in items:
            dist_to_p1 = float(np.linalg.norm(item.vector - pole1_vector))
            dist_to_p2 = float(np.linalg.norm(item.vector - pole2_vector))

            if dist_to_p1 < dist_to_p2:
                left_child.items.append(item)
            else:
                right_child.items.append(item)

        # если все упало в одну сторону кидаем один элемент в пустую чтобы не крашнулось
        if not left_child.items:
            left_child.items.append(right_child.items.pop())
        elif not right_child.items:
            right_child.items.append(left_child.items.pop())

        # текущая нода становится веткой и забирает новых детей
        node.is_leaf = False
        node.items = []
        node.children = [left_child, right_child]

        # обновляем геометрию
        self._update_node_geometry(left_child)
        self._update_node_geometry(right_child)
        self._update_node_geometry(node)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        """поиск k ближайших соседей через очередь с приоритетом"""
        # куча для хранения топ k кандидатов
        # в питоне heapq это min-heap поэтому кладем отрицательные дистанции
        # на нулевом индексе всегда лежит самый худший из лучших кандидатов
        best_candidates: list[tuple[float, int, tuple[str, ...], Item]] = []

        def _search_recursive(node: Node, path: tuple[str, ...]) -> None:
            # если куча полная берем дистанцию с вершины убирая минус
            worst_dist = (
                -best_candidates[0][0] if len(best_candidates) == k else float("inf")
            )

            # математика отсечения ветки
            if node.centroid is not None:
                dist_to_center = float(np.linalg.norm(node.centroid - query))
                # если сфера слишком далеко выходим
                if dist_to_center - node.radius > worst_dist:
                    return

            current_path = path + (node.node_id,)

            if node.is_leaf:
                if not node.items:
                    return

                # считаем дистанции до всех векторов листа разом
                vectors = np.stack([item.vector for item in node.items])
                distances = np.linalg.norm(vectors - query, axis=1)

                for idx, dist in enumerate(distances):
                    dist_f = float(dist)
                    item = node.items[idx]

                    if len(best_candidates) < k:
                        # id item нужен чтобы питон не пытался сравнивать объекты если дистанции равны
                        heapq.heappush(
                            best_candidates, (-dist_f, id(item), current_path, item)
                        )
                    else:
                        # если новый элемент ближе худшего в куче заменяем его
                        if dist_f < -best_candidates[0][0]:
                            heapq.heappushpop(
                                best_candidates, (-dist_f, id(item), current_path, item)
                            )
                return

            # мы в ветке сортируем детей по близости к запросу
            valid_children = [c for c in node.children if c.centroid is not None]
            valid_children.sort(key=lambda c: float(np.linalg.norm(c.centroid - query)))

            for child in valid_children:
                _search_recursive(child, current_path)

        # запускаем обход
        _search_recursive(self.root, ())

        # собираем результат достаем из кучи и сортируем от лучших к худшим
        best_candidates.sort(key=lambda x: -x[0])

        results = []
        for neg_dist, _, path, item in best_candidates:
            dist = -neg_dist
            results.append(
                SearchResult(
                    id=str(item.item_id),
                    distance=dist,
                    score=1.0 / (1.0 + dist),
                    path=path,
                    payload=item.payload,
                )
            )
        return results


# эти функции лежат на уровне модуля
def validate_record(data: dict) -> Record:
    """парсер для загрузки из json"""
    return Record(id=int(data["id"]), text=str(data["text"]))


def build_tree_from_records(
    records: list[Record], embeddings: Any, t: int = 8
) -> SemanticBTree:
    """фабрика для сборки дерева из массива текстов"""
    tree = SemanticBTree(t=t)

    if not records:
        return tree

    # прогоняем тексты через модельку
    texts = [r.text for r in records]
    vectors = embeddings.embed_batch(texts)

    # вставляем векторы в дерево один за другим
    for record, vector in zip(records, vectors):
        tree.insert(vector=vector, item_id=record.id, payload=record)

    return tree


def linear_search(
    vectors: np.ndarray, payloads: list[Record], query: np.ndarray, k: int = 5
) -> list[SearchResult]:
    """наивный перебор всех векторов на чистом питоне без магии numpy"""
    if len(vectors) == 0:
        return []

    best_candidates = []

    # перебираем каждый вектор ручками
    for i in range(len(vectors)):
        vector = vectors[i]

        # считаем евклидово расстояние честным циклом
        dist_sq = 0.0
        for j in range(len(query)):
            diff = float(vector[j]) - float(query[j])
            dist_sq += diff * diff
        dist = math.sqrt(dist_sq)

        record = payloads[i]
        best_candidates.append((dist, record))

    # сортируем весь список по дистанции
    best_candidates.sort(key=lambda x: x[0])

    # собираем финальный топ k
    results = []
    for dist, record in best_candidates[:k]:
        results.append(
            SearchResult(
                id=str(record.id),
                distance=dist,
                score=1.0 / (1.0 + dist),
                path=("linear_search",),  # у прямого перебора нет пути
                payload=record,
            )
        )
    return results
