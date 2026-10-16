# импорты
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Any

import numpy as np
import plotly.graph_objects as go  # type: ignore
import streamlit as st
from streamlit_echarts import st_echarts

from embedding import Embeddings

from main import (
    Node,
    Record,
    SearchResult,
    SemanticBTree,
    build_tree_from_records,
    validate_record,
    Item,
)


class _EChartsItem(TypedDict):
    name: str
    value: int | None
    itemStyle: dict[str, str]


class _EChartsNode(TypedDict):
    name: str
    collapsed: bool
    children: list[Any]
    itemStyle: dict[str, str]


# генерация текста для узлов
def build_node_label(name: str, size: int, radius: float) -> str:
    return f"🔹 {name}\nвекторов: {size}\nрадиус: {radius:.3f}"


def build_root_label(node: Node) -> str:
    return (
        f"👑 {node.node_id}\n"
        f"векторов: {node.size}\n"
        f"радиус: {node.radius:.3f}\n"
        f"max_id: {node.max_item_id}"
    )


def build_leaf_label(text: str) -> str:
    # лимит длины текста
    short_text = text if len(text) < 45 else text[:42] + "..."
    return f"💬 {short_text}"


# конвертация дерева
def tree_to_echarts(
    node: Node,
    depth: int = 0,
    is_root: bool = False,
) -> _EChartsNode:

    label = (
        build_root_label(node)
        if is_root
        else build_node_label(
            name=node.node_id,
            size=node.size,
            radius=node.radius,
        )
    )

    # цвета уровней
    if is_root:
        color = "#7c3aed"
    elif node.is_leaf:
        color = "#059669"
    else:
        color = "#2563eb"

    item_style = {"color": color, "borderColor": "#1e293b", "borderWidth": "2"}

    if node.is_leaf:
        children = [
            _EChartsItem(
                name=build_leaf_label(
                    text=item.payload.text if item.payload is not None else ""
                ),
                value=item.payload.id if item.payload is not None else None,
                itemStyle={
                    "color": "#f59e0b",
                    "borderColor": "#b45309",
                    "borderWidth": "2",
                },
            )
            for item in node.items
        ]
        return _EChartsNode(
            name=label, collapsed=depth >= 2, children=children, itemStyle=item_style
        )

    return _EChartsNode(
        name=label,
        collapsed=depth >= 2,
        children=[tree_to_echarts(child, depth=depth + 1) for child in node.children],
        itemStyle=item_style,
    )


# рендер графа
def render_tree(tree: SemanticBTree) -> None:
    data = tree_to_echarts(tree.root, is_root=True)

    options = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
        },
        "series": [
            {
                "type": "tree",
                "data": [data],
                "top": "5%",
                "left": "10%",
                "bottom": "5%",
                "right": "25%",
                "symbol": "circle",
                "symbolSize": 14,
                "orient": "LR",
                "expandAndCollapse": True,
                "initialTreeDepth": -1,
                "edgeShape": "curve",
                "roam": True,
                "lineStyle": {
                    "width": 2.5,
                    "curveness": 0.4,
                    "color": "#cbd5e1",
                },
                "label": {
                    "position": "top",
                    "verticalAlign": "middle",
                    "align": "center",
                    "fontSize": 12,
                    "fontWeight": "bold",
                    "lineHeight": 18,
                    "color": "#0f172a",
                    "backgroundColor": "rgba(255, 255, 255, 0.95)",
                    "borderColor": "#94a3b8",
                    "borderWidth": 1.5,
                    "borderRadius": 6,
                    "padding": [6, 8],
                    "shadowColor": "rgba(0, 0, 0, 0.15)",
                    "shadowBlur": 6,
                },
                "leaves": {
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "backgroundColor": "#fffbeb",
                        "borderColor": "#fde68a",
                    }
                },
                "animationDuration": 400,
                "animationDurationUpdate": 400,
            }
        ],
    }

    st_echarts(options=options, height="850px")


# логика ui


def load_records(file_path: Path) -> list[Record]:
    with file_path.open("r", encoding="utf-8") as file:
        raw: list[dict] = json.load(file)
    return [validate_record(entry) for entry in raw]


def initialize_records(file_path: Path) -> None:
    if "records" not in st.session_state:
        st.session_state.records = load_records(file_path)


def invalidate_tree() -> None:
    st.session_state.pop("tree", None)
    st.session_state.pop("tree_key", None)


def get_next_record_id() -> int:
    tree: SemanticBTree | None = st.session_state.get("tree")
    if tree is not None and tree.root.max_item_id is not None:
        return tree.root.max_item_id + 1
    records: list[Record] = st.session_state.records
    numeric_ids = [record.id for record in records]
    return max(numeric_ids, default=0) + 1


def add_record(text: str) -> tuple[bool, str]:
    normalized_text = text.strip()
    if not normalized_text:
        return False, "заполните text"
    next_record_id = get_next_record_id()
    records: list[Record] = st.session_state.records
    st.session_state.records = [
        *records,
        Record(id=next_record_id, text=normalized_text),
    ]
    invalidate_tree()
    return True, f"запись id={next_record_id} добавлена"


def delete_record(record_id: str) -> tuple[bool, str]:
    normalized_id = record_id.strip()
    if not normalized_id:
        return False, "укажите id"
    records: list[Record] = st.session_state.records
    filtered_records = [record for record in records if str(record.id) != normalized_id]
    if len(filtered_records) == len(records):
        return False, f"запись id={normalized_id} не найдена"
    st.session_state.records = filtered_records
    invalidate_tree()
    return True, f"запись id={normalized_id} удалена"


def find_item_path(tree: SemanticBTree, item_id: int) -> list[str] | None:
    def dfs(node: Node, path: list[str]) -> list[str] | None:
        current_path = [*path, node.node_id]
        if node.is_leaf:
            if any(str(item.item_id) == str(item_id) for item in node.items):
                return current_path
            return None
        for child in node.children:
            child_path = dfs(child, current_path)
            if child_path is not None:
                return child_path
        return None

    return dfs(tree.root, [])


def render_feedback(tree: SemanticBTree | None) -> None:
    pending_added_id: int | None = st.session_state.pop("pending_added_id", None)
    if pending_added_id is not None and tree is not None:
        path = find_item_path(tree=tree, item_id=pending_added_id)
        if path is not None:
            st.session_state.feedback = (
                "success",
                f"запись id={pending_added_id} добавлена. путь: {' -> '.join(path)}",
            )
        else:
            st.session_state.feedback = (
                "success",
                f"запись id={pending_added_id} добавлена",
            )

    feedback: tuple[str, str] | None = st.session_state.pop("feedback", None)
    if feedback is not None:
        level, message = feedback
        if level == "success":
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)


def get_or_build_tree(
    records: list[Record], embedder: Embeddings, t_value: int
) -> SemanticBTree | None:
    if not records:
        invalidate_tree()
        return None
    records_signature = tuple(str(record.id) for record in records)
    current_key = f"tree_t_{t_value}_{records_signature}"
    if st.session_state.get("tree_key") != current_key:
        st.session_state.tree = build_tree_from_records(
            records=records, embeddings=embedder, t=t_value
        )
        st.session_state.tree_key = current_key
    return st.session_state.tree  # type: ignore[no-any-return]


def render_record_controls() -> None:
    with st.sidebar.expander("управление", expanded=True):
        with st.form("add_record_form", clear_on_submit=True):
            st.markdown("**добавить**")
            text = st.text_area("text", key="add_record_text", height=120)
            if st.form_submit_button("добавить"):
                success, message = add_record(text=text)
                if success:
                    added_id = message.split("id=")[1].split()[0]
                    st.session_state.pending_added_id = int(added_id)
                    st.rerun()
                else:
                    st.session_state.feedback = ("error", message)

        with st.form("delete_record_form", clear_on_submit=True):
            st.markdown("**удалить**")
            record_id_to_delete = st.text_input("id", key="delete_record_id")
            if st.form_submit_button("удалить"):
                success, message = delete_record(record_id=record_id_to_delete)
                st.session_state.feedback = ("success" if success else "error", message)
                st.rerun()


def render_records_overview(records: list[Record]) -> None:
    st.sidebar.caption(f"всего: {len(records)}")
    st.sidebar.caption(f"след id: {get_next_record_id()}" if records else "след id: 1")
    st.sidebar.caption("id: " + ", ".join(str(record.id) for record in records[:12]))
    if len(records) > 12:
        st.sidebar.caption("показаны первые 12")


def render_search_results(
    tree: SemanticBTree | None, embedder: Embeddings, query: str, k_value: int
) -> None:
    if tree is None:
        st.info("пусто")
        return
    if st.button("искать"):
        query_vector = embedder.embed(query)
        results: list[SearchResult] = tree.search(query_vector, k=k_value)
        st.subheader("результаты")
        for result in results:
            if result.payload is not None:
                st.write(f"**id={result.payload.id}**")
                st.write(result.payload.text)
            st.caption(f"d={result.distance:.4f} | путь={' -> '.join(result.path)}")


def render_tree_panel(tree: SemanticBTree | None) -> None:
    if tree is None:
        st.info("пусто")
        return
    render_tree(tree)


# pca проекция


class PCAProjector:
    def __init__(self, vectors: np.ndarray):
        self.mean = np.mean(vectors, axis=0)
        centered = vectors - self.mean
        if len(vectors) > 1 and not np.allclose(centered, 0):
            u, s, self.vt = np.linalg.svd(centered, full_matrices=False)
        else:
            self.vt = np.eye(vectors.shape[1])

    def project(self, vectors: np.ndarray) -> np.ndarray:
        centered = vectors - self.mean
        result = centered @ self.vt.T[:, :2]
        return np.array(result, dtype=np.float32)


def get_all_items(node: Node) -> list[Item]:
    if node.is_leaf:
        return node.items
    items = []
    for c in node.children:
        items.extend(get_all_items(c))
    return items


def get_all_nodes(node: Node) -> list[Node]:
    nodes = [node]
    if not node.is_leaf:
        for c in node.children:
            nodes.extend(get_all_nodes(c))
    return nodes


def render_pca_panel(tree: SemanticBTree | None) -> None:
    if tree is None or not tree.root.items and not tree.root.children:
        st.info("пусто")
        return

    all_items = get_all_items(tree.root)
    if len(all_items) < 2:
        st.info("нужно от 2 элементов")
        return

    show_only_leaves = st.checkbox("🌿 только листы", value=True)

    vectors = np.stack([item.vector for item in all_items])
    projector = PCAProjector(vectors)
    projected_points = projector.project(vectors)
    all_nodes = get_all_nodes(tree.root)

    fig = go.Figure()

    shapes = []

    # сферы под точками
    for node in all_nodes:
        if node.centroid is not None:
            if show_only_leaves and not node.is_leaf:
                continue

            proj_c = projector.project(node.centroid[np.newaxis, :])[0]
            r = node.radius

            if node.is_leaf:
                fill_color = "rgba(34, 197, 94, 0.15)"
                line_color = "rgba(21, 128, 61, 0.8)"
            else:
                fill_color = "rgba(239, 68, 68, 0.05)"
                line_color = "rgba(185, 28, 28, 0.6)"

            shapes.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=proj_c[0] - r,
                    y0=proj_c[1] - r,
                    x1=proj_c[0] + r,
                    y1=proj_c[1] + r,
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=2 if node.is_leaf else 1.5),
                    layer="below",
                )
            )

            # подпись
            fig.add_trace(
                go.Scatter(
                    x=[proj_c[0]],
                    y=[proj_c[1]],
                    mode="text",
                    text=[f"<b>{node.node_id}</b>"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                    textfont=dict(color=line_color, size=11),
                )
            )

    fig.update_layout(shapes=shapes)

    # точки
    hover_texts = [
        f"<b>id: {item.item_id}</b><br><i>{item.payload.text if item.payload else ''}</i>"
        for item in all_items
    ]

    fig.add_trace(
        go.Scatter(
            x=projected_points[:, 0],
            y=projected_points[:, 1],
            mode="markers",
            name="эмбеддинги",
            marker=dict(
                color="#2563eb",
                size=10,
                line=dict(color="white", width=1.5),
                symbol="circle",
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        dragmode="pan",
        hovermode="closest",
        plot_bgcolor="white",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Semantic B-Tree")

    data_path = Path(__file__).with_name("augmented_data.json")
    initialize_records(data_path)
    embedder = Embeddings()

    t_value = st.sidebar.slider("b-tree t", 2, 50, 8)
    k_value = st.sidebar.slider("top-k", 1, 20, 5)
    render_record_controls()

    records: list[Record] = st.session_state.records
    render_records_overview(records)

    tree = get_or_build_tree(records=records, embedder=embedder, t_value=t_value)
    render_feedback(tree)

    query = st.text_input("запрос", "космический корабль и спутники")

    left, right = st.columns([1, 2])

    with left:
        render_search_results(tree, embedder, query, k_value)

    with right:
        tab1, tab2 = st.tabs(["дерево", "pca"])
        with tab1:
            render_tree_panel(tree)
        with tab2:
            render_pca_panel(tree)


if __name__ == "__main__":
    main()
