"""token 前缀 → 解码状态快照的复用表。

旧实现的 radix 思路保留（按 token 前缀建树、命中即复用、LRU 淘汰），但保存的
不再是分页 KV 的页号，而是新架构整套池状态的快照（engine.memory.PrefixState）：
命中后由 engine 把它 restore 到 scratch cache，再用稠密 prefill 续跑剩余
prompt——数值上与整段 prefill 等价（max|Δlogit| ≈ 1e-6）。

树是"一 token 一条边"的字典树：state 挂在 prefill 的每个 stride 边界（以及
请求末尾）上，中间节点只承担路径。带状态的节点超过 max_states 时按 LRU 丢掉
最久未用的 state，并顺手剪掉因此空掉的枝干（节点数因此有界在 O(状态数 × 深度)）。
"""

from __future__ import annotations

from typing import Optional

from .memory import PrefixState, state_bytes


class _Node:
    __slots__ = ("children", "state", "depth", "stamp", "parent", "token")

    def __init__(self, depth: int, parent: Optional["_Node"] = None, token: int = -1):
        self.children: dict = {}
        self.state: Optional[PrefixState] = None
        self.depth = depth
        self.stamp = 0
        self.parent = parent
        self.token = token

    def path(self) -> list:
        """从根到本节点的 token 序列。"""
        out = []
        node = self
        while node.parent is not None:
            out.append(node.token)
            node = node.parent
        out.reverse()
        return out


class RadixPrefixCache:
    """按 token 前缀复用的状态快照表（max_states 控制 LRU 容量）。"""

    def __init__(self, max_states: int = 32, max_nodes: Optional[int] = None):
        # max_nodes 只作为兼容参数保留：节点数由"状态数 × 深度"隐式定界
        self.max_states = max(1, int(max_states))
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._root = _Node(0)
        self._n_nodes = 1
        self._n_states = 0
        self._tick = 0
        self.stats = {
            "lookups": 0,
            "hits": 0,
            "hit_tokens": 0,
            "inserts": 0,
            "cached_states": 0,
            "cached_tokens": 0,
        }

    # ------------------------------------------------------------------
    def has_room(self) -> bool:
        return self._n_states < self.max_states

    def match(self, tokens: list) -> tuple:
        """返回 (命中的最长前缀长度, 对应快照)；未命中为 (0, None)。"""
        self.stats["lookups"] += 1
        node = self._root
        best_n, best = 0, None
        for t in tokens:
            child = node.children.get(int(t))
            if child is None:
                break
            node = child
            if node.state is not None:
                best_n, best = node.depth, node.state
                node.stamp = self._now()
        if best is None:
            return 0, None
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += best_n
        return best_n, best

    def insert(self, tokens: list, state: PrefixState) -> None:
        node = self._root
        for t in tokens:
            t = int(t)
            child = node.children.get(t)
            if child is None:
                child = _Node(node.depth + 1, node, t)
                node.children[t] = child
                self._n_nodes += 1
            node = child
        if node.state is None:
            self._n_states += 1
        node.state = state
        node.stamp = self._now()
        self.stats["inserts"] += 1
        self._maybe_evict()
        self._refresh_stats()

    # ------------------------------------------------------------------
    def _now(self) -> int:
        self._tick += 1
        return self._tick

    def _state_nodes(self, root: Optional[_Node] = None) -> list:
        out = []
        stack = [root or self._root]
        while stack:
            node = stack.pop()
            if node.state is not None:
                out.append(node)
            stack.extend(node.children.values())
        return out

    def _maybe_evict(self) -> None:
        while self._n_states > self.max_states:
            nodes = self._state_nodes()
            if not nodes:
                self._n_states = 0
                break
            victim = min(nodes, key=lambda n: n.stamp)
            victim.state = None
            self._n_states -= 1
            self._prune_up(victim)

    def _prune_up(self, node: "_Node") -> None:
        """从 node 往上剪掉没有 state、也没有孩子的空枝（节点数因此有界）。"""
        while node is not self._root and node.state is None and not node.children:
            parent = node.parent
            if parent is None:
                break
            parent.children.pop(node.token, None)
            self._n_nodes -= 1
            node = parent

    def _refresh_stats(self) -> None:
        self.stats["cached_states"] = self._n_states
        self.stats["cached_tokens"] = max((n.depth for n in self._state_nodes()), default=0)

    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        return sum(state_bytes(n.state) for n in self._state_nodes())

    def __len__(self) -> int:
        return self._n_states
