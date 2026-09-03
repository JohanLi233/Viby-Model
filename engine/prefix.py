"""页对齐 radix 前缀树：只在 page 边界分叉，部分命中最多浪费 page_size-1 个 token。"""

from __future__ import annotations

from .memory import PagePool


class _Node:
    __slots__ = ("children", "tokens", "pages", "last_access")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.tokens: list[int] = []
        self.pages: list[int] = []
        self.last_access: int = 0


class RadixPrefixCache:
    def __init__(self, page_size: int, pool: PagePool):
        self.page_size = int(page_size)
        self.pool = pool
        self.root = _Node()
        self._tick = 0

    def match(self, tokens: list[int]) -> tuple[int, list[int]]:
        """返回 (命中 token 数, 覆盖这些 token 的 page id)。"""
        self._tick += 1
        node = self.root
        i = 0
        pages: list[int] = []
        n = len(tokens)
        ps = self.page_size
        while i < n:
            key = tokens[i]
            child = node.children.get(key)
            if child is None:
                break
            edge = child.tokens
            k = 0
            while k < len(edge) and i + k < n and tokens[i + k] == edge[k]:
                k += 1
            child.last_access = self._tick
            if k == len(edge):
                pages.extend(child.pages)
                i += k
                node = child
                continue
            n_full = (k // ps) * ps
            if n_full == 0:
                break
            pages.extend(child.pages[: n_full // ps])
            i += n_full
            break
        return i, list(pages)

    def insert(self, tokens: list[int], pages: list[int]) -> None:
        if not tokens or not pages:
            return
        ps = self.page_size
        expect = (len(tokens) + ps - 1) // ps
        if len(pages) < expect:
            raise ValueError("insert pages 不足以覆盖 tokens")
        pages = pages[:expect]
        self._tick += 1
        node = self.root
        i = 0
        pcur = 0
        n = len(tokens)
        while i < n:
            key = tokens[i]
            child = node.children.get(key)
            if child is None:
                leaf = _Node()
                leaf.tokens = list(tokens[i:])
                leaf.pages = list(pages[pcur:])
                leaf.last_access = self._tick
                self.pool.retain(leaf.pages)
                node.children[key] = leaf
                return
            edge = child.tokens
            k = 0
            while k < len(edge) and i + k < n and tokens[i + k] == edge[k]:
                k += 1
            child.last_access = self._tick
            if k == len(edge):
                i += k
                pcur += len(child.pages)
                node = child
                continue
            n_full = (k // ps) * ps
            if n_full == 0:
                # 页内分叉：不拆共享页，剩余当作新叶（调用方会自己 prefill）
                return
            mid = self._split_child(node, child, n_full)
            i += n_full
            pcur += n_full // ps
            node = mid
        # 已全部走完已有边，若还有多出来的 pages 不应发生

    def _split_child(self, parent: _Node, child: _Node, n_full: int) -> _Node:
        ps = self.page_size
        n_pages = n_full // ps
        mid = _Node()
        mid.tokens = child.tokens[:n_full]
        mid.pages = child.pages[:n_pages]
        mid.last_access = self._tick
        rest = _Node()
        rest.tokens = child.tokens[n_full:]
        rest.pages = child.pages[n_pages:]
        rest.children = child.children
        rest.last_access = child.last_access
        self.pool.retain(mid.pages)
        # child 的 pages 原树引用改挂到 mid+rest：先释放 child 整段，再 retain rest
        self.pool.free_pages(child.pages)
        self.pool.retain(rest.pages)
        parent.children[mid.tokens[0]] = mid
        if rest.tokens:
            mid.children[rest.tokens[0]] = rest
        return mid

    def reset(self) -> None:
        pages: list[int] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            pages.extend(n.pages)
            stack.extend(n.children.values())
        self.pool.free_pages(pages)
        self.root = _Node()
