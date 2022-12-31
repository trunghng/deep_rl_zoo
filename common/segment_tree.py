import operator


class SegmentTree(object):
    '''
    Segment tree data structure implementation according to
    https://cp-algorithms.com/data_structures/segment_tree.html
    '''


    def __init__(self,
                size: int,
                operator,
                neutral_element):
        '''
        The (binary) tree is an array, with the root at index 0
        Each node, with index i, has two child nodes of index 2i, 2i + 1
        '''
        self._size = size
        self._operator = operator
        self._tree = list(range(4 * size))
        self._neutral_element = neutral_element


    def build(self, a, v: int, tl: int, tr: int) -> None:
        '''
        Parameters
        ----------
        a: input array
        v: current vertex's index
        tl: left boundary of the current segment
        tr: right boundary of the current segment
        '''
        if tl == tr:
            self._tree[v] = a[tl]
        else:
            tm = (tl + tr) // 2
            self.build(a, v * 2, tl, tm)
            self.build(a, v * 2 + 1, tm + 1, tr)
            self._tree[v] = self._tree[v * 2] + self._tree[v * 2 + 1]


    def operate(self, v: int, tl: int, tr: int, l: int, r: int):
        '''
        Parameters
        ----------
        v: current vertex's index
        tl: left boundary of the current segment
        tr: right boundary of the current segment
        l: left boundary of the query
        r: right boundary of the query
        '''
        if l > r:
            return self._neutral_number
        if l == tl and r == tr:
            return self._tree[v]
        tm = (tr + tl) // 2
        return self._operator(self.operate(v * 2, tl, tm, l, min(r, tm)),
                self.operate(v * 2 + 1, tm + 1, tr, max(l, tm  + 1), r))


    def update(self, v: int, tl: int, tr: int, pos: int, value) -> None:
        '''
        Parameters
        ----------
        v: current vertex's index
        tl: left boundary of the current segment
        tr: right boundary of the current segment
        pos: position of the element
        value: new value of the element
        '''
        if tl == tr:
            self._tree[v] = value
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(v * 2, tl, tm, pos, value)
            else:
                self.update(v * 2 + 1, tm + 1, tr, pos, value)
            self._tree[v] = self._operator(self._tree[v * 2], self._tree[v * 2 + 1])


class SumSegmentTree(SegmentTree):


    def __init__(self, size: int):
        super().__init__(size, operator.add, 0)


class MinSegmentTree(SegmentTree):


    def __init__(self, size: int):
        super().__init__(size, operator.min, float('inf'))
