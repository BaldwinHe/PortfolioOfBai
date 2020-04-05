import numpy as np

class SumTree():
    pointer = 0
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.parent_size = buffer_size - 1
        self.buffer_data = np.empty(self.buffer_size,dtype=object)
        self.tree_data = np.zeros(self.parent_size + self.buffer_size)
        self.N = 0

    def add(self, priority, state):
        data_index = self.pointer + self.parent_size
        self.buffer_data[self.pointer] = state
        self.update(data_index, priority)

        self.pointer += 1
        if (self.pointer >= self.buffer_size):
            self.pointer = 0

        if (self.N < self.buffer_size):
            self.N += 1

    def update(self, data_index, priority):
        change = priority - self.tree_data[data_index]
        self.tree_data[data_index] = priority

        while( data_index != 0 ):
            data_index = ( data_index - 1) // 2
            self.tree_data[data_index] += change

    def get_leaf(self, value):
        parent_index = 0
        while(True):
            left_index = parent_index * 2 + 1
            right_index = left_index + 1

            if(left_index >= len(self.tree_data)):
                leaf_index = parent_index
                break
            else:
                if(value <= self.tree_data[left_index]):
                    parent_index = left_index
                else:
                    value -= self.tree_data[left_index]
                    parent_index = right_index
        data_index = leaf_index - self.parent_size
        return leaf_index, self.tree_data[leaf_index], self.buffer_data[data_index]

    @property
    def total(self):
        return self.tree_data[0]


