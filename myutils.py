import math

class SmallIterator:
    def __init__(self,x,y,bs):
        self.x = x
        self.y = y
        self.curr = 0
        self.bs = bs
        self.tot_size = len(self.x)
        self.size = math.ceil(self.tot_size / self.bs)
        
    def __iter__(self):
        self.curr = 0
        return self

    def __len__(self):
        return self.size
    
    def __next__(self):
        if self.curr < self.tot_size:
            tmp = self.curr
            self.curr += self.bs
            return self.x[tmp:self.curr], self.y[tmp:self.curr]

        else:
            self.curr = 0
            raise StopIteration
