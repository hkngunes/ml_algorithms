'''
Linear Regression Logic
It is supervised learning algorithm, so there are some xs and ys to train.
The purpose is to find best fitting line in the space build by given xs and
ys. First i will try to find that line, then i will use sckit learn to find
that line.
ys = c + m * xs
Finding c and m is finding the line.
'''
"""
My logic:
While finding the line, i will use iterative approach and determine a tolerance to avoid
long execution times.
My purpose is to to minimize the difference between the total distance of the left side data
and the right side data until getting the difference under tolerance.
Initialize:
tolerance = 5% of distance of nearest data
to initialize c and m let use 2 arbitrary data and find the c and m values
of the line lay above those data.
orth_line = y = const  + -1/m * x
"""
import matplotlib.pyplot as plt

class Dot:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]

class Line:
    def __init__(self, c=0, m=0):
        self.c = c
        self.m = m

class LinReg:
    def __init__(self, data, tolerance_rate = 0.001, max_learn_count = 100):
        self.tolerance_rate = tolerance_rate
        self.max_learn_count = max_learn_count
        self.dots = []
        self.dot_avg = Dot([0, 0])
        self.dot_avg_up = Dot([0, 0])
        self.dot_avg_lo = Dot([0, 0])
        self.init_dots(data)
        self.line = Line()
        self.init_line()
        self.cost = 0
        self.update_cost()
        self.prev_cost = 0
        self.prev_m = 0
        self.prev_c = 0
        self.plt_ms = []
        self.plt_ms_costs = []
        self.plt_cs = []
        self.plt_cs_costs = []

    def init_dots(self, data):
        for dt in data:
            self.dots.append(Dot(dt))
        self.init_avg_dot()
        self.init_avg_upper_lower()

    def init_avg_dot(self):
        sum_x, sum_y = (0, 0)
        for dot in self.dots:
            sum_y += dot.y
            sum_x += dot.x
        dot_count = len(self.dots)
        self.dot_avg = Dot([sum_x/dot_count, sum_y/dot_count])

    def init_avg_upper_lower(self):
        sum_up_x, sum_up_y, sum_lo_x, sum_lo_y = (0, 0, 0, 0)
        up_counter, lo_counter = (0, 0)
        for dot in self.dots:
            if dot.x <= self.dot_avg.x:
                sum_lo_x += dot.x
                sum_lo_y += dot.y
                lo_counter += 1
                continue
            else:
                sum_up_x += dot.x
                sum_up_y += dot.y
                up_counter += 1
        self.dot_avg_lo = Dot([sum_lo_x/lo_counter, sum_lo_y/lo_counter])
        self.dot_avg_up = Dot([sum_up_x/up_counter, sum_up_y/up_counter])

    def init_line(self):
        self.line.m = (self.dot_avg_up.y - self.dot_avg_lo.y) / (self.dot_avg_up.x - self.dot_avg_lo.x)
        self.line.c = self.dot_avg.y - self.line.m * self.dot_avg.x

    def update_cost(self):
        self.prev_cost = self.cost
        self.cost = 0
        for dot in self.dots:
            pred_y = self.line.c + self.line.m * dot.x
            self.cost += (pred_y - dot.y)**2

    def update_line_m(self):
        diff_m = self.line.m - self.prev_m
        diff_cost = self.cost - self.prev_cost
        self.prev_m = self.line.m
        if diff_cost >= 0:
            self.line.m -= (diff_m / 2)
        else:
            self.line.m += diff_m
        self.line.c = self.dot_avg.y - self.line.m * self.dot_avg.x

    def update_line_c(self):
        diff_c = self.line.c - self.prev_c
        diff_cost = self.cost - self.prev_cost
        self.prev_c = self.line.c
        if diff_cost >= 0:
            self.line.c -= (diff_c / 2)
        else:
            self.line.c += diff_c

    def train_with_m(self):
        self.prev_m = self.line.m
        self.line.m *= 1.5
        self.line.c = self.dot_avg.y - self.line.m * self.dot_avg.x
        self.update_cost()
        counter = 0
        while counter < self.max_learn_count:
            counter += 1
            self.update_line_m()
            self.update_cost()
            self.plt_ms.append(self.line.m)
            self.plt_ms_costs.append(self.cost)
            if abs((self.line.m - self.prev_m)/self.line.m) < self.tolerance_rate:
                break
        print("m_train_count : " + str(counter))

    def train_with_c(self):
        self.prev_c = self.line.c
        self.line.c *= 2
        self.update_cost()
        counter = 0
        while counter < self.max_learn_count:
            counter += 1
            self.update_line_c()
            self.update_cost()
            self.plt_cs.append(self.line.c)
            self.plt_cs_costs.append(self.cost)
            if abs(self.line.c - self.prev_c) < self.tolerance_rate:
                break
        print("c_train_count : " + str(counter))

    def train_line(self):
        self.train_with_m()
        self.train_with_c()

    def plt_cost_m(self, m):
        self.line.m = m
        self.line.c = self.dot_avg.y - self.dot_avg.x * self.line.m
        self.update_cost()
        return self.cost

    def plt_m(self, color):
        x_list = list(map(lambda x:x/100, range(40,130)))
        y_list = list(map(self.plt_cost_m, x_list))
        plt.plot(x_list, y_list, color)

    def plt_cost_c(self, c):
        self.line.c = c
        self.update_cost()
        return self.cost

    def plt_c(self,color):
        x_list = list(map(lambda a: a/100, range(1, 300)))
        y_list = list(map(self.plt_cost_c, x_list))
        plt.plot(x_list, y_list, color)

data =[
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 5),
    (3, 3),
    (3, 4),
]

lin_reg = LinReg(data, tolerance_rate=0.0001)
lin_reg.plt_m('r')
lin_reg.train_with_m()
plt.plot(lin_reg.plt_ms, lin_reg.plt_ms_costs, 'b')
plt.show()
lin_reg.plt_c('r')
lin_reg.train_with_c()
plt.plot(lin_reg.plt_cs, lin_reg.plt_cs_costs, 'b')
plt.show()
