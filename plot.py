#问题1 初始时刻板凳龙示意图

import matplotlib.pyplot as plt
import numpy as np
pi = 3.1415926535
head_len = 3.41 - 0.55
body_len = 2.2 - 0.55

class point:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def Polar_to_Rectangular(r, theta):  # 极坐标转直角坐标，输入的theta需为角度值
    # theta = theta * (np.pi / 180)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_prev_board(x,y,otheta,len):
    theta = otheta
    l = theta 
    r = theta + pi
    a = 0.55 / 2 / pi
    for _ in range(30):
        mid = (l + r) / 2
        tx, ty = Polar_to_Rectangular(a * mid, mid)
        if(point_distance(x, y, tx, ty) < len):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta
plt.figure(figsize=(6, 6))
line_x = []
line_y = []
for i in range(5000):
    atheta = 45 * pi / 5000 * i
    arho = atheta * 0.55 / 2 / pi
    ax, ay = Polar_to_Rectangular(arho, atheta)
    line_x.append(ax)
    line_y.append(ay)
plt.plot(line_x,line_y)

start_point = point(8.8, 0)
start_theta = 32 * pi
plt.scatter(start_point.x, start_point.y, color = 'green')

tx = start_point.x
ty = start_point.y
ttheta = start_theta

x,y,theta = get_prev_board(tx,ty,ttheta,head_len)
plt.scatter(x,y,color = 'green')
plt.plot([tx,x],[ty,y], linewidth = 3, color = 'red')

tx = x
ty = y
ttheta = theta
for i in range(222):
    x,y,theta = get_prev_board(tx,ty,ttheta,body_len)
    plt.scatter(x,y,color = 'green')
    plt.plot([tx,x],[ty,y], linewidth = 3, color = 'red')
    tx = x
    ty = y
    ttheta = theta
plt.savefig("plot_1.eps", dpi=300)
plt.show()
