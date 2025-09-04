import time
import numpy as np
import matplotlib.pyplot as plt
pi = 3.1415926535

class point:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y

class rect:
    p = [point(0,0),point(0,0),point(0,0),point(0,0),point(0,0)]

    def __init__(self,a,b,c,d,e,f,g,h):
        self.p = [point(a,b),point(c,d),point(e,f),point(g,h),point(a,b)]
        
    def print_point(self):
        print("Rect{")
        for i in range(4):
            print(f"({self.p[i].x},{self.p[i].y})")
        print("}")

class vec:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y

def normalize(v):
    len = np.sqrt(v.x*v.x+v.y*v.y)
    return vec(v.x / len, v.y / len)

def Rectangular_to_Polar(x, y): 
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta

def Polar_to_Rectangular(r, theta): 
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def arc_len(mid):
    a = 0.55 / 2 / pi
    return a / 2 * (mid * np.sqrt(mid * mid + 1) + np.log(mid + np.sqrt(mid * mid + 1)))

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_nxt_point(x,y,otheta,len):
    theta = otheta
    l = 0
    r = theta
    a = 0.55 / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(theta) - arc_len(mid) < len):
            r = mid
        else:
            l = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_point(x,y,otheta,lenn):
    theta = otheta
    l = theta 
    r = theta + 2 * pi
    a = 0.55 / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(mid) - arc_len(theta) < lenn):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_board(x,y,otheta,len):
    theta = otheta
    l = theta 
    r = theta + pi
    a = 0.55 / 2 / pi
    for _ in range(50):
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

def get_rectangle(x,y,a,b):
    line_vec = normalize(vec(x-a,y-b))
    up_vec = normalize(vec(line_vec.y, -line_vec.x))
    
    point1_x = x + line_vec.x * 0.275 + up_vec.x * 0.15
    point1_y = y + line_vec.y * 0.275 + up_vec.y * 0.15
    
    point2_x = x + line_vec.x * 0.275 + up_vec.x * -0.15
    point2_y = y + line_vec.y * 0.275 + up_vec.y * -0.15
    
    point3_x = a + line_vec.x * -0.275 + up_vec.x * -0.15
    point3_y = b + line_vec.y * -0.275 + up_vec.y * -0.15
    
    point4_x = a + line_vec.x * -0.275 + up_vec.x * 0.15
    point4_y = b + line_vec.y * -0.275 + up_vec.y * 0.15
    
    return rect(point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y)

def quick_judge(a,b,c,d): # 快速排斥，不相交返回 False，不能判断不相交返回 True
    if (max(a.x,b.x) < min(c.x,d.x) or
        max(c.x,d.x) < min(a.x,b.x) or
        max(a.y,b.y) < min(c.y,d.y) or
        max(c.y,d.y) < min(a.y,b.y)) :
        return False
    else:
        return True

def xmult(a,b,c,d): # 叉乘
    vectorAx = b.x - a.x
    vectorAy = b.y - a.y
    vectorBx = d.x - c.x
    vectorBy = d.y - c.y
    return (vectorAx * vectorBy - vectorAy * vectorBx)
def cross(a,b,c,d):
    if not quick_judge(a,b,c,d):
        return False
    xmult1 = xmult(c,d,c,a)
    xmult2 = xmult(c,d,c,b)
    xmult3 = xmult(a,b,a,c)
    xmult4 = xmult(a,b,a,d)
    if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
        return True
    else:
        return False

def collide_detect(s,t):
    for i in range(4):
        for j in range(4):
            if(cross(s.p[i], s.p[i + 1], t.p[j], t.p[j + 1])):
                return True
    return False

def check(s):
    data = np.zeros((448))
    rect_array = []
    x = 0.55 * 16
    y = 0
    theta = 32 * pi
    x,y,theta = get_nxt_point(x,y,theta,s)
    data[0] = x
    data[1] = y
    # print(s,x,y,theta)
    x,y,theta = get_prev_board(x,y,theta,3.41 - 0.55)
    data[2] = x
    data[3] = y
    rect_array.append(get_rectangle(data[0],data[1],data[2],data[3]))
    # print(len(rect_array),rect_array[0].p[0].x)
    # rect_array[0].print_point() 
    for i in range(222):
        x,y,theta = get_prev_board(x,y,theta, 2.2 - 0.55)
        data[(i + 2) * 2] = x
        data[(i + 2) * 2 + 1] = y
        rect_array.append(get_rectangle(data[(i+2)*2-2],data[(i+2)*2-1],data[(i+2)*2],data[(i+2)*2+1]))
    
    # for i in rect_array:
    #     i.print_point()
    
    for i in range(len(rect_array)):
        for j in range(i+2, len(rect_array)):
            if(collide_detect(rect_array[i],rect_array[j])):
                # print("true")
                return True
    # print("false")
    return False

L = 0
R = arc_len(32*pi)
# print(R)

for _ in range(50):
    mid = (L + R) / 2
    if(check(mid)):
        R = mid
    else:
        L = mid

print(L)

import random 
print("[L - 10, L) test:") # 稳定性测试 1
for _ in range(1, 101):
    test_mid = L - 10 + random.random() * 10
    if check(test_mid):
        print(f"#{_} test = {test_mid}, collided, model failed.")
        exit(0)
    else:
        print(f"#{_} test = {test_mid}, pass")

print("[L - 1, L) test:") # 稳定性测试 2
for _ in range(1, 101):
    test_mid = L - 1 + random.random() * 1
    if check(test_mid):
        print(f"#{_} test = {test_mid}, collided, model failed.")
        exit(0)
    else:
        print(f"#{_} test = {test_mid}, pass")

data = np.zeros((448))
data_nxt = np.zeros((448))
data_prev = np.zeros((448))
data_theta_prev = np.zeros((224))
data_theta_nxt = np.zeros((224))
speed = np.zeros((224))

rect_array = []

x = 0.55 * 16
y = 0
theta = 32 * pi
T = 0.00001

x,y,theta = get_nxt_point(x,y,theta,L)
data[0] = x
data[1] = y
data_prev[0], data_prev[1], data_theta_prev[0] = get_prev_point(x,y,theta,T)
data_nxt[0], data_nxt[1], data_theta_nxt[0] = get_nxt_point(x,y,theta,T)
x,y,theta = get_prev_board(x,y,theta,3.41 - 0.55)
data[2] = x
data[3] = y
data_prev[2], data_prev[3], data_theta_prev[1] = get_prev_board(data_prev[0], data_prev[1], data_theta_prev[0], 3.41 - 0.55)
data_nxt[2], data_nxt[3], data_theta_nxt[1] = get_prev_board(data_nxt[0], data_nxt[1], data_theta_nxt[0], 3.41 - 0.55)
rect_array.append(get_rectangle(data[0],data[1],data[2],data[3]))
for i in range(222):
    x,y,theta = get_prev_board(x,y,theta, 2.2 - 0.55)
    data[(i + 2) * 2] = x
    data[(i + 2) * 2 + 1] = y
    data_prev[(i + 2) * 2], data_prev[(i + 2) * 2 + 1], data_theta_prev[i + 2] = get_prev_board(data_prev[(i + 2) * 2 - 2], data_prev[(i + 2) * 2 - 1], data_theta_prev[i+1], 2.2 - 0.55)
    data_nxt[(i + 2) * 2], data_nxt[(i + 2) * 2 + 1], data_theta_nxt[i + 2] = get_prev_board(data_nxt[(i + 2) * 2 - 2], data_nxt[(i + 2) * 2 - 1], data_theta_prev[i+1], 2.2 - 0.55)
    rect_array.append(get_rectangle(data[(i+2)*2-2],data[(i+2)*2-1],data[(i+2)*2],data[(i+2)*2+1]))

rect_array[0].print_point()
rect_array[8].print_point()

for i in range(224):
    speed[i] = (arc_len(data_theta_prev[i]) - arc_len(data_theta_nxt[i])) / T / 2

data_xlsx = np.zeros((224,3))

for i in range(224):
    data_xlsx[i][0] = data[i * 2]
    data_xlsx[i][1] = data[i * 2 + 1]
    data_xlsx[i][2] = speed[i]

import pandas as pd
df = pd.DataFrame(data_xlsx)
df.to_excel("output2.xlsx",index=False)
print("Done.")
