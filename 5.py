import time
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import trange,tqdm
pi = 3.1415926535
A = 1.7
arc_circle_1 = 9.080829937621441
arc_circle_2 = 9.080829937621441 / 2
radius_circle_1 = 3.005417667789
radius_circle_2 = 3.005417667789 / 2

class point:
    x = 0
    y = 0
    def __init__(self,x):
        self.x = x[0]
        self.y = x[1]

def Rectangular_to_Polar(x, y):
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta

def Polar_to_Rectangular(r, theta):
    # theta = theta * (np.pi / 180)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
# 绕 pointx,pointy 逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return nRotatex, nRotatey
# 绕 pointx,pointy 顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return sRotatex,sRotatey

def arc_len(mid):
    a = A / 2 / pi
    arc_in = a / 2 * (mid * np.sqrt(mid * mid + 1) + np.log(mid + np.sqrt(mid * mid + 1)))
    return arc_in

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_nxt_point(x,y,otheta,len):
    # rho, theta = Rectangular_to_Polar(x, y)
    # print(rho, theta)
    theta = otheta
    l = 0
    r = theta
    a = A / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(theta) - arc_len(mid) < len):
            r = mid
        else:
            l = mid
    ntheta = l
    nrho = a * ntheta
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta

def get_prev_point(x,y,otheta,lenn):
    # rho, theta = Rectangular_to_Polar(x, y)
    # print(rho, theta)
    theta = otheta
    l = theta 
    r = theta + 200 * pi
    a = A / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(mid) - arc_len(theta) < lenn):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta

def get_prev_board(x,y,otheta,len):
    theta = otheta
    l = theta 
    r = theta + pi
    a = A / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        tx, ty = Polar_to_Rectangular(a * mid, mid)
        if(point_distance(x, y, tx, ty) < len):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta

A_point = point((-2.7118558637066594, -3.591077522761074))
H = point((-0.7600091166555,-1.3057264263462)) # 大圆圆心
G = point((1.7359324901811, 2.4484019745536)) # 小圆圆心
C = point((0.9039519545689, 1.1970258409204)) # 两圆相切处
A_r = np.sqrt(A_point.x * A_point.x + A_point.y * A_point.y)
A_theta = 2 * pi * A_r / A
# print(arc_len(A_theta)) 

ax, ay, start_theta= get_prev_point(A_point.x,A_point.y,A_theta,100)
# print(arc_len(start_theta))
 
def position(x):
    if x < 0:
        return get_prev_point(ax, ay, start_theta, -x)[0:2]
    elif x == 0:
        return ax, ay
    elif x <= 100:
        return get_nxt_point(ax, ay, start_theta, x)[0:2]
    elif x > 100 and x <= 100 + arc_circle_1:
        len = x - 100
        return Srotate(len / radius_circle_1, A_point.x, A_point.y, H.x, H.y)
    elif x > 100 + arc_circle_1 and x <= 100 + arc_circle_1 + arc_circle_2:
        len = x - (100 + arc_circle_1)
        return Nrotate(len / radius_circle_2, C.x, C.y, G.x, G.y)
    else:
        t_x, t_y, t_theta = get_prev_point(A_point.x, A_point.y, A_theta, x - (100 + arc_circle_1 + arc_circle_2))
        return -t_x, -t_y#, t_theta

def get_prev_board_position(pos,len):
    U = 0.1
    t_pos = pos - U
    pos_x, pos_y = position(pos)
    while True:
        t_x1, t_y1 = position(t_pos)
        t_x2, t_y2 = position(t_pos - U)
        
        if (point_distance(pos_x, pos_y, t_x1, t_y1) - len) * (point_distance(pos_x, pos_y, t_x2, t_y2) - len) <= 0:
            L = t_pos - U
            R = t_pos
            for _ in range(25):
                mid = (L + R) / 2
                mid_x, mid_y = position(mid)
                if(point_distance(pos_x, pos_y, mid_x, mid_y) < len):
                    R = mid
                else:
                    L = mid
            # print(f"binary search result: L = {t_pos-U}, R = {t_pos}, ANS = {L}")
            return L
        t_pos -= U

class data:
    t = -999
    pos = point((0,0))
    nxt = 0
    prev = 0
    speed = 0
    def __init__(self,t,x,a,b):
        self.t = t
        self.pos = x
        self.prev = a
        self.nxt = b

T = 0.0001

# x = ax
# y = ay
# theta = start_theta

# org_x = x
# org_y = y
# org_theta = theta
# data_array = []

def check(head_speed):
    t_init = 113.62124490643217
    delta_t = 0
    max_speed = 0
    while delta_t <= 3:
        data_array = []
        t = t_init + delta_t
        
        x, y = position(t)
        data_array.append(data(0, point((x, y)), t + head_speed * T, t - head_speed * T))
        t = get_prev_board_position(t, 3.41 - 0.55)
        x, y = position(t)
        prev_data = data_array[len(data_array) - 1]
        
        speed = (get_prev_board_position(prev_data.prev, 3.41 - 0.55) - get_prev_board_position(prev_data.nxt, 3.41 - 0.55)) / T / 2
        # print(speed)
        if(speed > 2):
            print(f"head speed = {head_speed}, t = {t} exceeded, false")
            return False
        max_speed = max(max_speed, speed)
        # data_array.append(data(Time, point((x, y)),
        #                 get_prev_board_position(prev_data.prev, 2.2 - 0.55), 
        #                 get_prev_board_position(prev_data.nxt, 2.2 - 0.55)))
        delta_t += 0.001
    print(f"head speed = {head_speed}, OK, true, max speed = {max_speed} < 2")
    return True

# L = 1
# R = 1.5 # 第一次二分
L = 1.15
R = 1.35

# check(1)
# exit(0)
for _ in range(30):
    mid = (L + R) / 2
    print(f"binary search L = {L}, R = {R}")
    if check(mid):
        L = mid
    else:
        R = mid

print(L) # 1.2838430792093278

# for t in trange(0, 3):
#     Time = t
#     t *= 1.2
#     x, y = position(t)
#     data_array.append(data(Time, point((x, y)), t + 1.2*T, t - 1.2*T))
    
#     t = get_prev_board_position(t, 3.41 - 0.55)
#     x, y = position(t)
#     prev_data = data_array[len(data_array) - 1]
    
#     data_array.append(data(Time, point((x, y)), 
#                            get_prev_board_position(prev_data.prev, 3.41 - 0.55), 
#                            get_prev_board_position(prev_data.nxt, 3.41 - 0.55)))
    
#     for i in range(222):
#         t = get_prev_board_position(t, 2.2 - 0.55)
#         x, y = position(t)
#         prev_data = data_array[len(data_array) - 1]
#         data_array.append(data(Time, point((x, y)),
#                         get_prev_board_position(prev_data.prev, 2.2 - 0.55), 
#                         get_prev_board_position(prev_data.nxt, 2.2 - 0.55)))

# for i in trange(len(data_array)):
#     # print(data_array[i].t, data_array[i].prev,data_array[i].nxt, (data_array[i].prev-data_array[i].nxt)/T/2)
#     data_array[i].speed = (data_array[i].prev - data_array[i].nxt) / T / 2

# result_xlsx = np.zeros((448,201))
# speed_xlsx = np.zeros((224,201))

# for i in range(len(data_array)):
#     ans = data_array[i]
#     result_xlsx[i * 2 - ans.t * 448][ans.t] = ans.pos.x
#     result_xlsx[i * 2 + 1 - ans.t * 448][ans.t] = ans.pos.y
#     speed_xlsx[i - ans.t * 224][ans.t] = ans.speed
        
# import pandas as pd
# df = pd.DataFrame(result_xlsx)
# df.to_excel("output_4_1_2.xlsx",index=False)
# print("Done. (1 / 2)")
# df = pd.DataFrame(speed_xlsx)
# df.to_excel("output_4_speed_1_2.xlsx",index=False)
# print("Done. (2 / 2)")
