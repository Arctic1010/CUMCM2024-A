import time
import numpy as np
import matplotlib.pyplot as plt
pi = 3.1415926535
# class point:
#     x = 0
#     y = 0
#     theta = 0
#     rho = 0

def Rectangular_to_Polar(x, y):  
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta

def Polar_to_Rectangular(r, theta):  
    # theta = theta * (np.pi / 180)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def arc_len(mid):
    a = 0.55 / 2 / pi
    return a / 2 * (mid * np.sqrt(mid * mid + 1) + np.log(mid + np.sqrt(mid * mid + 1)))

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_nxt_point(x,y,otheta,len):
    # rho, theta = Rectangular_to_Polar(x, y)
    # print(rho, theta)
    theta = otheta
    l = theta - 2 * pi
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
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta

def get_prev_point(x,y,otheta,lenn):
    # rho, theta = Rectangular_to_Polar(x, y)
    # print(rho, theta)
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
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
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
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta

# v = 1m/s
# t = t
# s = vt

# print(get_nxt_point(x,y,theta),sep=",")

# plt.plot(x,y,marker=".")
# print(x,y,sep=",")
# 448 * 301
T = 0.00001

data = np.zeros((448, 301))
data_nxt = np.zeros((448, 301))
data_prev = np.zeros((448, 301))
data_theta_prev = np.zeros((224, 301))
data_theta_nxt = np.zeros((224, 301))

speed = np.zeros((224, 301))

x = 0.55 * 16
y = 0
theta = 32 * pi
org_x = x
org_y = y
org_theta = theta
for t in range(301):   
    if(t != 0):
        x = org_x
        y = org_y
        theta = org_theta
        x,y,theta = get_nxt_point(x,y,theta,1)
        org_x = x
        org_y = y
        org_theta = theta
    data[0][t] = x
    data[1][t] = y
    data_prev[0][t], data_prev[1][t], data_theta_prev[0][t] = get_prev_point(x,y,theta,T)
    data_nxt[0][t], data_nxt[1][t], data_theta_nxt[0][t] = get_nxt_point(x,y,theta,T)
    # plt.plot(x,y,marker=".")
    x,y,theta = get_prev_board(x,y,theta,3.41 - 0.55)
    data[2][t] = x
    data[3][t] = y
    data_prev[2][t], data_prev[3][t], data_theta_prev[1][t] = get_prev_board(data_prev[0][t], data_prev[1][t], data_theta_prev[0][t], 3.41 - 0.55)
    data_nxt[2][t], data_nxt[3][t], data_theta_nxt[1][t] = get_prev_board(data_nxt[0][t], data_nxt[1][t], data_theta_nxt[0][t], 3.41 - 0.55)
    # plt.plot(x,y,marker=".")
    for i in range(222):
        x,y,theta = get_prev_board(x,y,theta, 2.2 - 0.55)
        data[(i + 2) * 2][t] = x
        data[(i + 2) * 2 + 1][t] = y
        data_prev[(i + 2) * 2][t], data_prev[(i + 2) * 2 + 1][t], data_theta_prev[i + 2][t] = get_prev_board(data_prev[(i + 2) * 2 - 2][t], data_prev[(i + 2) * 2 - 1][t], data_theta_prev[i+1][t], 2.2 - 0.55)
        data_nxt[(i + 2) * 2][t], data_nxt[(i + 2) * 2 + 1][t], data_theta_nxt[i + 2][t] = get_prev_board(data_nxt[(i + 2) * 2 - 2][t], data_nxt[(i + 2) * 2 - 1][t], data_theta_prev[i+1][t], 2.2 - 0.55)
    
        # data_prev[(i + 2) * 2][t], data_prev[(i + 2) * 2 + 1][t] = get_prev_point(x,y,theta,T)
        # data_nxt[(i + 2) * 2][t], data_nxt[(i + 2) * 2 + 1][t] = get_nxt_point(x,y,theta,T)
        
        # plt.plot(x,y,marker=".")

for t in range(301):
    for i in range(224):
        # print("prev",data_prev[(i*2)][t],data_prev[i*2+1][t])
        # print("just",data[(i*2)][t],data[i*2+1][t])
        # print("nxt",data_nxt[(i*2)][t],data_nxt[i*2+1][t])
        
        # print(arc_len(data_theta_prev[i][t]),arc_len(data_theta_nxt[i][t]))
        
        speed[i][t] = (arc_len(data_theta_prev[i][t]) - arc_len(data_theta_nxt[i][t])) / T / 2
 
    # print(x,y,sep=",")
    # plt.plot(x,y,marker=".")
# for i in range(224):
#     print(f"{data[i * 2][0]}, {data[i * 2 + 1][0]}")
import pandas as pd
df = pd.DataFrame(speed)
df.to_excel("output_speed.xlsx",index=False)
print("Done.")