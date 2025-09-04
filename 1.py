import time
import numpy as np
import matplotlib.pyplot as plt
pi = 3.1415926535
# class point:
#     x = 0
#     y = 0
#     theta = 0
#     rho = 0

def Rectangular_to_Polar(x, y):  # 直角坐标转极坐标，输出的thata为角度值
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta

def Polar_to_Rectangular(r, theta):  # 极坐标转直角坐标，输入的thata需为角度值
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
    for _ in range(30):
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
    # print(nrho, ntheta)
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    # print(nx,ny)
    return nx, ny, ntheta


# print(get_nxt_point(x,y,theta),sep=",")

# plt.plot(x,y,marker=".")
# print(x,y,sep=",")
# 448 * 301

data = np.zeros((448, 301))
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
    # plt.plot(x,y,marker=".")
    x,y,theta = get_prev_board(x,y,theta,3.41 - 0.55)
    data[2][t] = x
    data[3][t] = y
    # plt.plot(x,y,marker=".")
    for i in range(222):
        x,y,theta = get_prev_board(x,y,theta, 2.2 - 0.55)
        data[(i + 2) * 2][t] = x
        data[(i + 2) * 2 + 1][t] = y
        # plt.plot(x,y,marker=".")

    # print(x,y,sep=",")
    # plt.plot(x,y,marker=".")
# for i in range(224):
#     print(f"{data[i * 2][0]}, {data[i * 2 + 1][0]}")
import pandas as pd
df = pd.DataFrame(data)
df.to_excel("output.xlsx",index=False)
print("Done.")