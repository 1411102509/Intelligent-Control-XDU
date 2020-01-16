import random
import numpy as np

class CAR:
    def __init__(self,i): # d:[0,90] v:[20,40]
        super().__init__()
        self.name = i
        self.v = random.randint(20,40)
        self.v_new = 0
        self.d = random.randint(0,90)
        self.cross_time = 0

    def action(self,act):
        self.v_new = (1 + act*0.4)*self.v

    def caclu_time(self):
        self.cross_time = (90-self.d)/self.v + (110 + self.d)/self.v_new
        return self.cross_time
    
    def print(self):
        print("car_%d:(%.2f,%.2f)"%(self.name,self.d,self.v),"cross_time:",self.cross_time)   #(d,v)

def judge_crash(cars):
    crash = 0
    # car_1与car_2
    t_1 = (cars[0].d+15)/cars[0].v_new
    t_2 = (cars[1].d+5 )/cars[1].v_new
    if(abs(t_1-t_2)<1):
        crash = 1
    # car_1与car_4
    t_1 = (cars[0].d+5 )/cars[0].v_new
    t_4 = (cars[3].d+15)/cars[3].v_new
    if(abs(t_1-t_4)<1):
        crash = 1
    # car_3与car_2
    t_3 = (cars[2].d+5 )/cars[2].v_new
    t_2 = (cars[1].d+15)/cars[1].v_new
    if(abs(t_3-t_2)<1):
        crash = 1
    # car_3与car_4
    t_3 = (cars[2].d+15)/cars[2].v_new
    t_4 = (cars[3].d+5 )/cars[3].v_new
    if(abs(t_3-t_4)<1):
        crash = 1

    return crash

def traffic_game():
    # 随机初始化车辆信息
    cars = [CAR(i+1) for i in range(4)]
    # 问题1，指定数据
    # cars[0].d,cars[0].v = 45,23
    # cars[1].d,cars[1].v = 71,24
    # cars[2].d,cars[2].v = 47,28
    # cars[3].d,cars[3].v = 76,28
    
    for car in cars:
        car.print()

    # 所有决策
    all_actions = []
    for i in [1,0,-1]:
        for j in[1,0,-1]:
            all_actions.append([i,j])
    # print(all_actions)

    # 开始遍历所有的决策组合
    loss_array = [[0 for j in range(9)] for i in range(9)]
    i = j = 0
    for player_1 in all_actions:
        for player_2 in all_actions:
            # player_1决策,控制2，4
            cars[1].action(player_1[0])
            cars[3].action(player_1[1])
            # player_2决策，控制1，3
            cars[0].action(player_2[0])
            cars[2].action(player_2[1])
            
            # 判断是否撞击
            if judge_crash(cars):
                time_loss = float('inf')
            else:
                time_loss = 0
                for car in cars:
                    time_loss += car.caclu_time()

            loss_array[i][j] = time_loss/4

            j += 1
        i += 1
        j = 0

    # 传统方式的通过时间
    t_old = max(200/cars[0].v,200/cars[2].v) + max(200/cars[1].v,200/cars[3].v)
    return all_actions,loss_array,t_old       



if __name__ == "__main__":
    for i in range(10):
        print("*****************",i,"********************")
        actions,loss,time_old = traffic_game()

        # 找出最小决策组合
        min_loss = float('inf')
        player1_action = player2_action = []
        print("损失矩阵：")
        for i in range(len(loss)):
            for j in range(len(loss[i])):
                if min_loss > loss[i][j]:
                    min_loss = loss[i][j]
                    player1_action = actions[i]
                    player2_action = actions[j]
                print("%6.2f"%loss[i][j],end=' ')
            print('')
        
        print("博弈方法：min = %6.2f \nplayer1决策："%min_loss,player1_action," player2决策：",player2_action)
        print("\n传统方法：min_old = %6.2f"%time_old)
            