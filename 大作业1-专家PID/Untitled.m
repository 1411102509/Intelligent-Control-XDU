 %智能控制:专家PID
 clc                %清除命令行窗口的内容
 clear              %清除工作空间的所有变量
 close              %关闭当前的figure窗口
 
 ts = 0.001;        %采样时间
 
 sys = tf(5.235e005,[1,87.35,1.047e004,0]); %传递函数
 dsys = c2d(sys,ts,'z');                    %将连续的时间模型转换成离散的时间模型,z变换
 [num,den] = tfdata(dsys,'v');              %将传递函数的分子分母分别放入num,den中
        
 [y0,t,x] = step(dsys,0.2);                 %计算系统的阶跃响应(0-0.5s)，返回输入响应y,模拟时间向量t,状态轨迹x
 
 %绘制该传递函数的阶跃响应
 figure('name','阶跃响应');
 title('阶跃响应');
 plot([0 0.2],[1 1],'b',t,y0,'r');

 u_1 = 0.0;
 u_2 = 0.0;
 u_3 = 0.0;
 y_1 = 0;
 y_2 = 0;
 y_3 = 0;
 
 x = [0 0 0]';
 x2_1 = 0;
 
 %手动设定PID
 kp=0.6; ki=0.01; kd=0.01;
 
 error_1 = 0;
 for k = 1:1:500
    time(k) = k*ts;
    r(k)= 1.0;     %期望值                       
    u(k)= kp*x(1)+kd*x(2)+ki*x(3);          %PID控制输出

    %专家PID控制规则
    if abs(x(1))>0.8                        %规则1：误差值本身特别大
        u(k) = 0.45;
        elseif abs(x(1))>0.40
            u(k) = 0.40;
        elseif abs(x(1))>0.20 
            u(k) = 0.12;    
        elseif abs(x(1))>0.01
            u(k) = 0.10;   
    end

    if x(1)*x(2)>0||(x(2)==0)               %规则2：误差趋于增大    
        if abs(x(1))>=0.05 
            u(k)=u_1+2*kp*x(1); 
        else
            u(k)=u_1+0.4*kp*x(1);
        end
    end

    if (x(1)*x(2)<0&&x(2)*x2_1>0)||(x(1)==0)%规则3：误差趋于减小
        u(k)=u(k);
    end

    if x(1)*x(2)<0&&x(2)*x2_1<0             %规则4：峰值点   
        if abs(x(1))>=0.05          
            u(k)=u_1+2*kp*error_1;   
        else     
            u(k)=u_1+0.6*kp*error_1;
        end
    end

    if abs(x(1))<=0.001                     %规则5：误差值本身特别小
        u(k)=0.5*x(1)+0.010*x(3); 
    end

    %计算输出和误差
    y(k)=-den(2)*y_1-den(3)*y_2-den(4)*y_3+num(1)*u(k)+num(2)*u_1+num(3)*u_2+num(4)*u_3;
    error(k) = r(k)-y(k);     

    %参数更新
    u_3 =u_2;
    u_2 =u_1;
    u_1 =u(k);
    y_3 =y_2;
    y_2 =y_1;
    y_1 =y(k);  

    x(1)= error(k);                %计算P
    x2_1= x(2);
    x(2)= (error(k)-error_1)/ts;   %计算D
    x(3)= x(3)+error(k)*ts;        %计算I

    error_1 = error(k);
 end
 %绘制输出曲线
 figure(2);  
 plot(time,r,'b',time,y,'r');   
 xlabel('time(s)');ylabel('y');  
 %绘制误差曲线
 figure(4);  
 plot(time,r-y,'r');   
 xlabel('time(s)');ylabel('error');
