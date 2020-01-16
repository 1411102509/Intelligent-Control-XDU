 %���ܿ���:ר��PID
 clc                %��������д��ڵ�����
 clear              %��������ռ�����б���
 close              %�رյ�ǰ��figure����
 
 ts = 0.001;        %����ʱ��
 
 sys = tf(5.235e005,[1,87.35,1.047e004,0]); %���ݺ���
 dsys = c2d(sys,ts,'z');                    %��������ʱ��ģ��ת������ɢ��ʱ��ģ��,z�任
 [num,den] = tfdata(dsys,'v');              %�����ݺ����ķ��ӷ�ĸ�ֱ����num,den��
        
 [y0,t,x] = step(dsys,0.2);                 %����ϵͳ�Ľ�Ծ��Ӧ(0-0.5s)������������Ӧy,ģ��ʱ������t,״̬�켣x
 
 %���Ƹô��ݺ����Ľ�Ծ��Ӧ
 figure('name','��Ծ��Ӧ');
 title('��Ծ��Ӧ');
 plot([0 0.2],[1 1],'b',t,y0,'r');

 u_1 = 0.0;
 u_2 = 0.0;
 u_3 = 0.0;
 y_1 = 0;
 y_2 = 0;
 y_3 = 0;
 
 x = [0 0 0]';
 x2_1 = 0;
 
 %�ֶ��趨PID
 kp=0.6; ki=0.01; kd=0.01;
 
 error_1 = 0;
 for k = 1:1:500
    time(k) = k*ts;
    r(k)= 1.0;     %����ֵ                       
    u(k)= kp*x(1)+kd*x(2)+ki*x(3);          %PID�������

    %ר��PID���ƹ���
    if abs(x(1))>0.8                        %����1�����ֵ�����ر��
        u(k) = 0.45;
        elseif abs(x(1))>0.40
            u(k) = 0.40;
        elseif abs(x(1))>0.20 
            u(k) = 0.12;    
        elseif abs(x(1))>0.01
            u(k) = 0.10;   
    end

    if x(1)*x(2)>0||(x(2)==0)               %����2�������������    
        if abs(x(1))>=0.05 
            u(k)=u_1+2*kp*x(1); 
        else
            u(k)=u_1+0.4*kp*x(1);
        end
    end

    if (x(1)*x(2)<0&&x(2)*x2_1>0)||(x(1)==0)%����3��������ڼ�С
        u(k)=u(k);
    end

    if x(1)*x(2)<0&&x(2)*x2_1<0             %����4����ֵ��   
        if abs(x(1))>=0.05          
            u(k)=u_1+2*kp*error_1;   
        else     
            u(k)=u_1+0.6*kp*error_1;
        end
    end

    if abs(x(1))<=0.001                     %����5�����ֵ�����ر�С
        u(k)=0.5*x(1)+0.010*x(3); 
    end

    %������������
    y(k)=-den(2)*y_1-den(3)*y_2-den(4)*y_3+num(1)*u(k)+num(2)*u_1+num(3)*u_2+num(4)*u_3;
    error(k) = r(k)-y(k);     

    %��������
    u_3 =u_2;
    u_2 =u_1;
    u_1 =u(k);
    y_3 =y_2;
    y_2 =y_1;
    y_1 =y(k);  

    x(1)= error(k);                %����P
    x2_1= x(2);
    x(2)= (error(k)-error_1)/ts;   %����D
    x(3)= x(3)+error(k)*ts;        %����I

    error_1 = error(k);
 end
 %�����������
 figure(2);  
 plot(time,r,'b',time,y,'r');   
 xlabel('time(s)');ylabel('y');  
 %�����������
 figure(4);  
 plot(time,r-y,'r');   
 xlabel('time(s)');ylabel('error');
