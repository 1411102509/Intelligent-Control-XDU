# 用深度强化学习玩Atari游戏

## 深度Q网络

在$Q$学习算法中，若状态空间与动作空间是离散且维度不高时，可以使用$Q$表储存每个状态动作对应的Q值。当状态空间是高维空间时，由于状态太大使用表格式方法变得十分困难甚至无法完成。在这种情况，将Q表的更新转换为函数拟合问题，通过拟合一个函数代替Q表来估计Q值，使得相近状态下输出相近的动作。用于替代Q表的函数被称为动作值函数，通常简单称其为Q函数。

因此，在状态空间规模非常大时，通常将Q表的更新转换为函数拟合问题。由于神经网络的表达能力强，是用于拟合动作值函数良好选择。因此可以使用深度神经网络来拟合动作值函数来代替Q表产生Q值，使得相近的状态得到相近的输出动作。

深度学习与强化学习会存在以下问题：

- 深度学习是监督学习需要训练集，而强化学习不需要训练集，只通过环境返回回报值；同时也存在着噪声和延迟的问题；也存在很多状态的回报值都是0，也即样本稀疏的问题；
- 深度学习中通常每个样本之间是相互独立的，而强化学习中当前状态的状态值依赖后面的状态返回值；
- 使用非线性网络来表示值函数可能出现不稳定。


使用深度神经网络拟合动作值函数近似的方法称为**深度Q网络**（**Deep Q Network, DQN**）。在DQN中，对上述问题的解决方案为：

- 通过Q学习使用回报来构造标签；
- 通过**经验回放**（**experience replay**）来解决样本状态相关性以及非静态分布问题；
- 使用一个网络产生当前的Q值，而使用另一个目标网络产生对应的目标Q值。

### 状态动作值函数近似

状态动作值函数$Q(\cdot, \cdot)$的输入为状态$s$，输出 $Q(s,a),\forall a \in A$，$A$ 是动作空间。根据状态动作值函数，使用 **$\epsilon$-贪婪** 策略来选择动作。训练时，环境会产生观测，智能体根据状态动作值函数得到该观测的Q值，然后应用上述策略确定动作，环境接收到此动作后会反馈奖励及下一个观测。对于$\epsilon$-贪婪策略可以形式化的描述如下：
$$a_t = \begin{cases} \mathrm{rand(A)} & \text{if } p < \epsilon, \\
\arg \max_{a'} Q(s_t, a') & \text{otherwise}, \end{cases}$$
其中$p$是一个随机数，$\mathrm{rand(A)}$指从动作空间$A$中随机选择一个动作。

状态动作值函数神经网络的训练需要定义真值。由于环境模型未知，因此无法得到状态动作值函数的真值。在这种情况下，获得状态动作值函数的方法一般是随机采样，如蒙特卡罗采样，将每一**幕**场景（**episode**）运行一遍，然后采样得到各个状态的值函数。有终止状态的场景称为分幕式场景，其中一幕指的是从某个初始状态到终止状态的整个序列，如一局游戏从开始到结束。

最优动作值函数遵守强化学习中称为Bellman方程的等式。直观地说，如果序列$s'$在下一时间步的最优值$Q^{*}(s', a')$对所有可能动作$a'$已知，最优策略是选择使得$r+\gamma Q^{*}(s', a')$期望值最大的动作$a'$：
$$Q^{*}(s, a) = \mathbb{E}_{s'} \Big[ r + \gamma_{a'} \max_{a'} Q^{*}(s', a') \big| s, a \Big].$$

许多强化学习算法背后的基本思想都是基于Bellman方程以迭代更新的方式估计动作值函数：
$$Q_{i+1} (s, a) = \mathbb{E}_{s'} \Big[ r + \gamma_{a'} \max_{a'} Q_{i}^{*}(s', a') \big| s, a \Big].$$
该值迭代算法当$i \to \infty$时收敛到最优动作值函数，即$Q_i \to Q^{*}$。然而在实践中，这个基本方法不实用，因为动作值函数是对每一个序列单独估计的，不具备推广性。作为替代方案，常用的方法是使用函数近似来估计动作值函数，即$Q(s, a; w) \approx Q^{*}(s, a)$，式中的$w$是函数的参数。

深度Q网络可以通过最小化Bellman方程的均方误差来调整第$i$次迭代的参数$w_i$，最优目标值$r + \gamma \max_{a'} Q^{*}(s', a')$使用近似的目标值$y = r + \gamma \max_{a'} Q(s', a'; w_{i}^{-})$来替代，其中使用的参数$w_{i}^{-}$来自前面某轮迭代。这就引出了在每次迭代$i$中不断变化的损失函数序列$L_i(w_i)$：
$$L_i(w_i) = \mathbb{E}_{s, a, r} \Big[ \big( \mathbb{E}_{s'}[y|s, a] - Q(s, a; w_i) \big)^{2} \Big].$$

在Q学习中，使用 $\epsilon$-贪婪策略来生成动作 $a_{t+1}$；但用来计算状态动作值函数的是使得 $Q(s_{t+1}, a_{t+1})$ 最大的动作。这种产生行为的策略和进行评估的策略不同的方法称为**异策略**（**off-policy**）方法。DQN中也使用了异策略方法。不同的是，Q学习中用来计算目标和预测值的Q是同一个Q，即使用了相同的神经网络。这样带来的一个问题就是，每次更新神经网络时，目标网络也被更新，容易导致神经网络参数不收敛。因此DQN在原本的 Q 网络的基础上引入了一个目标Q网络，用来计算目标值。它和Q网络结构一样，初始的权重也一样，只是Q网络每次迭代都会更新，而目标Q网络每隔一段时间才会更新。

### 经验回放

Q学习是一种异策略的方法，既可以学习当前经验也可以学习过去经验。因此在学习过程中随机地加入之前的学习经验会让神经网络的训练更有效率，**经验回放**(experience replay)缓冲区记录的就是过去的学习经历。

经验回放解决了相关性问题以及非静态分布的问题。通过在每个时刻智能体与环境交互得到的状态转移样本 $(s_t, a_t, r_t, s_{t+1})$ 储存到回放缓冲区，在训练时就随机拿出来一个批次的样本，这样可以打乱状态之间的相关性。

### 算法流程

1. 首先初始化经验回放缓冲区D，它的容量为N;
2. 初始化Q网络，随机生成权重 $w$;
3. 初始化目标Q网络，权重为 $w^- = w$;
4. 循环遍历 $\mathrm{episode} = 1, 2,\dots, M$:
5. 初始化初始状态$s_1$;
6. 循环遍历 $t = 1,2,\dots, T$:
    - 用 $\epsilon$−贪婪策略生成动作 $a_t$：以 $\epsilon$ 概率选择一个随机的动作，或选则动作$a_t = \max_{a} Q(s_t,a;w^-)$;
    - 执行动作 $a_t$，接收奖励 $r_t$ 及新的状态 $s_{t+1}$;
    - 将样本 $(s_t, a_t, r_t, s_{t+1})$ 存入 $D$ 中；
    - 从 $D$ 中随机抽取一个 mini-batch 的数据 $\{ (s_j, a_j, r_j, s_{j+1})_k \}_{k=1}^{K}$；
    - 令 $y_j=r_j$，如果 $j+1$ 步是终止的话，否则，令 $y_j = r_j+\gamma \max_{a'}Q(s_{t+1},a'; w^-)$；
    - 对 $[y_j − Q(s_t,a_j; w)]^2$ 关于 $w$ 使用梯度下降法进行更新；
    - 每隔 $C$ 步更新目标Q 网络 $w^- = w$。

## 算法实现

我们将在这里复现深度强化学习中的DQN模型，论文原文：[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)。模型接收游戏的图像作为输入，采用端到端的模型直接预测下一步要执行的控制，本项目需要在GPU环境下运行。

这里的实现主要参考了百度AI Studio项目 <https://aistudio.baidu.com/aistudio/projectdetail/169455>。考虑到教学内容的需要，对代码进行了一定的修改。

### 基于PaddlePaddle的深度Q网络

深度Q网络在文件`DQN_agent.py`文件中实现为类`DQNModel`。该深度卷积神经网络的输入包括了状态(`state`)，动作(`action`)，回报(`reward`)，下一个状态(`next_s`)以及游戏是否结束(`isOver`)的信息。这里实现的神经网络结构包括四个卷积(`conv`)池化(`max pool`)层接一个全连接(`fc`)层。你可以根据你所学的深度学习知识进行修改。

下面是该类中各函数的作用：

- `__init__`：初始化类实例；
- `_get_inputs`：对网络的输入数据进行处理，使之适合于PaddlePaddle框架；
- `_build_net`：建立深度神经网络，包括了三个不同的执行程序，预测程序（对应于目标Q网络）、训练程序（对应于学习的Q网络）、以及同步程序（同步Q网络参数至目标Q网络）；
- `get_DQN_prediction`：计算目标Q网络的预测；
- `act`：根据状态输入选择合适的动作；
- `train`：训练神经网络；
- `sync_target_network`：向目标Q网络同步参数；
- `save_inference_model`：将训练得到的网络参数存储到指定目录。

### 经验回放缓冲

经验回放缓冲区在文件`expreplay.py`中实现。其中`Experience`定义为状态、动作、回报以及游戏是否结束的四元组。而类`ReplayMemory`用于管理这些经验样本。经验样本被存放于一个循环队列中，当队列填充满后，新的经验将覆盖旧的经验样本。

类`ReplayMemory`中的主要函数功能描述如下：

- `append`：向缓冲区尾填充经验样本；
- `recent_state`：获取最近的状态；
- `sample`：根据指定序号获取经验样本；
- `sample_batch`：获取一个小批次(mini-batch)的经验样本，用于训练神经网络。

### 深度Q学习的实现

首先安装并导入需要使用到的Python程序包。各个程序包的作用为：

- `gym`：强化学习的训练场，用于模拟各种环境，本次实验中用于模拟Atari 2600视频游戏；
- `atari_py`：模拟Atari 2600视频游戏，与`gym`一起提供强化学习环境；
- `expreplay`：经验回放缓冲区；
- `DQN_agent`：实现DQN模型；
- `numpy`：科学计算中的各种数组及其基本运算；
- `os`：与系统交互，文件路径管理；
- `tqdm`：显示训练的进度。

由于百度AI Studio环境没有`gym`和`tqdm`，因此需要在第一次运行本项目时执行下述脚本进行安装。包安装成功后，请重启执行器，否则无法导入新安装的程序包。

请去掉下面单元格开头的`#`，执行单元格，包安装成功再加上`#`，然后从菜单栏重启执行器。


```python
!pip install gym[atari]==0.15.4 tqdm
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Requirement already satisfied: gym[atari]==0.15.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.15.4)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (4.36.1)
    Requirement already satisfied: pyglet<=1.3.2,>=1.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (1.3.2)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (1.3.0)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (1.12.0)
    Requirement already satisfied: cloudpickle~=1.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (1.2.1)
    Requirement already satisfied: numpy>=1.10.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (1.16.4)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (4.1.1.26)
    Requirement already satisfied: Pillow; extra == "atari" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (6.2.0)
    Requirement already satisfied: atari-py~=0.2.0; extra == "atari" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym[atari]==0.15.4) (0.2.6)
    Requirement already satisfied: future in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pyglet<=1.3.2,>=1.2.0->gym[atari]==0.15.4) (0.18.0)


导入需要使用的类和库。


```python
import os
from tqdm import tqdm
import numpy as np

import gym
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from expreplay import ReplayMemory, Experience
from DQN_agent import DQNModel
```

定义函数中需要用到的一些参数。这些参数的含义将在后面用到的时候介绍。这部分参数通常使用目前给定的值而无需修改，你也可以尝试修改部分值，探索它们对算法性能的影响。


```python
MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
GAMMA = 0.99

ACTION_REPEAT = 4  # aka FRAME_SKIP
UPDATE_FREQ = 4
SYNC_TARGET_FREQ = 10000 // UPDATE_FREQ

BATCH_SIZE = 64
USE_CUDA = True
```

### 基本辅助函数

下面定义两个函数，用于辅助实现$\epsilon$-贪婪策略。

- `action_random`：从动作空间中随机选择一个动作；
- `action_policy`：根据已经学习获得的目标Q网络选择一个已经习得的策略意义下的最优动作。


```python
def action_random(env):
    action = env.action_space.sample()
    return action

def action_policy(agent, state, exp):
    context = exp.recent_state()
    context.append(state)
    context = np.stack(context, axis=0)
    action = agent.act(context)
    return action
```

下面的函数`train_agent`的功能是从经验回放缓冲区中随机抽取`BATCH_SIZE`个样本，用于训练智能体的策略网络（深度Q网络）。由于视频游戏一帧图像缺少运动信息，会导致控制策略学习的困难。为解决这个问题，将连续`CONTEXT_LEN`帧图像合并在一起作为深度Q网络的输入。


```python
def train_agent(agent, exp):
    batch_all_state, batch_action, batch_reward, batch_isOver = exp.sample_batch(BATCH_SIZE)
    batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
    batch_next_state = batch_all_state[:, 1:, :, :]
    agent.train(batch_state, batch_action, batch_reward,
                batch_next_state, batch_isOver)
```

函数`eval_agent`生成新的游戏以评估当前学到的目标Q网络的性能。为了得到性能的准确估计，该函数使用测试环境(`test_env`)玩`n_episodes`局（幕）游戏，取其平均回报。


```python
def eval_agent(agent, env, n_episodes=32):
    episode_reward = []
    for _ in range(n_episodes):
        step = 0
        total_reward = 0
        state = env.reset()
        while True:
            step += 1
            action = agent.act(state)
            state, reward, isOver, info = env.step(action)
            total_reward += reward
            if isOver:
                break
        episode_reward.append(total_reward)
    eval_reward = np.mean(episode_reward)
    return eval_reward
```

### 分幕式学习

函数`train_episode`使用一局游戏，或称为一幕数据，来训练深度Q网络。该函数是整个算法最为核心的部分。在开始，对环境进行重置启动游戏，并循环赶到游戏结束。在每一步中，首先执行$\epsilon$-贪婪策略来选择动作。执行动作后收集训练样本，放入经验回放缓冲区。在训练的前面部分，$\epsilon$(`g_epsilon`)值不断减小，以减少随机探索产生的样本。

每隔`UPDATE_FREQ`步，从经验回放缓冲区中提取样本训练深度Q网络。每训练`SYNC_TARGET_FREQ`个批次，将Q网络的参数同步到目标Q网络。若某一步后游戏结束，则该幕（局）的训练结束。

请你补充$\epsilon$-贪婪策略的实现代码。即分别把

- `action = action_policy(agent, state, exp)`
- `action = action_random(env)`

放到注释块“epsilon 贪婪”的合适位置。

请你补充训练智能和同步网络的代码，即分别把

- `agent.sync_target_network()`
- `train_agent(agent, exp)`

放到注释块“epsilon 贪婪”的合适位置。


```python
def train_episode(agent, env, exp, warmup=False):
    global g_epsilon
    global g_train_batches
    step = 0
    total_reward = 0
    state = env.reset()
    while True:
        step += 1
        # epsilon greedy action
        prob = np.random.random()
        # ======= 将 epsilon 贪婪 代码补充到这里
        if prob < g_epsilon:
            action = action_random(env)
        else:
            action = action_policy(agent, state, exp)
        # ======= 补充代码结束
        next_state, reward, isOver, _ = env.step(action)
        exp.append(Experience(state, action, reward, isOver))
        g_epsilon = max(0.1, g_epsilon - 1e-6)

        # train model
        if not warmup and len(exp) > MEMORY_WARMUP_SIZE:
            # ======= 将 训练智能体 代码补充到这里
            if step % UPDATE_FREQ == 0:
                train_agent(agent, exp)
                if g_train_batches % SYNC_TARGET_FREQ == 0:
                    agent.sync_target_network()
                g_train_batches += 1
            # ======= 补充代码结束
    
        total_reward += reward
        state = next_state
        if isOver:
            break
    return total_reward, step
```

### 创建游戏环境

在Atari 2600视频游戏中，环境给出的游戏状态是 210×160 的图像，每个像素有128种可能的颜色。这是一个非常大的输入空间。为了降低复杂度，我们将图像转换为灰度，并将其大小调整为为84×84。该操作可以使用`gym`封装的类`AtariPreprocessing`很方便的使用。我们选择的游戏的例子是`Pong`，我们需要输入环境名`PongNoFrameskip-v0`，其中`NoFrameskip`表示没有跳帧，这是使用`AtariPreProcessing`的约束。

你可以下载视频`DQN-Pong.avi`了解一下`Pong`游戏。

下面首先创建用于训练深度Q网络的环境`env`。


```python
env_name = 'PongNoFrameskip-v0'
env = gym.make(env_name)
# env = FireResetEnv(env)
env = AtariPreprocessing(env)
action_dim = env.action_space.n
```

接下来创建用于评估当前目标Q网络性能的测试环境`test_env`。由于环境每次仅产生一帧图像，而深度Q网络输入为连续`CONTEXT_LEN`帧图像，这里使用封装类`FrameStack`来累积图像。在训练环境中，图像累积是由经验回放缓冲区实现的，因此无需使用`FrameStack`。


```python
test_env = gym.make(env_name)
# test_env = FireResetEnv(test_env)
test_env = AtariPreprocessing(test_env)
test_env = FrameStack(test_env, CONTEXT_LEN)
```

你也可以尝试其它游戏。深度Q网络的优点是你无需关注游戏的细节知识，可以直接使用目前的框架进行学习就可以在大多数游戏上取得很好的效果。

### 训练深度Q网络控制Atari游戏

首先初始化主要的变量$\epsilon$-贪婪策略参数`g_epsilon`以及控制同步目标Q网络参数的变量`g_train_batches`。

生成经验回放缓冲区`exp`，该缓冲区可以容纳`MEMORY_SIZE`幅大小为`IMAGE_SIZE`的图像。

最后创建基于深度Q网络的智能体`agent`，模型处理的图像大小为`IMAGE_SIZE`，动作空间维度为`action_dim`，折扣回报系数为`GAMMA`。

参数`USE_CUDA`表明是否使用GPU进行训练。由于该算法复杂度高，需要使用GPU进行训练。


```python
g_epsilon = 1.1
g_train_batches = 0

exp = ReplayMemory(int(MEMORY_SIZE), IMAGE_SIZE, CONTEXT_LEN)
agent = DQNModel(IMAGE_SIZE, action_dim, GAMMA, CONTEXT_LEN, USE_CUDA)
```

使用随机策略进行`MEMORY_WARMUP_SIZE`步游戏对经验回放缓冲区进行热身。


```python
pbar = tqdm(total=MEMORY_WARMUP_SIZE, desc='Memory warmup')
while len(exp) < MEMORY_WARMUP_SIZE:
    total_reward, step = train_episode(agent, env, exp, warmup=True)
    pbar.update(step)
pbar.close()
```

    Memory warmup: 50329it [00:58, 863.16it/s]                             


定义训练使用游戏总步数`TOTAL_STEPS`（注意不是局数）。定义在训练中期望评估目标Q网络的次数`N_EVALS`。注意这里的`N_EVALS`与前面的`n_episodes`的差异。

在达到总的游戏步数前，反复执行分幕式学习以训练深度Q网络。每隔`TEST_EVERY_STEPS`步评估一次目标Q网络的性能，若其平均回报优于此前学到的最佳模型，则将网络参数使用`save_inference_model`记录下来。持久化的模型存储在目录`saved_models`目录下，每个模型一个文件夹，且文件夹的名称包含了第几次评估及对应的平均回报值。

#### 提示

- 在执行以下程序时，建议先**跳转到思考题**部分，根据需要完成的思考题来调整程序参数。
- 由于AI Studio的环境原因，`tqdm`的进度条可能会出现不断换行的问题。重启执行器可以避免这个问题，但是训练过程需要从头开始。
- 由于训练时间很长，需要使用GPU进行训练，请设置`USE_CUDA=True`。目前，每天运行一次AI Studio项目，百度会赠送12小时GPU训练时间(GPU算力卡点数)；在一定条件下赠送48个GPU算力卡点数。
- 如果在训练过程中你需要关闭浏览器（或计算机），请检查AI Studio中左侧工具栏中的“**设置**”，将“**当您关闭当前页面时, 您希望后台环境保持运行到**:”选择为你期望的时间。


```python
# ======= 请修改以下参数到你期望的值
N_EVALS = 5
TOTAL_STEPS = 1000000
# ======= 参数修改结束
TEST_EVERY_STEPS = TOTAL_STEPS // N_EVALS

test_flag = 0
pbar = tqdm(total=TOTAL_STEPS, desc='Training DQN agent')
total_step = 1
max_reward = None
while total_step < TOTAL_STEPS:
    # start epoch
    pbar.set_postfix(stage='TRAIN', epsilon=g_epsilon, refresh=False)
    total_reward, step = train_episode(agent, env, exp)
    total_step += step

    if total_step // TEST_EVERY_STEPS == test_flag:
        test_flag += 1
        pbar.set_postfix(stage="EVAL {}".format(test_flag), refresh=False)
        eval_reward = eval_agent(agent, test_env)
        pbar.set_postfix(stage="EVAL {} Reward {:.2f}".format(test_flag, eval_reward), refresh=False)

        if max_reward is None or eval_reward > max_reward:
            max_reward = eval_reward
            model_folder = '{}-{}-{}_{:.2f}'.format('DQN', 'Pong', test_flag, eval_reward)
            save_path = os.path.join('saved_models', model_folder)
            agent.save_inference_model(save_path)
    pbar.update(step)
pbar.close()
```

    Training DQN agent: 1001646it [3:14:50, 85.68it/s, stage=EVAL 6 Reward -5.41]                                


## 用训练好的智能体玩Atari游戏

脚本`play.py`提供了使用保存的DQN模型玩Atari游戏并将其保存为视频的功能。请使用如下命令行使用保存的智能体模型玩一下Pong游戏：
```bash
!python3 play.py --use_cuda --game {env_name} --model_path {model_path} --viz
```
其中`{env_name}`为游戏对应的环境名称，这里我们使用的是`PongNoFrameskip-v0`。`{model_path}`对应的是保存模型的路径，一个可能的路径为`saved_models/DQN-Pong-46-0.62`。`--viz`表示是否要生成视频，不加`--viz`参数将仅报告玩游戏的回报；`--viz`生成的视频会保存在`videos`目录下，请下载查看。参数`--use_cuda`表示在模型推理过程中是否使用GPU。


```python
!python3 play.py --use_cuda --game PongNoFrameskip-v0 --model_path saved_models/DQN-Pong-5_-5.59 --viz
```

    W1207 18:14:03.577538   391 device_context.cc:259] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
    W1207 18:14:03.581506   391 device_context.cc:267] device: 0, cuDNN Version: 7.3.
    eval agent: 100%|█████████████████████████████████| 1/1 [00:18<00:00, 18.80s/it]
    Average reward of epidose: -8.0


## 思考题

1. 评估模型`eval_agent`得到的回报值意味着什么？
    - 结合`Pong`游戏的特点（请下载查看视频`DQN-Pong.avi`），说明为什么刚开始训练时游戏的回报是负值？
    - 是否平均回报值达到21时，智能体才能赢得游戏？为什么？

答：enal_agent得到的回报值是智能体一局中赢的回合数-输得回合数。
1. 训练刚开始是，由于g_epsilon设置初值为1.1的原因，随即探索概率大，智能体在每一回合中赢的概率很小，所以刚开始总体一局下来，赢的回合数小于输的回合数，回报值为负值。
2. 平均回报值不需要达到21，智能体就能赢得游戏。因为一局游戏是根据谁先得到21的得分来判断的输赢的，当游戏结束时，只要回报大于0，就能说明智能体先赢得了21回合，就可能赢得游戏。


2. 训练的总步数`TOTAL_STEPS`对学到的模型性能有什么影响？请尝试增加`TOTAL_STEPS`的值，得到更好的模型。提示：在`Pong`游戏上，DQN模型的回报有可能达到21。

答：训练的总步数越多，模型的平均回报值越高，智能体的反应越灵敏，越可能赢得游戏。

3. 是否可以使用玩游戏的幕数来控制训练的过程，控制总步数与控制游戏幕数两者有何差异？结合DeepMind训练打砖块智能体使用的训练幕数来估计训练出达到人类玩家水平的智能体的总步数。

答：
1. 可以使用玩游戏的幕数来控制训练过程。控制总步数可以控制整个训练过程中智能体决策的次数，控制游戏幕数可以控制整个训练过程中一共玩游戏的局数，相比来说，控制总步数更加的精准。
2. 由于训练到1000000次左右时，回报在-5附近，故估计训练出达到人类玩家水平的智能体的总步数（回报为0）在1500000步左右

4. 你最终训练得到的智能体在评估阶段的平均回报是多少？请使用你获得的最优模型生成玩游戏的视频。请下载你所得到的最优模型的参数以及用它玩游戏的视频，并上传到项目文件夹下。

答：最终训练了1000000步，智能体再评估阶段的平均回报为-5.41。
- 视频路径 videos/1.avi
- 模型路径 saved_models/DQN-Pong-6_-5.41

5. 函数`train_episode`中有一句

```python
        g_epsilon = max(0.1, g_epsilon - 1e-6)
```

- 它的作用是什么？
- 如果去掉这句，学习算法是否还能够收敛，为什么？
- $\epsilon$的每步减小量$10^{-6}$有什么含义吗？

答：
1. 使得在训练的前面部分，g_epsilon值不断减小，以减少随机探索产生的样本。但又保证了随机探索率不会低于0.1，保证了学习算法随机性。
2. 去掉这句，算法将不收敛。因为在训练的后部分，随机探索率不断减小趋近于0，决策仅依靠智能体学到的东西，不再探索未知空间，最后决策就固定在一个值上，导致不收敛。
3. $\epsilon$的每步减小量$10^{-6}$可能和训练步数有关，按照次方法，可以使随即探索率在一个合适的速率减少，直到步数达到$10^{6}$，随机探索率才达到最小值0.1。


6. 模型评估对所学模型性能的影响
    - `N_EVALS`值对整个训练过程所需要的时间有何影响？
    - `N_EVALS`值对于所学到的模型的平均回报是否有影响，为什么？
    - 执行`eval_agent`的时机对于所学到的模型性能是否有影响，为什么？
    - 是否有更好的方法选择评估模型的时机？

答：
1. `N_EVALS`值增大会拖慢整个训练过程所需要的时间，评估Q网络性能也需要耗费大量时间，评价越多导致整个过程变慢。
2. `N_EVALS`值会对所学到的模型的平均回报产生影响，因为模型评估之后才保存的，且只保存平均回报值更高的模型，故评估次数越多，测试出平均回报值高的可能性就越高，更有几率保存的到更好的模型。
3. 执行`eval_agent`的时机对于所学到的模型性能有影响，因为在不同的时机对Q网络进行测试，所得到的模型平均回报值是不同的，在合适的时机测试，使得保存性能更好的模型几率更高。
4. 在训练的前半部分不做测试，因为回报很大几率为负值，测试浪费时间没有太大意义，在训练进入后半部分再做测试。

### 附加题

7. 算法记录了评估过程中不断改善的各个模型，尝试使用所记录的模型，对比不同平均回报的模型玩游戏的效果。

答：不同回报模型玩游戏的效果差别很大，在平均回报为负的时候，智能体很大几率接不到球，基本赢不了游戏，但平均回报越高，对局越多，智能体赢的回合数越多；虽然没有得到回报为正的模型，但猜测为正的时候可能会赢得游戏，并且模型平均回报越大，智能体出现接不到球的概率越小，比分差距越大，智能体赢的越快。

8. 游戏`Pong`需要执行`FIRE`动作才会开始。如果不执行该动作，游戏会停止在没有发球的阶段。类`FireResetEnv`封装了强制环境产生第一个动作为`FIRE`的游戏动作序列。

    - 使用和不使用`FireResetEnv`对DQN学习有何影响，为什么？

```python
class _FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)
    
def FireResetEnv(env):
    if isinstance(env, gym.Wrapper):
        baseenv = env.unwrapped
    else:
        baseenv = env
    if 'FIRE' in baseenv.get_action_meanings():
        return _FireResetEnv(env)
    return env
```

答：

9. 折扣回报系数`GAMMA`的值对算法性能有什么影响？

答：

10. 尝试使用同样的模型学习能够玩打砖块游戏的智能体。

答：

11. 对于第一次作业中的出租车问题，能够使用DQN模型来解决？

答：

## 使用`gym`的代码片段


```python
import gym
```

### 列出游戏环境的观测空间、动作空间以及各个动作的含义。


```python
env = gym.make('Pong-v4')
print(env.observation_space)
print(env.action_space)
print(env.unwrapped.get_action_meanings())
env.close()
```

    Box(210, 160, 3)
    Discrete(6)
    ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']



```python
env = gym.make('Breakout-v4')
print(env.observation_space)
print(env.action_space)
print(env.unwrapped.get_action_meanings())
env.close()
```

    Box(210, 160, 3)
    Discrete(4)
    ['NOOP', 'FIRE', 'RIGHT', 'LEFT']


### 列出`gym`支持的环境名称


```python
names = [env.id for env in gym.envs.registry.all()]
list(filter(lambda x: x.find('Pong') >= 0, names))
```




    ['Pong-v0',
     'Pong-v4',
     'PongDeterministic-v0',
     'PongDeterministic-v4',
     'PongNoFrameskip-v0',
     'PongNoFrameskip-v4',
     'Pong-ram-v0',
     'Pong-ram-v4',
     'Pong-ramDeterministic-v0',
     'Pong-ramDeterministic-v4',
     'Pong-ramNoFrameskip-v0',
     'Pong-ramNoFrameskip-v4']



### 使用`Monitor`录制视频

使用`Monitor`是录制视频最便捷的方案。由于百度AI Studio环境中没有提供`ffmpeg`，因此只能使用`opencv`提供的功能以编程方式实现图像到视频的转换。


```python
from gym.wrappers import Monitor

env_name = 'Pong-v0'
env = gym.make(env_name)
env = Monitor(env, './monitor', force=True)

obs = env.reset()
for _ in range(1024):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='rgb_array')
    if done:
        break
env.close()
```


    ---------------------------------------------------------------------------

    DependencyNotInstalled                    Traceback (most recent call last)

    <ipython-input-20-245ae340f58d> in <module>
          5 env = Monitor(env, './monitor', force=True)
          6 
    ----> 7 obs = env.reset()
          8 for _ in range(1024):
          9     action = env.action_space.sample()


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitor.py in reset(self, **kwargs)
         37         self._before_reset()
         38         observation = self.env.reset(**kwargs)
    ---> 39         self._after_reset(observation)
         40 
         41         return observation


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitor.py in _after_reset(self, observation)
        186         self.stats_recorder.after_reset(observation)
        187 
    --> 188         self.reset_video_recorder()
        189 
        190         # Bump *after* all reset activity has finished


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitor.py in reset_video_recorder(self)
        207             enabled=self._video_enabled(),
        208         )
    --> 209         self.video_recorder.capture_frame()
        210 
        211     def _close_video_recorder(self):


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitoring/video_recorder.py in capture_frame(self)
        114                 self._encode_ansi_frame(frame)
        115             else:
    --> 116                 self._encode_image_frame(frame)
        117 
        118     def close(self):


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitoring/video_recorder.py in _encode_image_frame(self, frame)
        160     def _encode_image_frame(self, frame):
        161         if not self.encoder:
    --> 162             self.encoder = ImageEncoder(self.path, frame.shape, self.frames_per_sec)
        163             self.metadata['encoder_version'] = self.encoder.version_info
        164 


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/gym/wrappers/monitoring/video_recorder.py in __init__(self, output_path, frame_shape, frames_per_sec)
        253             self.backend = 'ffmpeg'
        254         else:
    --> 255             raise error.DependencyNotInstalled("""Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")
        256 
        257         self.start()


    DependencyNotInstalled: Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.



```python
!ls -l monitor/*.mp4
```
