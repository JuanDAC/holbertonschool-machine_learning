# 0x03. Policy Gradients

## Resources

Read or watch:

- [How Policy Gradient Reinforcement Learning Works](https://intranet.hbtn.io/rltoken/hfQFTtnxkkdO7AjwJvJb6Q)

- [Policy Gradients in a Nutshell](https://intranet.hbtn.io/rltoken/tZE8YdLjS3u2VPMGSo7UTg)

- [RL Course by David Silver - Lecture 7: Policy Gradient Methods](https://intranet.hbtn.io/rltoken/Ehf_ISuQx-hUUB21P4uMvQ)

- [Reinforcement Learning 6: Policy Gradients and Actor Critics](https://intranet.hbtn.io/rltoken/wxP1EioedlosWi-op63zLA)

- [Policy Gradient Algorithms](https://intranet.hbtn.io/rltoken/EiARIynXiIJXqw9P8o0jtg)

## Learning Objectives

- What is Policy?
- How to calculate a Policy Gradient?
- What and how to use a Monte-Carlo policy gradient?

## Requirements

### General

- Allowed editors: `vi` , `vim` , `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), and `gym` (version 0.7)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation ( `python3 -c 'print(__import__("my_module").__doc__)'` )
- All your classes should have documentation ( `python3 -c 'print(__import__("my_module").MyClass.__doc__)'` )
- All your functions (inside and outside a class) should have documentation ( `python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'` )
- All your files must be executable
- Your code should use the minimum number of operations

## Tasks

### 0. Simple Policy function

          Level: 0                    Manual review           Progress vs Score  Task Body Write a function that computes to policy with a weight of a matrix.

- Prototype: `def policy(matrix, weight):`

```bash
$ cat 0-main.py
#!/usr/bin/env python3
"""
Main file
"""
import numpy as np
from policy_gradient import policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01],
    [1.14374817e-04, 3.02332573e-01],
    [1.46755891e-01, 9.23385948e-02],
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$
$ ./0-main.py
[[0.50351642 0.49648358]]
$

```

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/0x03-policy_gradients`
- File: `policy_gradient.py`
- Code language: ` ` (project based)  
  Self-paced manual review Panel footer - Controls

### 1. Compute the Monte-Carlo policy gradient

          Level: 0                    Manual review           Progress vs Score  Task Body By using the previous function created   ` policy `  , write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

- Prototype: `def policy_gradient(state, weight):` \* `state` : matrix representing the current observation of the environment
- `weight` : matrix of random weight

- Return: the action and the gradient (in this order)

```bash
$ cat 1-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$
$ ./1-main.py
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]
$

```

_Results can be different since `weight` is randomized _
Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/0x03-policy_gradients`
- File: `policy_gradient.py`
- Code language: ` ` (project based)  
  Self-paced manual review Panel footer - Controls

### 2. Implement the training

          Level: 0                    Manual review           Progress vs Score  Task Body By using the previous function created   ` policy_gradient `  , write a function that implements a full training.

- Prototype: `def train(env, nb_episodes, alpha=0.000045, gamma=0.98):` \* `env` : initial environment
- `nb_episodes` : number of episodes used for training
- `alpha` : the learning rate
- `gamma` : the discount factor

- Return: all values of the score (sum of all rewards during one episode loop)
  Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use `end="\r", flush=False` of the print function.
  With the following main file, you should have this result plotted:
  ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/e2fff0551f5173b824a8ee1b2e67aff72d7309e2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221126T201007Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3c10f43db6fabe00f7b85901fcbe0377f579bdabbced820ded3177d727f4cdc5)

```bash
$ cat 2-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$
$ ./2-main.py

```

Results can be different we have multiple randomization
Also, we highly encourage you to play with `alpha` and `gamma` to change the trend of the plot
Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/0x03-policy_gradients`
- File: `train.py`
- Code language: ` ` (project based)  
  Self-paced manual review Panel footer - Controls

### 3. Animate iteration

          Level: 0                    Manual review           Progress vs Score  Task Body Update the prototype of the   ` train `   function by adding a last optional parameter   ` show_result `   (default:   ` False `  ).

When this parameter is `True` , render the environment every 1000 episodes computed.

```bash
$ cat 3-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$
$ ./3-main.py

```

Results can be different we have multiple randomization
Result after few episodes:
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/51a3d986d9c96960ddd0c009f7eaac5a2ce9f549.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221126T201007Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c3e372c88901c3ca377055bb4d898ebd5b5652c55c267631f2e1a88bbc8c8257)

Result after more episodes:
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/8dadd3f7918aa188cde1b5c6ac2aafddac8a081f.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221126T201007Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=651b7821193d17f69c0fdf6e364ac290b6a96c260e1095eb2d6f64644fa1a2c7)

Result after 10000 episodes:
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/da9d7deed16c5c9aec05e26bf14cf8b76e70dcce.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221126T201007Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=288ef534d1a3fd5bd3b48a2fd1d6700c86f2bf2b3aca350cc45cc4bc644699a5)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/0x03-policy_gradients`
- File: `train.py`
- Code language: ` ` (project based)  
   Self-paced manual review Panel footer - Controls
  Copyright © 2022 Holberton Inc, All rights reserved.
