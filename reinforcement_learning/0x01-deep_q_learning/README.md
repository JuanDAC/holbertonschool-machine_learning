Curriculum Specializations Average: 67.44%\* [

                  Foundations


                  Average: 103.37%

](https://intranet.hbtn.io/curriculums/1/observe)

- [Specializations Average: 67.44%](https://intranet.hbtn.io/curriculums/2/observe)

              You are currently connected as Student            Switch to:            [Staff](https://intranet.hbtn.io/users/switch_viewing_as_permission_group?group=staff)

|  # 0x01. Deep Q-learning

## Details

By: Alexa Orrico, Software Engineer at Holberton School Weight: 6Project over - took place fromOct 26, 2022 12:00 AMto Nov 4, 2022 12:00 AMManual QA review must be done(request it when you are done with the project) ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/8/9239a27ccd609cb9092aba0e6bb55ba7b5cf0b6b.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221107T182424Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=be0b46e8827b8aabcd4b148e4fb9c58e1a707a68592f77683c4db5aef7bf95c6)

## Resources

Read or watch :

- [Deep Q-Learning - Combining Neural Networks and Reinforcement Learning](https://intranet.hbtn.io/rltoken/vf8M2yFL9vWcFftBWFG2KQ)

- [Replay Memory Explained - Experience for Deep Q-Network Training](https://intranet.hbtn.io/rltoken/LciKBr548xY_iD4QkUatNw)

- [Training a Deep Q-Network - Reinforcement Learning](https://intranet.hbtn.io/rltoken/ZwReaNdr4Ei4GxWr-56oFg)

- [Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning](https://intranet.hbtn.io/rltoken/xAP3VzSnw0HLwjrBRn46Xw)

References :

- [Setting up anaconda for keras-rl](https://intranet.hbtn.io/rltoken/Q8hBeid5HHPA_YToSl5evg)

- [keras-rl](https://intranet.hbtn.io/rltoken/mSQhyiu7FEaFi_qTft1G2w)
- [rl.policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py)

- [rl.memory](https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py)

- [rl.agents.dqn](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)

- [Playing Atari with Deep Reinforcement Learning](https://intranet.hbtn.io/rltoken/SekcqEIbg0hxdEvoQSB-kA)

## Learning Objectives

- What is Deep Q-learning?
- What is the policy network?
- What is replay memory?
- What is the target network?
- Why must we utilize two separate networks during training?
- What is keras-rl? How do you use it?

## Requirements

### General

- Allowed editors: `vi` , `vim` , `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), `gym` (version 0.17.2), `keras` (version 2.2.5), and `keras-rl` (version 0.4.2)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation ( `python3 -c 'print(__import__("my_module").__doc__)'` )
- All your classes should have documentation ( `python3 -c 'print(__import__("my_module").MyClass.__doc__)'` )
- All your functions (inside and outside a class) should have documentation ( `python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'` )
- All your files must be executable
- Your code should use the minimum number of operations

## Installing Keras-RL

`pip install --user keras-rl ` ### Dependencies (that should already be installed)
`pip install --user keras==2.2.4 pip install --user Pillow pip install --user h5py ` ## Tasks

### 0. Breakout

          mandatory         Progress vs Score  Task Body Write a python script   ` train.py `   that utilizes   ` keras `  ,   ` keras-rl `  , and   ` gym `   to train an agent that can play Atari’s Breakout:

- Your script should utilize `keras-rl` ‘s `DQNAgent` , `SequentialMemory` , and `EpsGreedyQPolicy`
- Your script should save the final policy network as `policy.h5`
  Write a python script `play.py` that can display a game played by the agent trained by `train.py` :
- Your script should load the policy network saved in `policy.h5`
- Your agent should use the `GreedyQPolicy`
  Task URLs Github information Repo:
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/0x01-deep_q_learning`
- File: `train.py, play.py`
  Self-paced manual review Panel footer - Controls

### Ready for manual review

Now that you are ready to be reviewed, share your link to your peers. You can find some [here](https://intranet.hbtn.io/projects/784#available-reviewers-modal)
.
×#### Contact one of your peers
https://intranet.hbtn.io/corrections/411233/correct[]()
Don't forget to[review one of them](https://intranet.hbtn.io/corrections/to_review)
. Reviews are due by Nov 11, 2022 12:00 AM
Copyright © 2022 Holberton Inc, All rights reserved.
