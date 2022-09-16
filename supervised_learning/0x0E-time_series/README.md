Curriculum Specializations Average: 36.35%\* [

                  Foundations


                  Average: 103.37%





](https://intranet.hbtn.io/curriculums/1/observe)

- [
                  Specializations


                  Average: 36.35%








](https://intranet.hbtn.io/curriculums/2/observe)

            You are currently connected as Student            Switch to:            [Staff](https://intranet.hbtn.io/users/switch_viewing_as_permission_group?group=staff)

|  # 0x0E. Time Series Forecasting

## Details

By: Alexa Orrico, Software Engineer at Holberton School Weight: 4Ongoing second chance project - startedSep 5, 2022 12:00 AM, must end bySep 19, 2022 12:00 AMManual QA review must be done(request it when you are done with the project) ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/3b16b59e54876f2cc4fe9dcf887ac40585057e2c.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220916%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220916T065617Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=166d05dce5f2c349695e8a8d55672eafdcffe54c923e74c44f97b31226616c8f)

## Resources

Read or watch:

- [Time Series Prediction](https://intranet.hbtn.io/rltoken/HmkmzkQ7_A-h5KKzFQ_tJg)

- [Time Series Forecasting](https://intranet.hbtn.io/rltoken/_QoRZ53rwY7yYVV2SM3frw)

- [Time Series Talk : Stationarity](https://intranet.hbtn.io/rltoken/jLo-utlk8pzUzIMRbOJAPA)

- [tf.data: Build TensorFlow input pipelines](https://intranet.hbtn.io/rltoken/ulRRdAVAZr2KYM2ghlBRNQ)

- [Tensorflow Datasets](https://intranet.hbtn.io/rltoken/7H-EjwlfVHGCoWHDCjIU-g)

Definitions to skim

- [Time Series](https://intranet.hbtn.io/rltoken/eDzuZndaRfiXvecn4KvoHQ)

- [Stationary Process](https://intranet.hbtn.io/rltoken/JN26Hp5uM1OgIPUkF1gsYA)

References:

- [tf.keras.layers.SimpleRNN](https://intranet.hbtn.io/rltoken/1aM6PvPAN3kdBtvLB_hnrw)

- [tf.keras.layers.GRU](https://intranet.hbtn.io/rltoken/PUtluakWAs8wcw3rsmYJ2A)

- [tf.keras.layers.LSTM](https://intranet.hbtn.io/rltoken/0Cocg6XxDqjxeAUKYQLhGg)

- [tf.data.Dataset](https://intranet.hbtn.io/rltoken/Wzagcu07guZFjx88UTmIBA)

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://intranet.hbtn.io/rltoken/hNvLXJZduqjX5i_nOB7iAg)
, without the help of Google :

### General

- What is time series forecasting?
- What is a stationary process?
- What is a sliding window?
- How to preprocess time series data
- How to create a data pipeline in tensorflow for time series data
- How to perform time series forecasting with RNNs in tensorflow

## Requirements

### General

- Allowed editors: `vi` , `vim` , `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), `tensorflow` (version 2.4.1) and pandas (version 1.1.5)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should follow the `pycodestyle` style (version 2.4)
- All your modules should have documentation ( `python3 -c 'print(__import__("my_module").__doc__)'` )
- All your classes should have documentation ( `python3 -c 'print(__import__("my_module").MyClass.__doc__)'` )
- All your functions (inside and outside a class) should have documentation ( `python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'` )

## Tasks

### 0. When to Invest

          mandatory         Progress vs Score  Task Body Bitcoin (BTC) became a trending topic after its  [price](https://intranet.hbtn.io/rltoken/vjTWl4bomgHoPdlYDGJM0w)

peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.
Given the [coinbase](https://intranet.hbtn.io/rltoken/_-9LQxYpc6qTM7K_AI58-g)
and [bitstamp](https://intranet.hbtn.io/rltoken/0zZKYc5-xlxGFbxTfCVrBA)
datasets, write a script, `forecast_btc.py` , that creates, trains, and validates a keras model for the forecasting of BTC:

- Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
- The datasets are formatted such that every row represents a 60 second time window containing:\* The start time of the time window in Unix time
- The open price in USD at the start of the time window
- The high price in USD within the time window
- The low price in USD within the time window
- The close price in USD at end of the time window
- The amount of BTC transacted in the time window
- The amount of Currency (USD) transacted in the time window
- The [volume-weighted average price](https://intranet.hbtn.io/rltoken/79YPxEkzc7Q1rc92f1MOOQ)
  in USD for the time window

- Your model should use an RNN architecture of your choosing
- Your model should use mean-squared error (MSE) as its cost function
- You should use a `tf.data.Dataset` to feed data to your model
  Because the dataset is [raw](https://intranet.hbtn.io/rltoken/Keixv8XzPLglpNSCkUiOpQ)
  , you will need to create a script, `preprocess_data.py` to preprocess this data. Here are some things to consider:
- Are all of the data points useful?
- Are all of the data features useful?
- Should you rescale the data?
- Is the current time window relevant?
- How should you save this preprocessed data?
  Task URLs Github information Repo:
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/0x0E-time_series`
- File: `README.md, forecast_btc.py, preprocess_data.py`
  Self-paced manual review Panel footer - Controls

### 1. Everyone wants to know

          mandatory         Progress vs Score  Task Body Everyone wants to know how to make money with BTC! Write a blog post explaining your process in completing the task above:

- An introduction to Time Series Forecasting
- An explanation of your preprocessing method and why you chose it
- An explanation of how you set up your `tf.data.Dataset` for your model inputs
- An explanation of the model architecture that you used
- A results section containing the model performance and corresponding graphs
- A conclusion of your experience, your thoughts on forecasting BTC, and a link to your github with the relevant code
  Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.
  When done, please add all URLs below (blog post, shared link, etc.)
  Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.
  Task URLs #### Add URLs here:
  Save Github information Self-paced manual review Panel footer - Controls
  Ready for a manual reviewCopyright © 2022 Holberton Inc, All rights reserved.
