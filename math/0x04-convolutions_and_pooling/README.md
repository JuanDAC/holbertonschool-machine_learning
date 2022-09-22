Curriculum Specializations Average: 42.53%\* [ Foundations Average: 103.37% ](https://intranet.hbtn.io/curriculums/1/observe)

- [ Specializations Average: 42.53%](https://intranet.hbtn.io/curriculums/2/observe)

            You are currently connected as Student Switch to: [Staff](https://intranet.hbtn.io/users/switch_viewing_as_permission_group?group=staff)

|  # 0x04. Convolutions and Pooling

## Details

By: Alexa Orrico, Software Engineer at Holberton School Weight: 1Project over - took place fromJun 1, 2022 12:00 AMto Jun 3, 2022 12:00 AM An auto review will be launched at the deadline#### In a nutshell…

- Auto QA review: 0.0/53 mandatory
- Altogether:  0.0%\* Mandatory: 0.0%
- Optional: no optional tasks

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=69c0ae8aecd0b7b1a6d218cbb9891cc9f1afad9c222d18f38d0b4150cbb16468)

## Resources

Read or watch :

- [Image Kernels](https://intranet.hbtn.io/rltoken/Qeq8i5dhkR9Tlp-IgFDzQw)

- [Undrestanding Convolutional Layers](https://intranet.hbtn.io/rltoken/g8kHsJFzC51whRSEupvidw)

- [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://intranet.hbtn.io/rltoken/Q6_n8Jkv0JdHDoR9NpvnLQ)

- [What is max pooling in convolutional neural networks?](https://intranet.hbtn.io/rltoken/crEEAb4sDHc30ntPwY-qsQ)

- [Edge Detection Examples](https://intranet.hbtn.io/rltoken/nV4RcnhzFvjLfl7z2k5-Cw)

- [Padding](https://intranet.hbtn.io/rltoken/WZ_a9ntwdJ_AU51W46KOlw)

- [Strided Convolutions](https://intranet.hbtn.io/rltoken/yupMT890fCjD5XVyogDkmg)

- [Convolutions over Volumes](https://intranet.hbtn.io/rltoken/vdFQg1m-0BJ_s0lg8b3fkg)

- [Pooling Layers](https://intranet.hbtn.io/rltoken/Z0dPond1Oi9a04MiWsbgXA)

- [Implementing ‘SAME’ and ‘VALID’ padding of Tensorflow in Python](https://intranet.hbtn.io/rltoken/gJgrOuiHHqu6aNVZoX7iBA)
- NOTE: In this document, there is a mistake regarding valid padding. Floor rounding should be used for valid padding instead of ceiling

Definitions to skim:

- [Convolution](https://intranet.hbtn.io/rltoken/xbzvTRaBX2LUOM7A1NazVQ)

- [Kernel (image processing)](https://intranet.hbtn.io/rltoken/lsI2xbijDWAiKDFuCYkcAA)

References:

- [numpy.pad](https://intranet.hbtn.io/rltoken/8eMV-Jb3O0SSvzu_4BiNIw)

- [A guide to convolution arithmetic for deep learning](https://intranet.hbtn.io/rltoken/ZJItcZYPPp4e6bAV-xaMkw)

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://intranet.hbtn.io/rltoken/k7p5CwKUX9xs-MF60MPOxA)
, without the help of Google :

### General

- What is a convolution?
- What is max pooling? average pooling?
- What is a kernel/filter?
- What is padding?
- What is “same” padding? “valid” padding?
- What is a stride?
- What are channels?
- How to perform a convolution over an image
- How to perform max/average pooling over an image

## Requirements

### General

- Allowed editors: `vi` , `vim` , `emacs`
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.8)
- Your files will be executed with `numpy` (version 1.19.2)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.6)
- All your modules should have documentation ( `python3 -c 'print(__import__("my_module").__doc__)'` )
- All your classes should have documentation ( `python3 -c 'print(__import__("my_module").MyClass.__doc__)'` )
- All your functions (inside and outside a class) should have documentation ( `python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'` )
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
- You are not allowed to use `np.convolve`
- All your files must be executable
- The length of your files will be tested using `wc`

## More Info

### Testing

Please download [this dataset](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/animals_1.npz)
for use in some of the following main files.

## Tasks

### 0. Valid Convolution

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve_grayscale_valid(images, kernel): `   that performs a valid convolution on grayscale images:

- `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images

- `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- You are only allowed to use two `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./0-main.py
(50000, 28, 28)
(50000, 26, 26)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=abdc363d1e925cf6c50473b8b9821fa8f57a980de40eed97779c13be2cde96d9)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d1feb309395031105df7aea323db02a75098854291db194b22353f4a652ce5b4)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `0-convolve_grayscale_valid.py`
  Self-paced manual review Panel footer - Controls

### 1. Same Convolution

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve_grayscale_same(images, kernel): `   that performs a same convolution on grayscale images:

- `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images

- `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- if necessary, the image should be padded with 0’s
- You are only allowed to use two `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 1-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./1-main.py
(50000, 28, 28)
(50000, 28, 28)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=abdc363d1e925cf6c50473b8b9821fa8f57a980de40eed97779c13be2cde96d9)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=41562bc016ff36b40b638b1dbd5c44132c442f9540edaff88c6879a599b948c7)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `1-convolve_grayscale_same.py`
  Self-paced manual review Panel footer - Controls

### 2. Convolution with Padding

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve_grayscale_padding(images, kernel, padding): `   that performs a convolution on grayscale images with custom padding:

- `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images

- `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- `padding` is a tuple of `(ph, pw)` \* `ph` is the padding for the height of the image
- `pw` is the padding for the width of the image
- the image should be padded with 0’s

- You are only allowed to use two `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 2-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./2-main.py
(50000, 28, 28)
(50000, 30, 34)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=abdc363d1e925cf6c50473b8b9821fa8f57a980de40eed97779c13be2cde96d9)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=73ffc7eb77ce3d2e2753abf2f0d1bf03a309ba41332a20814e6e7bd35e42315e)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `2-convolve_grayscale_padding.py`
  Self-paced manual review Panel footer - Controls

### 3. Strided Convolution

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): `   that performs a convolution on grayscale images:

- `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images

- `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- `padding` is either a tuple of `(ph, pw)` , ‘same’, or ‘valid’\* if ‘same’, performs a same convolution
- if ‘valid’, performs a valid convolution
- if a tuple:\* `ph` is the padding for the height of the image
- `pw` is the padding for the width of the image

- the image should be padded with 0’s

- `stride` is a tuple of `(sh, sw)` \* `sh` is the stride for the height of the image
- `sw` is the stride for the width of the image

- You are only allowed to use two `for` loops; any other loops of any kind are not allowed Hint: loop over `i` and `j`
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 3-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./3-main.py
(50000, 28, 28)
(50000, 13, 13)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=abdc363d1e925cf6c50473b8b9821fa8f57a980de40eed97779c13be2cde96d9)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/036ccba7dccf211dab76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=faa71d3c36a2842cf7a8629e706034f58f2fdc2fcf9350eac1b2933cba85e248)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `3-convolve_grayscale.py`
  Self-paced manual review Panel footer - Controls

### 4. Convolution with Channels

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve_channels(images, kernel, padding='same', stride=(1, 1)): `   that performs a convolution on images with channels:

- `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images
- `c` is the number of channels in the image

- `kernel` is a `numpy.ndarray` with shape `(kh, kw, c)` containing the kernel for the convolution\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- `padding` is either a tuple of `(ph, pw)` , ‘same’, or ‘valid’\* if ‘same’, performs a same convolution
- if ‘valid’, performs a valid convolution
- if a tuple:\* `ph` is the padding for the height of the image
- `pw` is the padding for the width of the image

- the image should be padded with 0’s

- `stride` is a tuple of `(sh, sw)` \* `sh` is the stride for the height of the image
- `sw` is the stride for the width of the image

- You are only allowed to use two `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 4-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    images_conv = convolve_channels(images, kernel, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0])
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./4-main.py
(10000, 32, 32, 3)
(10000, 30, 30)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=905a8ba323f4929f8341f9939f202e7157c88febc1b71cfe6028d51b5aed807b)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/8bc039fb38d60601b01a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8c5b2dba08dc8b768a5fabfaff4ec633bebb443b5640fcdcf9d195a8046dd81f)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `4-convolve_channels.py`
  Self-paced manual review Panel footer - Controls

### 5. Multiple Kernels

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def convolve(images, kernels, padding='same', stride=(1, 1)): `   that performs a convolution on images using multiple kernels:

- `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images
- `c` is the number of channels in the image

- `kernels` is a `numpy.ndarray` with shape `(kh, kw, c, nc)` containing the kernels for the convolution\* `kh` is the height of a kernel
- `kw` is the width of a kernel
- `nc` is the number of kernels

- `padding` is either a tuple of `(ph, pw)` , ‘same’, or ‘valid’\* if ‘same’, performs a same convolution
- if ‘valid’, performs a valid convolution
- if a tuple:\* `ph` is the padding for the height of the image
- `pw` is the padding for the width of the image

- the image should be padded with 0’s

- `stride` is a tuple of `(sh, sw)` \* `sh` is the stride for the height of the image
- `sw` is the stride for the width of the image

- You are only allowed to use three `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the convolved images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 5-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve = __import__('5-convolve').convolve


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                       [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                       [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])

    images_conv = convolve(images, kernels, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 1])
    plt.show()
    plt.imshow(images_conv[0, :, :, 2])
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./5-main.py
(10000, 32, 32, 3)
(10000, 30, 30, 3)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=905a8ba323f4929f8341f9939f202e7157c88febc1b71cfe6028d51b5aed807b)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6d6319bb470e3566e885.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e347828ce025c60fe7256a603486925459a4f3f0c832fc4ee02c35e2e31efaf4)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/1370dd6200e942eee8f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6cbf571a7c22a681fb7a278d49cea9b98e5bb049daf658044f6b2aa05871dca4)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/a24b7d741b3c378f9f89.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c96614d61de2784547b892bfea2f7dc313f8f40a27b1c39f3c521f53e25f1c54)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `5-convolve.py`
  Self-paced manual review Panel footer - Controls

### 6. Pooling

          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def pool(images, kernel_shape, stride, mode='max'): `   that performs pooling on images:

- `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images\* `m` is the number of images
- `h` is the height in pixels of the images
- `w` is the width in pixels of the images
- `c` is the number of channels in the image

- `kernel_shape` is a tuple of `(kh, kw)` containing the kernel shape for the pooling\* `kh` is the height of the kernel
- `kw` is the width of the kernel

- `stride` is a tuple of `(sh, sw)` \* `sh` is the stride for the height of the image
- `sw` is the stride for the width of the image

- `mode` indicates the type of pooling\* `max` indicates max pooling
- `avg` indicates average pooling

- You are only allowed to use two `for` loops; any other loops of any kind are not allowed
- Returns: a `numpy.ndarray` containing the pooled images

```bash
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 6-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool = __import__('6-pool').pool


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_pool[0] / 255)
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./6-main.py
(10000, 32, 32, 3)
(10000, 16, 16, 3)

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=905a8ba323f4929f8341f9939f202e7157c88febc1b71cfe6028d51b5aed807b)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ab4705f939c3a8e487bb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220921T223145Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7a31c8cbd1fdadced24852b44785d640bfc54140c98b4b7aa8883f0da89795b8)

Task URLs Github information Repo:

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `math/0x04-convolutions_and_pooling`
- File: `6-pool.py`
  Self-paced manual review Panel footer - Controls
  Copyright © 2022 Holberton Inc, All rights reserved.
