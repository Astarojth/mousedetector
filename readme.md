这是一个基于yolov5的，针对小鼠经过专门训练的目标检测模型。请按照以下的教程进行配置环境和使用。

我们假设您的电脑已经过了基本的软件安装和配置，可以使用gpu进行训练。如果没有请您下载安装anaconda和vscode/pycharm等软件。
https://blog.csdn.net/ECHOSON/article/details/117220445 这里是一个关于anaconda的使用教程。

# part1：环境配置
以下是环境配置流程：

conda create -n yolo5 python==3.8.5
conda activate yolo5
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pycocotools

#cd到yolov5代码的目录下
pip install -r requirements.txt
pip install pyqt5
pip install labelme  #不需要自己标注数据集的话可以不用安装

到这里环境配置完成了。您可以使用以下代码测试。

python detect.py --source data/images/bus.jpg --weights pts/yolov5l.pt

# part2：数据集制作
如果您需要自己标注数据集，我推荐使用labelme工具进行标注，并使用transfer文件夹下的labelme2yolo.py脚本进行切换。

labelme #启动工具

https://blog.csdn.net/qq_40280673/article/details/127437581 这份教程详细说明了在标注yolo数据集时labelme的用法。

完成标注之后，将图片和json文件分别放入transfer文件下的images和json目录。运行：

python labelme2yolo.py
python change.py

yolo格式的标注文件会在txt文件夹中生成。

# part3：模型训练
将标记完成的数据按照下面的格式进行放置：

dataset
       ├─ images
       │    ├─ test # 下面放测试集图片
       │    ├─ train # 下面放训练集图片
       │    └─ val # 下面放验证集图片
       └─ labels
              ├─ test # 下面放测试集标签
              ├─ train # 下面放训练集标签
              ├─ val # 下面放验证集标签
              
我已经准备好了文件夹，将新的标注和图片放入即可，并修改data目录下的mouse_d.yaml文件的train和val的绝对路径。
如果需要增加新的class（如其它动物），请修改data目录下的mouse_data.yaml文件的nc和names。
我已经在model下建立了mouse.yaml配置文件。直接使用即可。
预训练模型和我已经训练好的模型的下载链接：https://pan.baidu.com/s/1t9L2fdGU_0exbdafCpIkvg?pwd=mice 
下载好后请将模型放入pts文件夹内。
执行下列代码运行程序即可：

python train.py --epoch xx --batch-size x --resume True/False

或

python train.py #默认epoch=5000，batchsize=16，可根据自己服务器情况选择，

resume的True或False代表是否从上次训练的模型继续训练，默认为False，意外中断需要继续训练时请改为True

# part4：模型使用
训练完后，请从runs/train中找到最近训练的模型best.pt并将其放置于pts文件夹内。
我们在这提供预训练模型的使用方式，更换模型修改即可。
如果使用我们训练好的模型，则不需要part2和part3，配置完环境后即可使用。
模型的使用全部集成在了detect.py文件，按照下面的指令：

 检测图片文件
  python detect.py  --weights pts/best_withouttail.pt --source file.jpg  # image 
 检测视频文件
   python detect.py --weights pts/best_withouttail.pt --source file.mp4  # video
 检测一个目录下的文件
  python detect.py --weights pts/best_withouttail.pt --source path  # directory

输出结果位于runs/detect文件夹中。


# debug说明：
1. 报错：AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
   解决方案： https://blog.csdn.net/weixin_43401024/article/details/124428432
   该问题由lib导致，只能配置完环境后手动修改
2. 报错：PermissionError: [Errno 13] Permission denied:
   解决方案：请使用绝对路径  
3. 训练时报错：OSError: [WinError 1455] 页面文件太小，无法完成操作。
   解决方案：https://blog.csdn.net/weixin_43817670/article/details/116748349
   num_workers已经修改，如果仍然有问题请加大虚拟内存。