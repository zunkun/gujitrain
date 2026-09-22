# guji-train

本项目测试古籍分割左右文本框效果

使用 yolo 进行处理，通过标记，训练，处理相关图片

本数据集采用 `Harvard drs 54114467 容齋隨筆 v.1.pdf` 进行训练

打包

```shell
pyinstaller -F -n cropyolo --hidden-import=torch._C --hidden-import=torch.nn.modules.batchnorm --hidden-import=cv2 --add-data "weights;weights" cropyolo.py

```
