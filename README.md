# Pytorch_Retinaface_Stop_Line
## train
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
```

## test
```shell
python detect.py --trained_model ./weights/mobilenet0.25_epoch_15.pth  --network mobile0.25 --input test.png
python detect.py --trained_model ./weights_stop_line_noupsample_0105/mobilenet0.25_epoch_245.pth  --network mobile0.25 --input test.png
CUDA_VISIBLE_DEVICES=3 python detect_color.py --trained_model ./weights_stop_line_color_0214/mobilenet0.25_Final.pth  --network mobile0.25 --input test.png
```

## test widerface val
```shell
#Generate txt file
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
#Evaluate txt results. Demo come from Here
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```


## 训练数据
```shell
#x y w h label0 x1 y1 x2 y2 x3 y3 x4 y4 label1
#0 312 638 98 1 637 316 2 319 1 405 636 406 1
#最后一位用作颜色检测 [bg=0, right color=1, wrong color=2]


#remove FPN中上下采样
#更改文件/home/wyw/Pytorch_Retinaface_Stop_Line_Noupsample/models/net.py下FPN中采用部分
#替换 nn.LeakyReLU(negative_slope= leaky,inplace=True)为nn.ReLU6(inplace=True)

#如果是四个关键点，在原来的Pytorch_Retinaface（https://github.com/biubug6/Pytorch_Retinaface）上需要改动的文件     最终参考https://github.com/gm19900510/Pytorch_Retina_License_Plate  https://github.com/Fanghc95/Plate-Landmarks-detection
#Pytorch_Retinaface_Stop_Line_Noupsample\data\wider_face.py  46、52、105行
#Pytorch_Retinaface_Stop_Line_Noupsample\data\data_augment.py  46、65、195行
#Pytorch_Retinaface_Stop_Line_Noupsample\utils\box_utils.py 202、256行
#Pytorch_Retinaface_Stop_Line_Noupsample\layers\modules\multibox_loss.py 67、75、94行
#Pytorch_Retinaface_Stop_Line_Noupsample\models\retinaface.py 43、52行
#Pytorch_Retinaface_Stop_Line_Noupsample\detect.py 137行



#检测标志线颜色 right_color(白黄)   wrong_color(其他)  修改的代码   参考https://github.com/wolfworld6/Pytorch-Retinaface-Mask-Detection
#更改Pytorch_Retinaface_Stop_Line_Noupsample\train.py 45行
#更改Pytorch_Retinaface_Stop_Line_Noupsample\models\retinaface.py 19、26行
#更改Pytorch_Retinaface_Stop_Line_Noupsample\data\wider_face.py 115行
#更改Pytorch_Retinaface_Stop_Line_Noupsample\layers\modules\multibox_loss.py 101、114行
#测试用detect_color.py





#為了方便起見我們直接更改 outputs 的位置，我們使用 onnx_edit.py 來更改，--outpus 第一個位置放 output name，[]裡放 output shape，例如 831[1,80,10,10]，點選 node 可以查看資訊。input id

$ wget https://raw.githubusercontent.com/d246810g2000/tensorrt/main/onnx_edit.py
$ python3 onnx_edit.py StopLineDetector_noupsample_sim.onnx stop_line_sim.onnx --outputs '476[1,4,80,80], 488[1,4,40,40], 500[1,4,20,20], 439[1,8,80,80], 451[1,8,40,40], 463[1,8,20,20], 513[1,16,80,80], 525[1,16,40,40], 537[1,16,20,20]'
```

## test result
![1386_1](https://github.com/Aruen24/Pytorch_Retinaface_Stop_Line/assets/27750891/e6123daf-c94a-44fa-9c33-918db07db52d)
![1639730557_horizon_1](https://github.com/Aruen24/Pytorch_Retinaface_Stop_Line/assets/27750891/413be330-8027-4c72-8a99-a8b39958fedd)


