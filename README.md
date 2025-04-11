![image](https://github.com/user-attachments/assets/a7b83d31-1b2b-45b0-bc59-b1ac5d3d6312)

# Code
《相似实例引导下融合异质图的医学影像报告生成》论文代码  
仓库内未放置iu，mimic等数据集，需要手动下载，且用到的sam等模型需要手动下载 放在指定路径下
# for_iu
在iu数据集上的相关代码。前期在进行代码编写的时候，出于方便，未作解耦处理。因此我们将对iu数据集的实验和mimic的实验分开放在两个文件夹下。实际上主体代码一致，只是一些数据的调用路径不一致。
# for_mimic
在mimic数据集上的相关代码。  
# 代码  
目录下main.py文件为代码入口，在main文件下设置各种参数，如数据所在路径，annotation文件所在路径等。  
## bert文件夹
计算余弦相似度，将图像特征映射到384空间
## data文件夹
原base模型r2gen存放data文件夹，现在存放两个数据集的下载链接
## graph文件夹
图相关操作
### dimensionIncrease.py
图节点特征升维操作，将20维特征恢复至512维
### dimensionReduction.py
图节点降维操作，方便进行图卷积
### graph_related.py
图构建操作
### heteroGnn.py
图卷积操作
## models文件夹
本文模型定义文件
## modules文件夹
模块实现细节，主要操作在encoder_decoder.py文件中，依赖于dataloader.py（数据加载）,visual_extractor.py（视觉特征提取）,trainer.py（训练计划）等文件
## radgraph_tool文件夹
医学实体-关系提取
## segment_anything文件夹
医学影像预分割处理

本文使用SAM对医学影像进行预分割处理，出于训练效率的考虑，我们提前将医学影像进行分割，放在数据同路径下。由于数据集过大，并未将数据上传至仓库，如有需要，可联系作者：3022594378@qq.com  
![image](https://github.com/user-attachments/assets/07e95222-7d14-45d3-b00c-9a8e4e6eec6e)

