# 🍎 Fruit Detection Dataset

This project uses a custom fruit detection dataset created with [Roboflow](https://roboflow.com/).

> Dataset provided by **Fruitsdetection** via Roboflow Universe  
> 🔗 [Fruits by YOLO Dataset]([[https://universe.roboflow.com/fruitsdetection/fruits-by-yolo](https://app.roboflow.com/foodobjectdetection-gbrbd/fruitsglavnidatasetmergani/1)](https://app.roboflow.com/foodobjectdetection-gbrbd/foodobjectdetectiondataset/1))

---

## 📦 Dataset Download

You can download the dataset directly from Roboflow using the following link:

👉 [Download Dataset from Roboflow]([https://universe.roboflow.com/fruitsdetection/fruits-by-yolo](https://app.roboflow.com/foodobjectdetection-gbrbd/foodobjectdetectiondataset/1/export))

Alternatively, you can paste the following code into your Colab notebook:

```python
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="gbnE2mI51TkywHQXfMAH")
project = rf.workspace("foodobjectdetection-gbrbd").project("foodobjectdetectiondataset")
version = project.version(1)
dataset = version.download("yolov8")
```           

