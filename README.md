# DeepFakeTorch

### Installation

You will need to install cmake first (required for dlib, which is used for face alignment).

```shell
conda create --name torchfakes
activate torchfakes
git clone https://github.com/IanSullivan/DeepFakeTorch.git
cd DeepFakeTorch
pip install requirments.txt
```

### Extract Faces
```shell
python facedetect.py
```

---

## Results
Training after 10,000 steps
<img src="images/swapped.gif">
<img src="images/b_to_a.gif">
