# Deep Fakes PyTorch

### Installation

You will need to install cmake first (required for dlib, which is used for face alignment).
You will also need to download shape_predictor_68_face_landmarks.dat for dlib, put it in the main folder

```shell
conda create --name torchfakes
activate torchfakes
git clone https://github.com/IanSullivan/DeepFakeTorch.git
cd DeepFakeTorch
pip install -r requirements.txt
```

### How it works
youtube video
https://www.youtube.com/watch?v=XqluthtTenI 

<iframe width="560" height="315" src="https://www.youtube.com/embed/XqluthtTenI" frameborder="0" allowfullscreen></iframe>

paper written by deepfacelab
https://arxiv.org/abs/2005.05535

### Extract Faces
```shell
python face_detect.py -video_src data_src.mp4 -out_name a
python face_detect.py -video_src data_dst.mp4 -out_name b
```

---

### Train Model
```shell
python train.py -face_a_dir a -face_b_dir b -n_steps 100000
```

---

### Write Video
```shell
python video_writer.py -original_video [video name] -model_location saved_models/model.pt -out_name myswappedvideo -decoder [either 'a' or 'b']
```

---

## Results
Training after 10,000 steps <br>
<img src="images/swapped.gif"> <br>
<img src="images/b_to_a.gif">

## TO DO LIST (if possible)
- [ ] Smooth images on swapped video
- [ ] Add colour correction to swapped faces
