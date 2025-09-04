# Ultra-wide-field Noninvasive Imaging through Scattering Media via Physics-guided Deep Learning

Welcome! This is the official implementation of the paper Ultra-wide-field Noninvasive Imaging through Scattering Media via Physics-guided Deep Learning.


## 💪 Get Started

### 1. Clone the repository:

   ```bash
   git clone https://github.com/LintaoPeng/ultra-wide-field-Noninvasive-imaging
   cd ultra-wide-field-Noninvasive-imaging
   ```

To set up the environment for this project, follow the steps below:

### 2. Create and Activate Conda Environment

```bash
conda create -n your_env_name python=3.10
conda activate your_env_name
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📊 Testing
For your convenience, we provide some example datasets in `./data/test` folder.  You can download the pretrain models in [Google Drive](https://drive.google.com/file/d/1EgJpwBgFsT2BLUUc5VTe0-nIMdG5fx32/view?usp=sharing). 

After downloading, extract the pretrained model into the `./ckpt` folder, and then run `test.py`. The code will use the pretrained model to automatically process all the images in the `./data/test/speckle` folder and output the results to the `./data/test/output` folder. 









