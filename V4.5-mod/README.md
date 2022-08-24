# Style2Paints V4.X-Mods

This is the source code of the Style2Paints V4. Some things have been altered to allow running on CUDA 11+, meaning RTX 3XXX series support.

# Install

You will need CUDA 11.1, CuDNN 8, Python 3.8

    cd s2p_v4_server
    pip install -r requirements.txt
    
    Get the tensorflow wheel from https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.15.4%2Bnv20.12/py38/CPU%2BGPU
    
    pip install tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
    pip install h5py==2.10.0
    pip install protobuf==3.20.0

Then download the model files

    https://drive.google.com/drive/folders/142ZFZUX1mpf2FOKGhf6Z7aUkQy2wNryF?usp=sharing

and put them like

    s2p_v4_server/nets/inception.net
    s2p_v4_server/nets/mat.npy
    s2p_v4_server/nets/norm.net
    ...

# Run

Simply run the python file like

    cd s2p_v4_server
    python server.py

Note that if you see something like 

    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.

Then just ignore it.

When the service is ready, you can use the software at

    http://127.0.0.1:8233/index.html
    
# EXE Build Instructions

*Anaconda environment is recommended!*
*Make sure you have CUDA 11.1 installed and CuDNN 8 installed!*

Setup Conda environment
    
    conda create -n s2p4 python==3.8
    cd into your s2p_v4_server directory
    pip install -r requirements.txt
    Extract wheel from https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.15.4%2Bnv20.12/py38/CPU%2BGPU
    pip install tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
    pip install h5py==2.10.0
    pip install protobuf==3.20.0
    conda install pyinstaller
    py -3.8 -m PyInstaller --noconfirm --onedir --console --icon ./game/favicon.ico --name "style2paintsV4" --add-data "./game;game/" --add-data "./linefiller;linefiller/" --add-data "./nets;nets/" --add-data "./refs;refs/" --add-data "./results;results/" --hidden-import "opencv-contrib-python" --hidden-import "bottle" --hidden-import "h5py" --hidden-import "keras" --hidden-import "scikit-learn" --hidden-import "scikit-image" --hidden-import "llvmlite" --hidden-import "numba" --hidden-import "tqdm" --hidden-import "paste" --hidden-import "tkinter" --collect-all "bottle"  "./server.py"
    EXE will be found in dist/style2paintsV4
    
or python environment

    Install python 3.8 normally...
    cd into your s2p_v4_server directory
    pip install -r requirements.txt
    Extract wheel from /TensorflowWheel...
    pip install /TensorflowWheel/tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
    pip install h5py==2.10.0
    pip install protobuf==3.20.0
    pip install pyinstaller
    pyinstaller --noconfirm --onedir --console --icon ./game/favicon.ico --name "style2paintsV4" --add-data "./game;game/" --add-data "./linefiller;linefiller/" --add-data "./nets;nets/" --add-data "./refs;refs/" --add-data "./results;results/" --hidden-import "opencv-contrib-python" --hidden-import "bottle" --hidden-import "h5py" --hidden-import "keras" --hidden-import "scikit-learn" --hidden-import "scikit-image" --hidden-import "llvmlite" --hidden-import "numba" --hidden-import "tqdm" --hidden-import "paste" --hidden-import "tkinter" --collect-all "bottle"  "./server.py"
    EXE will be found in dist/style2paintsV4

    
# Thanks

We thank [hepesu/LineFiller](https://github.com/hepesu/LineFiller) and [V-Sense/DeepNormals](https://github.com/V-Sense/DeepNormals) for the implementation of some basic algrithoms like flooding and normal lighting, though we do not use their models.
