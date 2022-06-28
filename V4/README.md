# Style2Paints V4

This is the source code of the Style2Paints V4. Some things have been altered to allow running on CUDA 11+, meaning RTX 3XXX series support.

# Install

You will need CUDA 11.1, CuDNN 8, Python 3.8

    cd s2p_v4_server
    pip install -r requirements.txt
    
    Extract the WHL file from TensorflowWheel
    
    pip install /TensorflowWheel/tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
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

# Thanks

We thank [hepesu/LineFiller](https://github.com/hepesu/LineFiller) and [V-Sense/DeepNormals](https://github.com/V-Sense/DeepNormals) for the implementation of some basic algrithoms like flooding and normal lighting, though we do not use their models.
