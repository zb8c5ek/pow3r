conda install -c conda-forge mesa-libgl-cos6-x86_64 mesa-libegl-cos6-x86_64
# 在容器中安装系统级 EGL 库
apt-get update && apt-get install -y libegl1-mesa-dev libgl1-mesa-dev
# 安装所需的 OpenGL 库
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglu1-mesa \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6