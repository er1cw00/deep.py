rm -rf .git
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
pip install -r requirements.txt

pip install numpy==1.26.4
pip cache purge
apt-get autoclean
apt-get clean



