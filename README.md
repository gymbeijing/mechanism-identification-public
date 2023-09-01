# mechanism-identification
## AWS
### Launch the instance
- Subnet: ```RESGAD-dev_public2_az2```

- Security groups: ```RESGAD-dev_dev_Base_Linux_SG, AdskVPN```

- Auto-assign public IP: ```Enable```

- Adding an IAM role (under Security tab) to the instance to get access to some resources, but should create a new one out of the existing one.

- Manage tags: ```key: adsk:moniker value: resgad-c-uw2```

### Connect to the instance
```ssh -i "guy1-mech-iden.pem" ec2-user/ubuntu@<private_ip_address>```

### Copy private key to the instance
```scp -i guy1-mech-iden.pem ~/.ssh/id_ed25519 ubuntu@<private_ip_address>:~/.ssh/```

### Create the virtual environment
```
sudo apt install python3.8-venv
python3 -m venv venv_py38

source venv_py38/bin/activate

# Install packages
pip install wheel
pip install salesforce-lavis
pip install torch
pip install scikit-learn
pip install python-tsp
pip install hdbscan
pip install umap-learn
pip install pytorch-lightning

# Connect to Jupyter Notebook
pip install jupyterlab
pip install notebook
jupyter notebook --generate-config
jupyter notebook password (123456)

jupyter notebook --no-browser --port=8889
ssh -i <key_pair>.pem -N -f -L localhost:8888:localhost:8889 ubuntu@<private_ip_address>
```
or run ```pip install -r requirements.txt```

## S3 Commands
### Download data from s3
```
aws s3 sync s3://resgad-lambouj-intern/0.0.4/multi_png_iso/0000/ <dest_folder>
aws s3 sync s3://resgad-lambouj-intern/clip_emb/ <dest_folder>
```

### Upload data to s3
```
aws s3 sync <src_folder> s3://resgad-lambouj-intern/<dest_folder>/
```

## Git Cheatsheet
- Revert git add: ```git reset```
- Tells Git to overwrite all changes in the working directory: ```git reset --hard```

## TensorBoard Cheatsheet
- In Terminal: ```tensorboard --logdir <tb_log_dir>```
- Listen to it through a port: ```ssh -i <key_pair>.pem -N -f -L localhost:8888:localhost:6006 ubuntu@<private_ip_address>```

## PyCharm Remote Development
```Command+Shift+A```: Upload current file to the remote server, and then run the code
