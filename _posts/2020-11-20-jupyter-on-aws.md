---
title: "Setting up a Jupyter Notebook on AWS"
date: 2020-11-20T14:03:00
categories:
  - blog
tags:
  - Data-science
classes: wide
header:
  teaser: "/assets/images/jupyter_teaser.png"
  overlay_image: /assets/images/jupyter_x_aws.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
---

Jupyter Notebooks are a great and widely used tool in data science. Quite often then run are run on localhosts or have to be accessed via SSH tunnelling. 

This is not very convenient when you want to share the results presented in the notebook to members of your team who are non-technical.

In this post, I show how to set up an EC2 instance on AWS, secure it with a password, add SSL encryption with a Let's Encrypt certificate and a publicly accessible name and set the notebook to run as a Linux service. 

# Setting up an EC2 instance

I'm not going to go into detail about this part of the process since there are already a few excellent blog posts that describe how in detail, such as [this one](https://medium.com/@alexjsanchez/python-3-notebooks-on-aws-ec2-in-15-mostly-easy-steps-2ec5e662c6c6) and [this one](https://chrisalbon.com/aws/basics/run_project_jupyter_on_amazon_ec2/).


# Setup a password

This can be done simply by running

```
jupyter notebook password
```

The password hash will be stored in the notebook config at `.jupyter/jupyter_notebook_config.json`

# Ensure the notebook is publically accessible

By default jupyter notebooks are only accessible on localhost,
we can change that by once again editing `.jupyter/jupyter_notebook_config.json`

```
{
  "NotebookApp": {
    "ip": "*",
    .
    .
    .
  }
}
```

# Elastic IP and a custom domain

Within the EC2 console, find "Elastic IPs" under "Network and Security", now either create a new Elastic IP or click "Associate Elastic IP address". Find your instance ID and associate it.

After associating the elastic IP, login to your DNS provider and create an A record 

| Type  | Name  | Content  | TTL  |
|---|---|---|---|---|
| A  | data  | 3.87.2.123  | Auto  |   |


For example, here I've directed `data.mydomain.com` to the elastic IP associated with my instance.

# SSH Config

To make your life easier you may want to setup an SSH config.
In the `~/.ssh/config` add an entry 

```
Host ec2-jupyter
    Hostname data.mydomain.com
    User ec2-user
    IdentityFile ~/.ssh/my-aws-key.pem
```

with the values replaced with your own.

Now you will be able to SSH into your server simply by typing
`ssh ec2-jupyter` instead of the full host and path to the key.

# Letsencrypt certificates and Jupyter

With the domain setup, I used letsencrypt to setup an SSL certificate.

## Installing certbot on Amazon Linux 2

First some prerequisites were needed

Navigate to your home directory (/home/ec2-user). Download EPEL with the following command.

```
sudo wget -r --no-parent -A 'epel-release-*.rpm' https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/
```

Install the repository packages as shown in the following command.

```
sudo rpm -Uvh dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-*.rpm
```

Enable EPEL as shown in the following command.

```
sudo yum-config-manager --enable epel*
```

Now actually install certbot

```
sudo yum install -y certbot python2-certbot-apache
```

## Generate the certificate

```bash
sudo certbot certonly --standalone --debug -d data.mydomain.com
```

Note, this `--standalone` option sets up a temporary webserver on port 80. In order to let it do that I had to do 2 things:

1. Disable Apache `sudo systemctl stop httpd`
2. Allow inbound traffic on port 80 in the EC2 security group temporarily


You may also want to follow [this guide from AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/SSL-on-amazon-linux-2.html#letsencrypt) but I experienced some issues and ended up doing it with `--standalone`.

The certificate and key get created in `/etc/letsencrypt/live/data.mydomain.com`.

To ensure Jupyter can access them I copied them into the home directly and changed their ownership

```
sudo cp /etc/letsencrypt/live/data.mydomain.com/cert.pem ssl/
sudo cp /etc/letsencrypt/live/data.mydomain.com/privkey.pem ssl/
cd ssl
chown ec2-user.ec2-user *
```

Next, I edited the Jupyter config:

```
vim .jupyter/jupyter_notebook_config.json
```

so it looks like

```
{
  "NotebookApp": {
    "password": "<hashed_password>",
    "ip": "*",
    "open_browser": false,
    "certfile": "/home/ec2-user/ssl/cert.pem",
    "keyfile": "/home/ec2-user/ssl/privkey.pem"
  }
}
```


# Linux Service


I would like the Jupyter Notebook to be ran as a service so that it can run in the background (without me having to SSH into the server to manually run it) Think Nginx or Apache.

First I created a `jupyter.service` file:

```bash
[Unit]
Description=Jupyter Notebook
[Service]
Type=simple
PIDFile=/run/jupyter.pid
ExecStart=/bin/bash -c ". /home/ec2-user/anaconda3/bin/activate;jupyter-notebook --notebook-dir=/home/ec2-user/"
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
~                              
```

Note that you can locate your Jupyter binary with `which jupyter-notebook`

and then I copied this file to the `systemd` directory and enabled the service

```bash
sudo cp jupyter.service /etc/systemd/system/
sudo systemctl enable jupyter.service
sudo systemctl daemon-reload
sudo systemctl start jupyter.service
```

The status of the service can be observed with the command

```bash
sudo systemctl status jupyter.service 
```

It should look something like

```
● jupyter.service - Jupyter Notebook
   Loaded: loaded (/etc/systemd/system/jupyter.service; enabled; vendor preset: disabled)
   Active: active (running) since Fri 2020-11-20 15:47:00 UTC; 42min ago
 Main PID: 16858 (bash)
   CGroup: /system.slice/jupyter.service
           ├─16858 /bin/bash -c . /home/ec2-user/anaconda3/bin/activate;jupyter-notebook --notebook-dir=/h...
           ├─16869 /home/ec2-user/anaconda3/bin/python /home/ec2-user/anaconda3/bin/jupyter-notebook --not...
           └─17192 /home/ec2-user/anaconda3/bin/python -m ipykernel_launcher -f /home/ec2-user/.local/shar...
```

If it didn't work could can stop the service, and try running it again manually with

```
jupyter notebook --no-browser
```

and check if all is well with the config.