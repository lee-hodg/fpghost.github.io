---
title: "Setting up Jupyterhub on AWS"
date: 2020-12-10T14:03:00
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

This guide will be about setting up the fiddly bits when deploying a Jupyter Hub to an AWS instance. It won't go into explicit detail about absolutely every step as the docs already do a great job of that. The purpose of this post is to discuss the things I found tricky after the install was complete.

First follow the [install instructions](https://tljh.jupyter.org/en/latest/install/amazon.html) for AWS. 
These amount to deploying an EC2 instance of type `Ubuntu Server 18.04 LTS (HVM), SSD Volume Type`, ensuring sufficient CPU and memory (I chose `t2.large`), adding the following snippet

```bash
#!/bin/bash
curl -L https://tljh.jupyter.org/bootstrap.py \
  | sudo python3 - \
    --admin <admin-user-name>
```

and also using a security group with SSH, HTTP and HTTPS open.

## Configuring an elastic IP and DNS

The next thing I did was to associate an Elastic IP address to the instance we recently setup. Go [here](https://console.aws.amazon.com/ec2/v2/home?#Addresses), click the IP you'd like to associate (or "Allocate IP address" if you currently have none free) then click "Actions > Associate Elastic IP address" before choosing the appropriate instance running Jupyterhub.

Now in the config of my DNS provider I added an A record, e.g. `jupyter.mycompany.com` pointing to this elastic IP address.

**Note:** you have to do this if you wish to use SSL.


## SSL

After around 10mins, you should be able to visit `http://jupyter.mycompany.com` (or just the public IP of the instance if you didn't set up a domain) and be greeted with a login screen. Choose the `<admin-user-name>` that you configured and pick a password. This will log you in as the admin.

TLJH has a [nice guide](https://tljh.jupyter.org/en/latest/howto/admin/https.html) on how to configure SSL. You will need to have set up
a public URL for this (as per above), and then it boils down to

```bash
sudo tljh-config set https.enabled true
sudo tljh-config set https.letsencrypt.email you@example.com
sudo tljh-config add-item https.letsencrypt.domains yourhub.yourdomain.edu
```

Check your config with

```bash
sudo tljh-config show
```

and reload the proxy

```bash
sudo tljh-config reload proxy
```

Now you will see the secure padlock if you visit your Jupyterhub.


## Jupyter lab as default

Jupyter lab is the more modern counterpart of the Jupyter notebook. If you would like to run this interface by default
TLJH also has a [guide](https://tljh.jupyter.org/en/latest/howto/env/notebook-interfaces.html). 

After logging in with the admin user(or via SSH to the EC2 instance) open up a new terminal and run

```bash
sudo tljh-config set user_environment.default_app jupyterlab
sudo tljh-config reload hub
```

You can still access regular Jupyter at `/user/<username>/tree` but by default your users will now see
`/user/<username>/lab`.


## OAuth with Github

Following the [TLJH guide](https://tljh.jupyter.org/en/latest/howto/auth/google.html)

Looking at the [docs](https://oauthenticator.readthedocs.io/en/latest/api/gen/oauthenticator.github.html) for Oauthenticator Github
suggest we should be able to restrict users with something like

```
sudo tljh-config set auth.GitHubOAuthenticator.allowed_users <allowed-user-1>
sudo tljh-config reload hub
```

Unfortunately, this restriction just did nothing for me despite the change showing in `/opt/tljh/config/config.yaml` (or equivalently
when running `sudo tljh-config show`)

Before finally figuring out the problem, I had also tried with a custom config snippet to manually add a Jupyterhub config [as described here](https://tljh.jupyter.org/en/latest/topic/escape-hatch.html). Any files in `/opt/tljh/config/jupyterhub_config.d` that end in .py will be loaded in alphabetical order as python files to provide configuration for JupyterHub. Then see [here](https://oauthenticator.readthedocs.io/en/latest/api/gen/oauthenticator.github.html)

```
# /opt/tljh/config/jupyterhub_config.d/oauth_config.py file
c = get_config()
# c.GitHubOAuthenticator.allowed_organizations = Set()
# c.GitHubOAuthenticator.admin_users = Set()
c.GitHubOAuthenticator.allowed_users = {'allowed-user-1'}
c.GitHubOAuthenticator.blocked_users = {'banned-user-2'}
```

This did not work either (I could still login as the banned user).

What did work in the end was

```
sudo tljh-config set auth.GitHubOAuthenticator.whitelist <allowed-user-1>
sudo tljh-config reload
```

The working config file looked like

```
auth:
  type: oauthenticator.github.GitHubOAuthenticator
  GitHubOAuthenticator:
    client_id: XXXX
    client_secret: XXXX
    oauth_callback_url: XXX
    whitelist:
    - allowed-user-1
    admin_users:
    - admin-user-2
```

I guess because they are still using an older version of `oauthenticator`, which uses the deprecated `whitelist` parameter.

You can test by first going to https://jupyter.yourcompany.com/hub/admin and deleting the user in question and also deleting
them from the Linux system with `sudo deluser --remove-home jupyter-allowed-user-1`, now change the whitelist to be say `some-other-user`, you will see in the admin user list that `some-other-user` is present and has "never" logged in and you will see "allowed-user-1" is not. If you try to authenticate with Github and the `allowed-user-1` you should see a 403 forbidden.
If now you switch back to `allowed-user-1` you can login.

## Switching back

If you do want to switch back authentication so that it uses the `FirstUseAuthenticator`, just enter:

```
sudo tljh-config set auth.type firstuseauthenticator.FirstUseAuthenticator
```

and then clean up the `config.yaml` so all that remains is

```yaml
auth:
  type: firstuseauthenticator.FirstUseAuthenticator
```

Using the `FirstUseAuthenticator` means an admin gets to add users, then when they login they choose their own password,
else if they try to login without such a username in the system, they are rejected.

## Add admin users

To add a user as an admin do

```
sudo tljh-config add-item users.admin <username>
sudo tljh-config reload
```


## Deploy keys

You may want to be able to push and pull from some Github repo from the Jupyterhub instance. Deploy keys allow you to achieve this.

On the EC2 instance (for example after SSH'ing in), run:

```
ssh-keygen -t rsa -b 4096 -C "{email}"
```

Copy it to `https://github.com/<SomeOrg>/<some-repo>/settings/keys/new`


Now ensure whatever jupyter user has access to those keys in their home dir, e.g. `/home/jupyter-some-user/.ssh`.
Copy the keys and ensure to `chown jupyter-some-user.jupyter-some-user -R /home/jupyter-some-user/.ssh`

Now `sudo su jupyter-some-user` and `cd ~` before finally cloning the git repo into this users directory.
If you checked `write access` on the deploy key this user will also be able to commit changes to the notebooks when done.

Another option would be keep the keys in a shared directory across many users [as described here](https://tljh.jupyter.org/en/latest/howto/content/share-data.html) and then just create a symlink to the `ida_rsa` files in a given users `~/.ssh/` directory.


## Conda packages

We can install `conda` packages following [this guide](https://tljh.jupyter.org/en/latest/howto/env/user-environment.html).

Basically login as admin, get a new terminal and run commands like

```bash
sudo -E conda install numpy
```

The `-E` flag is to preserve the environment.

See the note about doing this from SSH (`export PATH=/opt/tljh/user/bin:${PATH}`).


## Readonly postgres user on RDS

If you want to pull data from a Postgres database on RDS then `pycopg2` is a good choice. You may want to set up a dedicated user in the database than has only read-only permissions (for security and to avoid accidents, since it's very unlikely you will want to write data from a Jupyterlab notebook to the database)

To do that run the following commands:

```
# The role
cursor.execute('CREATE ROLE readonly;')
cursor.execute("GRANT USAGE ON SCHEMA public TO readonly;")
cursor.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;")
cursor.execute("GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readonly;")
cursor.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly;")
cursor.execute('GRANT CONNECT ON DATABASE your_db TO readonly;')
# The user
cursor.execute("CREATE USER readonly_user WITH PASSWORD 'XXXXXXXXXXXXXXXX';")
# The role to the user
cursor.execute("GRANT readonly TO readonly_user;")
```

After a successful login with this user, test no create perm with

```
cursor.execute("CREATE TABLE public.test1 (id serial PRIMARY KEY, num integer, data varchar);")
```



## DB via IAM conn rather than creds:


Rather than have to distribute database credentials to each user (either individually or via global env variables or a shared folder)
it would be better if we were [using the role itself to connect](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.Connecting.Python.html)


Things that you must do to use the IAM role for database access:

- Enable IAM/Password in the RDS instance settings.
- The IAM Role associated with instance to have RDS read-only access as a policy.
- Run `GRANT rds_iam TO <user>;`  (to revert `REVOKE rds_iam FROM <user>;`)


However, for me this did not work. I hit the following error

```
FATAL:  pg_hba.conf rejects connection for host "X.X.X.X", user "readonly_user", database "your_db", SSL off
```

I'd be interested to hear any solutions to that.

## Extensions

TLJH installs a really old node js (6.x) and this prevents the Jupyerlab Github extension from installing. You may need to upgrade your version of node to get those extensions working.