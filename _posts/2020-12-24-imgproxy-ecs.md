---
title: "Deploying imgproxy with AWS Fargate"
date: 2020-12-24T07:19:00
categories:
  - Blog
tags:
  - AWS
header:
  image: /assets/images/imgproxy.jpg
  teaser: "/assets/images/imgproxy_teaser.jpeg"
---

imgproxy is a fast and secure standalone server for resizing and converting remote images. The main principles of imgproxy are simplicity, speed, and security.

I wanted to use imgproxy with AWS to serve images up from an s3 bucket and have it also sit behind a Cloudfront distribution.

At the time, I didn't see any good guides online for how to do this. Imgproxy has a Docker image and for that reason [AWS Fargate](https://aws.amazon.com/fargate) seemed a good choice.

# AWS Fargate

I followed the [first run wizard](https://console.aws.amazon.com/ecs/home?region=us-east-1#/firstRun), choosing to configure the "custom" container.

## The container

I used the image `darthsim/imgproxy` and entered `8080` for the port mapping, since that is the default `imgproxy` port.

It looked like this:

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/imgproxy1.png)

## Environment variables

I set the environment variables to configure s3 access and signing as follows

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/imgproxy2.png)

**Note:** I tried allowing s3 access purely via the ECS task IAM role, but imgproxy gave me "Missing Credentials" errors. In the end, I just created an IAM User with a policy that allowed it read-only access to the relevant s3 bucket and copied its ID/Secret to the environment variables above.

Also the salt and key can be generated with `echo $(xxd -g 2 -l 64 -p /dev/random | tr -d '\n')`.

It was also useful to adjust the following environment variables from their defaults:

```
IMGPROXY_READ_TIMEOUT 30 (10 def)
IMGPROXY_DOWNLOAD_TIMEOUT 30 (10 def)
IMGPROXY_TTL 86400 (def 3600)
IMGPROXY_MAX_SRC_RESOLUTION  160  (def 16.8)
```


## The service

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/imgproxy3.png)

It was essential to use the load balancer or I got "Missing Endpoint" errors, and also it would not be possible later to put the cluster behind Cloudfront.

## Is it working?

To check it is working, wait for the task to go from `PENDING` to `RUNNING` and then copy the `DNS Name` from the [Load Balancer](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LoadBalancers:sort=loadBalancerName) that was created.

If you visit http://ec2co-XXXX.us-east-1.elb.amazonaws.com:8080/ you should be greeted with:

```
"Hey, I'm imgproxy!"
```

# Cloudfront

Choose the "web" distribution and for the `Origin Domain Name` choose the Elastic Load Balancer that is associated with the ECS. Also for the `HTTP Port` choose `8080`, since that is the port that the origin listens on.

Next choose `Redirect HTTP to HTTPS`, and optionally you can set `Alternate Domain Names (CNAMEs)`, e.g. `imgproxy.mycompany.com` (and go create a CNAME record in your DNS provider mapping that to the Cloudfront domain).

Now `https://imgproxy.mycompany.com` should greet you with the 

```
"Hey, I'm imgproxy!"
```

# Signing

Imgproxy provides [examples](https://github.com/imgproxy/imgproxy/blob/master/examples/signature.py) on how to generate the signed URL from the salt and key:

```python
import base64
import hashlib
import hmac
import textwrap

# Hex encoded key and salt set in the IMGPROXY_KEY, IMGPROXY_SALT env
key = bytes.fromhex("<hex_salt>")
salt = bytes.fromhex("<hex_key>")

# The s3 image url
url = b"s3://my-bucket/artworks/1.jpg"
    
encoded_url = base64.urlsafe_b64encode(url).rstrip(b"=").decode()
# You can trim padding spaces to get good-looking url
encoded_url = '/'.join(textwrap.wrap(encoded_url, 16))

path = "/{resize}/{width}/{height}/{gravity}/{enlarge}/{encoded_url}.{extension}".format(
    encoded_url=encoded_url,
    resize="fill",
    width=400,
    height=600,
    gravity="no",
    enlarge=1,
    extension="webp",
).encode()
digest = hmac.new(key, msg=salt+path, digestmod=hashlib.sha256).digest()

protection = base64.urlsafe_b64encode(digest).rstrip(b"=")

url = b'/%s%s' % (
    protection,
    path,
)

host= 'https://imgproxy.mycompany.com'
print(f'{host}{url.decode()}')
```