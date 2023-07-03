# Botbowl :heart: docker
[Docker](http://docker.com) is a platform designed to help developers build, share, and run modern applications. 

## Submitting a docker image to the Bot Bowl competition
Your image will be started with `docker run -p <host_port>:5100 -t <bot_image>`. Below follows detailed instructions to build and submit your bot in a docker image. In the example, we'll build a bot called "nuffle". 


Modify (../examples/containerized_bot.py) so it starts your bot instead of the scripted bot. 

Add your additional dependencies into (../examples/extra_requirements.txt), if you're bot is based on the [A2C example](a2c.md), you want to add `torch` for example. 

Build the docker image and tag it as `nuffle_bot_image`. Call this command in **the root folder of this repository**: 
```shell
docker build . -t nuffle_bot_image --file docker/Dockerfile.comptetition_bot
```

Confirm that it's working: 
```shell
docker run -p 5100:5100 -it nuffle_bot_image 
```
You should see it saying something like: `Agent listening on 4924d7d0fa7d:5100 using token 32`. 
`

Create the image file to be uploaded: 
```shell
docker save -o nuffle_bot_image.tar nuffle_bot_image
```
It will create the file `nuffle_bot_image.tar` which is the one you upload in the submission form. Optionally, the image can be compress, e.g. with gzip: `gzip nuffle_bot_image.tar`, it compresses the file quite a lot! 
 
## Troubleshooting: 
 - if you get `ModuleNotFoundError` when running your bot in the docker it could mean that your dependencies (e.g. PyTorch) weren't installed in the image. Make sure they are by adding the to (../examples/extra_requirements.txt)
- Python version. The python version installed in the docker image is specified on the first line of (../docker/Dockerfile.comptetition_bot), you can change it to another version.
