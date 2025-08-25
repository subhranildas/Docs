## What is Docker?

Ever faced the “works on my machine” problem? That’s exactly what **Docker** solves.

Docker is an **open-source platform** that packages applications and their dependencies into **containers**, lightweight, portable environments that run consistently across laptops, servers, and the cloud.

---

## Why Docker?

- **Consistency**: Same environment everywhere.
- **Portability**: Run anywhere Docker is installed.
- **Efficiency**: Faster and lighter than virtual machines.
- **Scalability**: Works seamlessly with Kubernetes and modern cloud platforms.

---

## Key Concepts

- **Image** → Blueprint of your app (code + dependencies).
- **Container** → Running instance of an image.
- **Docker Hub** → Library of ready-to-use images.

---

## Resources and Links

### Docker installation Docs

- _*[Docker Setup Overview](https://docs.docker.com/engine/install/)*_
- _*[Docker Setup MacOS](https://docs.docker.com/desktop/setup/install/mac-install/)*_
- _*[Docker Setup Windows](https://docs.docker.com/desktop/setup/install/windows-install/)*_

### Docker installation Guide for Older Systems

- _*[Docker Toolbox MacOS](https://github.com/docker-archive/toolbox/blob/master/docs/toolbox_install_mac.md)*_
- _*[Docker Toolbox Windows](https://github.com/docker-archive/toolbox/blob/master/docs/toolbox_install_windows.md)*_

## Quick Start

### Example 1

Install Docker, then run:

```bash
docker run hello-world
```

> This pulls a test image and runs it in a container — your first step into containerization.

### Example 2

Download and extract _**<a href="/DOCS/DOCKER_DOCS/Downloads/demo-app.zip" download>this</a>**_ simple project.

Extract the ZIP.
Go inside the Extracted folder.
Run the following Commands.

!> Make sure Docker is running in the background.

```bash
docker build .
```

The above command will finish and return an ID of an image (A Docker Image)
Now we can create a container out of this image by running the following command.

```bash
docker run -p 3000:3000 <imageID>
```

Now we can visit [localhost:3000](http://localhost:3000)

> The purpose of **-p 3000:3000** is to publish port 3000 onto port 3000.

Now we can stop the container by opening up a new terminal and running the following commands.

```bash
docker ps
```

The above command will list all the running Containers. We can take the name of our container and
run the following command to stop it.

```bash
docker stop <ContainerName>
```

## Conclusion

Docker has transformed software delivery by making apps easy to build, ship, and run anywhere.
Being a developer or DevOps engineer, Docker becomes absolutely essential and extremely useful.
