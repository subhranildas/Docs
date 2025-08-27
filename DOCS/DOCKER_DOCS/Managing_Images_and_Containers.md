## Creating an Image

A **Docker image** is a packaged blueprint of an application.

### Build from Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

Build the image:

```bash
docker build -t myapp:1.0 .
```

-t myapp:1.0 → names the image myapp with tag 1.0.

### Save a Container as Image

```bash
docker commit <container_id> myapp:snapshot
```

## Creating a Container

Containers are running instances of images.

```bash
docker run -it --name mycontainer ubuntu:20.04
```

-it → interactive terminal.
--name → custom name.
ubuntu:20.04 → base image.

Run in the background (detached mode):

```bash
docker run -d --name webserver -p 8080:80 nginx
```

## Stopping & Restarting Containers

Stop a container:

```bash
docker stop <container_id_or_name>
```

## Start a stopped container:

Restart (stop + start):

```bash
docker start <container_id_or_name>
```

## Understanding Attached & Detached Containers

Attached (default) → container output is shown in your terminal.

```bash
docker run -it ubuntu
```

Detached → runs in the background.

```bash
docker run -d nginx
```

Use docker logs <container> to see output from detached containers.

## Attaching to an Already-Running Container

Attach to a background container:

```bash
docker attach <container_id_or_name>
```

?>Exiting may stop the container (depends on how it was started).

## Entering Interactive Mode

Use exec to open a shell in a running container (preferred):

```bash
docker exec -it <container_id_or_name> /bin/bash
```

or,

```bash
docker exec -it <container_id_or_name> sh
```

?>This is safer than attach since it won’t disrupt the main process.

## Attached and Detached Containers

Attached → useful for debugging, interactive apps.
Detached → best for long-running services.

### Check container state:

List all the containers and their states:

```bash
docker ps -a
```

List the running containers:

```bash
docker ps
```

## Deleting Images and Containers

Remove a container:

```bash
docker rm <container_id_or_name>
```

Remove an image:

```bash
docker rmi <image_id_or_name>
```

If container still uses the image, stop & remove it first.

### Removing Containers Automatically

Run with --rm:

```bash
docker run --rm ubuntu echo "hello world"
```

Container is removed after it exits.

## Image Inspection

See metadata (layers, env, configs):

```bash
docker inspect <image_id_or_name>
```

Example:

```bash
docker inspect nginx
```

## Copying Files Into and From a Container

Copy file to container:

```bash
docker cp myfile.txt <container_id_or_name>:/app/
```

Copy file from container:

```bash
docker cp <container_id_or_name>:/app/output.log ./output.log
```

## Naming & Tagging Images and Containers

Tagging an image:

```bash
docker build -t myusername/myapp:2.0 .
```

Naming Containers while running:

```bash
docker run -it --name <container-name> <ImageID>
```

Renaming a container:

```bash
docker rename oldname newname
```

Tags are essential for versioning and sharing.

## Image Sharing

Docker Hub (or private registries) allows sharing images.
Docker Hub is a cloud-based registry where we can store and share container images.

### Login to Docker Hub

Before pushing or pulling private images, you need to authenticate:

```bash
docker login
```

It will prompt for:

- Username → your Docker Hub username
- Password / Personal Access Token (recommended)

```bash
docker info
```

> Username: `<username>` → will be visible if logged in.

### Tagging an Image for Docker Hub

Docker Hub requires the image to be named in the following format:

```md
<dockerhub-username>/<repository-name>:<tag>
```

For example, if the Docker Hub username is subhranil and we want to push an nginx-based image,
we can run the following command to do it.

```bash
docker tag nginx:latest subhranil/my-nginx:1.0
```

Here:

- subhranil → Docker Hub username
- my-nginx → repository name
- 1.0 → version tag

?>If no version tag is given, Docker defaults to :latest.

### Pushing an Image to Docker Hub

After tagging we can push it:

```bash
docker push subhranil/my-nginx:1.0
```

If successful, upload progress will be visible and the image will be available in the Docker Hub repository of the account owner.

### Pulling an Image from Docker Hub

Anyone (if the repo is public) or authenticated users (if private) can pull an image using the following command:

```bash
docker pull subhranil/my-nginx:1.0
```

We can then run it using the following command:

```bash
docker run -d -p 8080:80 subhranil/my-nginx:1.0
```

### Working with Private Repositories

If the repository is private, one must be logged in with docker login first.
If logged out, pulling will fail with a “denied: requested access to the resource is denied” error.

### Logging Out

If one is on a shared machine, log out after pushing/pulling can be done using the following command:

```bash
docker logout
```

### Quick Workflow Example

- Step 1: Login

```bash
docker login
```

- Step 2: Build your image

```bash
docker build -t myapp:latest .
```

- Step 3: Tag it for Docker Hub

```bash
docker tag myapp:latest subhranil/myapp:latest
```

- Step 4: Push it

```bash
docker push subhranil/myapp:latest
```

- Step 5: (On another machine) Pull it

```bash
docker pull subhranil/myapp:latest
docker run -d -p 8080:80 subhranil/myapp:latest
```
