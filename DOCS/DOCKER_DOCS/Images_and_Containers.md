## What is a Docker Image?

A **Docker Image** is like a blueprint or template for the application.  
It contains everything the application needs to run:

- Source code
- Dependencies
- System libraries
- Configurations

It is basically a **recipe** for creating containers.

?>Example: The official [Python image](https://hub.docker.com/_/python) includes Python, pip, and a base Linux distribution.

---

## What is a Docker Container?

A **Docker Container** is a running instance of an image.

- It is **lightweight** (shares the host OS kernel).
- It is **isolated** (each container has its own filesystem, networking, and process space).
- It can be started, stopped, removed, or moved easily.

?> If an **image is a recipe**, then a **container is the actual dish** prepared from it.

Example command to run a container:

```bash
docker run -it python:3.12
```

The above command launches a container with Python 3.12 inside.

## Using & Running External (Pre-Built) Images

Docker Hub provides thousands of ready-to-use images (databases, programming languages, web servers, etc.).

### Example: Running an NGINX server:

```bash
docker run -d -p 8080:80 nginx
```

?>-d → Detached mode

?>-p 8080:80 → Maps port 8080 on your machine to port 80 inside the container

?>nginx → Official NGINX image from Docker Hub

Now we can open http://localhost:8080 and see NGINX running.

## Building and Running Our Own Image

We can create custom images using a Dockerfile.

### Example: Dockerfile

```dockerfile
# Base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
CMD ["python", "app.py"]
```

Build and run the image:

```bash
docker build -t my-python-app .
docker run -p 5000:5000 my-python-app
```

## EXPOSE & A Little Utility Functionality

The EXPOSE instruction in a Dockerfile tells Docker which port(s) the app will use.

### Example:

```dockerfile
EXPOSE 5000
```

!>This doesn’t publish the port automatically, but it documents that the container expects traffic on port 5000. To actually access it, you still need -p.

### Example run:

```bash
docker run -p 5000:5000 my-python-app
```

## Images are Read-Only

Docker images are immutable, they can not be changed once built.

When a container runs:

- A read-only image layer is used.
- A read-write layer is added on top for runtime changes.
- This makes containers fast to start and ensures consistency across environments.

## Understanding Image Layers

Docker images are built in layers. Each instruction in a Dockerfile (FROM, RUN, COPY, etc.) creates a new layer.

### Benefits:

- Reusability → unchanged layers are cached and reused.
- Efficiency → smaller builds, faster deployments.

### Example:

```dockerfile
# Base layer
FROM ubuntu:20.04
# Layer for system packages
RUN apt-get update
# Another layer
RUN apt-get install -y python3
# Your app code layer
COPY . /app
```

If only the code is changed, then Docker reuses the base and system layers, rebuilding only the last layer, saving time.

## Conclusion

- Image = Blueprint (read-only).
- Container = Running instance of an image.
- External Images = Pulled from Docker Hub.
- Custom Images = Built with a Dockerfile.
- EXPOSE = Declares app ports.
- Layers = Efficient builds and caching.
