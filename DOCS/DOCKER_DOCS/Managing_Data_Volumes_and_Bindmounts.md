## What are Volumes in Docker?

In Docker, containers are **Impermanent** by default, meaning any data inside
a container is lost when it stops or is removed.

To persist data or share data between containers, Docker provides Volumes.
Volumes are the preferred way of handling persistent data in Docker.

### Why Use Volumes?

- Data persistence → Data survives container restarts and removal.
- Data sharing → Multiple containers can access the same volume.
- Isolation → Volumes are stored outside the container filesystem, managed by Docker.
- Performance → More efficient than bind mounts, especially on Docker Desktop (Windows/Mac).

## Types of Volumes

### Named Volumes

Created and managed by Docker.

Example:

```bash
docker volume create mydata
docker run -d -v mydata:/app/data nginx
```

?> Named volumes are not deleted by docker when the container is shut down. They are also not attached to any container. As long as a container is started with the same -v argument, the data persists.

### Anonymous Volumes

Created automatically when we use -v /path without naming the volume.
Harder to manage, as they have random names.

Example:

```bash
docker run -d -v /app/data nginx
```

?> Anonymous volumes are deleted as they are recreated whenever a container is created.

## Basic Commands

- Create a new volume

```bash
docker volume create <name>
```

- List all volumes

```bash
docker volume ls
```

- Inspect details of a volume

```bash
docker volume inspect <name>
```

- Remove a volume

```bash
docker volume rm <name>
```

- Remove all unused volumes

```bash
docker volume prune
```

## Using Volumes in Containers

### Mounting a Named Volume

```bash
docker run --name web -v myvolume:/usr/share/nginx/html nginx
```

Here:

- myvolume → volume name
- /usr/share/nginx/html → container path where it is mounted

## What are Bind Mounts in Docker?

Unlike Named Volumes, which are managed by Docker, bind mounts directly map a directory or file from the host machine into a container.
A Bind Mount is a type of volume where a file or directory on the host machine is directly mounted into a container at a specific path.
This gives the container immediate access to the host filesystem, making bind mounts useful for development scenarios where we want live code or data updates inside the container.

Bind mounts can be created using the -v or --mount option when running a container.

Using -v flag:

```bash
docker run -d -v /host/path:/container/path nginx
```

- /host/path → Path on the host machine
- /container/path → Path inside the container where the host directory is mounted

Using --mount flag (preferred for clarity):

```bash
docker run -d --mount type=bind,source=/host/path,target=/container/path nginx
```

## When to Use Bind Mounts?

Development → Keep source code in sync between host and container for live editing.
Testing → Quickly test changes without rebuilding the image.
Debugging → Mount logs or configuration files from host into a container.

### Example: Mounting Source Code

Suppose we are developing a Node.js app and want to run it inside Docker but edit the code on your host machine.
docker run -it --name node-app -v $(pwd):/usr/src/app -w /usr/src/app node:18 npm start.

Here:

- $(pwd) → current working directory on the host (your code)
- /usr/src/app → container directory where the code is mounted

Any changes made in the local files will be instantly reflected inside the container.

## Security Considerations

- Host access: Bind mounts give containers direct access to your host’s filesystem. Be careful about what you expose.
- Path existence: If the host path doesn’t exist, Docker may create a directory (but with root permissions), which can cause permission issues.
- Portability: Bind mounts rely on absolute paths, making them less portable across environments compared to named volumes.

## Key Differences: Bind Mounts vs Named Volumes

| Feature     | Bind Mounts                       | Named Volumes                                   |
| ----------- | --------------------------------- | ----------------------------------------------- |
| Location    | Specific path on the host machine | Managed by Docker in `/var/lib/docker/volumes/` |
| Portability | Low (depends on host path)        | High (Docker manages paths)                     |
| Use case    | Development, debugging            | Production, persistent storage                  |
| Management  | Host-administered                 | Docker-administered                             |

## Bind mounts Clean Up

Bind mounts don’t show up under docker volume ls because they’re not managed volumes. To stop using them, just stop or remove the container.
The underlying host data remains untouched.

## Conclusion

Bind mounts are a powerful way to share files between your host system and Docker containers. They are best suited for development and debugging but should be used with caution in production due to security and portability concerns.

For production-ready persistent storage, named volumes are generally the safer and more manageable choice.
