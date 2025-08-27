## Docker Build Arguments (ARG)

Docker provides build-time variables using the ARG instruction.
These allows us to pass temporary configuration values into the Docker build process.

Unlike environment variables (ENV), build arguments are not available at runtime (inside running containers).

## Defining ARG in Dockerfile

### Define a build argument with no default

ARG APP_VERSION

### Define with default value

ARG BASE_IMAGE=ubuntu:22.04

```dockerfile
FROM ${BASE_IMAGE}
RUN echo "Building version: ${APP_VERSION}"
```

### Passing ARG at Build Time

```dockerfile
docker build \
 --build-arg APP_VERSION=1.2.3 \
 --build-arg BASE_IMAGE=alpine:3.20 \
 -t myapp:1.2.3 .
```

## Default vs Overridden

If no value is passed, Docker uses the default defined in Dockerfile.
If overridden with --build-arg, that takes priority.

Example:

```dockerfile
ARG PORT=8080
RUN echo "Default port: $PORT"
```

```bash
docker build --build-arg PORT=5000 -t custom .
```

Output:

```bash
Default port: 5000
```

## Scope of ARG

ARG before FROM → available only for FROM.
ARG after FROM → available in subsequent instructions.

ENV is needed if we want the variable to be available at runtime.

Example:

```dockerfile
ARG BASE=alpine
FROM ${BASE}
ARG AUTHOR
ENV AUTHOR=${AUTHOR}
```

## Common Use Cases

Switching Base Images

```dockerfile
ARG BASE=alpine
FROM ${BASE}
```

Specifying Versions

```dockerfile
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}
```

Build Metadata

```dockerfile
ARG GIT_COMMIT
LABEL git_commit=${GIT_COMMIT}
```

Proxies & Package Mirrors

```dockerfile
ARG HTTP_PROXY
ARG HTTPS_PROXY
```

## Security Considerations

ARG values do not persist in the final image (unless copied into ENV or files).
They can still be seen in build history:

```bash
docker history myapp:latest
```

- Never pass secrets (API keys, passwords) as build arguments.
- Use Docker secrets, environment variables, or external secret managers instead.

## Example: Combining ARG and ENV

```dockerfile
FROM python:3.12
# Build-time argument
ARG APP_VERSION=1.0
# Runtime environment variable
ENV APP_VERSION=${APP_VERSION}
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
```

Build:

```bash
docker build --build-arg APP_VERSION=2.0 -t myapp:2.0 .
```

Run:

```bash
docker run myapp:2.0
Inside container:
echo $APP_VERSION
```
