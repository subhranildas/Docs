## What is the .dockerignore File in Docker?

When building Docker images, we often don’t want everything from our project directory to be copied into the image.
This is where the .dockerignore file comes in.

It works very much like a .gitignore file.

- Defines patterns for files and directories to exclude from the Docker build context.
- Keeps the image smaller, build faster, and more secure.

## Purpose of .dockerignore

- Reduce Image Size
- Prevents unnecessary files (logs, caches, build artifacts) from being copied.
- Speed Up Builds
- Less data sent to the Docker daemon → faster build times.
- Improve Security
- Excludes sensitive files (API keys, configs, .env files) from being baked into images.
- Avoid Unwanted Layer Changes
- Changing temporary/local files won’t invalidate Docker’s build cache.

## Syntax Rules

Each line is a pattern (like .gitignore).

Supports:

```yaml
# → comments
- → wildcards
! → negation (include something that was excluded)
```

## Example .dockerignore

```text
# Ignore node_modules (too big, will rebuild inside container)

node_modules

# Ignore logs and temp files

_.log
tmp/
_.tmp

# Ignore Git data

.git
.gitignore

# Ignore Docker-related files themselves

Dockerfile
.dockerignore

# Ignore environment configs (security)

.env
```

## Use Case: Node.js Application

Imagine a Node.js web app with thousands of files in node_modules.
If you run this Dockerfile:

```dockerfile
FROM node:18
WORKDIR /usr/src/app
COPY . .
RUN npm install
CMD ["node", "server.js"]
```

Without .dockerignore, Docker sends everything (including node_modules, .git/, logs).

- Build context is huge.
- Image is bloated.

Security risk if .env is copied.
With .dockerignore Only the essential source code and configs get copied.
Build is smaller, faster, and safer.

## Typical Patterns to Ignore

- node_modules, **pycache**, target/, bin/ (language-specific build dirs)

- .git, .svn, .hg (VCS metadata)

- _.log, _.tmp, \*.bak (temporary files)

- .env, _.key, _.pem (secrets and keys)

- Local config files not needed in container
