## What are Environment Variables in Docker?

Environment variables are one of the most common ways to configure Docker containers.
They allow us to pass configurations, secrets, and runtime settings into containers without hardcoding them into your image.

## Setting Environment Variables

- Using -e (inline)

```markdown
docker run -d \
 -e APP_MODE=production \
 -e DEBUG=false \
 nginx:latest
```

- Using --env-file

```markdown
APP_MODE=production
DEBUG=false
API_KEY=12345
```

## Running container with env file

```bash
docker run -d --env-file .env myapp:latest
```

## Why Use Environment Variables?

- Flexibility → Different configs for dev, test, prod.
- Decoupling → Same image, multiple environments.
- Secrets injection → API keys, DB passwords, tokens.

## Security Considerations

Environment variables can leak in unexpected ways.

- Visible in process lists
- docker inspect and docker ps --format can expose env variables.
- Included in image layers (if misused)
- If you ENV in Dockerfile with secrets → they become permanent.
- Gets stored in logs
- If apps log all env vars on startup, secrets leak to logs.

### Best Practices for Security

#### Dos

- Use --env-file or .env (not hardcoded in Dockerfile).
- Keep .env files out of version control (.gitignore).
- Use Docker secrets (for Swarm) or Kubernetes Secrets (for K8s).
- Limit container access with least privilege.
- Rotate and revoke secrets regularly.

#### Don'ts

- ENV PASSWORD=12345 in your Dockerfile.
- Committing .env files with sensitive data.
- Logging environment variables carelessly.

### Example: Secure vs Insecure

- Insecure Dockerfile

```dockerfile
FROM python:3.12
ENV DB_PASSWORD=supersecret
CMD ["python", "app.py"]
```

Anyone with image access can extract DB_PASSWORD.

- Secure Dockerfile

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
```

Run with:

```bash
docker run -d \
 -e DB_PASSWORD=${DB_PASSWORD} \
 myapp:latest
```
