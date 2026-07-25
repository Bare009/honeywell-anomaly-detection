# Analyst dashboard image: build the Vite app, serve the static bundle with nginx.
# Build context is the repository root (matches the other services):
#   docker build -f frontend/Dockerfile .

FROM node:20-alpine AS build
WORKDIR /app

# Dependencies first so source edits do not invalidate the install layer.
COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
