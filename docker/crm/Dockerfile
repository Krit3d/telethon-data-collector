FROM node:20-alpine AS builder

WORKDIR /app

COPY src/web/crm/package.json src/web/crm/package-lock.json* ./

RUN npm ci

COPY src/web/crm/ ./

ARG VITE_API_URL=/api/v1
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:alpine

COPY docker/crm/nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]