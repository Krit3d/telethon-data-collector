#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(collabrama.ru www.collabrama.ru api.collabrama.ru crm.collabrama.ru admin.collabrama.ru)
PRIMARY_DOMAIN="collabrama.ru"
COMPOSE_FILE="docker-compose.gateway.yml"
CERT_FILE="./certbot/conf/live/${PRIMARY_DOMAIN}/fullchain.pem"
EMAIL="${EMAIL:-}"

if [ -f "${CERT_FILE}" ]; then
  echo "Certificates already exist"
  exit 0
fi

docker compose -f "${COMPOSE_FILE}" down

mkdir -p ./certbot/conf ./certbot/www

DOMAIN_ARGS=()
for domain in "${DOMAINS[@]}"; do
  DOMAIN_ARGS+=(-d "${domain}")
done

if [ -n "${EMAIL}" ]; then
  EMAIL_FLAG="--email ${EMAIL}"
else
  EMAIL_FLAG="--register-unsafely-without-email"
fi

docker run --rm \
  -p 80:80 \
  -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
  -v "$(pwd)/certbot/www:/var/www/certbot" \
  certbot/certbot:latest certonly \
  --standalone \
  --preferred-challenges http \
  "${DOMAIN_ARGS[@]}" \
  --agree-tos \
  --non-interactive \
  ${EMAIL_FLAG}

echo "Certificates successfully issued for ${DOMAINS[*]}"
