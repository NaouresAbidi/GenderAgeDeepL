#!/bin/bash

echo "🧹 Cleaning up old containers and images..."

# Stop everything
docker-compose down -v

# Remove old images (handle both cases)
docker rmi genderagedeepl-age-gender-api 2>/dev/null || true
docker rmi genderagedeepL-age-gender-api 2>/dev/null || true

# Clean up
docker system prune -f

echo "🔨 Rebuilding with fixed Dockerfile..."

# Rebuild and start
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

echo "🔍 Checking service status:"
docker-compose ps

echo "🌐 Testing endpoints:"
echo "API Health: $(curl -s http://localhost:5000/health 2>/dev/null || echo 'Not ready yet')"
echo "MLflow: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:5001 2>/dev/null || echo 'Not ready yet')"
echo "Grafana: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:3001 2>/dev/null || echo 'Not ready yet')"

echo "✅ Services should be available at:"
echo "   - API: http://localhost:5000"
echo "   - MLflow: http://localhost:5001"  
echo "   - Grafana: http://localhost:3001"
echo "   - Prometheus: http://localhost:9090"