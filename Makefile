.PHONY: test lint run up

test:
	PROVIDER_MODE=fake ROUTER_API_KEY=ci-only-placeholder DB_URL=sqlite:// pytest -q

lint:
	ruff check

run:
	uvicorn app.api:app --reload

up:
	docker compose up --build
