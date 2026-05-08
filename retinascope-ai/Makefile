.PHONY: backend frontend validate

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && pnpm dev

validate:
	cd backend && python ../scripts/validate_checkpoints.py

