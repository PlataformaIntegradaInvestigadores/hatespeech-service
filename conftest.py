import os

# Los tests corren siempre con el backend mock y una base SQLite en memoria,
# sin GPU ni Postgres. Se fija antes de importar la app.
os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("JWT_SIGNING_KEY", "test-signing-key")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key")
