import os
import sys

def get_token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not tok:
        # repli sur secrets_local.py (git-ignore) si disponible
        try:
            import secrets_local
            tok = getattr(secrets_local, "MAPILLARY_TOKEN", "").strip()
        except ImportError:
            tok = ""
    if not tok:
        print(
            "ERREUR : variable d'environnement MAPILLARY_TOKEN absente.\n"
            "  Cree un token sur https://www.mapillary.com/dashboard/developers\n"
            '  puis : export MAPILLARY_TOKEN="MLY|xxxxx|yyyyy"',
            file=sys.stderr,
        )
        sys.exit(2)
    return tok


def slug(city: str, postal: str | None) -> str:
    base = city.lower().strip().replace(" ", "_")
    return f"{base}_{postal}" if postal else base
