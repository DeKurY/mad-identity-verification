import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class IdentityStore:
    """Supabase-first storage with local JSON fallback for Colab demos."""

    def __init__(self, base_dir: Path, supabase_url: str = "", supabase_key: str = ""):
        self.base_dir = Path(base_dir)
        self.local_path = self.base_dir / "storage" / "local_store.json"
        self._client = None
        self.last_error: Optional[str] = None
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                self._client = create_client(supabase_url, supabase_key)
            except Exception as exc:
                self.last_error = repr(exc)
                self._client = None

    @property
    def using_supabase(self) -> bool:
        return self._client is not None

    def _read_local(self) -> Dict[str, Any]:
        if not self.local_path.exists():
            return {"identity_profiles": [], "verification_attempts": []}
        try:
            data = json.loads(self.local_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.last_error = f"local_store_json_decode_error={exc}"
            return {"identity_profiles": [], "verification_attempts": []}
        data.setdefault("identity_profiles", [])
        data.setdefault("verification_attempts", [])
        return data

    def _write_local(self, data: Dict[str, Any]) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.local_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            tmp_path.chmod(0o600)
        except Exception:
            pass
        tmp_path.replace(self.local_path)
        try:
            self.local_path.chmod(0o600)
        except Exception:
            pass

    def create_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is not None:
            try:
                result = self._client.table("identity_profiles").upsert(
                    profile,
                    on_conflict="passport_series,passport_number",
                ).execute()
                if result.data:
                    return result.data[0]
            except Exception as exc:
                self.last_error = repr(exc)
        data = self._read_local()
        profile = dict(profile)
        existing = None
        for item in data["identity_profiles"]:
            if item.get("passport_series") == profile.get("passport_series") and item.get("passport_number") == profile.get("passport_number"):
                existing = item
                break
        if existing:
            existing.update(profile)
            profile = existing
        else:
            profile.setdefault("id", str(uuid.uuid4()))
            data["identity_profiles"].append(profile)
        self._write_local(data)
        return profile

    def list_profiles(self) -> List[Dict[str, Any]]:
        if self._client is not None:
            try:
                result = self._client.table("identity_profiles").select("*").limit(1000).execute()
                return result.data or []
            except Exception as exc:
                self.last_error = repr(exc)
        return self._read_local().get("identity_profiles", [])

    def save_attempt(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is not None:
            try:
                result = self._client.table("verification_attempts").insert(attempt).execute()
                if result.data:
                    return result.data[0]
            except Exception as exc:
                self.last_error = repr(exc)
        data = self._read_local()
        attempt = dict(attempt)
        attempt.setdefault("id", str(uuid.uuid4()))
        data["verification_attempts"].append(attempt)
        self._write_local(data)
        return attempt
