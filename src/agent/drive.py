from typing import Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from .config import Settings, now_ist

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def _creds(cfg: Settings) -> Credentials:
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(cfg.google_token_path, SCOPES)
    except Exception:
        pass
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(cfg.google_credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(cfg.google_token_path, "w") as f:
            f.write(creds.to_json())
    return creds

def _svc(cfg: Settings):
    return build("drive", "v3", credentials=_creds(cfg))

def _find_folder(svc, name: str, parent_id: Optional[str]) -> Optional[str]:
    # Escape single quotes for Drive query syntax
    name_escaped = name.replace("'", "\\'")
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name = '{name_escaped}' and trashed = false"
    )
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = svc.files().list(q=q, fields="files(id,name)", pageSize=10).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _ensure_folder(svc, name: str, parent_id: Optional[str]) -> str:
    found = _find_folder(svc, name, parent_id)
    if found:
        return found
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    f = svc.files().create(body=meta, fields="id").execute()
    return f["id"]

def ensure_hierarchy(cfg: Settings) -> str:
    svc = _svc(cfg)
    top = _ensure_folder(svc, cfg.top_folder_name, cfg.root_drive_parent_id)
    dname = now_ist().strftime("%d%m%Y")
    d = _ensure_folder(svc, dname, top)
    hour = str(now_ist().hour + 1)  # 1..24
    return _ensure_folder(svc, hour, d)

def upload_docx(cfg: Settings, folder_id: str, local_path: str, filename: str):
    service = _svc(cfg)
    media = MediaFileUpload(local_path, mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document", resumable=True)
    meta = {"name": filename, "parents":[folder_id]}
    return service.files().create(body=meta, media_body=media, fields="id,webViewLink,name").execute()