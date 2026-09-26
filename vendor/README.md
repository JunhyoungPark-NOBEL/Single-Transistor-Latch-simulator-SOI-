# Bundled Python installer

This directory contains the unmodified official CPython 3.13.15 Windows x64 installer.

- File: `python-3.13.15-amd64.exe`
- Official source: https://www.python.org/ftp/python/3.13.15/python-3.13.15-amd64.exe
- Release / published checksum: https://www.python.org/downloads/release/python-31315/
- Release date: 2026-08-05
- Bytes: 29,452,944
- SHA-256: `edec09c4853aeae9ac36efb8c9f95b6b8e2fee65eee56d9767a8b7c69c574403`
- Python license: `PYTHON-LICENSE.txt`; the installer includes its upstream licenses.

Biristor Studio's Windows bootstrap validates the pinned checksum and Windows Authenticode signature before executing the installer. It first searches for an existing compatible Python. Installation uses current-user options and an application-specific target directory. It is a registered CPython installation, not a portable or registry-free runtime. It does not alter PATH or add the shared Python launcher.

The Python installer is included; scientific and server packages are downloaded by pip on first setup. This is not a fully offline application package.

Installer options reference: https://docs.python.org/3.13/using/windows.html#installing-without-ui

Python and its third-party components retain their own licenses; the application repository's copyright statement does not replace these licenses.
