#!/usr/bin/env python3
"""Start the UI as the unprivileged "app" user.

Volumes created by older (root) images are root-owned, so ownership of the
UI's own data directory is fixed first; then privileges are dropped and the
command is exec'd. Pure Python, so it does not depend on setpriv/gosu being
present in the base image.
"""

import os
import pwd
import sys

DATA_DIR = "/data/kb-ui"

if os.getuid() == 0:
    app = pwd.getpwnam("app")
    os.makedirs(DATA_DIR, exist_ok=True)
    for root, dirs, files in os.walk(DATA_DIR):
        for name in [root, *(os.path.join(root, n) for n in dirs + files)]:
            os.chown(name, app.pw_uid, app.pw_gid)
    os.setgroups([])
    os.setgid(app.pw_gid)
    os.setuid(app.pw_uid)
    os.environ["HOME"] = app.pw_dir

os.execvp(sys.argv[1], sys.argv[1:])
