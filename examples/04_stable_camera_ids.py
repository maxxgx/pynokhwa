"""
Show stable camera IDs and verify that a camera can be opened from its unique_id.

On macOS and Windows, unique_id comes from the platform camera backend. On Linux,
pynokhwa prefers /dev/v4l/by-id and falls back to an index-based ID when no stable
udev symlink is available. The `id_stable` field indicates whether the ID is durable.
"""

import pynokhwa

cameras = pynokhwa.query(only_usable=True)
if not cameras:
    raise SystemExit("No cameras found")

print("Discovered cameras:")
for camera in cameras:
    stability = "stable" if camera.id_stable else "unstable"
    print(f"[{camera.index}] {camera.name}")
    print(f"    unique_id ({stability}): {camera.unique_id}")
    print(f"    description: {camera.description}")
    if camera.misc and camera.misc != camera.unique_id:
        print(f"    backend misc: {camera.misc}")

usable = [camera for camera in cameras if camera.can_open()]
if not usable:
    raise SystemExit("No usable cameras found")

camera_info = usable[0]
print(f"\nOpening camera by unique_id: {camera_info.unique_id}")
camera = pynokhwa.Camera(camera_info.unique_id)

formats = camera.get_format_options()
print(f"Opened {camera_info.name}; found {len(formats)} format options")

if not camera_info.id_stable:
    print(
        "This camera does not expose a durable platform ID. "
        "Its unique_id is only an index fallback and may change."
    )
