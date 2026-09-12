# World map regression checks

Run `scripts/check-world-map.sh` on macOS with Xcode installed. It compiles the
production map selection, navigation, networking and tile code into a temporary
executable. All HTTP requests are intercepted by a URLProtocol fixture; no API
key, internet access or live map requests are used.

Checks cover 80 geographic/clearance/movement combinations (including 0°,0° and
high latitudes), coverage holes, 2:1 neighbors, non-overlapping leaves, the 160-leaf
and estimated memory budgets, direction mapping, monotonic camera-centred LOD while
descending, atomic parent/child imagery replacement, speed changes, stereo frusta,
retry backoff and request cancellation. With a Metal device, it also checks complete
and partial satellite composites, invalid image dimensions, recovery, completed
mipmaps, texture replacement without mesh rebuilding, and terrain height
interpolation. Metal checks explicitly report SKIP when the environment cannot
expose a device.

The 256 MiB limit accounts for tile buffers/textures and pending uploads, with a
232 MiB conservative selected working set. It does not measure driver allocations,
render targets, or resources retained by previously submitted frames. Device GPU
frame time, total process memory, visual LOD transitions and head-motion comfort
still require a Vision Pro run.
