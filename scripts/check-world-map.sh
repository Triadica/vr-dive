#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/world-map-tests.XXXXXX")
trap 'rm -rf "$TEST_DIR"' EXIT
xcrun swiftc -O -module-cache-path "$TEST_DIR/modules" \
  "$ROOT/vr-dive/Demos/WorldMap/MapTileMath.swift" \
  "$ROOT/vr-dive/Demos/WorldMap/MapStreamingPolicy.swift" \
  "$ROOT/vr-dive/Demos/WorldMap/MapNetwork.swift" \
  "$ROOT/vr-dive/Demos/WorldMap/WorldMapTile.swift" \
  "$ROOT/scripts/world-map-tests/NetworkTests.swift" \
  "$ROOT/scripts/world-map-tests/main.swift" -o "$TEST_DIR/check-world-map"
"$TEST_DIR/check-world-map"
