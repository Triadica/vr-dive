#!/bin/sh
# Injects the untracked Google Maps API key from the repository-root .env into
# the built app bundle. The key is copied only into the build products
# directory; it is never written into a tracked source file.
set -e

ENV_FILE="${SRCROOT}/.env"
OUT_DIR="${BUILT_PRODUCTS_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
OUT_FILE="${OUT_DIR}/GoogleMaps.env"

if [ -z "${UNLOCALIZED_RESOURCES_FOLDER_PATH}" ]; then
  echo "warning: UNLOCALIZED_RESOURCES_FOLDER_PATH is empty; skipping Google key injection."
  exit 0
fi

mkdir -p "${OUT_DIR}"

if [ -f "${ENV_FILE}" ]; then
  cp "${ENV_FILE}" "${OUT_FILE}"
  echo "note: Injected Google Maps key from .env (untracked) into ${OUT_FILE}."
else
  rm -f "${OUT_FILE}"
  echo "warning: ${ENV_FILE} not found; Google satellite imagery disabled. Open DEM terrain still loads."
fi
