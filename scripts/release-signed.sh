#!/usr/bin/env bash
# Builds a release APK signed with the Uno Lab key and writes it, with a SHA256SUMS file, to dist/UnoLab-<version>/.
# Needs UNOLAB_RELEASE_STORE_FILE, UNOLAB_RELEASE_STORE_PASSWORD, UNOLAB_RELEASE_KEY_ALIAS and UNOLAB_RELEASE_KEY_PASSWORD.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd -P)
version=$(sed -n 's/.*versionName = "\(.*\)".*/\1/p' "$root/app/build.gradle.kts")
out="$root/dist/UnoLab-$version"
for name in UNOLAB_RELEASE_STORE_FILE UNOLAB_RELEASE_STORE_PASSWORD UNOLAB_RELEASE_KEY_ALIAS UNOLAB_RELEASE_KEY_PASSWORD; do
    [[ -n "${!name:-}" ]] || { echo "Missing $name" >&2; exit 1; }
done
case "$(cd "$(dirname "$UNOLAB_RELEASE_STORE_FILE")" && pwd -P)" in "$root"*) echo "The keystore must be outside the repository." >&2; exit 1;; esac
[[ ! -e "$out" ]] || { echo "Refusing to replace $out" >&2; exit 1; }
(cd "$root" && ./gradlew -q :app:assembleRelease)
mkdir -p "$out"
cp "$root/app/build/outputs/apk/release/app-release.apk" "$out/UnoLab-$version.apk"
(cd "$out" && shasum -a 256 "UnoLab-$version.apk" > SHA256SUMS.txt)
echo "Built $out"
