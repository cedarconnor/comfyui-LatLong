import os
import urllib.request

# Download Three.js assets required by the interactive panorama viewer.
#
# This mirrors the install behavior of:
# https://github.com/ProGamerGov/ComfyUI_preview360panorama
#
# It downloads into: <this_extension>/js/lib/

JS_FILES = [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.172.0/three.core.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.172.0/three.module.min.js",
]


def main() -> None:
    extension_path = os.path.dirname(__file__)
    js_lib_path = os.path.join(extension_path, "js", "lib")
    os.makedirs(js_lib_path, exist_ok=True)

    for url in JS_FILES:
        file_name = os.path.basename(url)
        file_path = os.path.join(js_lib_path, file_name)
        urllib.request.urlretrieve(url, file_path)


if __name__ == "__main__":
    main()

