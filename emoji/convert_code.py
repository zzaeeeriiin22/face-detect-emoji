import cv2
import base64
from pathlib import Path

current_dir = Path(__file__).parent
image_files = sorted(current_dir.glob("*.png"))

EMOJI_WIDTH = 160
EMOJI_HEIGHT = 160

points_table = {
    "surprise": [(56, 62), (104, 62), (80, 134)],
    "happy": [(56, 62), (104, 62), (80, 134)],
    "neutral": [(56, 62), (104, 62), (80, 134)],
    "angry": [(56, 62), (104, 62), (80, 134)],
}

# Generate C++ header file
with open("emoji.h", "w") as f:
    f.write("#ifndef EMOJI_H\n")
    f.write("#define EMOJI_H\n\n")

    f.write("#include \"geometry.h\"\n")

    f.write("namespace emoji {\n")

    for image in image_files:
        image_bgr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if image_bgr.shape[2] == 4:
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
        else:
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

        image_rgba = cv2.resize(image_rgba, (EMOJI_WIDTH, EMOJI_HEIGHT))
        pixels = image_rgba.flatten()

        pxiel_array = "{\n" + ", ".join([f"0x{pixel:02x}" for pixel in pixels]) + "}"    
        f.write(f"const unsigned char {image.stem.upper()}_EMOJI[] = {pxiel_array};\n")

        points_array = "{\n" + ", ".join([f"geo::Point{{{x}.f, {y}.f}}" for x, y in points_table[image.stem.lower()]]) + "\n}"
        f.write(f"const std::vector<geo::Point> {image.stem.upper()}_POINTS = {points_array};\n")

    f.write(f"const int EMOJI_WIDTH = {EMOJI_WIDTH};\n")
    f.write(f"const int EMOJI_HEIGHT = {EMOJI_HEIGHT};\n")
    f.write("}\n")
    f.write("\n#endif // EMOJI_H\n")

# Generate JavaScript module
js_output_path = current_dir.parent / "js" / "emoji_data.js"
with open(js_output_path, "w") as f:
    f.write("// Auto-generated emoji data for JavaScript backend\n\n")
    
    # Store emoji data as ImageData-compatible arrays
    emoji_data = {}
    
    for image in image_files:
        image_bgr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if image_bgr.shape[2] == 4:
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
        else:
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

        image_rgba = cv2.resize(image_rgba, (EMOJI_WIDTH, EMOJI_HEIGHT))
        
        emoji_name = image.stem.upper()
        emoji_data[emoji_name] = {
            'pixels': image_rgba.flatten().tolist(),
            'points': points_table[image.stem.lower()]
        }
    
    # Write emoji pixel data
    for name, data in emoji_data.items():
        f.write(f"export const {name}_EMOJI = [\n")
        pixels = data['pixels']
        # Write in chunks for readability
        chunk_size = 16
        for i in range(0, len(pixels), chunk_size):
            chunk = pixels[i:i+chunk_size]
            f.write("    " + ", ".join(str(p) for p in chunk))
            if i + chunk_size < len(pixels):
                f.write(",\n")
            else:
                f.write("\n")
        f.write("];\n\n")
    
    # Write emoji points
    for name, data in emoji_data.items():
        points = data['points']
        points_str = ", ".join([f"{{x: {x}, y: {y}}}" for x, y in points])
        f.write(f"export const {name}_POINTS = [{points_str}];\n")
    
    f.write(f"\nexport const EMOJI_WIDTH = {EMOJI_WIDTH};\n")
    f.write(f"export const EMOJI_HEIGHT = {EMOJI_HEIGHT};\n")
    
    # Export emotion to emoji mapping
    f.write("\n// Mapping from emotion class names to emoji data\n")
    f.write("export const EMOTION_TO_EMOJI = {\n")
    for name in emoji_data.keys():
        f.write(f"    '{name.lower()}': {{\n")
        f.write(f"        pixels: {name}_EMOJI,\n")
        f.write(f"        points: {name}_POINTS\n")
        f.write("    },\n")
    f.write("};\n")

print("Generated emoji.h and js/emoji_data.js successfully!")
